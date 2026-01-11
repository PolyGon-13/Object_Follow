import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy

class CoordinateCalculatorNode(Node):
    def __init__(self):
        super().__init__('coordinate_calculator_node')
        self.bridge=CvBridge()
        self.K_matrix=None

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.camera_info_sub=self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.camera_info_callback,10)

        # 2개의 서로 다른 토픽을 동시에 구독하기 위해 message_filters.Subscriber 생성
        self.detection_sub=message_filters.Subscriber(self,Detection2DArray,'/yolo/detections',qos_profile=qos_profile)
        self.depth_sub=message_filters.Subscriber(self,Image,'/camera/camera/aligned_depth_to_color/image_raw',qos_profile=qos_profile)

        # 2개의 Subscriber를 시간 동기화 장치에 등록 -> 타임스탬프가 거의 동일한 메시지 2개가 한 세트로 도착할 때까지 기다렸다가, 두 메시지를 묶어서 콜백함수로 전달
        self.ts=message_filters.TimeSynchronizer([self.detection_sub,self.depth_sub],30)
        self.ts.registerCallback(self.synced_callback)

    # 카메라 고유정보(내부 파라미터) 한 번만 수신
    # msg.k에는 [fx,0,cx,0,fy,cy,0,0,1] 형태의 9개까지 리스트 존재
    def camera_info_callback(self,msg):
        if self.K_matrix is None:
            self.K_matrix=np.array(msg.k).reshape((3,3))
            self.destroy_subscription(self.camera_info_sub)

    def synced_callback(self,detection_msg,depth_msg):
        if self.K_matrix is None:
            self.get_logger().warn('Camera intrinsics not available yet. Skipping frame.')
            return
        
        try:
            depth_image=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return
        
        fx,fy=self.K_matrix[0,0],self.K_matrix[1,1] # 초점거리 (3D 공간의 물체가 2D 이미지 센서에 얼마나 크거나 작게 맺히는지를 결정하는 배율)
        cx,cy=self.K_matrix[0,2],self.K_matrix[1,2] # 주점 (2D 이미지 픽셀 좌표계의 진짜 원점이 어딘지 알려줌)

        for detection in detection_msg.detections:
            bbox=detection.bbox
            x_norm=bbox.center.position.x
            y_norm=bbox.center.position.y

            u=int(x_norm*depth_image.shape[1]) # shape[1]에는 이미지의 너비 값
            # 전체길이가 128이고 x가 0.75이면 원하는 위치는 75% 지점인 960 
            v=int(y_norm*depth_image.shape[0]) # shape[0]에는 이미지의 높이 값

            depth_mm=depth_image[v,u]
            # 리얼센스는 거리를 측정할 수 없는 픽셀(너무 멀거나, 빛이 반사되는 표면)의 값을 0으로 반환
            if depth_mm==0:
                self.get_logger().warn(f'Depth at ({u},{v}) is zero. Skipping frame.')
                continue

            Z=float(depth_mm)/1000.0 # 리얼센스는 거리값을 mm 단위로 알려줌 
            # 역투영
            X=(u-cx)*Z/fx
            Y=(v-cy)*Z/fy

            self.get_logger().info(f'Orange 3D Coordinates (m) : [X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}]')

def main(args=None):
    rclpy.init(args=args)
    node=CoordinateCalculatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()