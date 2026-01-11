import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # ROS 이미지 메시지와 OpenCV 이미지 사이의 변환을 도와줌
from ultralytics import YOLO
import cv2

pt_path='../weights/yolov8n.pt'

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_test_node')

        self.model=YOLO(pt_path)
        
        self.bridge=CvBridge()
        # ROS2의 표준 이미지 메시지 형식은 opencv가 바로 처리할 수 없기 때문에 이 둘 사이의 데이터 형식을 변환해주는 역할

        qos_profile=QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, # 신뢰성보다 최신 데이터 전송을 우선 (한두 프레임 정도 유실되어도 괜찮지만, 지연이 없어야 하는 경우), (기본값은 RELIABLE로 모든 메시지를 반드시 전송해야 할 때 사용)
            history=QoSHistoryPolicy.KEEP_LAST, depth=1 # 버퍼에 몇 개의 메시지를 저장할지 설정 (여기서는 가장 최신 메시지 1개만 유지)
        )

        self.subscription=self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,10)
        self.publisher=self.create_publisher(Image,'/yolo/annotated_image',10)

    def image_callback(self,msg):
        try:
            # ROS 메시지 -> opencv 이미지 변환
            cv_image=self.bridge.imgmsg_to_cv2(msg,'bgr8')

            # YOLO 추론 수행
            results=self.model(cv_image,verbose=False,half=True,imgsz=640) # half=True로 FP16을 사용하여 모델의 정밀도를 약간 낮추고 추론 속도를 크게 높임

            # 결과 시각화
            annotated_frame=results[0].plot()

            # opencv 이미지 -> ROS 메시지 변환 및 퍼블리시
            annotated_msg=self.bridge.cv2_to_imgmsg(annotated_frame,'bgr8')
            annotated_msg.header.stamp=self.get_clock().now().to_msg()
            self.publisher.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    yolo_test_node=YoloNode()
    rclpy.spin(yolo_test_node)
    yolo_test_node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()