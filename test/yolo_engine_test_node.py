import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

engine_path='../weights/yolov8n.engine'

class YoloEngineNode(Node):
    def __init__(self):
        super().__init__('yolo_engine_test_node')

        if not os.path.exists(engine_path):
            temp_model=YOLO(pt_path)
            temp_model.export(format='engine',half=True,imgsz=640,device=0)

        self.model=YOLO(engine_path,task='detect')
        self.bridge=CvBridge()

        qos_profile=QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription=self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,qos_profile)
        self.publisher=self.create_publisher(Image,'/yolo/annotated_image',10)

    def image_callback(self,msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(msg,'bgr8')

            results=self.model(cv_image,verbose=False,half=True,imgsz=640)

            annotated_frame=results[0].plot()

            annotated_msg=self.bridge.cv2_to_imgmsg(annotated_frame,'bgr8')
            annotated_msg.header.stamp=self.get_clock().now().to_msg()
            self.publisher.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    yolo_test_node=YoloEngineNode()
    rclpy.spin(yolo_test_node)
    yolo_test_node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()