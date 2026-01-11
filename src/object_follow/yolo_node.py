import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray,Detection2D,BoundingBox2D,ObjectHypothesisWithPose
from ultralytics import YOLO
from cv_bridge import CvBridge
import os

pt_path='weights/yolov8n.pt'
engine_path='weights/yolov8n.engine'

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        if not os.path.exists(engine_path):
            self.get_logger().info(f'TensorRT engine not found. Creating a new one from {pt_path}')
            model_builder=YOLO(pt_path)
            model_builder.export(format='engine',half=True,imgsz=640,device=0)
            
        self.model=YOLO(engine_path,task='detect')
        self.bridge=CvBridge()

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.subscription=self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,qos_profile)
        self.detection_publisher=self.create_publisher(Detection2DArray,'/yolo/detections',qos_profile)

        self.get_logger().info('yolo_node has been started.')

    def image_callback(self,msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
            results=self.model(cv_image,verbose=False,conf=0.20) 

            detections_msg=Detection2DArray() 
            detections_msg.header=msg.header 
            for box in results[0].boxes:
                if int(box.cls)==49:
                    detection=Detection2D()
                    x_center,y_center,width,height=box.xywhn[0]

                    bbox=BoundingBox2D()
                    
                    bbox.center.position.x=float(x_center)
                    bbox.center.position.y=float(y_center)
                    bbox.size_x=float(width)
                    bbox.size_y=float(height)
                    detection.bbox=bbox

                    hypothesis=ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id=str(int(box.cls))
                    hypothesis.hypothesis.score=float(box.conf)
                    detection.results.append(hypothesis)

                    detections_msg.detections.append(detection)

            self.detection_publisher.publish(detections_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    yolo_node=YoloDetectionNode()

    try:
        rclpy.spin(yolo_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()