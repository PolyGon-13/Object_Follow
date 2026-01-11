import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
from vision_msgs.msg import Detection2DArray,Detection2D,BoundingBox2D,ObjectHypothesisWithPose
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

pt_path='weights/yolov8n.pt'
engine_path='weights/yolov8n.engine'

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        if not os.path.exists(engine_path):
            model_builder=YOLO(pt_path)
            model_builder.export(format='engine',half=True,imgsz=640,device=0)

        self.model=YOLO(engine_path,task='detect')
        self.bridge=CvBridge()
        self.tracker=DeepSort(max_age=90)

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.subscription=self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,qos_profile)
        self.detection_publisher=self.create_publisher(Detection2DArray,'/yolo/detections',qos_profile)

    def image_callback(self,msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
            H,W,_=cv_image.shape

            results=self.model(cv_image,verbose=False,conf=0.40)

            detections_for_deepsort=[]
            for box in results[0].boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                w,h=x2-x1,y2-y1
                confidence=float(box.conf)
                class_id=int(box.cls)
                detections_for_deepsort.append(([x1,y1,w,h],confidence,class_id))

            tracks=self.tracker.update_tracks(detections_for_deepsort,frame=cv_image)

            detections_msg=Detection2DArray()
            detections_msg.header=msg.header

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id=track.track_id
                class_id=track.det_class

                if class_id==49:
                    detection=Detection2D()
                    detection.id=str(track_id)

                    ltrb=track.to_ltrb()
                    x1,y1,x2,y2=map(int,ltrb)

                    w,h=x2-x1,y2-y1
                    x_center=(x1+w/2)/W
                    y_center=(y1+h/2)/H
                    width_norm=float(w)/W
                    height_norm=float(h)/H

                    bbox=BoundingBox2D()
                    bbox.center.position.x=x_center
                    bbox.center.position.y=y_center
                    bbox.size_x=width_norm
                    bbox.size_y=height_norm
                    detection.bbox=bbox

                    hypothesis=ObjectHypothesisWithPose()
                    if class_id is not None:
                        hypothesis.hypothesis.class_id=str(class_id)
                    else:
                        hypothesis.hypothesis.class_id="unknown"
                    if track.det_conf is not None:
                        hypothesis.hypothesis.score=float(track.det_conf)
                    else:
                        hypothesis.hypothesis.score=0.0
                    detection.results.append(hypothesis)
                    detections_msg.detections.append(detection)

            if len(detections_msg.detections)>0:
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
        #cv2.destroyAllWindows()

if __name__=='__main__':
    main()
