import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RealsenseDepthNode(Node):
    def __init__(self):
        super().__init__('realsense_depth_node')
        self.bridge=CvBridge()

        self.subscription=self.create_subscription(Image,'/camera/camera/depth/image_rect_raw',self.depth_image_callback,10)

    def depth_image_callback(self,msg):
        try:
            depth_image=self.bridge.imgmsg_to_cv2(msg,desired_encoding='16UC1')

            depth_colormap=cv2.convertScaleAbs(depth_image,alpha=0.15)

            cv2.imshow("Depth_Image",depth_colormap)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Failed to process depth image : {e}')

def main(args=None):
    rclpy.init(args=args)
    node=RealsenseDepthNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()