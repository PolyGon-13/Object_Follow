import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import message_filters
import traceback
import cv2
import numpy as np

class FindPositionAndMoveNode(Node):
    def __init__(self):
        super().__init__('find_position_and_move_node')
        self.bridge = CvBridge()

        self.K_matrix = None

        self.last_linear_error = 0.0
        self.integral_linear_error = 0.0
        self.last_angular_error = 0.0
        self.integral_angular_error = 0.0

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/camera/color/camera_info', 
            self.camera_info_callback, 
            10
        )

        detection_sub = message_filters.Subscriber(self, Detection2DArray, '/yolo/detections', qos_profile=qos_profile)
        color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos_profile)
        depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile)

        self.ts = message_filters.TimeSynchronizer([detection_sub, depth_sub, color_sub], 30)
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("find_position_and_move_node has been started (Pixel-based Control).")

    def camera_info_callback(self, msg):
        if self.K_matrix is None:
            self.K_matrix = np.array(msg.k).reshape((3, 3))
            self.destroy_subscription(self.camera_info_sub)
    
    def synced_callback(self, detection_msg, depth_msg, color_msg):
        try:
            if self.K_matrix is None:
                self.get_logger().warn('Camera intrinsics not available yet.')
                return
            
            if not detection_msg.detections:
                stop_twist = Twist()
                self.cmd_vel_publisher.publish(stop_twist)
                self.integral_linear_error = 0.0
                self.integral_angular_error = 0.0
                
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                    cv2.imshow("Find Position and Move", cv_image)
                    cv2.waitKey(1)
                except Exception:
                    pass
                return
            
            try:
                color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Failed to convert images: {e}')
                return
            
            cx, cy = self.K_matrix[0, 2], self.K_matrix[1, 2]
            
            detection = detection_msg.detections[0]
            bbox = detection.bbox
            
            img_h, img_w = color_image.shape[:2]
            
            u = int(bbox.center.position.x * img_w)
            v = int(bbox.center.position.y * img_h)

            angular_error = (cx - u) / float(img_w)
            linear_error = (cy - v) / float(img_h)

            linear_dead_zone = 0.05
            angular_dead_zone = 0.05

            lin_p, lin_i, lin_d = 0.5, 0.0, 0.05
            ang_p, ang_i, ang_d = 1.0, 0.0, 0.1

            linear_error_diff = linear_error - self.last_linear_error
            angular_error_diff = angular_error - self.last_angular_error

            twist_msg = Twist()

            if abs(linear_error) > linear_dead_zone:
                self.integral_linear_error += linear_error
                p = lin_p * linear_error
                i = lin_i * self.integral_linear_error
                d = lin_d * linear_error_diff
                twist_msg.linear.x = p + i + d
                twist_msg.linear.x = max(min(twist_msg.linear.x, 0.2), -0.2)
            else:
                self.integral_linear_error = 0.0
                twist_msg.linear.x = 0.0

            if abs(angular_error) > angular_dead_zone:
                self.integral_angular_error += angular_error
                p = ang_p * angular_error
                i = ang_i * self.integral_angular_error
                d = ang_d * angular_error_diff
                twist_msg.angular.z = p + i + d
                twist_msg.angular.z = max(min(twist_msg.angular.z, 0.5), -0.5)
            else:
                self.integral_angular_error = 0.0
                twist_msg.angular.z = 0.0

            self.cmd_vel_publisher.publish(twist_msg)

            self.last_linear_error = linear_error
            self.last_angular_error = angular_error

            self.get_logger().info(f'Pixel Error: Lin={linear_error:.3f}, Ang={angular_error:.3f} | Cmd: lin={twist_msg.linear.x:.2f}, ang={twist_msg.angular.z:.2f}')

            w_px = int(bbox.size_x * img_w)
            h_px = int(bbox.size_y * img_h)
            x1 = u - w_px // 2
            y1 = v - h_px // 2
            x2 = u + w_px // 2
            y2 = v + h_px // 2

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
            cv2.circle(color_image, (int(cx), int(cy)), 5, (255, 0, 0), -1)
            
            cv2.imshow("Find Position and Move", color_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Critical error in synced_callback: {e}')
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = FindPositionAndMoveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_twist = Twist()
        node.cmd_vel_publisher.publish(stop_twist)
        node.get_logger().info('Shutting down and stopping robot.')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

