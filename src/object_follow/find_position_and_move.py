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

        # PID Control Variables
        self.last_linear_error = 0.0
        self.integral_linear_error = 0.0
        self.last_angular_error = 0.0
        self.integral_angular_error = 0.0

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers and Subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/camera/color/camera_info', 
            self.camera_info_callback, 
            10
        )

        # Time Synchronizer for Image and Detection topics
        detection_sub = message_filters.Subscriber(self, Detection2DArray, '/yolo/detections', qos_profile=qos_profile)
        # Note: We still subscribe to color image for visualization, but Depth is less critical for pure visual servoing unless we want minimal distance checks.
        # But we keep it simple: sync RGB. (Removing depth sync to simplify or keeping it?)
        # Keeping structure similar but removing depth logic dependency for movement.
        color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos_profile)
        
        # Keep depth subscription slightly loosely to avoid breaking if it's there, but we won't use it for control.
        # To be safe, let's keep the sync as 3 if avail, but since user had issues, maybe just 2?
        # User's launch file provides all 3. Let's keep 3 but handle depth optionally / just read it.
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
            
            # If no detections, stop the robot
            if not detection_msg.detections:
                stop_twist = Twist()
                self.cmd_vel_publisher.publish(stop_twist)
                self.integral_linear_error = 0.0
                self.integral_angular_error = 0.0
                
                # Still show the image
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                    cv2.imshow("Find Position and Move", cv_image)
                    cv2.waitKey(1)
                except Exception:
                    pass
                return
            
            try:
                color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                # depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1') # Kept for potential use
            except Exception as e:
                self.get_logger().error(f'Failed to convert images: {e}')
                return
            
            # Camera intrinsics
            # cx, cy are the principal points (center of image usually)
            cx, cy = self.K_matrix[0, 2], self.K_matrix[1, 2]
            
            # Assume first detection is target
            detection = detection_msg.detections[0]
            bbox = detection.bbox
            
            img_h, img_w = color_image.shape[:2]
            
            # Helper to get pixel coordinates
            # bbox.center.position.x is likely normalized 0-1, OR already pixel.
            # In previous view_file of yolo_node.py (Turn 26), lines 42, 46:
            # bbox.center.position.x = float(x_center) where x_center comes from box.xywhn[0] (Normalized).
            # So find_position_and_move logic `u = int(bbox.center.position.x * depth_image.shape[1])` was correct.
            
            u = int(bbox.center.position.x * img_w)
            v = int(bbox.center.position.y * img_h)

            # CONTROL LOGIC (Pixel Based)
            # Goal: Keep (u, v) at (cx, cy)
            
            # 1. Angular Control (Horizontal)
            # If u < cx (Left), we want to turn Left (Positive Z).
            # Error = cx - u
            # NOTE: User feedback "Move opposite" suggests my previous "Positive Y -> Left -> Turn Left" was wrong OR 
            # maybe it was right but they wanted reverse behavior. 
            # Let's stick to standard Visual Servoing: 
            # Pixel Error > 0 (Target is Left of Center) -> Turn Left (+Z).
            
            angular_error = (cx - u) / float(img_w) # Normalize error -0.5 to 0.5 roughly
            
            # 2. Linear Control (Vertical)
            # If v < cy (Top/Far), we want to move Forward (Positive X).
            # Error = cy - v
            
            linear_error = (cy - v) / float(img_h) # Normalize error

            # Dead Zones
            linear_dead_zone = 0.05  # 5% of screen height
            angular_dead_zone = 0.05 # 5% of screen width

            # PID Gains (Tuned for normalized error)
            lin_p, lin_i, lin_d = 0.5, 0.0, 0.05
            ang_p, ang_i, ang_d = 1.0, 0.0, 0.1

            linear_error_diff = linear_error - self.last_linear_error
            angular_error_diff = angular_error - self.last_angular_error

            twist_msg = Twist()

            # Apply Logic
            if abs(linear_error) > linear_dead_zone:
                self.integral_linear_error += linear_error
                p = lin_p * linear_error
                i = lin_i * self.integral_linear_error
                d = lin_d * linear_error_diff
                twist_msg.linear.x = p + i + d
                twist_msg.linear.x = max(min(twist_msg.linear.x, 0.2), -0.2) # Limit linear speed
            else:
                self.integral_linear_error = 0.0
                twist_msg.linear.x = 0.0

            if abs(angular_error) > angular_dead_zone:
                self.integral_angular_error += angular_error
                p = ang_p * angular_error
                i = ang_i * self.integral_angular_error
                d = ang_d * angular_error_diff
                twist_msg.angular.z = p + i + d
                twist_msg.angular.z = max(min(twist_msg.angular.z, 0.5), -0.5) # Limit angular speed
            else:
                self.integral_angular_error = 0.0
                twist_msg.angular.z = 0.0

            self.cmd_vel_publisher.publish(twist_msg)

            self.last_linear_error = linear_error
            self.last_angular_error = angular_error

            self.get_logger().info(f'Pixel Error: Lin={linear_error:.3f}, Ang={angular_error:.3f} | Cmd: lin={twist_msg.linear.x:.2f}, ang={twist_msg.angular.z:.2f}')

            # Visualization
            w_px = int(bbox.size_x * img_w)
            h_px = int(bbox.size_y * img_h)
            x1 = u - w_px // 2
            y1 = v - h_px // 2
            x2 = u + w_px // 2
            y2 = v + h_px // 2

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
            # Draw Center
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
