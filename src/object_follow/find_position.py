import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from vision_msgs.msg import Detection2DArray

class FindPositionNode(Node):
    def __init__(self):
        super().__init__('find_position_node')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self.detection_callback,
            qos_profile
        )
        
        self.get_logger().info('Find_Position node has been started.')

    def detection_callback(self, msg):
        if not msg.detections:
            return

        # Assuming the first detection is the target
        detection = msg.detections[0]
        bbox = detection.bbox
        
        # Center x, y are normalized coordinates (0.0 to 1.0)
        x = bbox.center.position.x
        y = bbox.center.position.y

        # Determine quadrant
        # (0,0) is usually Top-Left in image coordinates
        # x increases to right, y increases to bottom
        
        quadrant = 0
        
        if x < 0.5:
            # Left side
            if y < 0.5:
                quadrant = 1 # Top-Left
            else:
                quadrant = 3 # Bottom-Left
        else:
            # Right side
            if y < 0.5:
                quadrant = 2 # Top-Right
            else:
                quadrant = 4 # Bottom-Right

        self.get_logger().info(f'Ball detected in Quadrant: {quadrant} (x={x:.2f}, y={y:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = FindPositionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
