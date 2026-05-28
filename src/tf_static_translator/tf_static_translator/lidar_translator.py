# 라이다 앞쪽에 왠지 모르게 dummy값들 나타나는데
# 이거 제거하고 다시 발행하는 코드

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_system_default, DurabilityPolicy

class Translator(Node):
    def __init__(self,
                 name="lidar", msg="scan",
                 tl_msg="ad/scan"):
        
        super().__init__(name+"translator")

        self.sub = self.create_subscription(
            LaserScan, msg, self.tl_callback,
            qos_profile_system_default
        )

        pub_qos_profile = qos_profile_system_default
        pub_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(
            LaserScan, tl_msg, pub_qos_profile
        )
    
    def tl_callback(self, data):
        # if rclpy.ok:
        data.ranges = [r if r > 0.9 else float('inf') for r in data.ranges]
        self.pub.publish(data)
        
def main(args=None):
    rclpy.init(args=args)
    node = Translator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()