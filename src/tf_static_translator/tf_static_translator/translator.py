import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_system_default, DurabilityPolicy

class Translator(Node):
    def __init__(self,
                 name="tf_static", msg="/tf_static",
                 tl_msg="/tf_static"):
        
        super().__init__(name+"translator")

        self.sub = self.create_subscription(
            TFMessage, msg, self.tl_callback,
            qos_profile_system_default
        )

        pub_qos_profile = qos_profile_system_default
        pub_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(
            TFMessage, tl_msg, pub_qos_profile
        )
    
    def tl_callback(self, data):
        # if rclpy.ok:
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