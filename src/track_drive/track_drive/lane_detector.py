import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class CamViewerNode(Node):
    def __init__(self):
        super().__init__('cam_viewer')

        self.bridge = CvBridge()

        self.image = None

        # Subscribers
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.img_callback, qos_profile_sensor_data)

        # Timer (30 FPS)
        self.timer = self.create_timer(0.03, self.process_images)

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def process_images(self):

        if any(v is None for v in self.image.values()):
            return
            
        h, w = 240, 320

        f = cv2.resize(self.image, (w, h))

        cv2.imshow("Front View", f)
        cv2.waitKey(1)

class 

def main(args=None):
    rclpy.init(args=args)
    node = CamViewerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
