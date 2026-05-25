import rclpy, time, cv2, os, math
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge

class TrackDriverNode(Node):

    def __init__(self):
        super().__init__("driver")
        self.get_logger().info("----- Hello World -----")

        self.image = None
        self.motor_msg = XycarMotor()
        self.lidar_images = None
        self.bridge = CvBridge()

        self.motor_pub = self.create_publisher(XycarMotor,'xycar_motor',10)