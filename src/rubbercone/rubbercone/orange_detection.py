#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
주황색 라바콘 감지 노드 (ROS2)
- 카메라 이미지에서 주황색 감지
- /is_orange 토픽으로 발행
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class OrangeDetection(Node):
    def __init__(self):
        super().__init__('orange_detection')

        # 구독
        self.subscription = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',   # 앞 카메라
            self.camera_cb,
            10
        )

        # 발행
        self.is_orange_pub = self.create_publisher(Bool, '/is_orange', 10)

        self.bridge = CvBridge()
        self.cv_image = None

        # 30Hz 루프
        self.timer = self.create_timer(1/30, self.timer_cb)

        self.get_logger().info('OrangeDetection node started')

    def camera_cb(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(str(e))

    def timer_cb(self):
        if self.cv_image is None:
            return

        self.detect_orange(self.cv_image)

    def detect_orange(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 주황색 HSV 범위
        lower_orange = np.array([0,   80,  80])
        upper_orange = np.array([25,  255, 200])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        pixel_count = np.count_nonzero(mask)

        msg = Bool()
        msg.data = pixel_count > 10000   # 픽셀 10000개 이상이면 주황색 있음
        self.is_orange_pub.publish(msg)

        # self.get_logger().info(f'orange pixels: {pixel_count}')


def main(args=None):
    rclpy.init(args=args)
    node = OrangeDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
