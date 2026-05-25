#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import cv2
import time
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class TrackDriverNode(Node):

    # =========================================
    # 초기화
    # =========================================
    def __init__(self):

        super().__init__('driver')

        self.image = None
        self.lidar_ranges = None

        self.bridge = CvBridge()

        self.motor_msg = XycarMotor()

        # steering 저장
        self.prev_angle = 0

        # PID
        self.prev_error = 0
        self.integral = 0

        # Publisher
        self.motor_pub = self.create_publisher(
            XycarMotor,
            'xycar_motor',
            10
        )

        # Camera Subscriber
        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        # LiDAR Subscriber
        self.sub_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("===== START =====")

    # =========================================
    # 카메라 콜백
    # =========================================
    def cam_callback(self, data):

        self.image = self.bridge.imgmsg_to_cv2(
            data,
            "bgr8"
        )

    # =========================================
    # 라이다 콜백
    # =========================================
    def lidar_callback(self, msg):

        self.lidar_ranges = msg.ranges

    # =========================================
    # 차량 제어
    # =========================================
    def drive(self, angle, speed):

        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)

        self.motor_pub.publish(self.motor_msg)

    # =========================================
    # 차선 주행
    # =========================================
    def lane_driving(self, frame):

        height, width, _ = frame.shape

        # =====================================
        # ROI
        # =====================================
        roi = frame[
            int(height * 0.55):height,
            :
        ]

        roi_h, roi_w, _ = roi.shape

        # =====================================
        # HSV 변환
        # =====================================
        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        # =====================================
        # 흰색 차선 범위
        # =====================================
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 90, 255])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # =====================================
        # morphology
        # =====================================
        kernel = np.ones((5, 5), np.uint8)

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # =====================================
        # Blur
        # =====================================
        blur = cv2.GaussianBlur(
            white_mask,
            (5, 5),
            0
        )

        # =====================================
        # Canny
        # =====================================
        edges = cv2.Canny(
            blur,
            50,
            150
        )

        # =====================================
        # Hough
        # =====================================
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            30,
            minLineLength=40,
            maxLineGap=50
        )

        # =====================================
        # 차선 없으면 이전값 유지
        # =====================================
        if lines is None:

            return self.prev_angle

        left_lines = []

        # =====================================
        # 왼쪽 흰 실선만 사용
        # =====================================
        for line in lines:

            x1, y1, x2, y2 = line[0]

            # 수직선 방지
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # 수평 제거
            if abs(slope) < 0.3:
                continue

            # 화면 왼쪽만 사용
            avg_x = (x1 + x2) // 2

            if avg_x > roi_w // 2:
                continue

            # 아래쪽 점 사용
            if y1 > y2:
                x_bottom = x1
            else:
                x_bottom = x2

            left_lines.append(x_bottom)

            cv2.line(
                roi,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

        # =====================================
        # 차선 못 찾으면 이전값 유지
        # =====================================
        if len(left_lines) == 0:

            return self.prev_angle

        # =====================================
        # 왼쪽 차선 위치
        # =====================================
        left_lane = int(np.mean(left_lines))

        # =====================================
        # 차선 중앙 offset
        # =====================================
        offset = 170

        lane_center = left_lane + offset

        # =====================================
        # 차량 목표 중앙
        # =====================================
        target = roi_w // 2

        # =====================================
        # 오차 계산
        # =====================================
        error = target - lane_center

        # =====================================
        # PID
        # =====================================
        kp = 0.45
        kd = 0.15
        ki = 0.0005

        self.integral += error

        derivative = error - self.prev_error

        angle = (
            kp * error +
            kd * derivative +
            ki * self.integral
        )

        self.prev_error = error

        # =====================================
        # smoothing
        # =====================================
        angle = (
            0.6 * self.prev_angle +
            0.4 * angle
        )

        self.prev_angle = angle

        # =====================================
        # steering 제한
        # =====================================
        angle = max(min(angle, 50), -50)

        # =====================================
        # 디버그
        # =====================================
        cv2.circle(
            roi,
            (left_lane, roi_h - 20),
            8,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            roi,
            (lane_center, roi_h - 40),
            10,
            (0, 0, 255),
            -1
        )

        cv2.line(
            roi,
            (target, 0),
            (target, roi_h),
            (255, 0, 0),
            2
        )

        cv2.imshow("white_mask", white_mask)
        cv2.imshow("edges", edges)
        cv2.imshow("lane_roi", roi)

        print("left_lane :", left_lane)
        print("lane_center :", lane_center)
        print("error :", error)
        print("angle :", angle)

        return angle

    # =========================================
    # 메인 루프
    # =========================================
    def main_loop(self):

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.01
            )

            if self.image is None:
                continue

            frame = self.image.copy()

            angle = self.lane_driving(frame)

            # 속도 조절
            speed = 4

            if abs(angle) > 20:
                speed = 3

            if abs(angle) > 35:
                speed = 2

            self.drive(angle, speed)

            cv2.imshow("camera", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # =========================================
    # 종료
    # =========================================
    def destroy(self):

        self.drive(0, 0)

        cv2.destroyAllWindows()

        self.destroy_node()


# =============================================
# main 함수
# =============================================
def main(args=None):

    rclpy.init(args=args)

    node = TrackDriverNode()

    try:

        node.main_loop()

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy()

        rclpy.shutdown()


if __name__ == '__main__':

    main()