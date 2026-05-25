#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# Xycar ROS2 자율주행
#
# 핵심 수정 사항
# 1. 차선 "선분 평균" 방식 제거
# 2. 화면 하단 기준 실제 차선 위치 사용
# 3. 차선 유지 PID 강화
# 4. 곡선에서도 차선 유지 가능
#
# 트랙 구조
# [흰 실선] 1차선 [노란 점선] 2차선 [흰 실선]
#
# 차량은:
# 왼쪽 흰 실선 + 중앙 노란 점선 사이
# (= 1차선)
# 를 유지하도록 설계
# =============================================

# hi

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

        self.prev_angle = 0

        # =====================================
        # PID
        # =====================================
        self.prev_error = 0
        self.integral = 0

        # =====================================
        # Publisher
        # =====================================
        self.motor_pub = self.create_publisher(
            XycarMotor,
            'xycar_motor',
            10
        )

        # =====================================
        # Camera Subscriber
        # =====================================
        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        # =====================================
        # LiDAR Subscriber
        # =====================================
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
        # 화면 아래만 사용
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
        # 흰 차선
        # =====================================
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 70, 255])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # =====================================
        # 노란 점선
        # =====================================
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        # =====================================
        # Blur
        # =====================================
        white_mask = cv2.GaussianBlur(
            white_mask,
            (5, 5),
            0
        )

        yellow_mask = cv2.GaussianBlur(
            yellow_mask,
            (5, 5),
            0
        )

        # =====================================
        # Canny
        # =====================================
        white_edges = cv2.Canny(
            white_mask,
            50,
            150
        )

        yellow_edges = cv2.Canny(
            yellow_mask,
            50,
            150
        )

        # =====================================
        # Hough
        # =====================================
        white_lines = cv2.HoughLinesP(
            white_edges,
            1,
            np.pi / 180,
            30,
            minLineLength=30,
            maxLineGap=40
        )

        yellow_lines = cv2.HoughLinesP(
            yellow_edges,
            1,
            np.pi / 180,
            15,
            minLineLength=15,
            maxLineGap=60
        )

        # =====================================
        # 차선 없으면 이전값 유지
        # =====================================
        if white_lines is None or yellow_lines is None:

            return self.prev_angle

        # =====================================
        # 실제 차선 위치 찾기
        # 화면 하단 기준
        # =====================================
        left_positions = []
        right_positions = []

        # =====================================
        # 흰 실선
        # 왼쪽 차선
        # =====================================
        for line in white_lines:

            x1, y1, x2, y2 = line[0]

            slope = 999

            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.2:
                continue

            # 아래쪽 점 사용
            if y1 > y2:
                x_bottom = x1
            else:
                x_bottom = x2

            left_positions.append(x_bottom)

            cv2.line(
                roi,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                3
            )

        # =====================================
        # 노란 점선
        # 오른쪽 차선
        # =====================================
        for line in yellow_lines:

            x1, y1, x2, y2 = line[0]

            slope = 999

            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.2:
                continue

            # 아래쪽 점 사용
            if y1 > y2:
                x_bottom = x1
            else:
                x_bottom = x2

            right_positions.append(x_bottom)

            cv2.line(
                roi,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

        # =====================================
        # 차선 부족
        # =====================================
        if len(left_positions) == 0:

            return self.prev_angle

        if len(right_positions) == 0:

            return self.prev_angle

        # =====================================
        # 실제 차선 위치
        # =====================================
        left_lane = int(np.mean(left_positions))
        right_lane = int(np.mean(right_positions))

        # =====================================
        # 1차선 중앙
        # =====================================
        lane_center = (left_lane + right_lane) // 2

        # =====================================
        # 차량 목표 중앙
        # =====================================
        target = roi_w // 2

        # =====================================
        # 오차
        # =====================================
        error = target - lane_center

        # =====================================
        # PID
        # =====================================
        kp = 0.40
        kd = 0.12
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
            0.7 * self.prev_angle +
            0.3 * angle
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
            (right_lane, roi_h - 20),
            8,
            (0, 255, 255),
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
        cv2.imshow("yellow_mask", yellow_mask)
        cv2.imshow("lane_roi", roi)

        print("left_lane :", left_lane)
        print("right_lane:", right_lane)
        print("lane_center:", lane_center)
        print("error:", error)
        print("angle:", angle)

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

            speed = 4

            # 커브 감속
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
# main
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