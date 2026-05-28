```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# ROS2 XYCar Advanced Lane Driving
#
# 기능
# 1. 2차선 중앙 유지
# 2. 오른쪽 차선 우선 유지
# 3. 커브 조기 인식
# 4. Look Ahead
# 5. 조향 스무딩
# 6. 커브에서 조향 강화
# 7. 커브 감속
# 8. 차선 하나만 보여도 유지
# =============================================

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from xycar_msgs.msg import Motor

WIDTH = 640
HEIGHT = 480

ROI_Y = 220
ROI_HEIGHT = 220

prev_angle = 0

class LaneDriver(Node):

    def __init__(self):

        super().__init__('lane_driver')

        self.motor_pub = self.create_publisher(
            Motor,
            'xycar_motor',
            10
        )

        self.cap = cv2.VideoCapture(0)

        self.timer = self.create_timer(
            0.03,
            self.run
        )

    # =========================================
    # 차량 제어
    # =========================================
    def drive(self, angle, speed):

        msg = Motor()

        msg.angle = float(angle)
        msg.speed = float(speed)

        self.motor_pub.publish(msg)

    # =========================================
    # 차선 검출
    # =========================================
    def detect_lane(self, frame):

        global prev_angle

        # ROI
        roi = frame[ROI_Y:ROI_Y + ROI_HEIGHT, :]

        # Gray
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # Blur
        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Canny
        edges = cv2.Canny(
            blur,
            50,
            150
        )

        # Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            30,
            minLineLength=40,
            maxLineGap=20
        )

        left_lines = []
        right_lines = []

        if lines is not None:

            for line in lines:

                x1, y1, x2, y2 = line[0]

                # 기울기 계산
                if x2 - x1 == 0:
                    continue

                slope = (y2 - y1) / (x2 - x1)

                # 너무 수평인 선 제거
                if abs(slope) < 0.3:
                    continue

                # 왼쪽 차선
                if slope < 0 and x2 < WIDTH // 2:
                    left_lines.append(line[0])

                # 오른쪽 차선
                elif slope > 0 and x1 > WIDTH // 2:
                    right_lines.append(line[0])

        left_x = None
        right_x = None

        # =====================================
        # 왼쪽 차선 평균
        # =====================================
        if len(left_lines) > 0:

            x_sum = 0

            for line in left_lines:

                x1, y1, x2, y2 = line

                x_sum += (x1 + x2)

                cv2.line(
                    roi,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

            left_x = x_sum // (len(left_lines) * 2)

        # =====================================
        # 오른쪽 차선 평균
        # =====================================
        if len(right_lines) > 0:

            x_sum = 0

            for line in right_lines:

                x1, y1, x2, y2 = line

                x_sum += (x1 + x2)

                cv2.line(
                    roi,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

            right_x = x_sum // (len(right_lines) * 2)

        # =====================================
        # 차선 중앙 계산
        # =====================================
        if left_x is not None and right_x is not None:

            lane_center = (left_x + right_x) // 2

        # 오른쪽 차선만 보일 경우
        elif right_x is not None:

            lane_center = right_x - 220

        # 왼쪽 차선만 보일 경우
        elif left_x is not None:

            lane_center = left_x + 220

        else:

            # 차선이 안 보이면 이전 조향 유지
            return prev_angle, 15

        # =====================================
        # Look Ahead 기반 오차 계산
        # =====================================
        image_center = WIDTH // 2

        error = lane_center - image_center

        # =====================================
        # 커브 감지
        # =====================================
        curve_strength = abs(error)

        # =====================================
        # 동적 gain 조절
        # =====================================
        kp = 0.25

        if curve_strength > 60:
            kp = 0.4

        if curve_strength > 100:
            kp = 0.55

        if curve_strength > 140:
            kp = 0.7

        # 조향 계산
        angle = kp * error

        # =====================================
        # Steering Clamp
        # =====================================
        angle = max(min(angle, 50), -50)

        # =====================================
        # Steering Smoothing
        # =====================================
        angle = (
            prev_angle * 0.7
            + angle * 0.3
        )

        prev_angle = angle

        # =====================================
        # 속도 제어
        # =====================================
        speed = 30

        if abs(angle) > 20:
            speed = 24

        if abs(angle) > 35:
            speed = 20

        if abs(angle) > 45:
            speed = 16

        # =====================================
        # 디버깅 화면
        # =====================================

        # 화면 중심
        cv2.line(
            roi,
            (image_center, 0),
            (image_center, ROI_HEIGHT),
            (0, 0, 255),
            2
        )

        # 차선 중심
        cv2.circle(
            roi,
            (lane_center, ROI_HEIGHT // 2),
            8,
            (0, 255, 255),
            -1
        )

        # 텍스트 출력
        cv2.putText(
            roi,
            f"Angle : {angle:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            roi,
            f"Speed : {speed}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Lane ROI", roi)
        cv2.imshow("Edges", edges)

        cv2.waitKey(1)

        return angle, speed

    # =========================================
    # 메인 루프
    # =========================================
    def run(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT)
        )

        angle, speed = self.detect_lane(frame)

        self.drive(angle, speed)

# =============================================
# main
# =============================================
def main(args=None):

    rclpy.init(args=args)

    node = LaneDriver()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
