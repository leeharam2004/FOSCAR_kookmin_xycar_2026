#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# ROS2 Xycar Sliding Window Lane Driving
#
# 수정 내용
# 1. 2차선 유지
# 2. 오른쪽 실선 기준 주행
# 3. 중앙 노란 점선 무시
# 4. 커브 조향 강화
# 5. steering smoothing 감소
# 6. 커브 대응 향상
# =============================================

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


# =============================================
# Sliding Window
# =============================================
class SlideWindow:

    def __init__(self):

        self.x_previous = 480

    def slidewindow(self, img):

        out_img = np.dstack((img, img, img))

        height = img.shape[0]
        width = img.shape[1]

        nonzero = img.nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # =====================================
        # 오른쪽 절반만 사용
        # =====================================
        right_region = img[
            height - 80:height,
            width // 2:
        ]

        histogram = np.sum(
            right_region,
            axis=0
        )

        # 시작점
        rightx_current = np.argmax(histogram) + width // 2

        # window 설정
        nwindows = 10
        window_height = 15

        margin = 45
        minpix = 20

        right_lane_inds = []

        # =====================================
        # Sliding Window
        # =====================================
        for window in range(nwindows):

            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            win_x_low = rightx_current - margin
            win_x_high = rightx_current + margin

            cv2.rectangle(
                out_img,
                (win_x_low, win_y_low),
                (win_x_high, win_y_high),
                (255, 0, 0),
                2
            )

            good_inds = (
                (
                    (nonzeroy >= win_y_low) &
                    (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_low) &
                    (nonzerox < win_x_high)
                ).nonzero()[0]
            )

            right_lane_inds.append(good_inds)

            # 다음 window 위치
            if len(good_inds) > minpix:

                rightx_current = int(
                    np.mean(nonzerox[good_inds])
                )

        # concatenate
        right_lane_inds = np.concatenate(right_lane_inds)

        # =====================================
        # 차선 못 찾으면 이전값 유지
        # =====================================
        if len(right_lane_inds) == 0:

            x_location = self.x_previous

            return out_img, x_location

        # =====================================
        # 오른쪽 실선 x
        # =====================================
        rightx = np.max(
            nonzerox[right_lane_inds]
        )

        # =====================================
        # 2차선 중앙 계산
        #
        # 오른쪽 실선 기준
        # 왼쪽으로 offset
        # =====================================
        lane_width_offset = 170

        x_location = int(
            rightx - lane_width_offset
        )

        self.x_previous = x_location

        # 디버그 표시
        cv2.circle(
            out_img,
            (rightx, height - 30),
            10,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            out_img,
            (x_location, height - 60),
            10,
            (0, 0, 255),
            -1
        )

        return out_img, x_location


# =============================================
# ROS2 Node
# =============================================
class TrackDriverNode(Node):

    def __init__(self):

        super().__init__('driver')

        self.image = None

        self.bridge = CvBridge()

        self.slidewindow = SlideWindow()

        self.motor_msg = XycarMotor()

        # PID
        self.prev_error = 0
        self.integral = 0

        # steering smoothing
        self.prev_angle = 0

        # publisher
        self.motor_pub = self.create_publisher(
            XycarMotor,
            'xycar_motor',
            10
        )

        # subscriber
        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            "===== 2nd Lane Driving Start ====="
        )

    # =========================================
    # camera callback
    # =========================================
    def cam_callback(self, data):

        self.image = self.bridge.imgmsg_to_cv2(
            data,
            "bgr8"
        )

    # =========================================
    # drive
    # =========================================
    def drive(self, angle, speed):

        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)

        self.motor_pub.publish(
            self.motor_msg
        )

    # =========================================
    # Bird Eye View
    # =========================================
    def bird_eye_view(self, frame):

        height, width = frame.shape[:2]

        src = np.float32([
            [220, 320],
            [420, 320],
            [100, 480],
            [540, 480]
        ])

        dst = np.float32([
            [180, 0],
            [460, 0],
            [180, 480],
            [460, 480]
        ])

        matrix = cv2.getPerspectiveTransform(
            src,
            dst
        )

        warped = cv2.warpPerspective(
            frame,
            matrix,
            (width, height)
        )

        return warped

    # =========================================
    # preprocessing
    # =========================================
    def preprocessing(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # 흰색
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 90, 255])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # 노란색
        lower_yellow = np.array([10, 70, 70])
        upper_yellow = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        # merge
        mask = cv2.bitwise_or(
            white_mask,
            yellow_mask
        )

        # morphology
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        mask = cv2.GaussianBlur(
            mask,
            (5, 5),
            0
        )

        return mask

    # =========================================
    # lane driving
    # =========================================
    def lane_driving(self, frame):

        # Bird Eye
        bev = self.bird_eye_view(frame)

        # binary
        binary = self.preprocessing(bev)

        # sliding window
        out_img, lane_center = (
            self.slidewindow.slidewindow(binary)
        )

        # 목표 중앙
        target = 320

        # 오차
        error = target - lane_center

        # =====================================
        # PID 강화
        # =====================================
        kp = 0.75
        kd = 0.25
        ki = 0.0003

        self.integral += error

        derivative = error - self.prev_error

        angle = (
            kp * error +
            kd * derivative +
            ki * self.integral
        )

        self.prev_error = error

        # =====================================
        # smoothing 감소
        # =====================================
        angle = (
            0.3 * self.prev_angle +
            0.7 * angle
        )

        self.prev_angle = angle

        # =====================================
        # steering 제한 증가
        # =====================================
        angle = max(min(angle, 70), -70)

        # debug
        cv2.line(
            out_img,
            (320, 0),
            (320, 480),
            (0, 255, 0),
            2
        )

        cv2.imshow("bev", bev)
        cv2.imshow("binary", binary)
        cv2.imshow("sliding_window", out_img)

        print("lane_center:", lane_center)
        print("error:", error)
        print("angle:", angle)

        return angle

    # =========================================
    # main loop
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

            # =================================
            # 속도 제어
            # =================================
            speed = 4

            if abs(angle) > 20:
                speed = 3

            if abs(angle) > 40:
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