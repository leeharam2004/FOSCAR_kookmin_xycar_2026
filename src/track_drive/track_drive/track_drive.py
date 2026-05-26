#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# ROS2 Xycar Dual Lane Driving
#
# 핵심 개선
# 1. 왼쪽/오른쪽 차선 동시 추적
# 2. 중앙 노란 점선 무시
# 3. 양쪽 실선 기반 차선 중앙 계산
# 4. 커브 안정성 향상
# 5. 직진 흔들림 감소
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

        self.prev_center = 320

    def slidewindow(self, img):

        out_img = np.dstack((img, img, img))

        height = img.shape[0]
        width = img.shape[1]

        nonzero = img.nonzero()

        nonzeroy = np.array(nonzero[0])

        nonzerox = np.array(nonzero[1])

        # =====================================
        # 좌우 영역 분리
        # =====================================
        left_region = img[
            height - 120:height,
            :width // 2
        ]

        right_region = img[
            height - 120:height,
            width // 2:width
        ]

        # histogram
        left_histogram = np.sum(
            left_region,
            axis=0
        )

        right_histogram = np.sum(
            right_region,
            axis=0
        )

        # 시작점
        leftx_current = np.argmax(
            left_histogram
        )

        rightx_current = (
            np.argmax(right_histogram)
            + width // 2
        )

        # =====================================
        # sliding window parameter
        # =====================================
        nwindows = 12

        window_height = int(
            height / nwindows
        )

        margin = 70

        minpix = 25

        left_lane_inds = []

        right_lane_inds = []

        # =====================================
        # sliding windows
        # =====================================
        for window in range(nwindows):

            win_y_low = (
                height - (window + 1) * window_height
            )

            win_y_high = (
                height - window * window_height
            )

            # left window
            win_xleft_low = (
                leftx_current - margin
            )

            win_xleft_high = (
                leftx_current + margin
            )

            # right window
            win_xright_low = (
                rightx_current - margin
            )

            win_xright_high = (
                rightx_current + margin
            )

            # draw left
            cv2.rectangle(
                out_img,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (255, 0, 0),
                2
            )

            # draw right
            cv2.rectangle(
                out_img,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0),
                2
            )

            # left lane pixel
            good_left_inds = (
                (
                    (nonzeroy >= win_y_low) &
                    (nonzeroy < win_y_high) &
                    (nonzerox >= win_xleft_low) &
                    (nonzerox < win_xleft_high)
                ).nonzero()[0]
            )

            # right lane pixel
            good_right_inds = (
                (
                    (nonzeroy >= win_y_low) &
                    (nonzeroy < win_y_high) &
                    (nonzerox >= win_xright_low) &
                    (nonzerox < win_xright_high)
                ).nonzero()[0]
            )

            left_lane_inds.append(
                good_left_inds
            )

            right_lane_inds.append(
                good_right_inds
            )

            # 다음 left window
            if len(good_left_inds) > minpix:

                leftx_current = int(
                    np.mean(
                        nonzerox[good_left_inds]
                    )
                )

            # 다음 right window
            if len(good_right_inds) > minpix:

                rightx_current = int(
                    np.mean(
                        nonzerox[good_right_inds]
                    )
                )

        # concatenate
        left_lane_inds = np.concatenate(
            left_lane_inds
        )

        right_lane_inds = np.concatenate(
            right_lane_inds
        )

        # =====================================
        # 양쪽 차선 모두 못 찾음
        # =====================================
        if (
            len(left_lane_inds) == 0 and
            len(right_lane_inds) == 0
        ):

            return out_img, self.prev_center

        # =====================================
        # 양쪽 차선 평균
        # =====================================
        if len(left_lane_inds) > 0:

            leftx = int(
                np.mean(
                    nonzerox[left_lane_inds]
                )
            )

        else:

            leftx = None

        if len(right_lane_inds) > 0:

            rightx = int(
                np.mean(
                    nonzerox[right_lane_inds]
                )
            )

        else:

            rightx = None

        # =====================================
        # 차선 중앙 계산
        # =====================================
        if leftx is not None and rightx is not None:

            lane_center = int(
                (leftx + rightx) / 2
            )

        elif leftx is not None:

            lane_center = leftx + 240

        elif rightx is not None:

            lane_center = rightx - 240

        else:

            lane_center = self.prev_center

        self.prev_center = lane_center

        # =====================================
        # debug
        # =====================================
        if leftx is not None:

            cv2.circle(
                out_img,
                (leftx, height - 20),
                10,
                (255, 255, 0),
                -1
            )

        if rightx is not None:

            cv2.circle(
                out_img,
                (rightx, height - 20),
                10,
                (0, 255, 255),
                -1
            )

        cv2.circle(
            out_img,
            (lane_center, height - 40),
            10,
            (0, 0, 255),
            -1
        )

        return out_img, lane_center


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
            "===== Dual Lane Driving Start ====="
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
    # preprocessing
    # =========================================
    def preprocessing(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # white
        lower_white = np.array([
            0, 0, 150
        ])

        upper_white = np.array([
            180, 90, 255
        ])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # yellow
        lower_yellow = np.array([
            10, 80, 80
        ])

        upper_yellow = np.array([
            40, 255, 255
        ])

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
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # blur
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

        # =====================================
        # ROI 확대
        # =====================================
        roi = frame[
            150:480,
            :
        ]

        # =====================================
        # binary lane image
        # =====================================
        binary = self.preprocessing(
            roi
        )

        # =====================================
        # sliding window
        # =====================================
        out_img, lane_center = (
            self.slidewindow.slidewindow(
                binary
            )
        )

        # =====================================
        # target
        # =====================================
        target = 320

        # =====================================
        # error
        # =====================================
        error = target - lane_center

        # =====================================
        # PID
        # =====================================
        kp = 1.15

        kd = 0.45

        ki = 0.0005

        self.integral += error

        derivative = (
            error - self.prev_error
        )

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
            0.1 * self.prev_angle +
            0.9 * angle
        )

        self.prev_angle = angle

        # =====================================
        # steering limit
        # =====================================
        angle = max(
            min(angle, 90),
            -90
        )

        # =====================================
        # debug
        # =====================================
        cv2.line(
            out_img,
            (target, 0),
            (target, out_img.shape[0]),
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "binary",
            binary
        )

        cv2.imshow(
            "sliding_window",
            out_img
        )

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

            angle = self.lane_driving(
                frame
            )

            # =================================
            # speed control
            # =================================
            speed = 4

            if abs(angle) > 25:

                speed = 3

            if abs(angle) > 50:

                speed = 2

            self.drive(
                angle,
                speed
            )

            cv2.imshow(
                "camera",
                frame
            )

            if cv2.waitKey(1) & 0xFF == 27:

                break

    # =========================================
    # destroy
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