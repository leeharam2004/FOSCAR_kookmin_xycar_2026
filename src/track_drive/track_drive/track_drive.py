#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# ROS2 Xycar Lane Driving
#
# 개선 내용
# 1. 더 먼 차선 탐지
# 2. 커브 조기 인식
# 3. 오른쪽 실선 기반 2차선 유지
# 4. 노란 점선 무시 강화
# 5. 조향 반응 강화
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

        self.x_previous = 320

    def slidewindow(self, img):

        out_img = np.dstack((img, img, img))

        height = img.shape[0]
        width = img.shape[1]

        nonzero = img.nonzero()

        nonzeroy = np.array(nonzero[0])

        nonzerox = np.array(nonzero[1])

        # =====================================
        # 더 넓은 영역으로 시작점 탐색
        # =====================================
        right_region = img[
            height - 120:height,
            int(width * 0.55):width
        ]

        histogram = np.sum(
            right_region,
            axis=0
        )

        rightx_current = (
            np.argmax(histogram)
            + int(width * 0.55)
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

            win_x_low = (
                rightx_current - margin
            )

            win_x_high = (
                rightx_current + margin
            )

            # draw window
            cv2.rectangle(
                out_img,
                (win_x_low, win_y_low),
                (win_x_high, win_y_high),
                (255, 0, 0),
                2
            )

            # lane pixels
            good_inds = (
                (
                    (nonzeroy >= win_y_low) &
                    (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_low) &
                    (nonzerox < win_x_high)
                ).nonzero()[0]
            )

            right_lane_inds.append(
                good_inds
            )

            # 다음 윈도우 위치 이동
            if len(good_inds) > minpix:

                rightx_current = int(
                    np.mean(
                        nonzerox[good_inds]
                    )
                )

        # concatenate
        right_lane_inds = np.concatenate(
            right_lane_inds
        )

        # =====================================
        # 차선 못 찾으면 이전값 유지
        # =====================================
        if len(right_lane_inds) == 0:

            return out_img, self.x_previous

        # =====================================
        # 가장 오른쪽 픽셀 사용
        # =====================================
        rightx = int(
            np.max(
                nonzerox[right_lane_inds]
            )
        )

        # =====================================
        # 2차선 중심 계산
        # =====================================
        lane_width_offset = 400

        x_location = (
            rightx - lane_width_offset
        )

        self.x_previous = x_location

        # =====================================
        # debug
        # =====================================
        cv2.circle(
            out_img,
            (rightx, height - 20),
            10,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            out_img,
            (x_location, height - 40),
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
            "===== Enhanced Lane Driving Start ====="
        )

    # =========================================
    # callback
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

        # =====================================
        # white lane
        # =====================================
        lower_white = np.array([
            0, 0, 150
        ])

        upper_white = np.array([
            180, 80, 255
        ])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # =====================================
        # yellow lane
        # =====================================
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
        # 더 먼 차선까지 보도록 ROI 확대
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
        # target center
        # =====================================
        target = 320

        # =====================================
        # error
        # =====================================
        error = target - lane_center

        # =====================================
        # PID
        # =====================================
        kp = 1.2

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
            (0, 255, 0),
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
        print("target:", target)
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