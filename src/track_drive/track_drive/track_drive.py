#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# ROS2 Xycar Sliding Window Lane Driving
#
# 기반:
# KUAC 2024 우승팀 slidewindow 알고리즘
#
# 수정:
# - ROS1 → ROS2 변환
# - 시뮬레이터용 튜닝
# - xycar_motor 적용
# - PID steering 추가
# - Bird Eye View 추가
#
# 특징:
# 1. 흰색/노란색 모두 인식
# 2. 곡선 차선 안정적
# 3. 양쪽 차선 모두 사용
# 4. 한쪽 차선만 보여도 추정 가능
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
# Sliding Window Class
# =============================================
class SlideWindow:

    def __init__(self):

        self.x_previous = 320

    def slidewindow(self, img):

        out_img = np.dstack((img, img, img))

        height = img.shape[0]
        width = img.shape[1]

        # nonzero pixel
        nonzero = img.nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # window parameter
        nwindows = 10
        window_height = 15

        margin = 40
        minpix = 20

        # start region
        left_region = img[height-80:height, :width//2]
        right_region = img[height-80:height, width//2:]

        # histogram
        left_hist = np.sum(left_region, axis=0)
        right_hist = np.sum(right_region, axis=0)

        # start x
        leftx_current = np.argmax(left_hist)

        rightx_current = np.argmax(right_hist) + width//2

        left_lane_inds = []
        right_lane_inds = []

        # =====================================
        # sliding windows
        # =====================================
        for window in range(nwindows):

            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            # left window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            # right window
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # draw windows
            cv2.rectangle(
                out_img,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                out_img,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (255, 0, 0),
                2
            )

            # good left inds
            good_left_inds = (
                (
                    (nonzeroy >= win_y_low) &
                    (nonzeroy < win_y_high) &
                    (nonzerox >= win_xleft_low) &
                    (nonzerox < win_xleft_high)
                ).nonzero()[0]
            )

            # good right inds
            good_right_inds = (
                (
                    (nonzeroy >= win_y_low) &
                    (nonzeroy < win_y_high) &
                    (nonzerox >= win_xright_low) &
                    (nonzerox < win_xright_high)
                ).nonzero()[0]
            )

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # next window position
            if len(good_left_inds) > minpix:

                leftx_current = int(
                    np.mean(nonzerox[good_left_inds])
                )

            if len(good_right_inds) > minpix:

                rightx_current = int(
                    np.mean(nonzerox[good_right_inds])
                )

        # concatenate
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # fallback
        if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:

            x_location = self.x_previous

            return out_img, x_location

        # lane x
        leftx = np.mean(nonzerox[left_lane_inds])
        rightx = np.mean(nonzerox[right_lane_inds])

        # lane center
        x_location = int((leftx + rightx) // 2)

        self.x_previous = x_location

        # draw center
        cv2.circle(
            out_img,
            (x_location, height-40),
            10,
            (0, 0, 255),
            -1
        )

        return out_img, x_location


# =============================================
# Main ROS2 Node
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

        # camera subscriber
        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("===== Sliding Window Start =====")

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

        self.motor_pub.publish(self.motor_msg)

    # =========================================
    # bird eye view
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

        matrix = cv2.getPerspectiveTransform(src, dst)

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

        # white
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 80, 255])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # yellow
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

        # bird eye
        bev = self.bird_eye_view(frame)

        # binary
        binary = self.preprocessing(bev)

        # sliding window
        out_img, lane_center = self.slidewindow.slidewindow(binary)

        # target center
        target = 320

        # error
        error = target - lane_center

        # PID
        kp = 0.40
        kd = 0.12
        ki = 0.0003

        self.integral += error

        derivative = error - self.prev_error

        angle = (
            kp * error +
            kd * derivative +
            ki * self.integral
        )

        self.prev_error = error

        # smoothing
        angle = (
            0.7 * self.prev_angle +
            0.3 * angle
        )

        self.prev_angle = angle

        # limit
        angle = max(min(angle, 50), -50)

        # debug
        cv2.line(
            out_img,
            (320, 0),
            (320, 480),
            (0, 255, 255),
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

            # speed control
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