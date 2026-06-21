#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
# ROS2 Xycar Lane Driving
#
# 개선 내용
# 1. 직선 구간 좌우 비틀거림 감소
# 2. S자 구간 차선 추적 강화
# 3. 오른쪽 실선 기반 2차선 유지
# 4. 중앙 노란 점선 무시 강화
# 5. 차선 신뢰도 낮을 때 감속
# 6. 조향 변화량 제한으로 흔들림 감소
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
        self.rightx_previous = 520

        # 오른쪽 실선에서 차량 중심까지 거리
        # 중앙선 쪽으로 붙으면 240~250
        # 오른쪽 실선 쪽으로 붙으면 270~290
        self.lane_width_offset = 260

        self.fit_previous = None
        self.lost_count = 0

    def slidewindow(self, img):

        out_img = np.dstack((img, img, img))

        height = img.shape[0]
        width = img.shape[1]

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 차선 픽셀이 거의 없으면 이전 위치 유지
        if len(nonzerox) < 80:
            self.lost_count += 1
            return out_img, self.x_previous, 0.0, 0

        # =====================================
        # 오른쪽 차선 시작점 탐색
        # 중앙 점선을 피하기 위해 오른쪽 영역 위주로 탐색
        # =====================================
        x_start = int(width * 0.50)
        x_end = int(width * 0.98)

        y_start = int(height * 0.55)

        right_region = img[
            y_start:height,
            x_start:x_end
        ]

        histogram = np.sum(right_region, axis=0)

        if np.max(histogram) < 1200:
            rightx_current = self.rightx_previous
        else:
            rightx_current = np.argmax(histogram) + x_start

        # =====================================
        # Sliding Window
        # S자 대응을 위해 window 개수와 margin 증가
        # =====================================
        nwindows = 14
        window_height = int(height / nwindows)

        margin = 90
        minpix = 15

        right_lane_inds = []

        for window in range(nwindows):

            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            win_x_low = rightx_current - margin
            win_x_high = rightx_current + margin

            win_x_low = max(0, win_x_low)
            win_x_high = min(width, win_x_high)

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

            if len(good_inds) > minpix:

                mean_x = int(np.mean(nonzerox[good_inds]))

                # 중앙 점선 쪽으로 튀는 것 방지
                if mean_x > int(width * 0.45):
                    rightx_current = mean_x

        right_lane_inds = np.concatenate(right_lane_inds)

        if len(right_lane_inds) < 70:
            self.lost_count += 1
            return out_img, self.x_previous, 0.2, 0

        lane_x = nonzerox[right_lane_inds]
        lane_y = nonzeroy[right_lane_inds]

        # 너무 왼쪽 픽셀 제거
        valid = lane_x > int(width * 0.43)

        lane_x = lane_x[valid]
        lane_y = lane_y[valid]

        if len(lane_x) < 70:
            self.lost_count += 1
            return out_img, self.x_previous, 0.2, 0

        # =====================================
        # 차선 곡선 피팅
        # =====================================
        try:
            fit = np.polyfit(lane_y, lane_x, 2)
        except:
            self.lost_count += 1
            return out_img, self.x_previous, 0.2, 0

        # 이전 fit과 섞어서 직선 구간 흔들림 감소
        if self.fit_previous is not None:
            fit = 0.35 * self.fit_previous + 0.65 * fit

        self.fit_previous = fit

        # =====================================
        # 가까운 점 / 중간 점 / 먼 점 사용
        # 직선에서는 가까운 점 중심
        # S자에서는 중간/먼 점도 반영
        # =====================================
        near_y = int(height * 0.90)
        mid_y = int(height * 0.65)
        far_y = int(height * 0.42)

        rightx_near = int(
            fit[0] * near_y ** 2 +
            fit[1] * near_y +
            fit[2]
        )

        rightx_mid = int(
            fit[0] * mid_y ** 2 +
            fit[1] * mid_y +
            fit[2]
        )

        rightx_far = int(
            fit[0] * far_y ** 2 +
            fit[1] * far_y +
            fit[2]
        )

        rightx_near = int(np.clip(rightx_near, 0, width - 1))
        rightx_mid = int(np.clip(rightx_mid, 0, width - 1))
        rightx_far = int(np.clip(rightx_far, 0, width - 1))

        # curve 값이 클수록 S자/커브로 판단
        curve = rightx_far - rightx_near

        # 직선 구간
        if abs(curve) < 35:

            rightx = int(
                0.75 * rightx_near +
                0.25 * rightx_mid
            )

            center_alpha = 0.30
            max_jump = 45

        # 곡선/S자 구간
        else:

            rightx = int(
                0.40 * rightx_near +
                0.40 * rightx_mid +
                0.20 * rightx_far
            )

            center_alpha = 0.65
            max_jump = 95

        rightx = int(np.clip(
            rightx,
            int(width * 0.42),
            width - 5
        ))

        self.rightx_previous = rightx

        # =====================================
        # 오른쪽 실선 기준 2차선 중심 계산
        # =====================================
        x_location = rightx - self.lane_width_offset

        x_location = int(np.clip(
            x_location,
            0,
            width - 1
        ))

        # =====================================
        # 중심값 급격한 변화 제한
        # 직선 좌우 비틀거림 방지 핵심
        # =====================================
        diff = x_location - self.x_previous

        if abs(diff) > max_jump:
            x_location = self.x_previous + np.sign(diff) * max_jump
            x_location = int(x_location)

        x_location = int(
            (1.0 - center_alpha) * self.x_previous +
            center_alpha * x_location
        )

        self.x_previous = x_location
        self.lost_count = 0

        # 차선 신뢰도
        confidence = min(1.0, len(lane_x) / 500.0)

        # =====================================
        # Debug Drawing
        # =====================================
        ploty = np.linspace(0, height - 1, height)

        fitx = (
            fit[0] * ploty ** 2 +
            fit[1] * ploty +
            fit[2]
        )

        pts = np.array(
            [np.transpose(np.vstack([fitx, ploty]))],
            dtype=np.int32
        )

        cv2.polylines(
            out_img,
            pts,
            False,
            (255, 255, 0),
            4
        )

        cv2.circle(
            out_img,
            (rightx_near, near_y),
            8,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            out_img,
            (rightx_mid, mid_y),
            8,
            (255, 255, 0),
            -1
        )

        cv2.circle(
            out_img,
            (rightx_far, far_y),
            8,
            (255, 0, 255),
            -1
        )

        cv2.circle(
            out_img,
            (x_location, mid_y),
            10,
            (0, 0, 255),
            -1
        )

        return out_img, x_location, confidence, curve


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

        # PID 관련 변수
        self.prev_error = 0
        self.integral = 0
        self.prev_angle = 0

        # 직선 흔들림 방지 필터
        self.filtered_error = 0
        self.filtered_derivative = 0

        # 차선 상태 저장
        self.last_confidence = 1.0
        self.last_curve = 0

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
            "===== Stabilized Lane Driving Start ====="
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
    # motor publish
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

        height = frame.shape[0]
        width = frame.shape[1]

        # =====================================
        # 흰색 실선
        # 직선에서 끊기지 않도록 V값을 조금 낮춤
        # =====================================
        lower_white = np.array([
            0, 0, 135
        ])

        upper_white = np.array([
            180, 95, 255
        ])

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # =====================================
        # 노란색 실선
        # 단, 중앙 노란 점선은 최대한 제거
        # =====================================
        lower_yellow = np.array([
            15, 70, 70
        ])

        upper_yellow = np.array([
            40, 255, 255
        ])

        yellow_mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        # 중앙 노란 점선 제거
        yellow_mask[:, :int(width * 0.55)] = 0

        # 흰색 + 오른쪽 노란색만 사용
        mask = cv2.bitwise_or(
            white_mask,
            yellow_mask
        )

        # 화면 왼쪽 노이즈 제거
        mask[:, :int(width * 0.35)] = 0

        # =====================================
        # Morphology
        # S자에서 차선이 끊겨도 조금 이어주기
        # =====================================
        kernel_close = np.ones(
            (5, 5),
            np.uint8
        )

        kernel_open = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel_close
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel_open
        )

        mask = cv2.GaussianBlur(
            mask,
            (5, 5),
            0
        )

        _, mask = cv2.threshold(
            mask,
            120,
            255,
            cv2.THRESH_BINARY
        )

        return mask

    # =========================================
    # lane driving
    # =========================================
    def lane_driving(self, frame):

        # =====================================
        # ROI
        # 너무 위쪽까지 보면 노이즈가 많아져서 흔들릴 수 있음
        # =====================================
        roi = frame[
            130:480,
            :
        ]

        binary = self.preprocessing(
            roi
        )

        out_img, lane_center, confidence, curve = (
            self.slidewindow.slidewindow(
                binary
            )
        )

        self.last_confidence = confidence
        self.last_curve = curve

        target = 320

        error = target - lane_center

        # =====================================
        # 직선 비틀거림 방지
        # 작은 오차는 무시
        # =====================================
        if abs(error) < 10:
            error = 0

        # error 필터링
        self.filtered_error = (
            0.70 * self.filtered_error +
            0.30 * error
        )

        derivative = self.filtered_error - self.prev_error

        self.filtered_derivative = (
            0.75 * self.filtered_derivative +
            0.25 * derivative
        )

        self.prev_error = self.filtered_error

        # =====================================
        # PID
        # 기존보다 약하게 조정해서 직선 흔들림 감소
        # =====================================
        kp = 0.60
        kd = 0.15
        ki = 0.0

        # S자 / 커브에서는 반응을 조금 키움
        if abs(curve) > 55:
            kp = 0.95
            kd = 0.28

        self.integral += self.filtered_error

        self.integral = max(
            min(self.integral, 5000),
            -5000
        )

        angle = (
            kp * self.filtered_error +
            kd * self.filtered_derivative +
            ki * self.integral
        )

        # =====================================
        # 차선 신뢰도가 낮을 때
        # 새 조향값을 강하게 믿지 않고 이전 조향 유지
        # =====================================
        if confidence < 0.25:
            angle = self.prev_angle * 0.85

        # =====================================
        # 조향 방향
        # 만약 차가 반대로 꺾이면 이 값만 -1.0으로 바꾸면 됨
        # =====================================
        STEER_SIGN = 1.0

        angle = STEER_SIGN * angle

        # =====================================
        # 조향 변화량 제한
        # 직선 비틀거림 방지 핵심
        # =====================================
        max_angle_change = 8

        if abs(curve) > 55:
            max_angle_change = 14

        angle_diff = angle - self.prev_angle

        if angle_diff > max_angle_change:
            angle = self.prev_angle + max_angle_change

        elif angle_diff < -max_angle_change:
            angle = self.prev_angle - max_angle_change

        # 최종 조향 제한
        angle = max(
            min(angle, 70),
            -70
        )

        self.prev_angle = angle

        # =====================================
        # Debug
        # =====================================
        cv2.line(
            out_img,
            (target, 0),
            (target, out_img.shape[0]),
            (0, 255, 0),
            2
        )

        cv2.circle(
            out_img,
            (lane_center, int(out_img.shape[0] * 0.65)),
            8,
            (0, 0, 255),
            -1
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
        print("filtered_error:", self.filtered_error)
        print("curve:", curve)
        print("confidence:", confidence)
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
            speed = 5

            # 차선 신뢰도가 낮으면 감속
            if self.last_confidence < 0.25:
                speed = 2

            # 조향이 크면 감속
            elif abs(angle) > 55:
                speed = 2

            elif abs(angle) > 35:
                speed = 3

            # S자 / 커브 감속
            elif abs(self.last_curve) > 80:
                speed = 3

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