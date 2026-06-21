#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 camera-based lane tracking and traffic-light-controlled driving."""

import time

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


# =============================================
# Sliding Window
# =============================================
class SlideWindow:

    def __init__(self):
        self.rightx_previous = 580
        self.rightx_lookahead_previous = 500
        self.right_lane_detected = False
        self.right_missing_windows = 10
        self.leftx_previous = 60
        self.leftx_lookahead_previous = 140
        self.left_lane_detected = False
        self.left_missing_windows = 10
        self.tracking_top_ratio = 0.28

    def slidewindow(self, img):
        # 이번 프레임에서 실제 차선 픽셀 검증을 끝내기 전까지는
        # 이전 좌표를 반환하더라도 미검출 상태로 취급한다.
        self.right_lane_detected = False
        self.right_missing_windows = 0
        out_img = np.dstack((img, img, img))
        height = img.shape[0]
        width = img.shape[1]
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # =====================================
        # 오른쪽 차선 시작점 탐색
        # 중앙선 오인식을 막기 위해 아래쪽 오른쪽 영역만 사용
        # =====================================
        right_search_x = int(width * 0.52)
        right_region = img[
            int(height * 0.55):height,
            right_search_x:width
        ]
        histogram = np.sum(
            right_region,
            axis=0
        )
        if len(histogram) == 0 or np.max(histogram) < 255 * 20:
            rightx_current = int(
                np.clip(
                    self.rightx_previous,
                    right_search_x,
                    width - 1
                )
            )
        else:
            smooth_histogram = np.convolve(
                histogram,
                np.ones(9),
                mode='same'
            )
            peak_value = np.max(
                smooth_histogram
            )
            candidate_x = (
                np.where(
                    smooth_histogram > peak_value * 0.35
                )[0] + right_search_x
            )
            if len(candidate_x) == 0:
                rightx_current = (
                    np.argmax(smooth_histogram)
                    + right_search_x
                )
            else:
                rightx_current = int(
                    candidate_x[
                        np.argmin(
                            np.abs(candidate_x - self.rightx_previous)
                        )
                    ]
                )
            if abs(rightx_current - self.rightx_previous) > 160:
                rightx_current = int(
                    0.65 * self.rightx_previous +
                    0.35 * rightx_current
                )
            rightx_current = int(
                np.clip(
                    rightx_current,
                    right_search_x,
                    int(width * 0.93)
                )
            )
        # =====================================
        # sliding window parameter
        # =====================================
        # 너무 먼 상단 영역은 장애물을 흰 차선으로 오인할 가능성이
        # 높으므로 추적하지 않는다. lookahead_y(42%)는 이 범위 안에 있다.
        tracking_top = int(height * self.tracking_top_ratio)
        nwindows = 10
        window_height = int(
            (height - tracking_top) / nwindows
        )
        # 차선 주변의 장애물 픽셀이 윈도우에 들어오지 않도록 폭을 제한한다.
        # 실제 바운딩 박스 너비는 margin의 두 배다.
        margin = 45
        minpix = 25
        right_lane_inds = []
        # =====================================
        # sliding windows
        # =====================================
        for window in range(nwindows):
            win_y_low = (
                height - (window + 1) * window_height
            )
            win_y_low = max(win_y_low, tracking_top)
            win_y_high = (
                height - window * window_height
            )
            # 아래쪽 윈도우는 반드시 오른쪽 차선 영역에서만 찾는다.
            # 위쪽은 커브를 따라갈 수 있도록 조금 더 열어둔다.
            if win_y_low > int(height * 0.45):
                min_lane_x = int(width * 0.50)
            else:
                min_lane_x = int(width * 0.30)
            win_x_low = max(
                rightx_current - margin, min_lane_x
            )
            win_x_high = min(
                rightx_current + margin, width - 1
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
            else:
                self.right_missing_windows += 1
        # concatenate
        if len(right_lane_inds) > 0:
            right_lane_inds = np.concatenate(
                right_lane_inds
            )
        # =====================================
        # 차선 못 찾으면 이전값 유지
        # =====================================
        if len(right_lane_inds) == 0:
            return (
                out_img,
                self.rightx_previous,
                self.rightx_lookahead_previous
            )
        # =====================================
        # 중앙선/잡음으로 점프하는 경우 방지
        # =====================================
        lane_x = nonzerox[right_lane_inds]
        lane_y = nonzeroy[right_lane_inds]
        lower_lane = lane_y > int(height * 0.55)
        if np.count_nonzero(lower_lane) < minpix:
            return (
                out_img,
                self.rightx_previous,
                self.rightx_lookahead_previous
            )
        lower_median_x = np.median(
            lane_x[lower_lane]
        )
        if lower_median_x < int(width * 0.55):
            return (
                out_img,
                self.rightx_previous,
                self.rightx_lookahead_previous
            )
        lookahead_y = int(height * 0.42)
        if len(lane_x) >= 50:
            fit = np.polyfit(
                lane_y,
                lane_x,
                2
            )
            rightx = int(
                np.polyval(
                    fit,
                    height - 1
                )
            )
            rightx_lookahead = int(
                np.polyval(
                    fit,
                    lookahead_y
                )
            )
        else:
            rightx = int(lower_median_x)
            rightx_lookahead = rightx
        rightx = int(
            np.clip(
                rightx,
                int(width * 0.55),
                width - 1
            )
        )
        rightx_lookahead = int(
            np.clip(
                rightx_lookahead,
                int(width * 0.30),
                width - 1
            )
        )
        self.rightx_previous = rightx
        self.rightx_lookahead_previous = rightx_lookahead
        self.right_lane_detected = True
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
            (rightx_lookahead, lookahead_y),
            8,
            (255, 255, 0),
            -1
        )
        return out_img, rightx, rightx_lookahead

    def detect_left_lane(
        self,
        img,
        out_img=None,
        right_lane_x=None,
        right_lane_lookahead=None,
        use_right_boundary=False,
    ):
        self.left_lane_detected = False
        self.left_missing_windows = 0
        height, width = img.shape[:2]
        tracking_top = int(height * self.tracking_top_ratio)
        nonzeroy, nonzerox = img.nonzero()
        # 왼쪽 차선 검출은 카메라 화면의 왼쪽 절반으로만 제한한다.
        left_limit = int(width * 0.50)
        left_region = img[int(height * 0.55):height, 0:left_limit]
        histogram = np.sum(left_region, axis=0)
        if len(histogram) == 0 or np.max(histogram) < 255 * 20:
            leftx_current = int(np.clip(
                self.leftx_previous,
                0,
                left_limit - 1
            ))
        else:
            smooth_histogram = np.convolve(
                histogram,
                np.ones(9),
                mode='same'
            )
            peak_value = np.max(smooth_histogram)
            candidate_x = np.where(
                smooth_histogram > peak_value * 0.35
            )[0]
            if len(candidate_x) == 0:
                leftx_current = int(np.argmax(smooth_histogram))
            else:
                leftx_current = int(candidate_x[
                    np.argmin(np.abs(candidate_x - self.leftx_previous))
                ])
            # 한 프레임 사이에 왼쪽 실선이 크게 이동할 수 없으므로,
            # 멀리 있는 오른쪽 실선만 후보로 잡힌 경우 이전 위치를 유지한다.
            if abs(leftx_current - self.leftx_previous) > 120:
                leftx_current = self.leftx_previous
        nwindows = 10
        window_height = max(1, int(
            (height - tracking_top) / nwindows
        ))
        margin = 45
        minpix = 25
        left_lane_inds = []
        for window in range(nwindows):
            win_y_low = max(
                height - (window + 1) * window_height,
                tracking_top
            )
            win_y_high = height - window * window_height
            max_lane_x = left_limit
            # 오른쪽 실선이 보이면 같은 선을 왼쪽 윈도우가 따라가지
            # 못하도록 현재 높이에서 예상한 오른쪽 선과 간격을 둔다.
            if use_right_boundary:
                window_center_y = (win_y_low + win_y_high) / 2.0
                lookahead_y = height * 0.42
                denominator = max(1.0, (height - 1) - lookahead_y)
                lookahead_ratio = (
                    (height - 1) - window_center_y
                ) / denominator
                expected_right_x = (
                    right_lane_x +
                    lookahead_ratio *
                    (right_lane_lookahead - right_lane_x)
                )
                minimum_gap = float(np.clip(
                    200.0 - 80.0 * lookahead_ratio,
                    100.0,
                    200.0
                ))
                max_lane_x = min(
                    max_lane_x,
                    int(expected_right_x - minimum_gap)
                )
            leftx_current = min(
                leftx_current,
                max(0, max_lane_x - margin)
            )
            win_x_low = max(leftx_current - margin, 0)
            win_x_high = max(
                win_x_low,
                min(leftx_current + margin, max_lane_x)
            )
            if out_img is not None:
                cv2.rectangle(
                    out_img,
                    (win_x_low, win_y_low),
                    (win_x_high, win_y_high),
                    (255, 0, 255),
                    2
                )
            good_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &
                (nonzerox < win_x_high)
            ).nonzero()[0]
            left_lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_inds]))
            else:
                self.left_missing_windows += 1
        left_lane_inds = np.concatenate(left_lane_inds)
        if len(left_lane_inds) == 0:
            return self.leftx_previous, self.leftx_lookahead_previous
        lane_x = nonzerox[left_lane_inds]
        lane_y = nonzeroy[left_lane_inds]
        lower_lane = lane_y > int(height * 0.55)
        if np.count_nonzero(lower_lane) < minpix:
            return self.leftx_previous, self.leftx_lookahead_previous
        lower_median_x = np.median(lane_x[lower_lane])
        if lower_median_x >= left_limit:
            return self.leftx_previous, self.leftx_lookahead_previous
        lookahead_y = int(height * 0.42)
        if len(lane_x) >= 50:
            fit = np.polyfit(lane_y, lane_x, 2)
            leftx = int(np.polyval(fit, height - 1))
            leftx_lookahead = int(np.polyval(fit, lookahead_y))
        else:
            leftx = int(lower_median_x)
            leftx_lookahead = leftx
        leftx = int(np.clip(leftx, 0, left_limit - 1))
        leftx_lookahead = int(np.clip(
            leftx_lookahead,
            0,
            left_limit - 1
        ))
        if (
            abs(leftx - self.leftx_previous) > 120 or
            abs(
                leftx_lookahead - self.leftx_lookahead_previous
            ) > 180
        ):
            return self.leftx_previous, self.leftx_lookahead_previous
        self.leftx_previous = leftx
        self.leftx_lookahead_previous = leftx_lookahead
        self.left_lane_detected = True
        return leftx, leftx_lookahead


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
        # 직선에서는 속도를 높이고 커브에서는 낮춘다. 속도는 아래
        # main_loop에서 연속적으로 보간해 급격한 단계 변화를 방지한다.
        self.straight_speed = 10.0
        self.curve_speed = 1.5
        self.current_speed = self.curve_speed
        self.max_speed_command = self.straight_speed
        self.straight_angle_threshold = 8.0
        self.straight_speed_delay = 2.0
        self.last_turn_time = time.monotonic()
        self.publisher_conflict_reported = False
        # 시작 시 정지한다. 초록불이면 주행 상태를 유지하고, 이후
        # 빨간불이 검출될 때만 다시 정지 상태로 전환한다.
        self.red_pixel_threshold = 50
        self.green_pixel_threshold = 30
        self.red_required_frames = 5
        self.green_required_frames = 3
        self.red_frame_count = 0
        self.green_frame_count = 0
        self.driving_enabled = False
        # 왼쪽·오른쪽 흰색 실선 사이의 전체 트랙 폭 추정값이다.
        self.road_width_bottom_estimate = 520.0
        self.road_width_lookahead_estimate = 360.0
        self.right_lane_distance_bottom = 200.0
        self.right_lane_distance_lookahead = 140.0
        self.lane_recovery = False
        self.lane_reacquire_frames = 0
        self.lane_reacquire_required = 2
        self.lane_recovery_angle = 90.0
        self.single_lane_recovery_angle = 45.0
        self.lane_direction_threshold = 12.0
        self.lane_direction_gain = 0.60
        self.lane_direction_min_angle = 20.0
        self.lane_direction_max_angle = 90.0
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
        self.traffic_light_sub = self.create_subscription(
            Int64MultiArray,
            '/traffic_light',
            self.traffic_light_callback,
            10
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

    def traffic_light_callback(self, message):
        if len(message.data) < 2:
            self.red_frame_count = 0
            self.green_frame_count = 0
            return
        red_visible = message.data[0] >= self.red_pixel_threshold
        green_visible = message.data[1] >= self.green_pixel_threshold
        # 빨강과 초록이 동시에 검출되면 초록 주행을 우선한다.
        if green_visible:
            self.green_frame_count = min(
                self.green_frame_count + 1,
                self.green_required_frames
            )
            self.red_frame_count = 0
            if (
                self.green_frame_count >= self.green_required_frames and
                not self.driving_enabled
            ):
                self.driving_enabled = True
                self.get_logger().info('GREEN detected: driving enabled')
        elif red_visible:
            self.red_frame_count = min(
                self.red_frame_count + 1,
                self.red_required_frames
            )
            self.green_frame_count = 0
            if (
                self.red_frame_count >= self.red_required_frames and
                self.driving_enabled
            ):
                self.driving_enabled = False
                self.get_logger().info('RED detected: vehicle stopped')
        else:
            # 신호등이 시야에서 사라져도 현재 주행 상태는 유지한다.
            self.red_frame_count = 0
            self.green_frame_count = 0

    # =========================================
    # drive
    # =========================================
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        # 이후 속도 로직이 변경되더라도 직선 최고 속도를 넘지 않는다.
        self.motor_msg.speed = float(np.clip(
            speed,
            -self.max_speed_command,
            self.max_speed_command
        ))
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
        height = white_mask.shape[0]
        width = white_mask.shape[1]
        road_roi = np.zeros_like(
            white_mask
        )
        road_polygon = np.array([[
            (0, height),
            (width - 1, height),
            (int(width * 0.95), int(height * 0.10)),
            (int(width * 0.05), int(height * 0.10))
        ]], np.int32)
        cv2.fillPoly(
            road_roi,
            road_polygon,
            255
        )
        white_lane = cv2.bitwise_and(
            white_mask,
            road_roi
        )
        yellow_center = cv2.bitwise_and(
            yellow_mask,
            road_roi
        )
        # morphology
        kernel = np.ones(
            (3, 3),
            np.uint8
        )
        white_lane = cv2.morphologyEx(
            white_lane,
            cv2.MORPH_OPEN,
            kernel
        )
        white_lane = cv2.morphologyEx(
            white_lane,
            cv2.MORPH_CLOSE,
            kernel
        )
        yellow_center = cv2.morphologyEx(
            yellow_center,
            cv2.MORPH_OPEN,
            kernel
        )
        # blur
        white_lane = cv2.GaussianBlur(
            white_lane,
            (5, 5),
            0
        )
        yellow_center = cv2.GaussianBlur(
            yellow_center,
            (5, 5),
            0
        )
        _, white_lane = cv2.threshold(
            white_lane,
            127,
            255,
            cv2.THRESH_BINARY
        )
        _, yellow_center = cv2.threshold(
            yellow_center,
            127,
            255,
            cv2.THRESH_BINARY
        )
        return white_lane, yellow_center

    @staticmethod
    def extract_solid_yellow_lanes(yellow_mask):
        """Keep long yellow boundaries while rejecting short dotted dashes."""
        height = yellow_mask.shape[0]
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 9)
        )
        connected_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            vertical_kernel
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            connected_mask,
            8
        )
        solid_mask = np.zeros_like(yellow_mask)
        minimum_height = int(height * 0.30)
        solid_component_count = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            component_height = stats[label, cv2.CC_STAT_HEIGHT]
            if area >= 120 and component_height >= minimum_height:
                solid_mask[labels == label] = 255
                solid_component_count += 1
        # 팽창된 연결 성분 전체가 아닌 원래 노란 픽셀만 반환한다.
        return (
            cv2.bitwise_and(yellow_mask, solid_mask),
            solid_component_count,
        )

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
        # 흰색 차선과 노란 중앙선을 따로 인식
        # =====================================
        white_binary, yellow_binary = self.preprocessing(
            roi
        )
        solid_yellow_binary, solid_yellow_count = (
            self.extract_solid_yellow_lanes(yellow_binary)
        )
        # 보호구역에서는 양쪽 경계가 함께 노란 실선으로 바뀐다.
        # 긴 성분이 하나뿐이면 중앙 점선 또는 잡음으로 보고 사용하지 않는다.
        solid_yellow_available = solid_yellow_count >= 2
        all_lane_binary = cv2.bitwise_or(
            white_binary,
            solid_yellow_binary
        )
        # =====================================
        # sliding window
        # =====================================
        out_img, right_lane_x, right_lane_lookahead = (
            self.slidewindow.slidewindow(
                white_binary
            )
        )
        right_lane_detected = self.slidewindow.right_lane_detected
        right_used_color_fallback = False
        if (
            solid_yellow_available and
            (
                not right_lane_detected or
                self.slidewindow.right_missing_windows > 6
            )
        ):
            # 흰색 경계가 부족하면 어린이 보호구역의 노란 실선을
            # 포함한 마스크로 같은 위치에서 다시 탐색한다.
            out_img, right_lane_x, right_lane_lookahead = (
                self.slidewindow.slidewindow(all_lane_binary)
            )
            right_lane_detected = self.slidewindow.right_lane_detected
            right_used_color_fallback = True
        left_lane_x, left_lane_lookahead = (
            self.slidewindow.detect_left_lane(
                white_binary,
                out_img,
                right_lane_x,
                right_lane_lookahead,
                True,
            )
        )
        left_lane_detected = self.slidewindow.left_lane_detected
        left_used_color_fallback = False
        if (
            solid_yellow_available and
            (
                not left_lane_detected or
                self.slidewindow.left_missing_windows > 6
            )
        ):
            left_lane_x, left_lane_lookahead = (
                self.slidewindow.detect_left_lane(
                    all_lane_binary,
                    out_img,
                    right_lane_x,
                    right_lane_lookahead,
                    True,
                )
            )
            left_lane_detected = self.slidewindow.left_lane_detected
            left_used_color_fallback = True
        left_missing_windows = self.slidewindow.left_missing_windows
        right_missing_windows = self.slidewindow.right_missing_windows
        left_valid_windows = max(0, 10 - left_missing_windows)
        right_valid_windows = max(0, 10 - right_missing_windows)
        # 두 검출기가 같은 실선을 잡은 경우 왼쪽 검출을 무효화한다.
        if left_lane_detected and right_lane_detected:
            bottom_separation = right_lane_x - left_lane_x
            lookahead_separation = (
                right_lane_lookahead - left_lane_lookahead
            )
            minimum_bottom_separation = max(
                260,
                int(self.road_width_bottom_estimate * 0.60)
            )
            minimum_lookahead_separation = max(
                150,
                int(self.road_width_lookahead_estimate * 0.55)
            )
            if (
                bottom_separation < minimum_bottom_separation or
                lookahead_separation < minimum_lookahead_separation
            ):
                left_lane_detected = False
                self.slidewindow.left_lane_detected = False
        lane_detected = right_lane_detected or left_lane_detected
        if lane_detected:
            if self.lane_recovery:
                self.lane_reacquire_frames += 1
                if (
                    self.lane_reacquire_frames >=
                    self.lane_reacquire_required
                ):
                    self.lane_recovery = False
            else:
                self.lane_reacquire_frames = 0
        else:
            self.lane_recovery = True
            self.lane_reacquire_frames = 0
        # =====================================
        # target center
        # =====================================
        target = 320
        # 오른쪽 실선이 보이면 설정한 간격으로 목표 주행선을 만든다.
        if right_lane_detected:
            bottom_lane_center = (
                right_lane_x - self.right_lane_distance_bottom
            )
            lookahead_lane_center = (
                right_lane_lookahead -
                self.right_lane_distance_lookahead
            )
        # 오른쪽 실선이 사라지면 최근 전체 트랙 폭과 왼쪽 흰색 실선으로
        # 동일한 목표 주행선을 추정한다.
        elif left_lane_detected:
            bottom_lane_center = (
                left_lane_x + self.road_width_bottom_estimate -
                self.right_lane_distance_bottom
            )
            lookahead_lane_center = (
                left_lane_lookahead + self.road_width_lookahead_estimate -
                self.right_lane_distance_lookahead
            )
        else:
            bottom_lane_center = float(target)
            lookahead_lane_center = float(target)
        if left_lane_detected and right_lane_detected:
            road_width_bottom = right_lane_x - left_lane_x
            road_width_lookahead = (
                right_lane_lookahead - left_lane_lookahead
            )
            if 300 <= road_width_bottom <= 630:
                self.road_width_bottom_estimate = (
                    0.85 * self.road_width_bottom_estimate +
                    0.15 * road_width_bottom
                )
            if 180 <= road_width_lookahead <= 550:
                self.road_width_lookahead_estimate = (
                    0.85 * self.road_width_lookahead_estimate +
                    0.15 * road_width_lookahead
                )
            # 양쪽이 모두 보이면 유효 윈도우가 많은 차선에 더 높은
            # 가중치를 주어 장애물이나 일시적인 오검출 영향을 줄인다.
            left_bottom_center = (
                left_lane_x + self.road_width_bottom_estimate -
                self.right_lane_distance_bottom
            )
            left_lookahead_center = (
                left_lane_lookahead + self.road_width_lookahead_estimate -
                self.right_lane_distance_lookahead
            )
            total_valid_windows = max(
                1,
                left_valid_windows + right_valid_windows
            )
            left_weight = left_valid_windows / total_valid_windows
            right_weight = right_valid_windows / total_valid_windows
            bottom_lane_center = (
                right_weight * bottom_lane_center +
                left_weight * left_bottom_center
            )
            lookahead_lane_center = (
                right_weight * lookahead_lane_center +
                left_weight * left_lookahead_center
            )
        # 직선에서는 가까운 중심을 안정적으로 따르고, 코너에서는 먼
        # 중심의 비중을 최대 50%까지 높여 조향을 더 일찍 시작한다.
        center_delta = abs(bottom_lane_center - lookahead_lane_center)
        curve_strength = float(np.clip(center_delta / 80.0, 0.0, 1.0))
        lookahead_weight = 0.30 + 0.20 * curve_strength
        bottom_weight = 1.0 - lookahead_weight
        lane_center = int(
            bottom_weight * bottom_lane_center +
            lookahead_weight * lookahead_lane_center
        )
        lane_center = int(
            np.clip(
                lane_center,
                0,
                white_binary.shape[1] - 1
            )
        )
        curve_delta = bottom_lane_center - lookahead_lane_center
        if curve_delta > 0.0:
            # 왼쪽 커브는 더 먼 시점부터 강하게 선행 조향한다.
            curve_feedforward = float(np.clip(
                curve_delta * 0.45,
                0,
                90
            ))
        else:
            curve_feedforward = float(np.clip(
                curve_delta * 0.32,
                -90,
                0
            ))
        # =====================================
        # error
        # =====================================
        error = target - lane_center
        # =====================================
        # PID
        # =====================================
        kp = 1.08
        kd = 0.32
        ki = 0.0005
        self.integral += error
        self.integral = max(min(self.integral, 500), -500)
        derivative = (
            error - self.prev_error
        )
        angle = (
            kp * error +
            kd * derivative +
            ki * self.integral +
            curve_feedforward
        )
        self.prev_error = error
        # =====================================
        # smoothing 감소
        # =====================================
        if abs(error) > 50:              # 커브: 빠른 반응
            angle = 0.45 * self.prev_angle + 0.55 * angle
        else:                            # 직선: 부드럽게
            angle = 0.70 * self.prev_angle + 0.30 * angle
        self.prev_angle = angle
        # =====================================
        # steering limit
        # =====================================
        angle = max(
            min(angle, 90),
            -90
        )
        angle *= -1
        # 차량이 왼쪽 커브에서 언더스티어하는 특성을 보정한다.
        # 곡률이 클수록 왼쪽 조향만 최대 40% 증폭한다.
        if angle < 0.0:
            left_curve_gain = 1.0 + 0.40 * curve_strength
            angle = max(angle * left_curve_gain, -90.0)
        # 미검출 중에는 오른쪽 최대 조향을 유지한다. 첫 재검출
        # 프레임은 직진하고, 두 번째 연속 검출부터 PID로 즉시 복귀한다.
        if self.lane_recovery:
            if lane_detected:
                angle = 0.0
            else:
                angle = self.lane_recovery_angle
            self.integral = 0.0
            cv2.putText(
                out_img,
                'BOTH LANES LOST - RECOVERING',
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        # 한쪽 실선이 사라져도 반대쪽 유효 윈도우가 4개 이상이면
        # 그 차선의 PID 조향 크기를 이용해 반대 방향으로 주행한다.
        if not left_lane_detected and right_lane_detected:
            if right_valid_windows >= 4:
                angle = -abs(angle)
            else:
                angle = -self.single_lane_recovery_angle
        elif not right_lane_detected and left_lane_detected:
            if left_valid_windows >= 4:
                angle = abs(angle)
            else:
                angle = self.single_lane_recovery_angle
        # 위쪽 차선 위치가 바깥 방향으로 이동하면 해당 커브 방향을
        # 우선한다. 실제 차량 기준 양수는 우측, 음수는 좌측 조향이다.
        right_direction_delta = right_lane_lookahead - right_lane_x
        left_direction_delta = left_lane_x - left_lane_lookahead
        right_turn_hint = (
            right_lane_detected and
            right_valid_windows >= 4 and
            right_direction_delta >= self.lane_direction_threshold
        )
        left_turn_hint = (
            left_lane_detected and
            left_valid_windows >= 4 and
            left_direction_delta >= self.lane_direction_threshold
        )
        lane_turn_hint = 'NONE'
        if right_turn_hint and (
            not left_turn_hint or
            right_direction_delta >= left_direction_delta
        ):
            hint_angle = float(np.clip(
                right_direction_delta * self.lane_direction_gain,
                self.lane_direction_min_angle,
                self.lane_direction_max_angle
            ))
            angle = max(abs(angle), hint_angle)
            lane_turn_hint = 'RIGHT'
        elif left_turn_hint:
            hint_angle = float(np.clip(
                left_direction_delta * self.lane_direction_gain,
                self.lane_direction_min_angle,
                self.lane_direction_max_angle
            ))
            angle = -max(abs(angle), hint_angle)
            lane_turn_hint = 'LEFT'
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
        cv2.putText(
            out_img,
            (
                f'LEFT LANE: {"TRACKED" if left_lane_detected else "LOST"} '
                f'[{"WHITE+SOLID-Y" if left_used_color_fallback else "WHITE"}]'
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255) if left_lane_detected else (0, 0, 255),
            2
        )
        cv2.putText(
            out_img,
            (
                f'RIGHT LANE: {"TRACKED" if right_lane_detected else "LOST"} '
                f'[{"WHITE+SOLID-Y" if right_used_color_fallback else "WHITE"}]'
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0) if right_lane_detected else (0, 0, 255),
            2
        )
        cv2.putText(
            out_img,
            f'VALID WINDOWS L:{left_valid_windows}/10 R:{right_valid_windows}/10',
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )
        cv2.putText(
            out_img,
            (
                f'SOLID YELLOW: {solid_yellow_count} '
                f'[{"ACTIVE" if solid_yellow_available else "IGNORED"}]'
            ),
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.putText(
            out_img,
            (
                f'TURN HINT: {lane_turn_hint} '
                f'R:{right_direction_delta:.0f} L:{left_direction_delta:.0f}'
            ),
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        if left_lane_detected:
            cv2.line(
                out_img,
                (left_lane_x, 0),
                (left_lane_x, out_img.shape[0]),
                (255, 0, 255),
                2
            )
            cv2.circle(
                out_img,
                (left_lane_lookahead, int(out_img.shape[0] * 0.42)),
                8,
                (255, 0, 255),
                -1
            )
        cv2.circle(
            out_img,
            (lane_center, out_img.shape[0] - 60),
            8,
            (255, 0, 255),
            -1
        )
        binary = all_lane_binary
        cv2.imshow(
            "binary",
            binary
        )
        cv2.imshow(
            "solid_yellow_lanes",
            solid_yellow_binary
        )
        cv2.imshow(
            "sliding_window",
            out_img
        )
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
            # 조향이 작을수록 직선 속도에 가까워진다. 단계별 if문 대신
            # 연속 보간과 저역통과 필터를 사용해 속도 급변을 막는다.
            # 작은 조향에서는 감속을 최소화하고, 큰 조향부터 빠르게
            # 커브 속도로 내려가도록 비선형 속도 곡선을 사용한다.
            turn_ratio = float(
                np.clip(abs(angle) / 45.0, 0.0, 1.0) ** 1.6
            )
            target_speed = (
                self.straight_speed -
                turn_ratio * (self.straight_speed - self.curve_speed)
            )
            # 조향이 풀려도 마지막 커브 조향 후 2초 동안은 저속을
            # 유지한다. 연속으로 직진이 확인된 뒤에만 가속을 허용한다.
            now = time.monotonic()
            if abs(angle) > self.straight_angle_threshold:
                self.last_turn_time = now
            if now - self.last_turn_time < self.straight_speed_delay:
                target_speed = min(target_speed, self.curve_speed)
            self.current_speed = (
                0.92 * self.current_speed +
                0.08 * target_speed
            )
            speed = self.current_speed
            if not self.driving_enabled:
                speed = 0.0
            # 둘 이상의 노드가 같은 모터 토픽을 발행하면 명령이 서로
            # 덮어써지므로, 충돌이 해소될 때까지 이 노드는 정지 명령만 보낸다.
            if self.count_publishers('/xycar_motor') > 1:
                speed = 0.0
                if not self.publisher_conflict_reported:
                    self.get_logger().error(
                        'Multiple /xycar_motor publishers detected; stopping vehicle'
                    )
                    self.publisher_conflict_reported = True
            else:
                self.publisher_conflict_reported = False
            self.drive(angle, speed)
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
