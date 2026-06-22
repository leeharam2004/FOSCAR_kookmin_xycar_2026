# -1,521 +1,521 @@
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#아오 커밋어떻게해
#아직도 안
# =============================================
# ROS2 Xycar Lane Driving
#
# 개선 내용
# 1. 더 먼 차선 탐지
# 2. 커브 조기 인식
# 3. 오른쪽 실선 기반 2차선 유지
# 4. 노란 점선 중앙선 별도 인식
# 5. 조향 반응 강화
# 6. 커브길 곡선 2번째 까진 안정적 통과후 직진후 차선이탈 문제
# =============================================

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

# True로 바꾸면 신호등 없이 즉시 주행 시작 (신호등 콜백 무시)
IGNORE_TRAFFIC_LIGHT = True


# =============================================
# Sliding Window
# =============================================
class SlideWindow:

    def __init__(self):

        self.x_previous = 320
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
                self.x_previous,
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
                self.x_previous,
                self.rightx_previous,
                self.rightx_lookahead_previous
            )

        lower_median_x = np.median(
            lane_x[lower_lane]
        )

        if lower_median_x < int(width * 0.55):

            return (
                out_img,
                self.x_previous,
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

        # =====================================
        # 2차선 중심 계산
        # =====================================
        lane_width_offset_bottom = 260
        lane_width_offset_lookahead = 215

        bottom_center = (
            rightx - lane_width_offset_bottom
        )

        lookahead_center = (
            rightx_lookahead - lane_width_offset_lookahead
        )

        x_location = int(
            0.65 * bottom_center +
            0.35 * lookahead_center
        )

        if abs(x_location - self.x_previous) > 110:

            x_location = int(
                0.65 * self.x_previous +
                0.35 * x_location
            )

        self.rightx_previous = rightx
        self.rightx_lookahead_previous = rightx_lookahead

        self.x_previous = x_location
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

        cv2.circle(
            out_img,
            (x_location, height - 40),
            10,
            (0, 0, 255),
            -1
        )

        return out_img, x_location, rightx, rightx_lookahead

    def detect_left_lane(
        self,
        img,
        out_img=None,
        right_lane_x=None,
        right_lane_lookahead=None,
        right_lane_detected=False,
    ):

        self.left_lane_detected = False
        self.left_missing_windows = 0
        height, width = img.shape[:2]
        tracking_top = int(height * self.tracking_top_ratio)
        nonzeroy, nonzerox = img.nonzero()

        # 왼쪽 실선이 카메라 중앙 부근까지 들어오는 S자 구간도 허용한다.
        left_limit = int(width * 0.55)
        left_region = img[int(height * 0.55):height, 0:left_limit]
        histogram = np.sum(left_region, axis=0)

        if len(histogram) == 0 or np.max(histogram) < 255 * 20:
            leftx_current = int(np.clip(
                self.leftx_previous,
                0,
                left_limit
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

            if win_y_low > int(height * 0.45):
                max_lane_x = int(width * 0.58)
            else:
                max_lane_x = int(width * 0.78)

            # 오른쪽 실선이 보이면 같은 선을 왼쪽 윈도우가 따라가지
            # 못하도록 현재 높이에서 예상한 오른쪽 선과 간격을 둔다.
            if right_lane_detected:
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
        if lower_median_x > int(width * 0.55):
            return self.leftx_previous, self.leftx_lookahead_previous

        lookahead_y = int(height * 0.42)
        if len(lane_x) >= 50:
            fit = np.polyfit(lane_y, lane_x, 2)
            leftx = int(np.polyval(fit, height - 1))
            leftx_lookahead = int(np.polyval(fit, lookahead_y))
        else:
            leftx = int(lower_median_x)
            leftx_lookahead = leftx

        leftx = int(np.clip(leftx, 0, int(width * 0.55)))
        leftx_lookahead = int(np.clip(
            leftx_lookahead,
            0,
            int(width * 0.78)
        ))
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
        self.publisher_conflict_reported = False

        # 시작 시 정지한다. 초록불이면 주행 상태를 유지하고, 이후
        # 빨간불이 검출될 때만 다시 정지 상태로 전환한다.
        self.red_pixel_threshold = 50
        self.green_pixel_threshold = 30
        self.red_required_frames = 5
        self.green_required_frames = 3
        self.red_frame_count = 0
        self.green_frame_count = 0
        self.driving_enabled = IGNORE_TRAFFIC_LIGHT

        self.yellowx_previous = None
        self.yellow_lookahead_previous = None

        self.yellow_miss_count = 0

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

        self.pedestrian_blocking = False

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

        if IGNORE_TRAFFIC_LIGHT:
            return

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

    # =========================================
    # yellow dotted centerline
    # =========================================
    def detect_yellow_centerline(self, yellow_mask):

        height = yellow_mask.shape[0]
        width = yellow_mask.shape[1]
        lookahead_y = int(height * 0.42)

        center_roi = np.zeros_like(
            yellow_mask
        )

        center_polygon = np.array([[
            (int(width * 0.02), height),
            (int(width * 0.76), height),
            (int(width * 0.66), int(height * 0.12)),
            (int(width * 0.08), int(height * 0.12))
        ]], np.int32)

        cv2.fillPoly(
            center_roi,
            center_polygon,
            255
        )

        yellow_roi = cv2.bitwise_and(
            yellow_mask,
            center_roi
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            yellow_roi,
            8
        )

        valid_mask = np.zeros_like(
            yellow_roi
        )

        for label in range(1, num_labels):

            area = stats[label, cv2.CC_STAT_AREA]
            dash_width = stats[label, cv2.CC_STAT_WIDTH]
            dash_height = stats[label, cv2.CC_STAT_HEIGHT]

            if area >= 16 and dash_width >= 3 and dash_height >= 5:

                valid_mask[labels == label] = 255

        nonzero = valid_mask.nonzero()

        if len(nonzero[0]) < 25:

            self.yellow_miss_count += 1

            if self.yellow_miss_count > 5:

                self.yellowx_previous = None
                self.yellow_lookahead_previous = None

            return None, yellow_roi

        yellow_y = np.array(nonzero[0])
        yellow_x = np.array(nonzero[1])

        lower_part = yellow_y > int(height * 0.35)

        if np.count_nonzero(lower_part) >= 15:

            fit_y = yellow_y[lower_part]
            fit_x = yellow_x[lower_part]

        else:

            fit_y = yellow_y
            fit_x = yellow_x

        try:

            degree = 2 if len(yellow_x) >= 80 else 1

            fit = np.polyfit(
                yellow_y,
                yellow_x,
                degree
            )

            yellow_x_bottom = int(
                np.polyval(
                    fit,
                    height - 1
                )
            )

            yellow_x_lookahead = int(
                np.polyval(
                    fit,
                    lookahead_y
                )
            )

        except Exception:

            yellow_x_bottom = int(
                np.median(
                    fit_x
                )
            )

            yellow_x_lookahead = yellow_x_bottom

        yellow_x_bottom = int(
            np.clip(
                yellow_x_bottom,
                0,
                int(width * 0.72)
            )
        )

        yellow_x_lookahead = int(
            np.clip(
                yellow_x_lookahead,
                yellow_x_bottom - 145,
                yellow_x_bottom + 145
            )
        )

        yellow_x_lookahead = int(
            np.clip(
                yellow_x_lookahead,
                0,
                int(width * 0.76)
            )
        )

        if self.yellowx_previous is not None:

            yellow_x_bottom = int(
                0.75 * self.yellowx_previous +
                0.25 * yellow_x_bottom
            )

        if self.yellow_lookahead_previous is not None:

            yellow_x_lookahead = int(
                0.80 * self.yellow_lookahead_previous +
                0.20 * yellow_x_lookahead
            )

        self.yellowx_previous = yellow_x_bottom
        self.yellow_lookahead_previous = yellow_x_lookahead

        self.yellow_miss_count = 0

        yellow_info = {
            "bottom_x": yellow_x_bottom,
            "lookahead_x": yellow_x_lookahead,
            "lookahead_y": lookahead_y
        }

        return yellow_info, valid_mask

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

        # =====================================
        # sliding window
        # =====================================
        out_img, lane_center, right_lane_x, right_lane_lookahead = (
            self.slidewindow.slidewindow(
                white_binary
            )
        )
        right_lane_detected = self.slidewindow.right_lane_detected
        left_lane_x, left_lane_lookahead = (
            self.slidewindow.detect_left_lane(
                white_binary,
                out_img,
                right_lane_x,
                right_lane_lookahead,
                right_lane_detected,
            )
        )
        left_lane_detected = self.slidewindow.left_lane_detected
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
            if bottom_separation < 220 or lookahead_separation < 120:
                left_lane_detected = False
                self.slidewindow.left_lane_detected = False

        yellow_info, yellow_debug = self.detect_yellow_centerline(
            yellow_binary
        )
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
        yellow_center_x = None
        yellow_lookahead_x = None

        if yellow_info is not None:
            yellow_center_x = yellow_info["bottom_x"]
            yellow_lookahead_x = yellow_info["lookahead_x"]

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

        curve_delta = (
            bottom_lane_center - lookahead_lane_center
        )

        if curve_delta > 0.0:
            # 왼쪽 커브는 더 먼 시점부터 강하게 선행 조향한다.
            curve_feedforward = float(np.clip(
                curve_delta * 0.45,
                0,
                40
            ))
        else:
            curve_feedforward = float(np.clip(
                curve_delta * 0.32,
                -30,
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

        kd = 0.62

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
            min(angle, 100),
            -100
        )

        angle *= -1

        # 차량이 왼쪽 커브에서 언더스티어하는 특성을 보정한다.
        # 곡률이 클수록 왼쪽 조향만 최대 40% 증폭한다.
        if angle < 0.0:
            left_curve_gain = 1.0 + 0.40 * curve_strength
            angle = max(angle * left_curve_gain, -100.0)

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
        # 그 차선을 신뢰한다. 4개 미만일 때만 LOST 방향으로 복구한다.
        if (
            not left_lane_detected and
            right_lane_detected and
            right_valid_windows < 4
        ):
            angle = -self.single_lane_recovery_angle
        elif (
            not right_lane_detected and
            left_lane_detected and
            left_valid_windows < 4
        ):
            angle = self.single_lane_recovery_angle

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
            f'LEFT WHITE: {"TRACKED" if left_lane_detected else "LOST"}',
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255) if left_lane_detected else (0, 0, 255),
            2
        )
        cv2.putText(
            out_img,
            f'RIGHT WHITE: {"TRACKED" if right_lane_detected else "LOST"}',
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

        if yellow_center_x is not None:

            cv2.line(
                out_img,
                (yellow_center_x, 0),
                (yellow_center_x, out_img.shape[0]),
                (0, 255, 255),
                2
            )

            cv2.circle(
                out_img,
                (yellow_lookahead_x, yellow_info["lookahead_y"]),
                8,
                (0, 255, 255),
                -1
            )

        cv2.circle(
            out_img,
            (lane_center, out_img.shape[0] - 60),
            8,
            (255, 0, 255),
            -1
        )

        binary = cv2.bitwise_or(
            white_binary,
            yellow_binary
        )

        cv2.imshow(
            "binary",
            binary
        )

        cv2.imshow(
            "yellow_centerline",
            yellow_debug
        )

        cv2.imshow(
            "sliding_window",
            out_img
        )

        print("lane_center:", lane_center)
        print("target:", target)
        print("error:", error)
        print("angle:", angle)
        print("yellow_center_x:", yellow_center_x)
        print("yellow_lookahead_x:", yellow_lookahead_x)
        print("right_lane_x:", right_lane_x)
        print("right_lane_lookahead:", right_lane_lookahead)
        print("left_lane_x:", left_lane_x)
        print("left_lane_lookahead:", left_lane_lookahead)
        print("left_lane_detected:", left_lane_detected)
        print("right_lane_detected:", right_lane_detected)
        print("curve_feedforward:", curve_feedforward)

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
            self.current_speed = (
                0.92 * self.current_speed +
                0.08 * target_speed
            )
            speed = self.current_speed

            if not self.driving_enabled or self.pedestrian_blocking:
                speed = 0.0

            # 둘 이상의 노드가 같은 모터 토픽을 발행하면 명령이 서로
            # 덮어써지므로, 충돌이 해소될 때까지 이 노드는 퍼블리시 자체를 건너뛴다.
            if self.count_publishers('/xycar_motor') > 1:
                if not self.publisher_conflict_reported:
                    self.get_logger().error(
                        'Multiple /xycar_motor publishers detected; skipping publish'
                    )
                    self.publisher_conflict_reported = True
                cv2.waitKey(1)
                continue
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
