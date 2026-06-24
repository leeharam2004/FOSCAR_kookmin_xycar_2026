#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int64MultiArray
from xycar_msgs.msg import XycarMotor

# 0: 창 없음, 1: 핵심 창만, 2: HSV 트랙바와 모든 마스크
TRAFFIC_DEBUG_LEVEL = 1


class TrafficDetection(Node):
    """Detect red and green circular traffic lights from the front camera."""

    STATE_DRIVE = 'DRIVE'
    STATE_SIGNAL_WAIT = 'SIGNAL_WAIT'
    STATE_STOP = 'STOP'
    STATE_LEFT_TURN = 'LEFT_TURN'

    # 신호등은 영상 상단에 있고 라바콘은 주로 노면 쪽에 나타난다.
    TRAFFIC_ROI_TOP_RATIO = 0.0
    TRAFFIC_ROI_BOTTOM_RATIO = 0.55

    # 좌측에서 교차로로 진입하는 경찰차의 경광등 검출 영역이다.
    POLICE_ROI_X_START_RATIO = 0.0
    POLICE_ROI_X_END_RATIO = 0.75
    POLICE_ROI_Y_START_RATIO = 0.42
    POLICE_ROI_Y_END_RATIO = 0.92

    def __init__(self):
        super().__init__('traffic_detection')

        self.bridge = CvBridge()

        # 신호가 보이지 않는 구간에서는 차선 주행 명령을 그대로 통과시킨다.
        # 빨강은 정지, 초록은 출발, 빨강+초록은 좌회전으로 판정한다.
        self.declare_parameter('red_pixel_threshold', 50)
        self.declare_parameter('green_pixel_threshold', 50)
        self.declare_parameter('confirmation_frames', 3)
        self.declare_parameter('left_turn_angle', -100.0)
        self.declare_parameter('left_turn_speed', 3.0)
        self.declare_parameter('signal_left_turn_approach_sec', 1.5)
        self.declare_parameter('signal_wait_timeout_sec', 3.0)
        self.declare_parameter('no_signal_left_turn_hold_sec', 7.0)
        self.declare_parameter('police_blue_pixel_threshold', 12)
        self.declare_parameter('police_red_pixel_threshold', 12)
        self.declare_parameter('police_confirmation_frames', 3)
        self.declare_parameter('police_clear_frames', 10)
        self.declare_parameter('lane_reacquire_frames', 3)
        self.declare_parameter('left_turn_timeout_frames', 150)
        self.red_pixel_threshold = int(
            self.get_parameter('red_pixel_threshold').value
        )
        self.green_pixel_threshold = int(
            self.get_parameter('green_pixel_threshold').value
        )
        self.confirmation_frames = max(
            1,
            int(self.get_parameter('confirmation_frames').value),
        )
        self.left_turn_angle = float(
            self.get_parameter('left_turn_angle').value
        )
        self.left_turn_speed = float(
            self.get_parameter('left_turn_speed').value
        )
        self.signal_left_turn_approach_sec = max(
            0.0,
            float(self.get_parameter('signal_left_turn_approach_sec').value),
        )
        self.signal_wait_timeout_sec = max(
            0.0,
            float(self.get_parameter('signal_wait_timeout_sec').value),
        )
        self.no_signal_left_turn_hold_sec = max(
            0.0,
            float(self.get_parameter('no_signal_left_turn_hold_sec').value),
        )
        self.police_blue_pixel_threshold = max(
            1,
            int(self.get_parameter('police_blue_pixel_threshold').value),
        )
        self.police_red_pixel_threshold = max(
            1,
            int(self.get_parameter('police_red_pixel_threshold').value),
        )
        self.police_confirmation_frames = max(
            1,
            int(self.get_parameter('police_confirmation_frames').value),
        )
        self.police_clear_frames = max(
            1,
            int(self.get_parameter('police_clear_frames').value),
        )
        self.lane_reacquire_frames = max(
            1,
            int(self.get_parameter('lane_reacquire_frames').value),
        )
        self.left_turn_timeout_frames = max(
            1,
            int(self.get_parameter('left_turn_timeout_frames').value),
        )

        self.driving_state = self.STATE_DRIVE
        self.pending_signal_state = None
        self.pending_signal_frames = 0
        self.lane_was_lost_during_turn = False
        self.require_lane_loss_before_reacquire = True
        self.lane_reacquisition_frames = 0
        self.left_turn_frame_count = 0
        self.left_turn_timeout_enabled = True
        self.no_signal_left_turn = False
        self.no_signal_lane_hold_started_at = None
        self.signal_left_turn_approach_started_at = None
        self.police_car_detected = False
        self.police_detection_frames = 0
        self.police_clear_frame_count = 0
        self.stop_line_detected = False
        self.left_lane_detected = False
        self.right_lane_detected = False
        self.signal_wait_started_at = None

        self.camera_subscription = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.traffic_light_pub = self.create_publisher(
            Int64MultiArray,
            '/traffic_light',
            1,
        )
        self.red_center_pub = self.create_publisher(
            Int64MultiArray,
            '/red_center',
            1,
        )
        self.green_center_pub = self.create_publisher(
            Int64MultiArray,
            '/green_center',
            1,
        )
        self.police_car_pub = self.create_publisher(
            Bool,
            '/police_car_detected',
            10,
        )
        self.motor_subscription = self.create_subscription(
            XycarMotor,
            '/lane_motor_cmd',
            self.motor_callback,
            10,
        )
        self.stop_line_subscription = self.create_subscription(
            Bool,
            '/stop_line',
            self.stop_line_callback,
            10,
        )
        self.lane_detection_subscription = self.create_subscription(
            Int64MultiArray,
            '/lane_detection_status',
            self.lane_detection_callback,
            10,
        )
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10,
        )

        if TRAFFIC_DEBUG_LEVEL >= 2:
            self._create_debug_windows()
        self.get_logger().info('ROS2 traffic-light detector started')

    def stop_line_callback(self, message):
        was_detected = self.stop_line_detected
        self.stop_line_detected = bool(message.data)

        # 같은 정지선이 여러 프레임 유지돼도 한 번만 정지한다. 정지선이
        # 화면에서 사라진 뒤 다음 정지선이 나타나면 다시 활성화된다.
        if (
            self.stop_line_detected
            and not was_detected
            and self.driving_state == self.STATE_DRIVE
        ):
            self._set_driving_state(self.STATE_SIGNAL_WAIT)

    def lane_detection_callback(self, message):
        if len(message.data) < 2:
            return

        self.left_lane_detected = bool(message.data[0])
        self.right_lane_detected = bool(message.data[1])
        self.update_lane_reacquisition()

    def motor_callback(self, lane_command):
        """Apply the traffic-light state to the lane-driving command."""
        self._finish_no_signal_left_turn_hold_if_elapsed()
        output_command = XycarMotor()

        if self.driving_state in (self.STATE_STOP, self.STATE_SIGNAL_WAIT):
            output_command.angle = float(lane_command.angle)
            output_command.speed = 0.0
        elif self.driving_state == self.STATE_LEFT_TURN:
            if self._signal_left_turn_approach_is_active():
                output_command.angle = 0.0
            else:
                output_command.angle = self.left_turn_angle
            output_command.speed = self.left_turn_speed
        else:
            output_command.angle = float(lane_command.angle)
            output_command.speed = float(lane_command.speed)

        self.motor_pub.publish(output_command)

    def update_driving_state(
        self,
        red_pixel_count,
        green_pixel_count,
    ):
        """Debounce light detections and select drive, stop, or left turn."""
        # 좌회전을 시작한 뒤에는 신호가 화면에서 사라져도 새 차선을 찾을
        # 때까지 신호 검출 결과로 상태를 바꾸지 않는다.
        if self.driving_state == self.STATE_LEFT_TURN:
            return

        red_detected = red_pixel_count >= self.red_pixel_threshold
        green_detected = green_pixel_count >= self.green_pixel_threshold

        # 정지선을 만나기 전에는 신호를 주행 상태에 반영하지 않는다.
        if self.driving_state not in (self.STATE_SIGNAL_WAIT, self.STATE_STOP):
            self.pending_signal_state = None
            self.pending_signal_frames = 0
            return

        if red_detected and green_detected:
            candidate_state = self.STATE_LEFT_TURN
        elif red_detected:
            candidate_state = self.STATE_STOP
        elif green_detected:
            candidate_state = self.STATE_DRIVE
        else:
            candidate_state = None

        # 경찰차가 좌회전 충돌 구역에 있으면 신호 좌회전을 시작하지
        # 않는다. 차량이 사라지면 현재 신호를 다시 확인해 출발한다.
        if (
            candidate_state == self.STATE_LEFT_TURN
            and self.police_car_detected
        ):
            candidate_state = self.STATE_STOP

        if candidate_state is None:
            self.pending_signal_state = None
            self.pending_signal_frames = 0
            self._start_left_turn_if_signal_timed_out()
            return

        if candidate_state == self.pending_signal_state:
            self.pending_signal_frames += 1
        else:
            self.pending_signal_state = candidate_state
            self.pending_signal_frames = 1

        if self.pending_signal_frames < self.confirmation_frames:
            return

        if candidate_state != self.driving_state:
            self._set_driving_state(candidate_state)

    def _start_left_turn_if_signal_timed_out(self):
        """Start a left turn when no signal is visible after stopping."""
        if (
            self.driving_state != self.STATE_SIGNAL_WAIT
            or self.signal_wait_started_at is None
        ):
            return

        elapsed_sec = (
            self.get_clock().now() - self.signal_wait_started_at
        ).nanoseconds / 1e9
        if elapsed_sec >= self.signal_wait_timeout_sec:
            self.get_logger().info(
                'No signal detected for '
                f'{self.signal_wait_timeout_sec:.1f}s: starting left turn'
            )
            self._set_driving_state(self.STATE_LEFT_TURN, emit_log=False)
            # 왼쪽 차선이 안정적으로 검출된 뒤에도 설정된 시간 동안
            # 최대 좌조향을 유지하고 나서 차선 조향으로 복귀한다.
            self.left_turn_timeout_enabled = False
            self.require_lane_loss_before_reacquire = False
            self.no_signal_left_turn = True
            self.signal_left_turn_approach_started_at = None

    def _signal_left_turn_approach_is_active(self):
        if (
            self.driving_state != self.STATE_LEFT_TURN
            or self.no_signal_left_turn
            or self.signal_left_turn_approach_started_at is None
        ):
            return False

        elapsed_sec = (
            self.get_clock().now() - self.signal_left_turn_approach_started_at
        ).nanoseconds / 1e9
        if elapsed_sec < self.signal_left_turn_approach_sec:
            return True

        self.signal_left_turn_approach_started_at = None
        self.lane_was_lost_during_turn = False
        self.lane_reacquisition_frames = 0
        self.left_turn_frame_count = 0
        self.get_logger().info(
            'Signal intersection approach completed: maximum-left turn started'
        )
        return False

    def _finish_no_signal_left_turn_hold_if_elapsed(self):
        if (
            self.driving_state != self.STATE_LEFT_TURN
            or not self.no_signal_left_turn
            or self.no_signal_lane_hold_started_at is None
        ):
            return

        elapsed_sec = (
            self.get_clock().now() - self.no_signal_lane_hold_started_at
        ).nanoseconds / 1e9
        if elapsed_sec >= self.no_signal_left_turn_hold_sec:
            self.get_logger().info(
                'No-signal maximum-left hold completed: lane driving resumed'
            )
            self._set_driving_state(self.STATE_DRIVE)

    def _set_driving_state(self, state, emit_log=True):
        self.driving_state = state
        self.pending_signal_state = None
        self.pending_signal_frames = 0
        self.no_signal_left_turn = False
        self.no_signal_lane_hold_started_at = None
        self.signal_left_turn_approach_started_at = None
        self.signal_wait_started_at = (
            self.get_clock().now()
            if state == self.STATE_SIGNAL_WAIT
            else None
        )

        if state == self.STATE_LEFT_TURN:
            self.signal_left_turn_approach_started_at = self.get_clock().now()
            self.lane_was_lost_during_turn = False
            self.require_lane_loss_before_reacquire = True
            self.lane_reacquisition_frames = 0
            self.left_turn_frame_count = 0
            self.left_turn_timeout_enabled = True
            if emit_log:
                self.get_logger().info(
                    'Red and green confirmed: straight intersection approach '
                    'started'
                )
        elif state == self.STATE_SIGNAL_WAIT and emit_log:
            self.get_logger().info(
                'Stop line confirmed: vehicle stopped, checking signal'
            )
        elif state == self.STATE_STOP and emit_log:
            self.get_logger().info('Stop line confirmed: vehicle stopped')
        elif state == self.STATE_DRIVE and emit_log:
            self.get_logger().info('Lane driving resumed')

    def update_lane_reacquisition(self):
        """Return control when the required lane detectors are stable."""
        if self.driving_state != self.STATE_LEFT_TURN:
            return

        if self._signal_left_turn_approach_is_active():
            return

        self.left_turn_frame_count += 1
        if self.no_signal_left_turn:
            if self.no_signal_lane_hold_started_at is not None:
                return

            if self.left_lane_detected:
                self.lane_reacquisition_frames += 1
            else:
                self.lane_reacquisition_frames = 0

            if self.lane_reacquisition_frames >= self.lane_reacquire_frames:
                self.no_signal_lane_hold_started_at = self.get_clock().now()
                self.get_logger().info(
                    'Left lane reacquired: maintaining maximum-left turn for '
                    f'{self.no_signal_left_turn_hold_sec:.1f}s'
                )
            return

        if self.require_lane_loss_before_reacquire:
            lane_reacquired = (
                self.left_lane_detected and self.right_lane_detected
            )
        else:
            lane_reacquired = self.left_lane_detected

        if not lane_reacquired:
            self.lane_was_lost_during_turn = True
            self.lane_reacquisition_frames = 0
        elif (
            not self.require_lane_loss_before_reacquire
            or self.lane_was_lost_during_turn
        ):
            self.lane_reacquisition_frames += 1
            if self.lane_reacquisition_frames >= self.lane_reacquire_frames:
                self._set_driving_state(self.STATE_DRIVE)
                return

        # 차선을 끝내 찾지 못한 채 최대 조향을 계속하는 상황을 제한한다.
        if (
            self.left_turn_timeout_enabled
            and self.left_turn_frame_count >= self.left_turn_timeout_frames
        ):
            self.get_logger().warning(
                'Lane reacquisition timed out: vehicle stopped'
            )
            self._set_driving_state(self.STATE_STOP, emit_log=False)

    def _create_debug_windows(self):
        cv2.namedWindow('Red Trackbars', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('H_min_red1', 'Red Trackbars', 0, 179, self.nothing)
        cv2.createTrackbar('H_max_red1', 'Red Trackbars', 10, 179, self.nothing)
        cv2.createTrackbar('S_min_red1', 'Red Trackbars', 170, 255, self.nothing)
        cv2.createTrackbar('S_max_red1', 'Red Trackbars', 255, 255, self.nothing)
        cv2.createTrackbar('V_min_red1', 'Red Trackbars', 120, 255, self.nothing)
        cv2.createTrackbar('V_max_red1', 'Red Trackbars', 255, 255, self.nothing)

        cv2.namedWindow('Green Trackbars', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('H_min_green1', 'Green Trackbars', 38, 179, self.nothing)
        cv2.createTrackbar('H_max_green1', 'Green Trackbars', 102, 179, self.nothing)
        cv2.createTrackbar('S_min_green1', 'Green Trackbars', 170, 255, self.nothing)
        cv2.createTrackbar('S_max_green1', 'Green Trackbars', 255, 255, self.nothing)
        cv2.createTrackbar('V_min_green1', 'Green Trackbars', 120, 255, self.nothing)
        cv2.createTrackbar('V_max_green1', 'Green Trackbars', 255, 255, self.nothing)

    @staticmethod
    def nothing(_value):
        pass

    def _update_police_detection(self, candidate_detected):
        was_detected = self.police_car_detected

        if candidate_detected:
            self.police_detection_frames += 1
            self.police_clear_frame_count = 0
            if self.police_detection_frames >= self.police_confirmation_frames:
                self.police_car_detected = True
        else:
            self.police_detection_frames = 0
            if self.police_car_detected:
                self.police_clear_frame_count += 1
                if self.police_clear_frame_count >= self.police_clear_frames:
                    self.police_car_detected = False
                    self.police_clear_frame_count = 0

        if self.police_car_detected and not was_detected:
            self.get_logger().warning(
                'Police car detected in left-turn path: holding position'
            )
            # 신호 좌회전 진입 중 발견한 경우에도 즉시 정지한다.
            # 신호 미검출 교차로의 별도 좌회전 시퀀스는 유지한다.
            if (
                self.driving_state == self.STATE_LEFT_TURN
                and not self.no_signal_left_turn
            ):
                self._set_driving_state(self.STATE_STOP, emit_log=False)
        elif was_detected and not self.police_car_detected:
            self.get_logger().info(
                'Police car cleared: traffic-light decision resumed'
            )

    def detect_police_car(self, hsv_image):
        """Detect paired emergency-light colors in the conflict ROI."""
        height, width = hsv_image.shape[:2]
        x_start = int(width * self.POLICE_ROI_X_START_RATIO)
        x_end = int(width * self.POLICE_ROI_X_END_RATIO)
        y_start = int(height * self.POLICE_ROI_Y_START_RATIO)
        y_end = int(height * self.POLICE_ROI_Y_END_RATIO)

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y_start:y_end, x_start:x_end] = 255

        blue_mask = cv2.inRange(
            hsv_image,
            np.array([90, 120, 80], dtype=np.uint8),
            np.array([135, 255, 255], dtype=np.uint8),
        )
        red_mask_low = cv2.inRange(
            hsv_image,
            np.array([0, 150, 100], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
        )
        red_mask_high = cv2.inRange(
            hsv_image,
            np.array([170, 150, 100], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)

        blue_mask = cv2.bitwise_and(blue_mask, roi_mask)
        red_mask = cv2.bitwise_and(red_mask, roi_mask)
        light_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            light_kernel,
        )
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            light_kernel,
        )

        blue_pixel_count = int(np.count_nonzero(blue_mask))
        red_pixel_count = int(np.count_nonzero(red_mask))
        candidate_detected = (
            blue_pixel_count >= self.police_blue_pixel_threshold
            and red_pixel_count >= self.police_red_pixel_threshold
        )
        self._update_police_detection(candidate_detected)
        police_message = Bool()
        police_message.data = self.police_car_detected
        self.police_car_pub.publish(police_message)

        return {
            'blue_mask': blue_mask,
            'red_mask': red_mask,
            'blue_pixel_count': blue_pixel_count,
            'red_pixel_count': red_pixel_count,
            'roi': (x_start, y_start, x_end, y_end),
        }

    def camera_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        except CvBridgeError as error:
            self.get_logger().warning(f'Image conversion failed: {error}')
            return

        self.detect_traffic_light(image)
        if TRAFFIC_DEBUG_LEVEL > 0:
            cv2.waitKey(1)

    @staticmethod
    def filter_circular_contours(
        contours,
        circularity_threshold=0.45,
        min_area=30,
        aspect_ratio_min=0.55,
        aspect_ratio_max=1.8,
    ):
        filtered_contours = []
        contour_areas = []
        contour_circularities = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0 or area < min_area:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            _, _, width, height = cv2.boundingRect(contour)
            if width == 0 or height == 0:
                continue

            aspect_ratio = width / float(height)
            # 실제 램프는 빛 번짐 때문에 완전한 원이 아닐 수 있지만,
            # 가로·세로 비율은 대체로 정사각형에 가깝다.
            shape_is_light_like = (
                circularity >= circularity_threshold and
                aspect_ratio_min <= aspect_ratio <= aspect_ratio_max
            )
            if shape_is_light_like:
                filtered_contours.append(contour)
                contour_areas.append(area)
                contour_circularities.append(circularity)

        return filtered_contours, contour_areas, contour_circularities

    @staticmethod
    def get_contour_centers(contours):
        centers = []

        for contour in contours:
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue

            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            centers.append((center_x, center_y))

        return centers

    def _trackbar_range(self, prefix, window):
        if TRAFFIC_DEBUG_LEVEL < 2:
            default_ranges = {
                'red1': (
                    np.array([0, 170, 120], dtype=np.uint8),
                    np.array([10, 255, 255], dtype=np.uint8),
                ),
                'green1': (
                    np.array([38, 170, 120], dtype=np.uint8),
                    np.array([102, 255, 255], dtype=np.uint8),
                ),
            }
            return default_ranges[prefix]

        lower = np.array([
            cv2.getTrackbarPos(f'H_min_{prefix}', window),
            cv2.getTrackbarPos(f'S_min_{prefix}', window),
            cv2.getTrackbarPos(f'V_min_{prefix}', window),
        ], dtype=np.uint8)
        upper = np.array([
            cv2.getTrackbarPos(f'H_max_{prefix}', window),
            cv2.getTrackbarPos(f'S_max_{prefix}', window),
            cv2.getTrackbarPos(f'V_max_{prefix}', window),
        ], dtype=np.uint8)
        return lower, upper

    def detect_traffic_light(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        police_result = self.detect_police_car(hsv_image)

        height, width = image.shape[:2]
        roi_top = int(height * self.TRAFFIC_ROI_TOP_RATIO)
        roi_bottom = int(height * self.TRAFFIC_ROI_BOTTOM_RATIO)
        traffic_roi = np.zeros((height, width), dtype=np.uint8)
        traffic_roi[roi_top:roi_bottom, :] = 255

        lower_red, upper_red = self._trackbar_range('red1', 'Red Trackbars')
        lower_green, upper_green = self._trackbar_range(
            'green1',
            'Green Trackbars',
        )

        red_mask_low = cv2.inRange(hsv_image, lower_red, upper_red)
        # OpenCV HSV에서 빨강은 H=0과 H=179 경계에 걸쳐 있다.
        lower_red_high = np.array(
            [170, lower_red[1], lower_red[2]],
            dtype=np.uint8,
        )
        upper_red_high = np.array(
            [179, upper_red[1], upper_red[2]],
            dtype=np.uint8,
        )
        red_mask_high = cv2.inRange(
            hsv_image,
            lower_red_high,
            upper_red_high,
        )
        red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # 노면 쪽 빨간 물체는 신호등 후보에서 먼저 제외한다.
        red_mask = cv2.bitwise_and(red_mask, traffic_roi)
        green_mask = cv2.bitwise_and(green_mask, traffic_roi)

        # LED 점이나 노출 차이로 끊어진 램프 영역을 하나로 합친다.
        light_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, light_kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, light_kernel)

        red_contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        green_contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        red_filtered, _, _ = self.filter_circular_contours(red_contours)
        green_filtered, _, _ = self.filter_circular_contours(green_contours)

        red_result = np.zeros_like(red_mask)
        # 초록등과 초록 화살표를 모양으로 구분하지 않고 동일한 초록색
        # 픽셀로 사용한다.
        green_result = green_mask.copy()
        cv2.drawContours(red_result, red_filtered, -1, 255, cv2.FILLED)

        # 원형 조건을 통과한 후보만 주행 상태 판단에 사용한다.
        # 기존에는 필터링 전 색상 픽셀을 세어 라바콘도 포함됐다.
        red_pixel_count = int(np.count_nonzero(red_result))
        green_pixel_count = int(np.count_nonzero(green_result))
        red_centers = self.get_contour_centers(red_filtered)
        green_centers = self.get_contour_centers(green_filtered)
        self.update_driving_state(
            red_pixel_count,
            green_pixel_count,
        )

        cv2.putText(
            green_result,
            f'green pixels: {green_pixel_count}',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            255,
            2,
        )
        cv2.putText(
            red_result,
            f'red pixels: {red_pixel_count}',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            255,
            2,
        )

        traffic_message = Int64MultiArray()
        traffic_message.data = [red_pixel_count, green_pixel_count]
        self.traffic_light_pub.publish(traffic_message)

        red_centers_message = Int64MultiArray()
        red_centers_message.data = [
            coordinate
            for center in red_centers
            for coordinate in center
        ]
        self.red_center_pub.publish(red_centers_message)

        green_centers_message = Int64MultiArray()
        green_centers_message.data = [
            coordinate
            for center in green_centers
            for coordinate in center
        ]
        self.green_center_pub.publish(green_centers_message)

        debug_image = image.copy()
        cv2.rectangle(
            debug_image,
            (0, roi_top),
            (width - 1, max(roi_top, roi_bottom - 1)),
            (255, 255, 0),
            2,
        )
        if self.driving_state == self.STATE_DRIVE:
            state_color = (0, 255, 0)
        elif self.driving_state == self.STATE_LEFT_TURN:
            state_color = (0, 255, 255)
        elif self.driving_state == self.STATE_SIGNAL_WAIT:
            state_color = (255, 255, 0)
        else:
            state_color = (0, 0, 255)
        cv2.putText(
            debug_image,
            f'driving: {self.driving_state}',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            state_color,
            2,
        )
        police_x_start, police_y_start, police_x_end, police_y_end = (
            police_result['roi']
        )
        cv2.rectangle(
            debug_image,
            (police_x_start, police_y_start),
            (police_x_end - 1, police_y_end - 1),
            (0, 0, 255) if self.police_car_detected else (255, 128, 0),
            2,
        )
        cv2.putText(
            debug_image,
            'police: '
            f'{"DETECTED" if self.police_car_detected else "clear"} '
            f'(B:{police_result["blue_pixel_count"]} '
            f'R:{police_result["red_pixel_count"]})',
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255) if self.police_car_detected else (255, 255, 255),
            2,
        )
        police_debug = cv2.bitwise_or(
            police_result['blue_mask'],
            police_result['red_mask'],
        )
        if TRAFFIC_DEBUG_LEVEL >= 1:
            cv2.imshow('src', debug_image)
            cv2.imshow('green_result', green_result)
            cv2.imshow('police_result', police_debug)
        if TRAFFIC_DEBUG_LEVEL >= 2:
            cv2.imshow('red_mask', red_mask)
            cv2.imshow('green_mask', green_mask)
            cv2.imshow('red_result', red_result)
            

def main(args=None):
    rclpy.init(args=args)
    node = TrafficDetection()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
