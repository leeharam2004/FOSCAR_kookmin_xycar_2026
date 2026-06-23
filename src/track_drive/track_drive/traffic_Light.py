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

from track_drive.track_drive import (
    ROI_Y_END,
    ROI_Y_START,
    Preprocessing,
    SlideWindow,
)


class TrafficDetection(Node):
    """Detect red and green circular traffic lights from the front camera."""

    STATE_DRIVE = 'DRIVE'
    STATE_RED_APPROACH = 'RED_APPROACH'
    STATE_STOP = 'STOP'
    STATE_LEFT_TURN = 'LEFT_TURN'

    # 신호등은 영상 상단에 있고 라바콘은 주로 노면 쪽에 나타난다.
    TRAFFIC_ROI_TOP_RATIO = 0.0
    TRAFFIC_ROI_BOTTOM_RATIO = 0.55

    def __init__(self):
        super().__init__('traffic_detection')

        self.bridge = CvBridge()

        # 신호가 보이지 않는 구간에서는 차선 주행 명령을 그대로 통과시킨다.
        # 빨강은 정지, 초록은 출발, 빨강+초록은 좌회전으로 판정한다.
        self.declare_parameter('red_pixel_threshold', 50)
        self.declare_parameter('green_pixel_threshold', 50)
        self.declare_parameter('arrow_pixel_threshold', 15)
        self.declare_parameter('confirmation_frames', 3)
        self.declare_parameter('left_turn_angle', -100.0)
        self.declare_parameter('left_turn_speed', 3.0)
        self.declare_parameter('lane_reacquire_frames', 3)
        self.declare_parameter('left_turn_timeout_frames', 150)
        self.red_pixel_threshold = int(
            self.get_parameter('red_pixel_threshold').value
        )
        self.green_pixel_threshold = int(
            self.get_parameter('green_pixel_threshold').value
        )
        self.arrow_pixel_threshold = int(
            self.get_parameter('arrow_pixel_threshold').value
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
        self.lane_reacquisition_frames = 0
        self.left_turn_frame_count = 0
        self.stop_line_detected = False

        # track_drive.py는 수정하지 않고 같은 검출기를 별도 상태로 사용한다.
        # 좌회전 중에만 실행하므로 평상시에는 중복 영상 처리 비용이 없다.
        self.lane_preprocessing = Preprocessing()
        self.lane_detector = SlideWindow()

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
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10,
        )

        self._create_debug_windows()
        self.get_logger().info('ROS2 traffic-light detector started')

    def stop_line_callback(self, message):
        self.stop_line_detected = bool(message.data)
        if (
            self.driving_state == self.STATE_RED_APPROACH
            and self.stop_line_detected
        ):
            self._set_driving_state(self.STATE_STOP)

    def motor_callback(self, lane_command):
        """Apply the traffic-light state to the lane-driving command."""
        output_command = XycarMotor()

        if self.driving_state == self.STATE_STOP:
            output_command.angle = float(lane_command.angle)
            output_command.speed = 0.0
        elif self.driving_state == self.STATE_LEFT_TURN:
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
        arrow_pixel_count=0,
    ):
        """Debounce light detections and select drive, stop, or left turn."""
        # 좌회전을 시작한 뒤에는 신호가 화면에서 사라져도 새 차선을 찾을
        # 때까지 신호 검출 결과로 상태를 바꾸지 않는다.
        if self.driving_state == self.STATE_LEFT_TURN:
            return

        red_detected = red_pixel_count >= self.red_pixel_threshold
        green_detected = green_pixel_count >= self.green_pixel_threshold
        arrow_detected = arrow_pixel_count >= self.arrow_pixel_threshold

        if red_detected and (green_detected or arrow_detected):
            candidate_state = self.STATE_LEFT_TURN
        elif red_detected:
            if self.driving_state == self.STATE_STOP:
                candidate_state = self.STATE_STOP
            else:
                candidate_state = self.STATE_RED_APPROACH
        elif green_detected:
            candidate_state = self.STATE_DRIVE
        else:
            candidate_state = None

        if candidate_state is None:
            self.pending_signal_state = None
            self.pending_signal_frames = 0
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

    def _set_driving_state(self, state, emit_log=True):
        self.driving_state = state
        self.pending_signal_state = None
        self.pending_signal_frames = 0

        if state == self.STATE_LEFT_TURN:
            self.lane_was_lost_during_turn = False
            self.lane_reacquisition_frames = 0
            self.left_turn_frame_count = 0
            if emit_log:
                self.get_logger().info(
                    'Left arrow confirmed: maximum-left turn started'
                )
        elif state == self.STATE_RED_APPROACH:
            if emit_log:
                self.get_logger().info(
                    'Red light confirmed: approaching stop line'
                )
            if self.stop_line_detected:
                self._set_driving_state(self.STATE_STOP, emit_log=emit_log)
        elif state == self.STATE_STOP and emit_log:
            self.get_logger().info('Stop line confirmed: vehicle stopped')
        elif state == self.STATE_DRIVE and emit_log:
            self.get_logger().info('Lane driving resumed')

    def update_lane_reacquisition(self, image):
        """Return control after the old lane is lost and a lane is reacquired."""
        if self.driving_state != self.STATE_LEFT_TURN:
            return

        self.left_turn_frame_count += 1
        roi = image[ROI_Y_START:ROI_Y_END, :].copy()
        lane_input = self.lane_preprocessing.run(roi)
        lane_result = self.lane_detector.run(lane_input['warped_white'])
        lane_detected = (
            lane_result['left_detected'] or lane_result['right_detected']
        )

        if not lane_detected:
            self.lane_was_lost_during_turn = True
            self.lane_reacquisition_frames = 0
        elif self.lane_was_lost_during_turn:
            self.lane_reacquisition_frames += 1
            if self.lane_reacquisition_frames >= self.lane_reacquire_frames:
                self._set_driving_state(self.STATE_DRIVE)
                return

        # 차선을 끝내 찾지 못한 채 최대 조향을 계속하는 상황을 제한한다.
        if self.left_turn_frame_count >= self.left_turn_timeout_frames:
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

    def camera_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        except CvBridgeError as error:
            self.get_logger().warning(f'Image conversion failed: {error}')
            return

        self.detect_traffic_light(image)
        self.update_lane_reacquisition(image)
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

    @staticmethod
    def build_arrow_search_mask(relaxed_green_mask, red_contours):
        """Keep relaxed green pixels near and to the right of a red lamp."""
        height, width = relaxed_green_mask.shape[:2]
        search_roi = np.zeros_like(relaxed_green_mask)

        for red_contour in red_contours:
            red_x, red_y, red_width, red_height = cv2.boundingRect(red_contour)
            vertical_margin = max(15, red_height * 2)
            search_left = min(width, red_x + red_width)
            search_right = min(width, red_x + int(width * 0.65))
            search_top = max(0, red_y - vertical_margin)
            search_bottom = min(
                height,
                red_y + red_height + vertical_margin,
            )

            if search_left < search_right and search_top < search_bottom:
                search_roi[
                    search_top:search_bottom,
                    search_left:search_right,
                ] = 255

        arrow_mask = cv2.bitwise_and(relaxed_green_mask, search_roi)
        # 멀리 있는 LED 점을 지우지 않고, 끊어진 화살표 픽셀만 합친다.
        arrow_mask = cv2.morphologyEx(
            arrow_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        return arrow_mask

    @staticmethod
    def _trackbar_range(prefix, window):
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
        hue, saturation, value = cv2.split(hsv_image)

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
        # 화살표는 작은 LED의 빛 번짐과 카메라 노출 때문에 일반 초록등보다
        # 채도와 밝기가 낮게 잡힐 수 있다. 빨간등 주변에서만 사용할 완화
        # 마스크이므로 일반 초록 판정보다 범위를 넓힌다.
        relaxed_green_mask = cv2.inRange(
            hsv_image,
            np.array([25, 60, 70], dtype=np.uint8),
            np.array([110, 255, 255], dtype=np.uint8),
        )

        # 노면 쪽 빨간 물체는 신호등 후보에서 먼저 제외한다.
        red_mask = cv2.bitwise_and(red_mask, traffic_roi)
        green_mask = cv2.bitwise_and(green_mask, traffic_roi)
        relaxed_green_mask = cv2.bitwise_and(
            relaxed_green_mask,
            traffic_roi,
        )

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
        arrow_mask = self.build_arrow_search_mask(
            relaxed_green_mask,
            red_filtered,
        )

        red_result = np.zeros_like(red_mask)
        green_result = np.zeros_like(green_mask)
        cv2.drawContours(red_result, red_filtered, -1, 255, cv2.FILLED)
        cv2.drawContours(green_result, green_filtered, -1, 255, cv2.FILLED)

        # 원형 조건을 통과한 후보만 주행 상태 판단에 사용한다.
        # 기존에는 필터링 전 색상 픽셀을 세어 라바콘도 포함됐다.
        red_pixel_count = int(np.count_nonzero(red_result))
        green_pixel_count = int(np.count_nonzero(green_result))
        arrow_pixel_count = int(np.count_nonzero(arrow_mask))
        red_centers = self.get_contour_centers(red_filtered)
        green_centers = self.get_contour_centers(green_filtered)
        self.update_driving_state(
            red_pixel_count,
            green_pixel_count,
            arrow_pixel_count,
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
        cv2.putText(
            arrow_mask,
            f'arrow pixels: {arrow_pixel_count}',
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
        elif self.driving_state == self.STATE_RED_APPROACH:
            state_color = (0, 165, 255)
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
        cv2.imshow('src', debug_image)
        cv2.imshow('red_mask', red_mask)
        cv2.imshow('green_mask', green_mask)
        cv2.imshow('red_result', red_result)
        cv2.imshow('green_result', green_result)
        cv2.imshow('arrow_mask', arrow_mask)
        cv2.imshow('h', hue)
        cv2.imshow('s', saturation)
        cv2.imshow('v', value)


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
