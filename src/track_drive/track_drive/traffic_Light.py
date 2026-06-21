#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

# False로 설정하면 모든 cv2 창을 띄우지 않고 하드코딩 상수로 HSV 범위를 사용한다.
DEBUG = False

_RED_LOWER  = np.array([  0, 170, 120], dtype=np.uint8)
_RED_UPPER  = np.array([ 10, 255, 255], dtype=np.uint8)
_GREEN_LOWER = np.array([ 38, 165, 120], dtype=np.uint8)
_GREEN_UPPER = np.array([102, 255, 255], dtype=np.uint8)

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray


class TrafficDetection(Node):
    """Detect red and green circular traffic lights from the front camera."""

    # 신호등은 영상 상단에 있고 라바콘은 주로 노면 쪽에 나타난다.
    TRAFFIC_ROI_TOP_RATIO = 0.0
    TRAFFIC_ROI_BOTTOM_RATIO = 0.55

    def __init__(self):
        super().__init__('traffic_detection')

        self.bridge = CvBridge()

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

        self._create_debug_windows()
        self.get_logger().info('ROS2 traffic-light detector started')

    def _create_debug_windows(self):
        if not DEBUG:
            return
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
        cv2.createTrackbar('S_min_green1', 'Green Trackbars', 165, 255, self.nothing)
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
        if DEBUG:
            cv2.waitKey(1)

    @staticmethod
    def filter_circular_contours(
        contours,
        circularity_threshold=0.45,
        min_area=30,
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
                0.55 <= aspect_ratio <= 1.8
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
    def _trackbar_range(prefix, window):
        if not DEBUG:
            if prefix == 'red1':
                return _RED_LOWER.copy(), _RED_UPPER.copy()
            return _GREEN_LOWER.copy(), _GREEN_UPPER.copy()
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
        green_result = np.zeros_like(green_mask)
        cv2.drawContours(red_result, red_filtered, -1, 255, cv2.FILLED)
        cv2.drawContours(green_result, green_filtered, -1, 255, cv2.FILLED)

        # 원형 조건을 통과한 후보만 주행 상태 판단에 사용한다.
        # 기존에는 필터링 전 색상 픽셀을 세어 라바콘도 포함됐다.
        red_pixel_count = int(np.count_nonzero(red_result))
        green_pixel_count = int(np.count_nonzero(green_result))
        red_centers = self.get_contour_centers(red_filtered)
        green_centers = self.get_contour_centers(green_filtered)

        if DEBUG:
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

        if DEBUG:
            debug_image = image.copy()
            cv2.rectangle(
                debug_image,
                (0, roi_top),
                (width - 1, max(roi_top, roi_bottom - 1)),
                (255, 255, 0),
                2,
            )
            cv2.imshow('src', debug_image)
            cv2.imshow('red_mask', red_mask)
            cv2.imshow('green_mask', green_mask)
            cv2.imshow('red_result', red_result)
            cv2.imshow('green_result', green_result)
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
