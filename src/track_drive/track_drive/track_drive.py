#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Int64MultiArray, String
from xycar_msgs.msg import XycarMotor

# =============================================
# 디버그 레벨
#   0 — 창 없음
#   1 — sliding_window 창만 (지금은 birds_eye_binary로 대체)
#   2 — 모든 창 표시
# =============================================
DEBUG_LEVEL         = 1
WHITE_TUNING        = False   # True로 바꾸면 white HSV 트랙바 창 활성화
YELLOW_TUNING       = False   # True로 바꾸면 yellow HSV 트랙바 창 활성화
OVERTAKE_CAR_TUNING = False   # True로 바꾸면 overtake 차량 HSV 트랙바 + 마스크 창 활성화
STANLEY_TUNING      = False   # True로 바꾸면 Stanley 파라미터 트랙바 창 활성화

# =============================================
# 파라미터 (튜닝 필요)
# =============================================
ROI_Y_START = 150
ROI_Y_END   = 480
ROI_H       = ROI_Y_END - ROI_Y_START   # 330
ROI_W       = 640

# BEV SRC: ROI 기준 사다리꼴 (좌하→우하→우상→좌상)

BOTTOM_DIST = 893
TOP_DIST = 190
Y_OFFSET = 50

SRC_POINTS = np.float32([
    [320 - BOTTOM_DIST, ROI_H - Y_OFFSET],
    [320 + BOTTOM_DIST, ROI_H - Y_OFFSET],
    [320 + TOP_DIST,   160 - Y_OFFSET],
    [320 - TOP_DIST,   160 - Y_OFFSET],
])

DST_MARGIN = 0
DST_POINTS = np.float32([
    [DST_MARGIN,         ROI_H],
    [ROI_W - DST_MARGIN, ROI_H],
    [ROI_W - DST_MARGIN,     0],
    [DST_MARGIN,             0],
])

WARPED_SIZE = (ROI_W, ROI_H)

# =============================================
# 슬라이딩 윈도우 파라미터 (BEV warped_binary 기준, 튜닝 필요)
# =============================================
SW_NWINDOWS           = 10
SW_MARGIN             = 45       # 윈도우 절반 폭 (px)
SW_MINPIX             = 25       # 윈도우 이동 최소 픽셀
SW_TOP_OFFSET         = 90       # 윈도우 추적 상단 한계 (px, 튜닝)
SW_LOOKAHEAD_Y_RATIO  = 0.75     # lookahead y 위치 비율
SW_BOTTOM_OFFSET      = 60       # 윈도우 시작을 바닥에서 이 px 위부터 (튜닝)
SW_VALID_Y_RATIO      = 0.8      # 활성 구간 상단에서 이 비율 아래에 픽셀 있어야 유효 (0=전체, 1=맨아래)
CURV_SCALE            = 6267     # curv → pos 단위 환산 (±0.03 → ±188)

# =============================================
# PID + Feedforward 파라미터 (튜닝 필요)
# =============================================
KP  = 2.0   # 비례
KD  = 30.0   # 미분
KI  = 0.0   # 적분
KSW = 1.0   # 차선 위치 오차 가중치
KFF = 0.0   # 커브 feedforward 게인

SPEED_MAX   = 19.5  # 직선 최대 속도
SPEED_MIN   = 5.0  # 커브 최소 속도
SPEED_KD    = 27.0  # 떨림 기반 감속 게인 (d_error 기준)

# =============================================
# 컨트롤러 선택 및 공용 파라미터
# =============================================
CONTROLLER       = 'stanley'                              # 'pid' or 'stanley'
CONTROLLER_LA_Y  = int(ROI_H * SW_LOOKAHEAD_Y_RATIO)     # BEV lookahead y (247)
CONTROLLER_Y_CAR = ROI_H - SW_BOTTOM_OFFSET              # BEV 차량 위치 y (270)
LANE_REF_R       = int(ROI_W * 0.75) - 28               # 우측 차선 기준 x (452)
LANE_REF_L       = int(ROI_W * 0.25) + 28               # 좌측 차선 기준 x (188)

# Stanley 파라미터 (튜닝 필요)
STANLEY_HEADING_SCALE = 56.0   # slope → angle 단위 변환 게인
STANLEY_K             = 1.7   # cross-track 게인
STANLEY_V_MIN         = 1.0    # 속도 하한 (0 나눗셈 방지)

# =============================================
# 어린이 보호구역 감지 파라미터 (튜닝 필요)
# =============================================
SCHOOL_ZONE_ENTER_FRAMES = 5
SCHOOL_ZONE_EXIT_FRAMES = 7
SCHOOL_ZONE_YELLOW_RATIO = 2.0
CENTER_MASK_X_START = 220
CENTER_MASK_X_END = 420

# =============================================
# 체크기 구간 감지 파라미터 (BEV warped_color 하단 절반 기준)
# =============================================
CHECKERED_DARK_V_THRESHOLD  = 20     # HSV V < 이 값 → 거의 검정 픽셀
CHECKERED_DARK_PIXEL_MIN    = 3000   # 검정 픽셀 최소 수 (튜닝)
CHECKERED_WHITE_PIXEL_MIN   = 1000   # 동시에 흰 픽셀도 있어야 함 (그림자 오발동 방지)
CHECKERED_ENTER_FRAMES      = 3
CHECKERED_EXIT_FRAMES       = 10

# =============================================
# 정지선 검출 파라미터 (BEV warped_white 기준)
# =============================================
STOP_LINE_Y_START        = 120
STOP_LINE_Y_END          = 240
STOP_LINE_X_MARGIN       = 70
STOP_LINE_MIN_WIDTH      = 260
STOP_LINE_MIN_HEIGHT     = 3
STOP_LINE_CONFIRM_FRAMES = 3


# =============================================
# HSVTuner
# YELLOW_TUNING / OVERTAKE_CAR_TUNING=True 일 때 생성된다.
# name으로 트랙바 창·마스크 창을 구분하므로 여러 인스턴스 동시 사용 가능.
# =============================================
class HSVTuner:

    def __init__(self, name, defaults, min_pixels=None):
        self._win        = f'{name} HSV Tuning'
        self._mask_win   = f'{name} Mask'
        self._min_pixels = min_pixels

        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        for key, val in defaults.items():
            hi = 179 if key.startswith('H') else 255
            cv2.createTrackbar(key, self._win, val, hi, lambda _: None)
        cv2.namedWindow(self._mask_win, cv2.WINDOW_NORMAL)

    def get_range(self):
        g = lambda n: cv2.getTrackbarPos(n, self._win)
        lower = np.array([g('H_min'), g('S_min'), g('V_min')], dtype=np.uint8)
        upper = np.array([g('H_max'), g('S_max'), g('V_max')], dtype=np.uint8)
        return lower, upper

    def show_mask(self, mask):
        display = mask.copy()
        label = f'pixels: {int(np.count_nonzero(mask))}'
        if self._min_pixels is not None:
            label += f'  min: {self._min_pixels}'
        cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        cv2.imshow(self._mask_win, display)

    def print_values(self):
        lo, hi = self.get_range()
        print(f'[HSVTuner:{self._win}] lower={lo.tolist()}  upper={hi.tolist()}')


# =============================================
# Preprocessing
# HSV 이진화 + Birds Eye View 변환
# ROS 없는 pure class — 이미지만 받고 결과 dict 반환
# =============================================
class Preprocessing:

    def __init__(self, src_points=SRC_POINTS, dst_points=DST_POINTS,
                 warped_size=WARPED_SIZE, white_tuner=None, yellow_tuner=None):

        self.M     = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        self.warped_size  = warped_size
        self._src_points  = src_points
        self.white_tuner  = white_tuner
        self.yellow_tuner = yellow_tuner

    # -----------------------------------------
    # 외부에서 호출하는 메인 메서드
    # roi: ROI 잘린 BGR 프레임
    # 반환: dict
    #   white_binary   — 흰 차선 이진 마스크
    #   yellow_binary  — 노란 차선 이진 마스크
    #   warped_binary  — BEV 변환된 합산 이진 마스크
    #   warped_color   — BEV 변환된 컬러 이미지
    #   annotated_roi  — SRC 사다리꼴 표시된 ROI
    # -----------------------------------------
    def run(self, roi):

        white_binary, yellow_binary = self._binary(roi)

        return {
            'white_binary':   white_binary,
            'yellow_binary':  yellow_binary,
            'warped_white':   self._warp(white_binary),
            'warped_yellow':  self._warp(yellow_binary),
            'warped_color':   self._warp(roi),
            'annotated_roi':  self._draw_src_polygon(roi),
        }

    # -----------------------------------------
    # HSV 이진화
    # -----------------------------------------
    def _binary(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.white_tuner is not None:
            lower_w, upper_w = self.white_tuner.get_range()
        else:
            lower_w = np.array([0,   0, 254])
            upper_w = np.array([1, 1, 255])
        white_mask = cv2.inRange(hsv, lower_w, upper_w)
        if self.white_tuner is not None:
            self.white_tuner.show_mask(white_mask)

        if self.yellow_tuner is not None:
            lower_y, upper_y = self.yellow_tuner.get_range()
        else:
            lower_y = np.array([10, 242,  80])
            upper_y = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_y, upper_y)
        if self.yellow_tuner is not None:
            self.yellow_tuner.show_mask(yellow_mask)

        h, w = white_mask.shape[:2]
        road_roi = np.zeros_like(white_mask)
        cv2.fillPoly(road_roi, np.array([[
            (0,              h),
            (w - 1,          h),
            (int(w * 0.95),  int(h * 0.10)),
            (int(w * 0.05),  int(h * 0.10)),
        ]], np.int32), 255)

        kernel = np.ones((3, 3), np.uint8)

        def clean(mask):
            m = cv2.bitwise_and(mask, road_roi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            m = cv2.GaussianBlur(m, (5, 5), 0)
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            return m

        return clean(white_mask), clean(yellow_mask)

    # -----------------------------------------
    # Perspective warp
    # -----------------------------------------
    def _warp(self, img):
        return cv2.warpPerspective(img, self.M, self.warped_size)

    # -----------------------------------------
    # BEV 좌표 → 원본 ROI 좌표 역변환
    # SlideWindow 결과를 원본 이미지에 표시할 때 사용
    # -----------------------------------------
    def unwarp_points(self, points):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)

    # -----------------------------------------
    # SRC 사다리꼴 시각화 (튜닝용)
    # -----------------------------------------
    def _draw_src_polygon(self, frame):
        overlay = frame.copy()
        pts = self._src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        for i, (x, y) in enumerate(self._src_points.astype(int)):
            cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(overlay, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return overlay


# =============================================
# StopLineDetector
# BEV 흰색 영상에서 차량 가까이에 있는 긴 가로선을 검출한다.
# =============================================
class StopLineDetector:

    def __init__(self):
        self.detection_frames = 0
        self.horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (41, 3),
        )

    def run(self, warped_white):
        height, width = warped_white.shape[:2]
        y_start = int(np.clip(STOP_LINE_Y_START, 0, height - 1))
        y_end = int(np.clip(STOP_LINE_Y_END, y_start + 1, height))
        x_start = int(np.clip(STOP_LINE_X_MARGIN, 0, width - 1))
        x_end = int(np.clip(width - STOP_LINE_X_MARGIN, x_start + 1, width))

        search = warped_white[y_start:y_end, x_start:x_end]
        horizontal = cv2.morphologyEx(
            search,
            cv2.MORPH_OPEN,
            self.horizontal_kernel,
        )
        contours, _ = cv2.findContours(
            horizontal,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            if (
                candidate_width >= STOP_LINE_MIN_WIDTH
                and candidate_height >= STOP_LINE_MIN_HEIGHT
            ):
                candidates.append(
                    (x + x_start, y + y_start, candidate_width, candidate_height)
                )

        if candidates:
            self.detection_frames += 1
        else:
            self.detection_frames = 0

        detected = self.detection_frames >= STOP_LINE_CONFIRM_FRAMES
        debug_img = cv2.cvtColor(warped_white, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(
            debug_img,
            (x_start, y_start),
            (x_end - 1, y_end - 1),
            (255, 255, 0),
            2,
        )
        for x, y, candidate_width, candidate_height in candidates:
            cv2.rectangle(
                debug_img,
                (x, y),
                (x + candidate_width, y + candidate_height),
                (0, 0, 255) if detected else (0, 255, 255),
                3,
            )
        cv2.putText(
            debug_img,
            f'stop line: {"DETECTED" if detected else "SEARCHING"}',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if detected else (0, 255, 0),
            2,
        )

        return {
            'detected': detected,
            'debug_img': debug_img,
            'candidates': candidates,
        }


# =============================================
# SchoolZoneDetector
# 흰색보다 노란색 차선이 우세하면 노란 양쪽 실선을 추종한다.
# 중앙 노란 점선은 차선으로 오인하지 않도록 제거한다.
# =============================================
class SchoolZoneDetector:

    def __init__(self):
        self.school_zone_mode = False
        self._frames = 0

    def run(self, warped_white, warped_yellow):
        yellow_px = int(np.count_nonzero(warped_yellow))
        white_px = int(np.count_nonzero(warped_white))
        is_yellow_dominant = (
            yellow_px > max(white_px, 1) * SCHOOL_ZONE_YELLOW_RATIO
        )

        if is_yellow_dominant:
            self._frames = max(self._frames, 0)
            self._frames = min(
                self._frames + 1,
                SCHOOL_ZONE_ENTER_FRAMES,
            )
        else:
            self._frames = min(self._frames, 0)
            self._frames = max(
                self._frames - 1,
                -SCHOOL_ZONE_EXIT_FRAMES,
            )

        if self._frames >= SCHOOL_ZONE_ENTER_FRAMES:
            self.school_zone_mode = True
        elif self._frames <= -SCHOOL_ZONE_EXIT_FRAMES:
            self.school_zone_mode = False

        if not self.school_zone_mode:
            return warped_white

        lane_image = warped_yellow.copy()
        lane_image[:, CENTER_MASK_X_START:CENTER_MASK_X_END] = 0
        return lane_image


# =============================================
# CheckeredZoneDetector
# BEV warped_color 하단 절반에서 검정+흰 픽셀이 동시에 많으면 체크기 구간으로 판정.
# 체크기 구간에서는 왼쪽 차선 감지를 강제로 실패 처리한다.
# =============================================
class CheckeredZoneDetector:

    def __init__(self):
        self.active = False
        self._frames = 0

    def run(self, roi, white_binary):
        height = roi.shape[0]
        bottom_roi   = roi[height // 3:, :]
        bottom_white = white_binary[height // 3:, :]

        hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
        dark_mask   = hsv[:, :, 2] < CHECKERED_DARK_V_THRESHOLD
        dark_count  = int(np.count_nonzero(dark_mask))
        white_count = int(np.count_nonzero(bottom_white))

        is_checkered = (
            dark_count  >= CHECKERED_DARK_PIXEL_MIN and
            white_count >= CHECKERED_WHITE_PIXEL_MIN
        )

        if is_checkered:
            self._frames = min(self._frames + 1, CHECKERED_ENTER_FRAMES)
        else:
            self._frames = max(self._frames - 1, -CHECKERED_EXIT_FRAMES)

        if self._frames >= CHECKERED_ENTER_FRAMES:
            self.active = True
        elif self._frames <= -CHECKERED_EXIT_FRAMES:
            self.active = False

        if DEBUG_LEVEL >= 2:
            debug_img = bottom_roi.copy()
            debug_img[dark_mask] = (0, 0, 255)   # 검정 픽셀 → 빨강으로 표시
            label = 'CHECKERED' if self.active else 'normal'
            color = (0, 0, 255) if self.active else (0, 255, 0)
            cv2.putText(debug_img, label,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(debug_img, f'dark:{dark_count} white:{white_count}',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow('checkered_zone', debug_img)


# =============================================
# SlideWindow
# warped_binary → 차선 탐지 + PID 직전 값까지 계산
#
# run() 반환 dict:
#   lane_center       — PID error = TARGET_X - lane_center
#   curve_feedforward — 커브 보정값 (부호 있음)
#   debug_img         — 시각화 이미지
#   right_detected    / left_detected
# =============================================
class SlideWindow:

    def __init__(self):

        # 오른쪽 차선 상태
        self.rightx_previous       = 480
        self.rightx_la_previous    = 420
        self.right_lane_detected   = False
        self.right_missing_windows = SW_NWINDOWS

        # 왼쪽 차선 상태
        self.leftx_previous       = 160
        self.leftx_la_previous    = 220
        self.left_lane_detected   = False
        self.left_missing_windows = SW_NWINDOWS


    # -----------------------------------------
    # 히스토그램 기반 차선 시작점 탐색
    # -----------------------------------------
    def _find_lane_startx(self, img, x_start, x_end, y_ratio_start,
                           x_previous, x_clip_max=None, jump_blend=True):
        height = img.shape[0]
        if x_clip_max is None:
            x_clip_max = x_end - 1

        region    = img[int(height * y_ratio_start):height, x_start:x_end]
        histogram = np.sum(region, axis=0)

        if len(histogram) == 0 or np.max(histogram) < 255 * 20:
            return int(np.clip(x_previous, x_start, x_clip_max))

        smooth     = np.convolve(histogram, np.ones(9), mode='same')
        peak       = np.max(smooth)
        candidates = np.where(smooth > peak * 0.35)[0] + x_start

        if len(candidates) == 0:
            startx = int(np.argmax(smooth)) + x_start
        else:
            startx = int(candidates[np.argmin(np.abs(candidates - x_previous))])

        if jump_blend and abs(startx - x_previous) > 160:
            startx = int(0.65 * x_previous + 0.35 * startx)

        return int(np.clip(startx, x_start, x_clip_max))

    # -----------------------------------------
    # 슬라이딩 윈도우 공통 탐색
    # get_x_bounds(idx, win_y_low, win_y_high) → (x_lower, x_upper)
    # -----------------------------------------
    def _run_sliding_window(self, img, startx, out_img=None,
                             get_x_bounds=None,
                             nwindows=SW_NWINDOWS, margin=SW_MARGIN,
                             minpix=SW_MINPIX, rect_color=(255, 0, 0)):
        height, width = img.shape[:2]
        tracking_top  = SW_TOP_OFFSET
        bottom        = height - SW_BOTTOM_OFFSET
        window_height = max(1, int((bottom - tracking_top) / nwindows))

        nz       = img.nonzero()
        nonzeroy = np.array(nz[0])
        nonzerox = np.array(nz[1])

        current_x = startx
        lane_inds = []
        missing   = 0

        for idx in range(nwindows):
            win_y_low  = max(bottom - (idx + 1) * window_height, tracking_top)
            win_y_high = bottom - idx * window_height

            if get_x_bounds is not None:
                x_lower, x_upper = get_x_bounds(idx, win_y_low, win_y_high)
            else:
                x_lower, x_upper = 0, width - 1

            current_x  = int(np.clip(current_x, x_lower, x_upper))
            win_x_low  = max(current_x - margin, x_lower)
            win_x_high = min(current_x + margin, x_upper)

            if out_img is not None:
                cv2.rectangle(out_img,
                              (win_x_low, win_y_low),
                              (win_x_high, win_y_high),
                              rect_color, 2)

            good = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) & (nonzerox < win_x_high)
            ).nonzero()[0]
            lane_inds.append(good)

            if len(good) > minpix:
                current_x = int(np.mean(nonzerox[good]))
            else:
                missing += 1

        all_inds = (
            np.concatenate(lane_inds) if lane_inds else np.array([], dtype=int)
        )
        return nonzerox[all_inds], nonzeroy[all_inds], missing

    # -----------------------------------------
    # Polyfit + 좌표 추출
    # validate_lower_median: lambda m: bool  (None이면 검증 생략)
    # -----------------------------------------
    def _fit_lane_curve(self, lane_x, lane_y, height, lookahead_y,
                         x_clip_lookahead,
                         validate_lower_median=None):
        lower_mask = lane_y > int(height * 0.55)

        if np.count_nonzero(lower_mask) < 25:
            return None, None

        lower_median = np.median(lane_x[lower_mask])

        if validate_lower_median is not None and not validate_lower_median(lower_median):
            return None, None

        if len(lane_x) >= 50:
            fit2 = np.polyfit(lane_y, lane_x, 2)   # cross-track lookahead 위치용
            fit1 = np.polyfit(lane_y, lane_x, 1)   # heading 방향용 (fit1[0] = slope)
            la   = int(np.polyval(fit2, lookahead_y))
        else:
            fit2 = None
            fit1 = None
            la   = int(lower_median)

        la = int(np.clip(la, x_clip_lookahead[0], x_clip_lookahead[1]))

        return fit2, fit1, la

    # -----------------------------------------
    # 오른쪽 차선 x 범위 콜백
    # -----------------------------------------
    def _make_right_x_bounds(self, height, width):
        def bounds(idx, win_y_low, win_y_high):
            if win_y_low > int(height * 0.45):
                return int(width * 0.50), width - 1
            else:
                return int(width * 0.30), width - 1
        return bounds

    # -----------------------------------------
    # 왼쪽 차선 x 범위 콜백 (오른쪽 간격 제약 포함)
    # -----------------------------------------
    def _make_left_x_bounds(self, height, width):
        def bounds(idx, win_y_low, win_y_high):
            if win_y_low > int(height * 0.45):
                return 0, int(width * 0.50)
            else:
                return 0, int(width * 0.70)
        return bounds

    # -----------------------------------------
    # 오른쪽 실선 탐지
    # 반환: (out_img, rightx, rightx_la)
    # -----------------------------------------
    def slidewindow_r(self, img):
        self.right_lane_detected   = False
        self.right_missing_windows = 0

        out_img       = np.dstack((img, img, img))
        height, width = img.shape[:2]
        la_y          = int(height * SW_LOOKAHEAD_Y_RATIO)

        startx = self._find_lane_startx(
            img,
            x_start       = int(width * 0.50),
            x_end         = width,
            y_ratio_start = 0.55,
            x_previous    = self.rightx_previous,
            x_clip_max    = int(width * 0.93),
            jump_blend    = True,
        )

        rx, ry, missing = self._run_sliding_window(
            img, startx,
            out_img      = out_img,
            get_x_bounds = self._make_right_x_bounds(height, width),
            rect_color   = (255, 0, 0),
        )
        self.right_missing_windows = missing

        active_mid_y = int(SW_TOP_OFFSET + (height - SW_BOTTOM_OFFSET - SW_TOP_OFFSET) * SW_VALID_Y_RATIO)

        if len(rx) == 0:
            return out_img, None, None, self.rightx_la_previous

        rx_arr, ry_arr   = np.array(rx), np.array(ry)
        bottom_right_cnt = int(np.sum((ry_arr >= active_mid_y) & (rx_arr >= width // 2)))
        if bottom_right_cnt < SW_MINPIX:
            return out_img, None, None, self.rightx_la_previous

        right_fit2, right_fit1, rightx_la = self._fit_lane_curve(
            rx, ry, height, la_y,
            x_clip_lookahead = (int(width * 0.30), width - 1),
            validate_lower_median = lambda m: m >= int(width * 0.50),
        )

        if rightx_la is None:
            return out_img, None, None, self.rightx_la_previous

        self.rightx_previous     = rightx_la
        self.rightx_la_previous  = rightx_la
        self.right_lane_detected = True

        cv2.circle(out_img, (rightx_la, la_y), 8, (255, 0, 0), -1)

        return out_img, right_fit2, right_fit1, rightx_la

    # -----------------------------------------
    # 왼쪽 실선 탐지 (slidewindow_r 대칭, 오프셋만 다름)
    # 반환: (left_curve, leftx_la)
    # -----------------------------------------
    def slidewindow_l(self, img, out_img):
        self.left_lane_detected   = False
        self.left_missing_windows = 0

        height, width = img.shape[:2]
        la_y          = int(height * SW_LOOKAHEAD_Y_RATIO)

        startx = self._find_lane_startx(
            img,
            x_start       = 0,
            x_end         = int(width * 0.50),
            y_ratio_start = 0.55,
            x_previous    = self.leftx_previous,
            x_clip_max    = int(width * 0.50) - 1,
            jump_blend    = True,
        )

        lx, ly, missing = self._run_sliding_window(
            img, startx,
            out_img      = out_img,
            get_x_bounds = self._make_left_x_bounds(height, width),
            rect_color   = (255, 0, 255),
        )
        self.left_missing_windows = missing

        active_mid_y = int(SW_TOP_OFFSET + (height - SW_BOTTOM_OFFSET - SW_TOP_OFFSET) * SW_VALID_Y_RATIO)

        if len(lx) == 0:
            return None, None, self.leftx_la_previous

        lx_arr, ly_arr  = np.array(lx), np.array(ly)
        bottom_left_cnt = int(np.sum((ly_arr >= active_mid_y) & (lx_arr < width // 2)))
        if bottom_left_cnt < SW_MINPIX:
            return None, None, self.leftx_la_previous

        left_fit2, left_fit1, leftx_la = self._fit_lane_curve(
            lx, ly, height, la_y,
            x_clip_lookahead = (0, int(width * 0.70)),
            validate_lower_median = lambda m: m <= int(width * 0.50),
        )

        if leftx_la is None:
            return None, None, self.leftx_la_previous

        self.leftx_previous     = leftx_la
        self.leftx_la_previous  = leftx_la
        self.left_lane_detected = True

        cv2.circle(out_img, (leftx_la, la_y), 8, (255, 0, 255), -1)

        return left_fit2, left_fit1, leftx_la

    # -----------------------------------------
    # 메인 실행
    # -----------------------------------------
    def run(self, warped_binary, skip_left=False):
        out_img, right_fit, right_fit1, rightx_la = self.slidewindow_r(warped_binary)

        if skip_left:
            left_fit, left_fit1, leftx_la = None, None, self.leftx_la_previous
            left_det = False
        else:
            left_fit, left_fit1, leftx_la = self.slidewindow_l(warped_binary, out_img)
            left_det = self.left_lane_detected

        right_det = self.right_lane_detected
        r_valid   = max(0, SW_NWINDOWS - self.right_missing_windows)
        l_valid   = max(0, SW_NWINDOWS - self.left_missing_windows)

        return {
            'right_fit':      right_fit,    # 2차 (cross-track lookahead)
            'left_fit':       left_fit,
            'right_fit1':     right_fit1,   # 1차 (heading slope)
            'left_fit1':      left_fit1,
            'right_la':       rightx_la,
            'left_la':        leftx_la,
            'r_valid':        r_valid,
            'l_valid':        l_valid,
            'right_detected': right_det,
            'left_detected':  left_det,
            'debug_img':      out_img,
        }


# =============================================
# PIDController
# =============================================
class PIDController:

    def __init__(self):
        self._prev_error = 0.0
        self._integral   = 0.0

    def compute(self, detection, ego_speed, lateral_offset=0.0, single_lane_ok=False):
        right_fit = detection['right_fit']
        left_fit  = detection['left_fit']
        right_det = detection['right_detected']
        left_det  = detection['left_detected']
        r_valid   = detection['r_valid']
        l_valid   = detection['l_valid']
        right_la  = detection['right_la']
        left_la   = detection['left_la']

        right_pos  = (right_la - (LANE_REF_R - lateral_offset)) if right_det else 0.0
        left_pos   = (left_la  - (LANE_REF_L - lateral_offset)) if left_det  else 0.0
        right_curv = (2.0 * right_fit[0] * CURV_SCALE) if (right_det and right_fit is not None) else 0.0
        left_curv  = (2.0 * left_fit[0]  * CURV_SCALE) if (left_det  and left_fit  is not None) else 0.0

        if right_det and left_det:
            total = max(1, r_valid + l_valid)
            error = (r_valid * right_pos  + l_valid * left_pos)  / total
            curv  = (r_valid * right_curv + l_valid * left_curv) / total
        elif right_det:
            error, curv = float(right_pos), float(right_curv)
        elif left_det:
            error, curv = float(left_pos), float(left_curv)
        else:
            error, curv = 0.0, 0.0

        error_total      = KSW * error + KFF * curv
        d_error          = error_total - self._prev_error
        self._integral   = float(np.clip(self._integral + error_total, -1000.0, 1000.0))
        self._prev_error = error_total

        raw_angle = KP * error_total + KD * d_error + KI * self._integral
        angle     = float(np.clip(raw_angle, -100.0, 100.0))

        t_angle = (abs(angle) / 100.0) ** 0.8
        t_kd    = min(1.0, abs(SPEED_KD * d_error) / 100.0)
        t       = max(t_angle, t_kd)
        speed   = SPEED_MAX - (SPEED_MAX - SPEED_MIN) * t
        if not single_lane_ok and (not right_det or not left_det):
            speed = SPEED_MIN

        return angle, speed


# =============================================
# StanleyController
# θ_e: 차량 위치(CONTROLLER_Y_CAR)에서의 polynomial 기울기 → heading error
# e: lookahead 위치의 cross-track error (pixel 단위)
# =============================================
class StanleyController:

    _WIN = 'Stanley Tuning'

    def __init__(self):
        self._prev_e = 0.0
        self._dbg    = {'e': 0.0, 'theta_e': 0.0, 'cross_correction': 0.0,
                        'right_cross': 0.0, 'left_cross': 0.0}
        if STANLEY_TUNING:
            self._setup_tuner()

    def _setup_tuner(self):
        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        bars = [
            ('heading_scale', int(STANLEY_HEADING_SCALE), 200),
            ('K_x10',         int(STANLEY_K * 10),        200),
            ('speed_max',     int(SPEED_MAX),              30),
            ('speed_min',     int(SPEED_MIN),              20),
            ('speed_kd',      int(SPEED_KD),               500),
        ]
        for name, val, hi in bars:
            cv2.createTrackbar(name, self._WIN, val, hi, lambda _: None)

    def _get_params(self):
        if not STANLEY_TUNING:
            return STANLEY_HEADING_SCALE, STANLEY_K, SPEED_MAX, SPEED_MIN, SPEED_KD
        g = lambda n: cv2.getTrackbarPos(n, self._WIN)
        return (
            float(g('heading_scale')),
            g('K_x10') / 10.0,
            float(g('speed_max')),
            float(g('speed_min')),
            float(g('speed_kd')),
        )

    def print_params(self):
        h, k, smax, smin, skd = self._get_params()
        print(f'[Stanley] heading_scale={h:.1f}  K={k:.1f}  speed_max={smax:.1f}  speed_min={smin:.1f}  speed_kd={skd:.1f}')

    def compute(self, detection, ego_speed, lateral_offset=0.0, single_lane_ok=False):
        heading_scale, k, speed_max, speed_min, speed_kd = self._get_params()

        right_fit1 = detection['right_fit1']   # 1차 (heading)
        left_fit1  = detection['left_fit1']
        right_det  = detection['right_detected']
        left_det   = detection['left_detected']
        r_valid    = detection['r_valid']
        l_valid    = detection['l_valid']
        right_la   = detection['right_la']
        left_la    = detection['left_la']

        # cross-track error (lookahead 기준, pixel)
        # lateral_offset > 0: 우측 이동 — 기준점을 왼쪽으로 당겨 우조향 유도
        right_pos = (right_la - (LANE_REF_R - lateral_offset)) if right_det else 0.0
        left_pos  = (left_la  - (LANE_REF_L - lateral_offset)) if left_det  else 0.0

        if right_det and left_det:
            total = max(1, r_valid + l_valid)
            e = (r_valid * right_pos + l_valid * left_pos) / total
        elif right_det:
            e = float(right_pos)
        elif left_det:
            e = float(left_pos)
        else:
            e = 0.0

        # heading: 1차 피팅 기울기를 부호 반전
        # BEV에서 y↓=근거리, y↑=전방이므로 우회전 시 fit1[0]=dx/dy < 0
        # 조향 convention(우회전=양수)과 맞추려면 부호 반전 필요
        r_slope = -right_fit1[0] if (right_det and right_fit1 is not None) else None
        l_slope = -left_fit1[0]  if (left_det  and left_fit1  is not None) else None

        if r_slope is not None and l_slope is not None:
            total = max(1, r_valid + l_valid)
            slope = (r_valid * r_slope + l_valid * l_slope) / total
        elif r_slope is not None:
            slope = r_slope
        elif l_slope is not None:
            slope = l_slope
        else:
            slope = 0.0

        v = max(STANLEY_V_MIN, float(ego_speed))

        theta_e          = slope * heading_scale
        cross_correction = k * e / v
        raw_angle        = theta_e + cross_correction
        angle            = float(np.clip(raw_angle, -100.0, 100.0))

        d_e          = e - self._prev_e
        self._prev_e = e

        t_angle = (abs(angle) / 100.0) ** 0.8
        t_kd    = min(1.0, abs(speed_kd * d_e) / 100.0)
        t       = max(t_angle, t_kd)
        speed   = speed_max - (speed_max - speed_min) * t
        if not single_lane_ok and (not right_det or not left_det):
            speed = speed_min

        self._dbg = {
            'e':                e,
            'theta_e':          theta_e,
            'cross_correction': cross_correction,
            'right_cross':      right_pos if right_det else None,
            'left_cross':       left_pos  if left_det  else None,
        }

        return angle, speed

# =============================================
# OvertakeDecision 게이트 파라미터
# =============================================
OVERTAKE_GATE_ENTER_FRAMES  = 5
OVERTAKE_GATE_EXIT_FRAMES   = 5
OVERTAKE_GATE_TIMEOUT_SEC   = 99999   # 사실상 비활성 — 필요 시 튜닝

OVERTAKE_CAR_HSV_LOWER      = np.array([22, 128, 128], dtype=np.uint8)
OVERTAKE_CAR_HSV_UPPER      = np.array([41, 193, 255], dtype=np.uint8)
OVERTAKE_CAR_MIN_PIXELS     = 200     # 노이즈 방지 최소 픽셀 수

# =============================================
# 카메라 추월 파라미터 (CameraOvertakeDecision)
# =============================================
CAMERA_OVERTAKE_ENTER_FRAMES   = 5    # 노란 차 N프레임 연속 감지 → LANE2 (튜닝)
CAMERA_OVERTAKE_LOSE_FRAMES    = 5   # 노란 차 M프레임 연속 미감지 → WAITING (튜닝)
CAMERA_OVERTAKE_RETURN_FRAMES  = 1   # WAITING 후 LANE1 복귀 대기 프레임 (튜닝)
CAMERA_OVERTAKE_LATERAL_OFFSET = 150  # Stanley 우측 offset px (LANE2, 튜닝)
CAMERA_OVERTAKE_LANE1_OFFSET   = -25 # Stanley 좌측 offset px (LANE1, 튜닝)

# =============================================
# LidarOvertakeDecision (구 OvertakeDecision)
# overtake_drive 노드의 제안값(state/motor_suggestion)을 받아
# PID 출력에 적용할지 결정한다.
# 게이트가 ACTIVE일 때만 제안값을 실제 출력에 반영한다.
# CameraOvertakeDecision과 교환 가능 — self.overtake 할당 클래스만 변경.
# =============================================
class LidarOvertakeDecision:

    def __init__(self, logger, tuner=None):
        self.logger     = logger
        self.tuner      = tuner
        self.state      = 'LANE1'
        self.suggestion = None

        self._gate_active  = False
        self._enter_streak = 0
        self._exit_streak  = 0
        self._active_since = None

    def update_state(self, msg):
        if self._gate_active and msg.data != self.state:
            self.logger.info(f'[overtake] state: {self.state} -> {msg.data}')
        self.state = msg.data

    def update_suggestion(self, msg):
        self.suggestion = msg
        if self._gate_active and self.state != 'LANE1':
            self.logger.info(
                f'[overtake] suggestion: angle={msg.angle:.1f}  speed={msg.speed:.1f}  state={self.state}')

    def _detect_yellow_car(self, roi):
        if self.tuner is not None:
            lower, upper = self.tuner.get_range()
        else:
            lower, upper = OVERTAKE_CAR_HSV_LOWER, OVERTAKE_CAR_HSV_UPPER
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        if self.tuner is not None:
            self.tuner.show_mask(mask)
        return int(np.count_nonzero(mask)) >= OVERTAKE_CAR_MIN_PIXELS

    def update_gate(self, roi, school_zone_active):
        car_visible = self._detect_yellow_car(roi)

        if not self._gate_active:
            self._enter_streak = self._enter_streak + 1 if car_visible else 0
            if self._enter_streak >= OVERTAKE_GATE_ENTER_FRAMES:
                self._gate_active  = True
                self._active_since = time.time()
                self._exit_streak  = 0
                self.logger.info('[overtake] gate ACTIVE')
        else:
            if time.time() - self._active_since >= OVERTAKE_GATE_TIMEOUT_SEC:
                self._gate_active  = False
                self._enter_streak = 0
                self.logger.info('[overtake] gate INACTIVE (timeout)')
                return

            self._exit_streak = self._exit_streak + 1 if school_zone_active else 0
            if self._exit_streak >= OVERTAKE_GATE_EXIT_FRAMES:
                self._gate_active  = False
                self._enter_streak = 0
                self.logger.info('[overtake] gate INACTIVE (school zone)')

    def get_lateral_offset(self):
        return 0.0

    def apply(self, angle, speed):
        if not self._gate_active or self.state == 'LANE1' or self.suggestion is None:
            return angle, speed
        return self.suggestion.angle, self.suggestion.speed


# =============================================
# CameraOvertakeDecision
# 카메라 HSV 기반 노란 차 감지로 추월 구간을 판단.
# overtake_drive 노드 없이 동작하며 LidarOvertakeDecision과 교환 가능.
#
# 상태:
#   DISABLED: 초기/비활성 — lateral_offset = 0
#   LANE2   : 우측 주행   — lateral_offset = +CAMERA_OVERTAKE_LATERAL_OFFSET
#   WAITING : 차 소실 대기 — lateral_offset 유지
#   LANE1   : 좌측 주행   — lateral_offset = CAMERA_OVERTAKE_LANE1_OFFSET (음수)
#
# 전환:
#   DISABLED → LANE2   : 노란 차 ENTER_FRAMES 연속 감지
#   LANE2    → WAITING : 노란 차 LOSE_FRAMES 연속 미감지
#   WAITING  → LANE2   : 노란 차 재감지
#   WAITING  → LANE1   : RETURN_FRAMES 경과
#   LANE1    : 노란 차 감지 무시
#   any      → DISABLED: school_zone_active → offset=0
# =============================================
class CameraOvertakeDecision:

    _DISABLED = 'DISABLED'
    _LANE1    = 'LANE1'
    _LANE2    = 'LANE2'
    _WAITING  = 'WAITING'

    def __init__(self, logger, tuner=None):
        self.logger          = logger
        self.tuner           = tuner
        self._state          = self._DISABLED
        self._lateral_offset = 0.0

        self._enter_streak = 0   # 연속 감지 프레임 (DISABLED→LANE2)
        self._lose_streak  = 0   # 연속 미감지 프레임 (LANE2→WAITING)
        self._wait_frames  = 0   # WAITING 경과 프레임

    def _detect_yellow_car(self, roi):
        if self.tuner is not None:
            lower, upper = self.tuner.get_range()
        else:
            lower, upper = OVERTAKE_CAR_HSV_LOWER, OVERTAKE_CAR_HSV_UPPER
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        if self.tuner is not None:
            self.tuner.show_mask(mask)
        return int(np.count_nonzero(mask)) >= OVERTAKE_CAR_MIN_PIXELS

    def _enter(self, state):
        if self._state != state:
            self.logger.info(f'[camera_overtake] {self._state} → {state}')
            self._state = state

    def update_gate(self, roi, school_zone_active):
        if school_zone_active:
            self._enter(self._DISABLED)
            self._lateral_offset = 0.0
            self._enter_streak = self._lose_streak = self._wait_frames = 0
            return

        car_visible = self._detect_yellow_car(roi)

        if self._state == self._DISABLED:
            self._enter_streak = self._enter_streak + 1 if car_visible else 0
            if self._enter_streak >= CAMERA_OVERTAKE_ENTER_FRAMES:
                self._enter(self._LANE2)
                self._lateral_offset = float(CAMERA_OVERTAKE_LATERAL_OFFSET)
                self._lose_streak = 0

        elif self._state == self._LANE2:
            if car_visible:
                self._lose_streak = 0
            else:
                self._lose_streak += 1
                if self._lose_streak >= CAMERA_OVERTAKE_LOSE_FRAMES:
                    self._enter(self._WAITING)
                    self._wait_frames = 0

        elif self._state == self._WAITING:
            if car_visible:
                self._enter(self._LANE2)
                self._lose_streak = 0
            else:
                self._wait_frames += 1
                if self._wait_frames >= CAMERA_OVERTAKE_RETURN_FRAMES:
                    self._enter(self._LANE1)
                    self._lateral_offset = float(CAMERA_OVERTAKE_LANE1_OFFSET)

        elif self._state == self._LANE1:
            pass  # 노란 차 재감지 무시

    def get_lateral_offset(self):
        return self._lateral_offset

    def apply(self, angle, speed):
        return angle, speed


# =============================================
# MainLoop — 실제 ROS2 노드
# Preprocessing → SlideWindow 순서로 호출
# =============================================
class MainLoop(Node):

    def __init__(self):

        super().__init__('track_drive_2')

        self.bridge        = CvBridge()
        self.image         = None
        self.new_image     = False
        self.stop_line_result = None
        self.lane_image = None
        white_tuner = HSVTuner(
            'White Lane',
            dict(H_min=0, H_max=180, S_min=0, S_max=80, V_min=150, V_max=255),
        ) if WHITE_TUNING else None
        yellow_tuner = HSVTuner(
            'Yellow Lane',
            dict(H_min=10, H_max=40, S_min=242, S_max=255, V_min=80, V_max=255),
        ) if YELLOW_TUNING else None
        overtake_tuner = HSVTuner(
            'Overtake Car',
            dict(H_min=13, H_max=54, S_min=128, S_max=193, V_min=128, V_max=255),
            min_pixels=OVERTAKE_CAR_MIN_PIXELS,
        ) if OVERTAKE_CAR_TUNING else None
        self.preprocessing = Preprocessing(white_tuner=white_tuner, yellow_tuner=yellow_tuner)
        self.school_zone = SchoolZoneDetector()
        self.checkered_zone = CheckeredZoneDetector()
        self.slidewindow   = SlideWindow()
        self.stop_line_detector = StopLineDetector()
        # 교환 포인트: CameraOvertakeDecision ↔ LidarOvertakeDecision
        self.overtake = CameraOvertakeDecision(self.get_logger(), tuner=overtake_tuner)

        self._pid        = PIDController()
        self._stanley    = StanleyController()
        self._last_speed = SPEED_MIN

        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self._cam_callback,
            qos_profile_sensor_data
        )
        # LidarOvertakeDecision 사용 시에만 활성화
        # self.sub_overtake_state = self.create_subscription(
        #     String, '/overtake/state', self.overtake.update_state, 10)
        # self.sub_overtake_suggestion = self.create_subscription(
        #     XycarMotor, '/overtake/motor_suggestion', self.overtake.update_suggestion, 10)

        self.pub_motor = self.create_publisher(XycarMotor, '/lane_motor_cmd', 10)
        self.pub_stop_line = self.create_publisher(Bool, '/stop_line', 10)
        self.pub_lane_detection = self.create_publisher(
            Int64MultiArray,
            '/lane_detection_status',
            10,
        )

        self.get_logger().info('track_drive_2 started')

    # -----------------------------------------
    # callback
    # -----------------------------------------
    def _cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        self.new_image = True

    # -----------------------------------------
    # debug 창 표시
    # DEBUG_LEVEL 0: 없음
    # DEBUG_LEVEL 1: sliding_window만 (미구현 시 birds_eye_binary 대체)
    # DEBUG_LEVEL 2: 전체
    # -----------------------------------------
    def _show_debug(self, prep, sw, stop_line, angle=0.0):

        if DEBUG_LEVEL == 0:
            return

        if DEBUG_LEVEL >= 1:
            if sw['debug_img'] is not None:
                debug_image = sw['debug_img'].copy()
                h, w = debug_image.shape[:2]
                ax = int(np.clip(w // 2 + angle, 0, w - 1))
                cv2.line(debug_image, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
                cv2.circle(debug_image, (ax, SW_TOP_OFFSET // 2), 10, (0, 255, 0), -1)
                cv2.putText(debug_image, f'angle: {angle:+.1f}  [{CONTROLLER}]',
                            (w // 2 - 120, SW_TOP_OFFSET // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if CONTROLLER == 'stanley':
                    dbg = self._stanley._dbg
                    # 우측 cross-track
                    rc = dbg['right_cross']
                    lc = dbg['left_cross']
                    rc_str = f'R cross: {rc:+.0f}px' if rc is not None else 'R cross: --'
                    lc_str = f'L cross: {lc:+.0f}px' if lc is not None else 'L cross: --'
                    cv2.putText(debug_image, rc_str,
                                (w - 210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 100), 2)
                    cv2.putText(debug_image, lc_str,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)
                    # heading / cross-track 합산
                    cv2.putText(debug_image,
                                f'θ_e: {dbg["theta_e"]:+.1f}  e: {dbg["e"]:+.1f}px',
                                (w // 2 - 120, SW_TOP_OFFSET // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 2)

                if self.school_zone.school_zone_mode:
                    cv2.putText(
                        debug_image,
                        'SCHOOL ZONE',
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                cv2.imshow('sliding_window', debug_image)
            else:
                cv2.imshow('sliding_window', prep['warped_white'])

        if DEBUG_LEVEL >= 2:
            cv2.imshow('stop_line', stop_line['debug_img'])
            cv2.imshow('original_roi',    prep['annotated_roi'])
            cv2.imshow('warped_white',    prep['warped_white'])
            cv2.imshow('warped_yellow',   prep['warped_yellow'])
            cv2.imshow('birds_eye_color', prep['warped_color'])

    # -----------------------------------------
    # main loop
    # -----------------------------------------
    def run(self):

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.01)

            if self.image is None:
                continue

            # 차선 PID는 기존과 같은 루프 주기로 실행한다. 새 프레임만
            # 처리하도록 제한하면 큰 KD가 프레임 간 노이즈에 반응해 조향이
            # 흔들리므로, 정지선 확인 횟수만 실제 카메라 프레임에 맞춘다.
            is_new_image = self.new_image
            self.new_image = False
            roi = self.image[ROI_Y_START:ROI_Y_END, :].copy()

            prep = self.preprocessing.run(roi)
            # 같은 카메라 프레임을 반복 처리해 debounce가 즉시 끝나지
            # 않도록 보호구역 판정은 새 영상마다 한 번만 갱신한다.
            if is_new_image or self.lane_image is None:
                self.lane_image = self.school_zone.run(
                    prep['warped_white'],
                    prep['warped_yellow'],
                )
                self.checkered_zone.run(roi, prep['white_binary'])
            sw = self.slidewindow.run(self.lane_image, skip_left=self.checkered_zone.active)
            if is_new_image or self.stop_line_result is None:
                lane_detection_message = Int64MultiArray()
                lane_detection_message.data = [
                    int(sw['left_detected']),
                    int(sw['right_detected']),
                ]
                self.pub_lane_detection.publish(lane_detection_message)

                self.stop_line_result = self.stop_line_detector.run(
                    prep['warped_white']
                )
                stop_line_message = Bool()
                stop_line_message.data = bool(
                    self.stop_line_result['detected']
                )
                self.pub_stop_line.publish(stop_line_message)
            stop_line = self.stop_line_result

            self.overtake.update_gate(roi, self.school_zone.school_zone_mode)
            lateral_offset = self.overtake.get_lateral_offset()

            single_lane_ok = sw['right_detected'] or sw['left_detected']
            ctrl = self._stanley if CONTROLLER == 'stanley' else self._pid
            angle, speed = ctrl.compute(sw, self._last_speed,
                                        lateral_offset=lateral_offset,
                                        single_lane_ok=single_lane_ok)

            # LidarOvertakeDecision 사용 시 활성화 (각도/속도 직접 override)
            # angle, speed = self.overtake.apply(angle, speed)
            if self.school_zone.school_zone_mode:
                speed = min(speed, 15.0)
            self._last_speed = float(speed)

            msg = XycarMotor()
            msg.angle = float(angle)
            msg.speed = float(speed)
            self.pub_motor.publish(msg)

            self._show_debug(prep, sw, stop_line, angle)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # -----------------------------------------
    # destroy
    # -----------------------------------------
    def destroy(self):
        cv2.destroyAllWindows()
        self.destroy_node()


# =============================================
# main
# =============================================
def main(args=None):

    rclpy.init(args=args)
    node = MainLoop()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
