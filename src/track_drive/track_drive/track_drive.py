#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from xycar_msgs.msg import XycarMotor

# =============================================
# 디버그 레벨
#   0 — 창 없음
#   1 — sliding_window 창만 (지금은 birds_eye_binary로 대체)
#   2 — 모든 창 표시
# =============================================
DEBUG_LEVEL = 1

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

SPEED_MAX   = 15.0  # 직선 최대 속도
SPEED_MIN   = 3.0  # 커브 최소 속도
SPEED_KD    = 100.0  # 떨림 기반 감속 게인 (d_error 기준)

# =============================================
# 어린이 보호구역 감지 파라미터 (튜닝 필요)
# =============================================
SCHOOL_ZONE_ENTER_FRAMES  = 5     # 연속 N프레임 yellow dominant → 진입
SCHOOL_ZONE_EXIT_FRAMES   = 15    # 연속 M프레임 not dominant → 탈출 (hysteresis)
SCHOOL_ZONE_YELLOW_RATIO  = 2.0   # yellow_px > white_px * ratio → dominant 판정
CENTER_MASK_X_START       = 220   # BEV 중앙 점선 마스크 시작 x (px, 튜닝)
CENTER_MASK_X_END         = 420   # BEV 중앙 점선 마스크 끝 x (px, 튜닝)

# =============================================
# 정지선 검출 파라미터 (BEV warped_white 기준)
# =============================================
STOP_LINE_Y_START        = 190
STOP_LINE_Y_END          = 300
STOP_LINE_X_MARGIN       = 70
STOP_LINE_MIN_WIDTH      = 260
STOP_LINE_MIN_HEIGHT     = 3
STOP_LINE_CONFIRM_FRAMES = 3


# =============================================
# Preprocessing
# HSV 이진화 + Birds Eye View 변환
# ROS 없는 pure class — 이미지만 받고 결과 dict 반환
# =============================================
class Preprocessing:

    def __init__(self, src_points=SRC_POINTS, dst_points=DST_POINTS,
                 warped_size=WARPED_SIZE):

        self.M     = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        self.warped_size  = warped_size
        self._src_points  = src_points

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

        white_mask  = cv2.inRange(hsv, np.array([0,  0,  150]),
                                       np.array([180, 80, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([10, 80,  80]),
                                       np.array([40, 255, 255]))

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
# warped_white / warped_yellow 비교 → 어린이 보호구역 감지 + debounce
# run() 반환: SlideWindow에 넘길 lane_img (white or 마스킹된 yellow)
# =============================================
class SchoolZoneDetector:

    def __init__(self):
        self.school_zone_mode = False
        self._frames = 0  # 양수: 진입 방향, 음수: 탈출 방향

    def run(self, warped_white, warped_yellow):
        yellow_px = int(np.count_nonzero(warped_yellow))
        white_px  = int(np.count_nonzero(warped_white))

        is_yellow_dominant = yellow_px > max(white_px, 1) * SCHOOL_ZONE_YELLOW_RATIO

        if is_yellow_dominant:
            self._frames = min(self._frames + 1, SCHOOL_ZONE_ENTER_FRAMES)
        else:
            self._frames = max(self._frames - 1, -SCHOOL_ZONE_EXIT_FRAMES)

        if self._frames >= SCHOOL_ZONE_ENTER_FRAMES:
            self.school_zone_mode = True
        elif self._frames <= -SCHOOL_ZONE_EXIT_FRAMES:
            self.school_zone_mode = False

        if not self.school_zone_mode:
            return warped_white

        lane_img = warped_yellow.copy()
        lane_img[:, CENTER_MASK_X_START:CENTER_MASK_X_END] = 0
        return lane_img


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

        # PID state
        self.prev_error = 0.0
        self.integral   = 0.0

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
            fit       = np.polyfit(lane_y, lane_x, 2)
            curvature = 2.0 * fit[0]
            la        = int(np.polyval(fit, lookahead_y))
        else:
            curvature = 0.0
            la        = int(lower_median)

        la = int(np.clip(la, x_clip_lookahead[0], x_clip_lookahead[1]))

        return curvature, la

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

        ref_r        = int(width * 0.75) - 28
        active_mid_y = int(SW_TOP_OFFSET + (height - SW_BOTTOM_OFFSET - SW_TOP_OFFSET) * SW_VALID_Y_RATIO)

        if len(rx) == 0:
            return out_img, 0.0, self.rightx_la_previous - ref_r

        rx_arr, ry_arr   = np.array(rx), np.array(ry)
        bottom_right_cnt = int(np.sum((ry_arr >= active_mid_y) & (rx_arr >= width // 2)))
        if bottom_right_cnt < SW_MINPIX:
            return out_img, 0.0, self.rightx_la_previous - ref_r

        right_curve, rightx_la = self._fit_lane_curve(
            rx, ry, height, la_y,
            x_clip_lookahead = (int(width * 0.30), width - 1),
            validate_lower_median = lambda m: m >= int(width * 0.50),
        )

        if right_curve is None:
            return out_img, 0.0, self.rightx_la_previous - ref_r

        self.rightx_previous     = rightx_la
        self.rightx_la_previous  = rightx_la
        self.right_lane_detected = True

        right_pos  = rightx_la - ref_r
        right_curv = right_curve * CURV_SCALE

        cv2.circle(out_img, (rightx_la, la_y), 8, (255, 0, 0), -1)
        cv2.putText(out_img, f'R curv: {right_curv:+.1f}',
                    (width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
        cv2.putText(out_img, f'R pos:  {right_pos:+d}',
                    (width - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        return out_img, right_curv, right_pos

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

        ref_l        = int(width * 0.25) + 28
        active_mid_y = int(SW_TOP_OFFSET + (height - SW_BOTTOM_OFFSET - SW_TOP_OFFSET) * SW_VALID_Y_RATIO)

        if len(lx) == 0:
            return 0.0, self.leftx_la_previous - ref_l

        lx_arr, ly_arr  = np.array(lx), np.array(ly)
        bottom_left_cnt = int(np.sum((ly_arr >= active_mid_y) & (lx_arr < width // 2)))
        if bottom_left_cnt < SW_MINPIX:
            return 0.0, self.leftx_la_previous - ref_l

        left_curve, leftx_la = self._fit_lane_curve(
            lx, ly, height, la_y,
            x_clip_lookahead = (0, int(width * 0.70)),
            validate_lower_median = lambda m: m <= int(width * 0.50),
        )

        if left_curve is None:
            return 0.0, self.leftx_la_previous - ref_l

        self.leftx_previous     = leftx_la
        self.leftx_la_previous  = leftx_la
        self.left_lane_detected = True

        left_pos  = leftx_la - ref_l
        left_curv = left_curve * CURV_SCALE

        cv2.circle(out_img, (leftx_la, la_y), 8, (255, 0, 255), -1)
        cv2.putText(out_img, f'L curv: {left_curv:+.1f}',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        cv2.putText(out_img, f'L pos:  {left_pos:+d}',
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        return left_curv, left_pos

    # -----------------------------------------
    # 메인 실행
    # -----------------------------------------
    def run(self, warped_binary):
        out_img, right_curv, right_pos = self.slidewindow_r(warped_binary)
        left_curv,  left_pos           = self.slidewindow_l(warped_binary, out_img)

        right_det = self.right_lane_detected
        left_det  = self.left_lane_detected
        r_valid   = max(0, SW_NWINDOWS - self.right_missing_windows)
        l_valid   = max(0, SW_NWINDOWS - self.left_missing_windows)

        # --- error & curv ---
        if right_det and left_det:
            total = max(1, r_valid + l_valid)
            error = (r_valid * right_pos  + l_valid * left_pos)  / total
            curv  = (r_valid * right_curv + l_valid * left_curv) / total
        elif right_det:
            error = float(right_pos)
            curv  = float(right_curv)
        elif left_det:
            error = float(left_pos)
            curv  = float(left_curv)
        else:
            error = 0.0
            curv  = 0.0

        # --- PID (curv feedforward을 error에 합산) ---
        error_total     = KSW * error + KFF * curv
        d_error         = error_total - self.prev_error
        self.integral   = float(np.clip(self.integral + error_total, -1000.0, 1000.0))
        self.prev_error = error_total

        raw_angle = KP * error_total + KD * d_error + KI * self.integral
        angle     = float(np.clip(raw_angle, -100.0, 100.0))

        # --- angle 시각화 ---
        height, width = warped_binary.shape[:2]
        ax = int(np.clip(width // 2 + angle, 0, width - 1))
        cv2.line(out_img, (width // 2, 0), (width // 2, height), (100, 100, 100), 1)
        cv2.circle(out_img, (ax, SW_TOP_OFFSET // 2), 10, (0, 255, 0), -1)
        cv2.putText(out_img, f'angle: {raw_angle:+.1f} -> {angle:+.1f}',
                    (width // 2 - 100, SW_TOP_OFFSET // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return {
            'angle':          angle,
            'error':          error,
            'curv':           curv,
            'd_error':        d_error,
            'debug_img':      out_img,
            'right_detected': right_det,
            'left_detected':  left_det,
        }


# =============================================
# MainLoop — 실제 ROS2 노드
# Preprocessing → SlideWindow 순서로 호출
# =============================================
class MainLoop(Node):

    def __init__(self):

        super().__init__('track_drive_2')

        self.bridge        = CvBridge()
        self.image         = None
        self.preprocessing = Preprocessing()
        self.school_zone   = SchoolZoneDetector()
        self.slidewindow   = SlideWindow()
        self.stop_line_detector = StopLineDetector()

        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self._cam_callback,
            qos_profile_sensor_data
        )

        self.pub_motor = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        self.pub_stop_line = self.create_publisher(Bool, '/stop_line', 10)

        self.get_logger().info('track_drive_2 started')

    # -----------------------------------------
    # callback
    # -----------------------------------------
    def _cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

    # -----------------------------------------
    # debug 창 표시
    # DEBUG_LEVEL 0: 없음
    # DEBUG_LEVEL 1: sliding_window만 (미구현 시 birds_eye_binary 대체)
    # DEBUG_LEVEL 2: 전체
    # -----------------------------------------
    def _show_debug(self, prep, sw, stop_line):

        if DEBUG_LEVEL == 0:
            return

        if DEBUG_LEVEL >= 1:
            if sw['debug_img'] is not None:
                dbg = sw['debug_img'].copy()
                if self.school_zone.school_zone_mode:
                    cv2.putText(dbg, 'SCHOOL ZONE', (10, dbg.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('sliding_window', dbg)
            else:
                cv2.imshow('sliding_window', prep['warped_white'])
            cv2.imshow('stop_line', stop_line['debug_img'])

        if DEBUG_LEVEL >= 2:
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

            roi = self.image[ROI_Y_START:ROI_Y_END, :].copy()

            prep     = self.preprocessing.run(roi)
            lane_img = self.school_zone.run(prep['warped_white'], prep['warped_yellow'])
            sw       = self.slidewindow.run(lane_img)
            stop_line = self.stop_line_detector.run(prep['warped_white'])

            stop_line_message = Bool()
            stop_line_message.data = bool(stop_line['detected'])
            self.pub_stop_line.publish(stop_line_message)

            t_angle = (abs(sw['angle']) / 100.0) ** 0.8
            t_kd    = min(1.0, abs(SPEED_KD * sw['d_error']) / 100.0)
            t       = max(t_angle, t_kd)
            speed   = SPEED_MAX - (SPEED_MAX - SPEED_MIN) * t
            if not sw['right_detected'] or not sw['left_detected']:
                speed = SPEED_MIN

            msg = XycarMotor()
            msg.angle = float(sw['angle'])
            msg.speed = float(speed)
            self.pub_motor.publish(msg)

            self._show_debug(prep, sw, stop_line)

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
