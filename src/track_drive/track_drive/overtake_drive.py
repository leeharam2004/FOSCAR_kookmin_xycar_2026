#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 차량 추월 미션 (라이다 기반 전방 차량 속도 판별 + 추종/추월 조향/속도 계산)
#   - 카메라 색상 인식(검정 차량 추종)은 더 이상 사용하지 않는다.
#   - /scan(LaserScan)에서 전방 +-LIDAR_FRONT_HALF_ANGLE_DEG 범위 안의 가장 가까운
#     장애물 거리를 매 프레임 추적해, 그 거리가 멀어지는 속도(away speed)로
#     "빠른 차(자차보다 빨리 달아나는 앞차)" / "느린 차(자차보다 느린 앞차)"를 판단한다.
#   - 빠른 차로 확정되면 그 차를 정확히 뒤따라가는 것을 "2차선 주행"으로 보고,
#     라이다 거리/각도 기반 추종 조향각/속도를 계산한다.
#   - 느린 차로 확정되면 고정된 조향각/속도로 항상 우측으로 비켜서("OVERTAKE_RIGHT")
#     추월 주행하고, 느린 차를 완전히 지나쳐 전방에서 더 이상 감지되지 않으면
#     "1차선 주행" 상태로 복귀한다.
#   - 빠른 차도 느린 차도 아니면(전방에 장애물이 없으면) "1차선 주행" 상태로 보고
#     조향/속도 제안 없이(0, 0) track_drive.py의 기본 주행을 그대로 따른다.
#   - 장애물까지의 "반경 거리(r)"만 추적하면, 차선 옆에 비스듬히 서 있는 정지 물체
#     (나무, 가로수, 표지판 등)도 가까이 지나칠 때 r이 가까워지다가 최근접점을 지나며
#     다시 멀어지는 것처럼 보이는 기하학적 착시가 생겨 "빠른 차"로 오인된다.
#     이를 막기 위해 r 대신 "진행방향 성분(x = r*cos(angle))"의 변화율을 상대속도로
#     쓴다 — 정지 물체는 측면 위치(각도)와 무관하게 항상 -자차속도로 일정하게 줄어든다.
#   - 이 상대속도에 /xycar_motor 속도(자차 속도 근사값)를 더하면 장애물의 절대속도가
#     되고, 절대속도가 거의 0인 정지 장애물은 빠른 차/느린 차 판단 및 라이다 인식 창
#     표시에서 모두 제외한다(움직이는 물체만 인식/표시).
#   - track_drive.py가 /xycar_motor로 차량을 직접 제어하고 있으므로,
#     이 노드는 모터를 건드리지 않고 계산값만 별도 토픽으로 publish한다.
#     (/overtake/state: 현재 상태, /overtake/motor_suggestion: 계산된 조향/속도값)
#     track_drive.py 파일/동작에는 전혀 영향을 주지 않는다.

import time
import math

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from xycar_msgs.msg import XycarMotor

DEBUG_LEVEL = 1   # 0: 인식 창 없음, 1: 라이다 인식 창 표시

# =============================================
# 라이다 전방 인식 범위
# 실제 인식 창으로 확인한 결과, 전방은 인덱스 0 기준 0도 회전 위치에 있다.
# =============================================
LIDAR_FRONT_INDEX_OFFSET_DEG = 0.0  # 인덱스 0 기준으로 전방까지의 각도 (음수면 반대 방향으로 회전)
LIDAR_FRONT_HALF_ANGLE_DEG   = 10.0   # 전방 +-10도 범위만 사용
LIDAR_MAX_VALID_RANGE_M      = 20.0   # 이보다 먼 값은 장애물 없음으로 취급 (라이다 사양에 맞게 튜닝 필요)

# =============================================
# "빠른 차" / "느린 차" 판단 파라미터
# 전방에서 가장 가까운 장애물까지의 거리가 프레임마다 멀어지는 속도(away speed)가
#   - LIDAR_FAST_AWAY_SPEED_MPS 이상으로 연속 유지되면 자차보다 빠르게 달아나는 "빠른 차"
#   - LIDAR_SLOW_AWAY_SPEED_MPS 이하(멀어지지 않거나 점점 가까워짐)로 연속 유지되면
#     자차보다 느린 "느린 차"로 판단한다.
# (둘 다 아니거나 장애물이 없으면 판단 보류 -> 1차선 유지)
# =============================================
LIDAR_FAST_AWAY_SPEED_MPS = 3.0 / 3.6  # 3km/h -> m/s. 이 값 이상으로 멀어지면 "빠른 차"
LIDAR_SLOW_AWAY_SPEED_MPS = 0.0        # 이 값 이하면(멀어지지 않음/가까워짐) "느린 차"
LIDAR_CONFIRM_FRAMES      = 3          # 위 조건을 이만큼 연속 만족해야 "빠른 차"/"느린 차"로 확정
LIDAR_MAX_JUMP_M          = 1.0        # 이전 프레임과 거리 차이가 이보다 크면 다른 물체로 보고 추적 리셋

# =============================================
# 절대속도(지면 기준) 필터 — 정지 장애물(벽, 콘 등) 제외용
# relative_speed + 자차 속도 = 장애물의 절대속도. 이 값의 절댓값이 아래 기준 미만이면
# "움직이지 않는" 장애물로 보고 빠른 차/느린 차 판단에서 제외한다.
# (자차 속도는 실측 센서가 없어 /xycar_motor의 명령 속도로 근사한다)
# =============================================
MOVING_OBJECT_MIN_ABS_SPEED_MPS = 1.0 / 3.6  # 1km/h -> m/s

STATE_LANE1         = 'LANE1'          # 빠른 차/느린 차 없음 -> 기본 주행 유지
STATE_LANE2         = 'LANE2'          # 빠른 차 확정 -> 그 차 바로 뒤를 추종
STATE_OVERTAKE_RIGHT = 'OVERTAKE_RIGHT'  # 느린 차 확정 -> 우측으로 비켜서 추월 주행

# =============================================
# 2차선(빠른 차 추종) 조향/속도 계산 파라미터 — 트랙에서 튜닝 필요
# =============================================
LIDAR_ANGLE_SIGN    = 1.0   # 인덱스가 커질수록(왼쪽->전방->오른쪽) 우측으로 본다고 가정한 부호.
                             # 인식 창에서 좌/우 반응이 뒤집혀 보이면 -1.0으로 변경한다.
FOLLOW_STEER_GAIN   = 2.0   # 전방 각도 오차(deg) -> angle 변환 게인
FOLLOW_ANGLE_LIMIT  = 50.0  # 계산되는 angle 제한

FOLLOW_DISTANCE_M = 1.0   # 빠른 차와 유지하려는 목표 거리(m)
FOLLOW_SPEED_GAIN = 6.0   # 목표 거리와의 오차(m) -> speed 보정 게인
FOLLOW_SPEED_MAX  = 12.0
FOLLOW_SPEED_MIN  = 3.0

# =============================================
# 느린 차 추월(우측 회피) 명령 파라미터 — 트랙에서 튜닝 필요
# 느린 차가 확정되면 고정된 조향각/속도로 항상 우측으로 비켜서 주행한다.
# =============================================
OVERTAKE_STEER_ANGLE = 20.0  # 우측으로 비켜가기 위한 고정 조향각(+ = 우측, FOLLOW_ANGLE_LIMIT와 동일 스케일)
OVERTAKE_SPEED       = 12.0  # 추월 중 유지할 속도
LIDAR_LOST_CONFIRM_FRAMES = LIDAR_CONFIRM_FRAMES  # 추월 중 전방 무감지가 이만큼 연속되면 완전히 지나친 것으로 보고 복귀


# =============================================
# 라이다 인덱스 -> 전방 인덱스 / 인덱스당 각도(도) 계산
# (find_front_obstacle과 디버그 시각화에서 동일하게 사용)
# =============================================
def lidar_front_index(scan):
    n = len(scan.ranges)
    increment = scan.angle_increment
    if n == 0 or not increment:
        return None, None

    degree_per_index = math.degrees(abs(increment))
    if degree_per_index <= 0:
        return None, None

    front_idx = int(round(LIDAR_FRONT_INDEX_OFFSET_DEG / degree_per_index)) % n
    return front_idx, degree_per_index


# =============================================
# 전방 +-half_angle_deg 범위에서 가장 가까운 유효 장애물 탐색
# 반환: {'distance': m, 'angle_deg': 전방 기준 각도(부호 포함)} 또는 None
# =============================================
def find_front_obstacle(scan, half_angle_deg, max_valid_range):
    ranges = scan.ranges
    n = len(ranges)
    front_idx, degree_per_index = lidar_front_index(scan)
    if front_idx is None:
        return None

    half_idx = max(1, int(round(half_angle_deg / degree_per_index)))
    lo = max(0, front_idx - half_idx)
    hi = min(n - 1, front_idx + half_idx)

    best_idx, best_dist = None, None
    for i in range(lo, hi + 1):
        d = ranges[i]
        if not math.isfinite(d) or d <= 0.0 or d > max_valid_range:
            continue
        if best_dist is None or d < best_dist:
            best_idx, best_dist = i, d

    if best_idx is None:
        return None

    angle_deg = (best_idx - front_idx) * degree_per_index
    return {'distance': best_dist, 'angle_deg': angle_deg}


# =============================================
# 빠른 차를 뒤따라가기 위한 (angle, speed) 계산.
# 실제 모터에는 publish하지 않고, 계산값만 돌려준다.
# =============================================
def compute_follow_command(obstacle):
    if obstacle is None:
        return 0.0, 0.0

    angle = float(np.clip(LIDAR_ANGLE_SIGN * FOLLOW_STEER_GAIN * obstacle['angle_deg'],
                           -FOLLOW_ANGLE_LIMIT, FOLLOW_ANGLE_LIMIT))

    dist_error = obstacle['distance'] - FOLLOW_DISTANCE_M
    speed = float(np.clip(FOLLOW_SPEED_MIN + FOLLOW_SPEED_GAIN * dist_error,
                           FOLLOW_SPEED_MIN, FOLLOW_SPEED_MAX))

    return angle, speed


# =============================================
# 느린 차를 우측으로 비켜서 추월하기 위한 (angle, speed). 고정값을 그대로 반환한다.
# =============================================
def compute_overtake_command():
    return OVERTAKE_STEER_ANGLE, OVERTAKE_SPEED


# =============================================
# OvertakeNode
# 모터 제어는 하지 않는 순수 인식/계산 노드 (track_drive.py와 토픽이 겹치지 않음)
# =============================================
class OvertakeNode(Node):

    def __init__(self):
        super().__init__('overtake_drive')

        self.state = STATE_LANE1
        self.scan  = None

        # 자차 속도(절대속도 계산용). 실측 센서가 없어 /xycar_motor 명령 속도로 근사한다.
        self.ego_speed_mps = 0.0

        # 라이다 추적 상태 ("빠른 차"/"느린 차" 판단용). 진행방향(x)/측면(y) 성분으로
        # 추적해야 비스듬한 정지 물체의 거리 착시(최근접점 부호반전)를 피할 수 있다.
        self._track_x      = None
        self._track_y      = None
        self._track_time   = None
        self._fast_streak  = 0
        self._slow_streak  = 0
        self._lost_streak  = 0

        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, qos_profile_sensor_data)
        self.sub_motor = self.create_subscription(
            XycarMotor, '/xycar_motor', self._motor_callback, 10)

        self.pub_state      = self.create_publisher(String, '/overtake/state', 10)
        self.pub_suggestion = self.create_publisher(
            XycarMotor, '/overtake/motor_suggestion', 10)

        self.fig = None
        if DEBUG_LEVEL > 0:
            self._init_debug_plot()

        self.get_logger().info('overtake_drive started (state=LANE1)')

    # -----------------------------------------
    # callback
    # -----------------------------------------
    def _scan_callback(self, msg):
        self.scan = msg

    def _motor_callback(self, msg):
        # /xycar_motor.speed는 실제 속도의 1/2 — 실제 km/h로 환산 후 m/s로 변환
        self.ego_speed_mps = float(msg.speed) * 2.0 / 3.6

    # -----------------------------------------
    # 인식 창 초기화 (라이다 포인트 / 전방 탐지 콘 / 검출 장애물 표시)
    # -----------------------------------------
    def _init_debug_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect('equal')
        lim = LIDAR_MAX_VALID_RANGE_M
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_title('overtake_drive lidar recognition')

        self._scan_scatter     = self.ax.scatter([], [], s=4, c='gray')
        self._cone_lines,      = self.ax.plot([], [], 'y--', linewidth=1)
        self._front_arrow,     = self.ax.plot([], [], 'b-', linewidth=2)
        self._obstacle_marker  = self.ax.scatter([], [], s=80, c='orange')
        self.ax.plot(0, 0, 'ko')

        self._status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        plt.ion()
        plt.show()

    # -----------------------------------------
    # 인식 창 갱신
    # -----------------------------------------
    def _update_debug_plot(self, moving_obstacle, angle, speed):
        if self.fig is None or self.scan is None:
            return

        front_idx, degree_per_index = lidar_front_index(self.scan)
        if front_idx is None:
            return

        ranges = np.array(self.scan.ranges, dtype=float)
        n = len(ranges)
        idx = np.arange(n)
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= LIDAR_MAX_VALID_RANGE_M)

        angles_rad = np.deg2rad((idx - front_idx) * degree_per_index)
        x = -ranges * np.cos(angles_rad)
        y = -ranges * np.sin(angles_rad)
        self._scan_scatter.set_offsets(np.c_[x[valid], y[valid]])

        half_rad = math.radians(LIDAR_FRONT_HALF_ANGLE_DEG)
        cone_len = LIDAR_MAX_VALID_RANGE_M
        cone_x = [-cone_len * math.cos(-half_rad), 0, -cone_len * math.cos(half_rad)]
        cone_y = [-cone_len * math.sin(-half_rad), 0, -cone_len * math.sin(half_rad)]
        self._cone_lines.set_data(cone_x, cone_y)
        self._front_arrow.set_data([0, -cone_len * 0.3], [0, 0])

        if moving_obstacle is not None:
            ang_rad = math.radians(moving_obstacle['angle_deg'])
            ox = -moving_obstacle['distance'] * math.cos(ang_rad)
            oy = -moving_obstacle['distance'] * math.sin(ang_rad)
            self._obstacle_marker.set_offsets([[ox, oy]])
            marker_color = {
                STATE_LANE2:          'green',   # 빠른 차 추종
                STATE_OVERTAKE_RIGHT: 'blue',    # 느린 차 추월(우측 회피)
            }.get(self.state, 'orange')          # 그 외(미확정 움직이는 물체)
            self._obstacle_marker.set_color(marker_color)
        else:
            self._obstacle_marker.set_offsets(np.empty((0, 2)))

        dist_txt = f"{moving_obstacle['distance']:.2f}m" if moving_obstacle is not None else '-'
        self._status_text.set_text(
            f"state: {self.state}\n"
            f"dist: {dist_txt}\n"
            f"angle: {angle:+.1f}  speed: {speed:.1f}"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # -----------------------------------------
    # 상태 전환
    # -----------------------------------------
    def _enter(self, state):
        if self.state != state:
            self.state = state
            self.get_logger().info(f'state -> {state}')

    # -----------------------------------------
    # 전방 장애물의 진행방향 성분(x = r*cos(angle)) 변화율로 상대속도(m/s)를 계산.
    # r(반경 거리)을 그대로 쓰면 비스듬히 서 있는 정지 물체(나무 등)를 가까이 지나칠 때
    # r이 가까워지다가 최근접점에서 다시 멀어지는 것처럼 보이는 착시가 생기지만,
    # x는 자차가 그 물체를 지나치는 동안 측면 위치(각도)와 무관하게 항상 -자차속도로
    # 일정하게 줄어들기 때문에 이 착시가 생기지 않는다.
    # 장애물이 없거나 추종 대상이 갑자기 바뀌면(jump) 추적을 리셋하고 None을 반환한다.
    # -----------------------------------------
    def _track_relative_speed(self, obstacle):
        now = time.time()

        if obstacle is None:
            self._track_x    = None
            self._track_y    = None
            self._track_time = None
            return None

        angle_rad = math.radians(obstacle['angle_deg'])
        x = obstacle['distance'] * math.cos(angle_rad)  # 진행방향 성분
        y = obstacle['distance'] * math.sin(angle_rad)  # 측면 성분

        if self._track_x is None or self._track_time is None:
            self._track_x, self._track_y, self._track_time = x, y, now
            return None

        if math.hypot(x - self._track_x, y - self._track_y) > LIDAR_MAX_JUMP_M:
            # 추종 대상이 갑자기 바뀜(다른 물체로 전환) -> 추적 리셋
            self._track_x, self._track_y, self._track_time = x, y, now
            return None

        dt             = max(now - self._track_time, 1e-3)
        relative_speed = (x - self._track_x) / dt

        self._track_x, self._track_y, self._track_time = x, y, now
        return relative_speed

    # -----------------------------------------
    # 상대속도로 "빠른 차"/"느린 차" 연속 프레임을 누적하고,
    # (fast_car, slow_car) 중 확정된 쪽을 obstacle로, 아니면 None으로 반환한다.
    # 절대속도가 거의 0인 정지 장애물은 moving_obstacle로도 잡히지 않는다.
    # (moving_obstacle은 라이다 인식 창에 표시할 "움직이는 물체"만 거른 값)
    # -----------------------------------------
    def _classify_obstacle(self, obstacle):
        relative_speed = self._track_relative_speed(obstacle)
        is_moving = (relative_speed is not None and
                     abs(relative_speed + self.ego_speed_mps) >= MOVING_OBJECT_MIN_ABS_SPEED_MPS)

        if not is_moving:
            # 추적 불가 또는 절대속도가 거의 0(정지 장애물) -> 판단 제외
            self._fast_streak = 0
            self._slow_streak = 0
        elif relative_speed >= LIDAR_FAST_AWAY_SPEED_MPS:
            self._fast_streak += 1
            self._slow_streak  = 0
        elif relative_speed <= LIDAR_SLOW_AWAY_SPEED_MPS:
            self._slow_streak += 1
            self._fast_streak  = 0
        else:
            self._fast_streak = 0
            self._slow_streak = 0

        fast_car = obstacle if self._fast_streak >= LIDAR_CONFIRM_FRAMES else None
        slow_car = obstacle if self._slow_streak >= LIDAR_CONFIRM_FRAMES else None
        moving_obstacle = obstacle if is_moving else None
        return fast_car, slow_car, moving_obstacle

    # -----------------------------------------
    # main loop
    # -----------------------------------------
    def run(self):
        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.01)

            if self.scan is None:
                continue

            obstacle_raw = find_front_obstacle(
                self.scan, LIDAR_FRONT_HALF_ANGLE_DEG, LIDAR_MAX_VALID_RANGE_M)

            fast_car, slow_car, moving_obstacle = self._classify_obstacle(obstacle_raw)

            self._lost_streak = 0 if obstacle_raw is not None else self._lost_streak + 1

            if fast_car is not None:
                self._enter(STATE_LANE2)
            elif slow_car is not None:
                self._enter(STATE_OVERTAKE_RIGHT)
            elif self.state == STATE_OVERTAKE_RIGHT:
                # 추월 중에는 잠시 전방 무감지가 있어도 유지하다가, 느린 차를 완전히
                # 지나친 것이 확인되면(연속 무감지) 1차선 주행으로 복귀한다.
                if self._lost_streak >= LIDAR_LOST_CONFIRM_FRAMES:
                    self._enter(STATE_LANE1)
            else:
                self._enter(STATE_LANE1)

            if self.state == STATE_LANE2:
                angle, speed = compute_follow_command(fast_car)
            elif self.state == STATE_OVERTAKE_RIGHT:
                angle, speed = compute_overtake_command()
            else:
                angle, speed = 0.0, 0.0

            state_msg = String()
            state_msg.data = self.state
            self.pub_state.publish(state_msg)

            suggestion = XycarMotor()
            suggestion.angle = float(angle)
            suggestion.speed = float(speed)
            self.pub_suggestion.publish(suggestion)

            self._update_debug_plot(moving_obstacle, angle, speed)

    def destroy(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.destroy_node()


# =============================================
# main
# =============================================
def main(args=None):

    rclpy.init(args=args)
    node = OvertakeNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
