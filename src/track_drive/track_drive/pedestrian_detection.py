#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 수정사항 보행자 인식 제대로 안됨 보행자 인식 확실히하고 정지 확실히하게 하기  정지된 물체(EX나무) 인식안되게 하기 보행자한테 벗어날때 속도 높여서 탈출하기
# 라이다 기반 보행자(이동 물체) 감지 정지 미션
#   - 카메라 색상 인식(검정 차량 추종)은 더 이상 사용하지 않는다.
#   - /scan(LaserScan)에서 전방 ROI 사각형(좌우 ROI_LATERAL_*_M, 전방거리 ROI_FORWARD_*_M)
#     안의 "모든" 물체(라이다 점 군집)를 매 프레임 각각 따로 추적한다.
#     (LaserScan은 2D라 높이(Z) 정보가 없어 ROI는 좌우/전방거리 2축에만 적용한다)
#     ROI 안에서 가장 가까운 물체 하나만 추적하면, 보행자보다 더 가까운 다른 정지
#     물체(라바콘, 벽 등)가 있을 때 추적이 그 물체에 고정되어 버려 보행자의 좌우
#     움직임을 영영 못 보는 문제가 생긴다. 이를 막기 위해 ROI 안의 모든 군집을
#     각각 독립적으로(여러 track으로) 추적해서, 그중 하나라도 "좌우로 움직이는
#     물체"로 확인되면 즉시 정지한다.
#   - 군집까지의 "반경 거리(r)"만 추적하면, 차선 옆에 비스듬히 서 있는 정지 물체
#     (나무, 가로수, 표지판 등)도 가까이 지나칠 때 r이 가까워지다가 최근접점을 지나며
#     다시 멀어지는 것처럼 보이는 기하학적 착시가 생겨 "움직이는 물체"로 오인된다.
#     또한 자차가 조향으로 좌우로 돌면(차체가 회전하면) 정지 장애물도 차량 좌표계
#     기준으로는 옆으로 쓸려가듯 움직인 것처럼 보인다.
#     이를 막기 위해 각 track을 진행방향(x)/측면(y) 성분으로 추적하면서, 매 프레임
#     "자차가 그동안 전진(ego_speed*dt)하고 회전(yaw_rate*dt)한 만큼"을 보정해
#     "정지 장애물이라면 지금 있어야 할 위치"를 예측하고, 실제 좌표와의 차이를
#     그 물체의 절대(지면 기준) 속도로 쓴다. 자차의 속도/회전각속도는
#     dead_reckoning.py(ad_tf_maker)가 publish하는 ODOM_TOPIC(nav_msgs/Odometry,
#     twist.twist.linear.x/angular.z)을 구독해서 얻는다.
#   - 절대속도 벡터(vx, vy) 중 측면(y, 좌우) 성분 |vy|가 LATERAL_SPEED_MIN_MPS
#     이상인 "좌우로 움직이는 물체"(예: 횡단하는 사람)만 정지 트리거로 인식한다.
#     진행방향(x)으로만 움직이는 물체(같은 차로를 나란히 달리는 차 등)나 정지 물체는
#     제외된다(라이다 인식 창 표시도 이 판정을 그대로 따른다).
#   - 노이즈로 인한 한 프레임짜리 오검출/누락으로 정지·재출발이 깜빡이지 않도록,
#     같은 track이 연속 DETECT_CONFIRM_FRAMES 프레임 동안 "좌우로 움직임"으로
#     판정되어야 실제로 정지 상태로 전환한다(확실히 정지).
#   - 좌우로 움직이는 물체가 ROI 안에서 확인되면 "STOP"으로 들어가 /xycar_motor에
#     직접 speed=0을 publish해 track_drive.py의 주행 명령을 덮어쓴다(angle은 항상
#     0). 정지를 유발한 그 물체(track)가 ROI에서 OBSTACLE_LOST_GRACE_SEC 이상
#     사라진 게 확인되면 "ESCAPE"로 들어가 그 순간부터 ESCAPE_BURST_DURATION_SEC
#     동안만 speed=ESCAPE_MOTOR_SPEED(약 ESCAPE_SPEED_KMH km/h)로 짧게 가속해 정지
#     구간을 벗어난다. 그 시간이 지나면 "IDLE"로 들어가 더 이상 /xycar_motor를
#     건드리지 않고 track_drive.py의 평소 주행에 맡긴다(보행자가 다시 감지되면
#     IDLE/ESCAPE 어느 상태에서든 즉시 STOP으로 복귀). STOP/ESCAPE 동안은
#     track_drive.py보다 빠른 주기로 반복 publish해 우선권을 유지한다.

import time
import math

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from xycar_msgs.msg import XycarMotor

DEBUG_LEVEL = 1   # 0: 인식 창 없음, 1: 라이다 인식 창 표시

ODOM_TOPIC = 'ad/odom'  # dead_reckoning.py(ad_tf_maker)가 publish하는 자차 odom 토픽

# =============================================
# 라이다 전방 인식 ROI
# 실제 인식 창으로 확인한 결과, 전방은 인덱스 0 기준 0도 회전 위치에 있다.
# ROI는 차량 기준 좌우(lateral, +면 우측)/전방거리(forward) 사각형으로 정의한다.
# (LaserScan은 2D라 높이(Z) 정보가 없어 Z축 ROI는 적용하지 않는다)
# =============================================
LIDAR_FRONT_INDEX_OFFSET_DEG = 0.0  # 인덱스 0 기준으로 전방까지의 각도 (음수면 반대 방향으로 회전)

ROI_LATERAL_MIN_M = -4.5   # 좌우 ROI 최소값(m), 폭 9m(-4.5~4.5)
ROI_LATERAL_MAX_M = 4.5    # 좌우 ROI 최대값(m), 폭 9m(-4.5~4.5)
ROI_FORWARD_MIN_M = 2.0    # 전방거리 ROI 최소값(m)
ROI_FORWARD_MAX_M = 14.0   # 전방거리 ROI 최대값(m)

# =============================================
# 물체 추적 파라미터
# 같은 track으로 묶는 기준이 너무 느슨하면, 라바콘처럼 물체가 줄지어 있을 때 차가
# 지나가면서 추적 대상이 옆 물체로 슬쩍 넘어갈 수 있다(둘 사이 거리가
# LIDAR_MAX_JUMP_M보다 가까우면 리셋되지 않음).
# 이렇게 다른 정지 물체로 추적이 넘어가면 실제로는 둘 다 정지해 있어도 그 사이의
# 위치 차이가 "이동"으로 계산되어 버린다. 이를 줄이기 위해
#   1) LIDAR_MAX_JUMP_M을 충분히 작게 잡아 다른 물체로의 전환을 더 잘 리셋하고,
#   2) 그래도 새어나온 비정상적으로 빠른 속도는 MAX_PLAUSIBLE_OBJECT_SPEED_MPS로
#      걸러서(사람이 낼 수 없는 속도면 추적 오류로 보고 무시) 무시한다.
# =============================================
LIDAR_MAX_JUMP_M = 0.5  # 이전 프레임과 좌표 차이가 이보다 크면 다른 물체로 보고 추적 리셋
MAX_PLAUSIBLE_OBJECT_SPEED_MPS = 3.0  # 사람이 뛰어도 못 낼 속도(약 10.8km/h) 이상이면 추적 오류로 무시

# 한 프레임 안에서 "같은 물체"로 묶을 인접 라이다 점들의 반경(거리) 차이 임계값(m).
# 나무/표지판처럼 표면이 둥글거나 거친 정지 물체는 최근접점 1개만 추적하면, 차가
# 지나가며 보는 각도가 바뀔 때 표면 위에서 최근접점이 옆으로 미끄러지듯 이동해
# 마치 좌우로 움직이는 것처럼 보이는 오검출이 생긴다. 최근접점 주변에서 반경
# 차이가 이 값 이내로 이어지는 점들을 한 물체(군집)로 보고 그 무게중심을 추적하면
# 표면 미끄러짐이 평균화되어 정지 물체가 훨씬 안정적으로 정지 물체로 인식된다.
CLUSTER_BREAK_M = 0.2

# =============================================
# 절대속도(지면 기준) 필터 — "좌우로 움직이는 물체"(보행자 등)만 정지 트리거로 인식
# 장애물의 절대속도 벡터(vx, vy. 자차의 전진/회전을 보정하고 남은 값)의 크기가
# ABS_SPEED_ZERO_EPS_MPS 미만이면 "절대속도 0"(완전히 정지한 물체)로 보고 무조건
# 배제한다. 그 외에 측면(y, 좌우) 성분의 크기가 LATERAL_SPEED_MIN_MPS 이상이어야
# "정지 트리거"로 인식한다. 진행방향(x) 성분만 있는 물체(예: 같은 차로로 나란히
# 움직이는 차)는 좌우로 움직이는 게 아니므로 정지 트리거에서 제외된다.
# 보행자 정지 반응을 빠르게 하기 위해 매 프레임 인스턴트 속도값을 그대로 쓴다
# (평활화/누적이동거리 없음 — 추가 지연 없이 DETECT_CONFIRM_FRAMES만큼만 걸린다).
# (자차 속도/회전각속도는 dead_reckoning.py가 publish하는 odom으로 얻는다)
#
# 자차가 좌/우회전 중이면 odom 회전각속도 추정 오차가 그대로 정지 장애물(나무 등)의
# 겉보기 속도로 남는데, 그 오차는 장애물이 멀수록(거리 r에 비례) 커진다. 이를
# 흡수하기 위해 "절대속도 0" 판정 허용 오차를 회전각속도·장애물 거리에 비례해
# 늘린다(직진 중에는 ABS_SPEED_ZERO_EPS_MPS만 적용되어 보행자 반응 속도에 영향 없음).
# =============================================
LATERAL_SPEED_MIN_MPS = 1.0 / 3.6  # 1km/h -> m/s. 측면 속도가 이 값 이상이면 "좌우로 움직임"
ABS_SPEED_ZERO_EPS_MPS = 0.05  # 이 값 미만이면 절대속도 0(완전 정지)으로 보고 무조건 배제
YAW_COMPENSATION_ERROR_RATIO = 0.1  # 회전각속도 추정 오차 비율(거리 m당, rad/s당 m/s 오차 여유)

# =============================================
# 정지/재출발 디바운스
# 노이즈로 인한 한 프레임짜리 오검출/누락으로 정지·재출발이 깜빡이지 않도록,
# 연속으로 이만큼 같은 판정이 나와야 실제로 상태를 전환한다.
# =============================================
DETECT_CONFIRM_FRAMES = 3

# 라이다 잡음/순간 가려짐(검은 옷의 IR 흡수, 다리 교차 등)으로 obstacle_raw가 단
# 한두 프레임만 끊겨도 "ROI에서 완전히 사라짐"으로 오인하면, 추적이 리셋되고
# STOP -> ESCAPE -> IDLE을 반복하는 오동작이 생긴다(보행자가 실제로는 계속 ROI
# 안에 있는데도). 마지막으로 obstacle_raw가 감지된 시각부터 이 시간 안에는 끊김을
# 무시하고 직전 판정을 그대로 유지한다.
OBSTACLE_LOST_GRACE_SEC = 0.3

# =============================================
# 정지 해제(탈출) 가속
# 보행자가 라이다에서 벗어난 게 확인된 "그 순간"만 ESCAPE_BURST_DURATION_SEC 동안
# 이 속도로 가속해 정지 구간을 빠르게 벗어난다. 그 뒤에는 모터를 더 이상 건드리지
# 않고 track_drive.py의 평소 주행에 다시 맡긴다.
# /xycar_motor의 speed 필드는 물리 단위가 아니라 모터 명령 스케일이라, m/s를 모터
# 명령으로 바꿀 때 dead_reckoning.py가 쓰는 보정값(모터 명령 ≈ m/s * 2.5)을 그대로
# 가져온다(실측 캘리브레이션은 아니므로 실제 트랙에서 속도가 다르면 튜닝이 필요하다).
# =============================================
MOTOR_SPEED_UNITS_PER_MPS = 2.5
ESCAPE_SPEED_KMH         = 10.0
ESCAPE_MOTOR_SPEED       = (ESCAPE_SPEED_KMH / 3.6) * MOTOR_SPEED_UNITS_PER_MPS
ESCAPE_BURST_DURATION_SEC = 1.5  # 가속을 유지할 시간(초). 트랙에서 실제로 벗어나는 데 걸리는 시간으로 튜닝 필요

STATE_STOP   = 'STOP'    # ROI 안에 좌우로 움직이는 물체(사람 등) 감지 -> 정지
STATE_ESCAPE = 'ESCAPE'  # 정지 직후 보행자가 사라짐 -> 짧게 가속해서 정지 구간을 벗어남
STATE_IDLE   = 'IDLE'    # 탈출 가속이 끝남 -> 모터를 건드리지 않고 track_drive.py에 맡김


# =============================================
# 라이다 인덱스 -> 전방 인덱스 / 인덱스당 각도(도) 계산
# (find_clusters_in_roi와 디버그 시각화에서 동일하게 사용)
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
# 전방 ROI 사각형(좌우 lateral_min~lateral_max, 전방거리 forward_min~forward_max)
# 안의 "모든" 물체(군집)를 찾아 각각의 무게중심 목록을 반환한다(가장 가까운 것 하나만
# 보면 그보다 가까운 다른 정지 물체에 추적이 고정되어 버려 보행자를 영영 못 보게
# 된다). 인접한 라이다 점들을 반경 차이가 CLUSTER_BREAK_M 이내로 이어지는 동안 같은
# 물체로 묶어 그 무게중심을 쓴다(나무 등 표면이 둥글거나 거친 정지 물체는 표면 위
# 최근접점이 미끄러지듯 이동해 보이는 오검출을 줄이기 위함).
# 반환: [{'distance': m, 'angle_deg': 전방 기준 각도(부호 포함)}, ...]
# =============================================
def find_clusters_in_roi(scan, lateral_min, lateral_max, forward_min, forward_max):
    ranges = scan.ranges
    n = len(ranges)
    front_idx, degree_per_index = lidar_front_index(scan)
    if front_idx is None:
        return []

    points = []  # (index, forward, lateral, distance), 인덱스 오름차순
    for i in range(n):
        d = ranges[i]
        if not math.isfinite(d) or d <= 0.0:
            continue

        angle_rad = math.radians((i - front_idx) * degree_per_index)
        forward = d * math.cos(angle_rad)
        lateral = d * math.sin(angle_rad)
        if not (forward_min <= forward <= forward_max):
            continue
        if not (lateral_min <= lateral <= lateral_max):
            continue

        points.append((i, forward, lateral, d))

    if not points:
        return []

    groups = [[points[0]]]
    for prev, curr in zip(points, points[1:]):
        same_cluster = (curr[0] - prev[0] == 1) and (abs(curr[3] - prev[3]) <= CLUSTER_BREAK_M)
        if same_cluster:
            groups[-1].append(curr)
        else:
            groups.append([curr])

    clusters = []
    for group in groups:
        centroid_forward = sum(p[1] for p in group) / len(group)
        centroid_lateral  = sum(p[2] for p in group) / len(group)
        distance  = math.hypot(centroid_forward, centroid_lateral)
        angle_deg = math.degrees(math.atan2(centroid_lateral, centroid_forward))
        clusters.append({'distance': distance, 'angle_deg': angle_deg})

    return clusters


# =============================================
# PedestrianDetectionNode
# 보행자(또는 그 외 좌우로 움직이는 물체) 감지 정지 노드.
#   - STOP: ROI 안에 좌우로 움직이는 물체가 감지되는 동안 매 루프 /xycar_motor에
#     직접 speed=0을 publish해 track_drive.py의 주행 명령을 덮어쓴다.
#   - ESCAPE: 보행자가 사라진 게 확인된 순간부터 ESCAPE_BURST_DURATION_SEC 동안만
#     ESCAPE_MOTOR_SPEED(약 10km/h)로 가속해 정지 구간을 벗어난다.
#   - IDLE: 탈출 가속이 끝나면 모터를 더 이상 건드리지 않고 track_drive.py의
#     평소 주행을 그대로 따른다(단, 보행자가 다시 감지되면 즉시 STOP으로 복귀).
# =============================================
class PedestrianDetectionNode(Node):

    def __init__(self):
        super().__init__('pedestrian_detection')

        self.state = STATE_IDLE
        self.scan  = None
        self._escape_start_time = None

        # 자차 속도/회전각속도(절대속도 계산용). dead_reckoning.py가 publish하는 odom에서 얻는다.
        self.ego_speed_mps      = 0.0
        self.ego_yaw_rate_radps = 0.0

        # 라이다 추적 상태 — ROI 안의 "모든" 물체(군집)를 각각 독립된 track으로 동시에
        # 추적한다(가장 가까운 것 하나만 추적하면 보행자보다 가까운 다른 정지 물체에
        # 추적이 고정되어 보행자를 영영 못 보는 문제가 생기기 때문). 각 track은
        # {'id', 'x'(진행방향), 'y'(측면), 'time', 'last_seen', 'moving_streak'}를 갖는다.
        # 라이다 잡음/순간 가려짐(검은 옷의 IR 흡수, 다리 교차 등)으로 어떤 track이 한두
        # 프레임 매칭에 실패해도 OBSTACLE_LOST_GRACE_SEC 동안은 곧바로 지우지 않고
        # 유지해서, "완전히 사라짐"으로 오인해 STOP -> ESCAPE -> IDLE을 반복하지 않게 한다.
        self._tracks         = []
        self._next_track_id  = 0
        self._stop_track_id  = None  # 현재 STOP을 유발한 track id(이 물체가 ROI를 벗어나야 탈출)

        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_callback, 10)

        self.pub_state = self.create_publisher(String, '/pedestrian_detection/state', 10)
        # 매 루프 track_drive.py의 주행 명령을 덮어쓰기 위한 직접 모터 publisher
        self.pub_motor = self.create_publisher(XycarMotor, '/xycar_motor', 10)

        self.fig = None
        if DEBUG_LEVEL > 0:
            self._init_debug_plot()

        self.get_logger().info('pedestrian_detection started (state=IDLE)')

    # -----------------------------------------
    # callback
    # -----------------------------------------
    def _scan_callback(self, msg):
        self.scan = msg

    def _odom_callback(self, msg):
        # dead_reckoning.py가 모터 명령/IMU로 추정해 채운 odom의 선속도, 회전각속도
        self.ego_speed_mps      = float(msg.twist.twist.linear.x)
        self.ego_yaw_rate_radps = float(msg.twist.twist.angular.z)

    # -----------------------------------------
    # 인식 창 초기화 (라이다 포인트 / 전방 ROI 사각형 / 검출 장애물 표시)
    # -----------------------------------------
    def _init_debug_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect('equal')
        lim = ROI_FORWARD_MAX_M
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_title('pedestrian_detection lidar recognition')

        self._scan_scatter     = self.ax.scatter([], [], s=4, c='gray')
        self._roi_lines,       = self.ax.plot([], [], 'y--', linewidth=1)
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
    def _update_debug_plot(self, moving_obstacle):
        if self.fig is None or self.scan is None:
            return

        front_idx, degree_per_index = lidar_front_index(self.scan)
        if front_idx is None:
            return

        ranges = np.array(self.scan.ranges, dtype=float)
        n = len(ranges)
        idx = np.arange(n)
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= ROI_FORWARD_MAX_M)

        angles_rad = np.deg2rad((idx - front_idx) * degree_per_index)
        x = -ranges * np.cos(angles_rad)
        y = -ranges * np.sin(angles_rad)
        self._scan_scatter.set_offsets(np.c_[x[valid], y[valid]])

        # ROI 사각형(좌우 x 전방거리) 표시. 플롯 좌표는 forward/lateral의 반대 부호를 쓴다
        # (전방이 플롯에서 -x 방향이 되도록 맞춘 기존 시각화 규약을 그대로 따른다).
        roi_forward = [ROI_FORWARD_MIN_M, ROI_FORWARD_MIN_M, ROI_FORWARD_MAX_M, ROI_FORWARD_MAX_M, ROI_FORWARD_MIN_M]
        roi_lateral = [ROI_LATERAL_MIN_M, ROI_LATERAL_MAX_M, ROI_LATERAL_MAX_M, ROI_LATERAL_MIN_M, ROI_LATERAL_MIN_M]
        roi_x = [-f for f in roi_forward]
        roi_y = [-l for l in roi_lateral]
        self._roi_lines.set_data(roi_x, roi_y)
        self._front_arrow.set_data([0, -ROI_FORWARD_MAX_M * 0.3], [0, 0])

        if moving_obstacle is not None:
            ang_rad = math.radians(moving_obstacle['angle_deg'])
            ox = -moving_obstacle['distance'] * math.cos(ang_rad)
            oy = -moving_obstacle['distance'] * math.sin(ang_rad)
            self._obstacle_marker.set_offsets([[ox, oy]])
            self._obstacle_marker.set_color('red' if self.state == STATE_STOP else 'orange')
        else:
            self._obstacle_marker.set_offsets(np.empty((0, 2)))

        dist_txt = f"{moving_obstacle['distance']:.2f}m" if moving_obstacle is not None else '-'
        self._status_text.set_text(
            f"state: {self.state}\n"
            f"dist: {dist_txt}"
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
    # ROI 안에서 찾은 모든 군집(clusters)을 이전 프레임의 track들과 매칭해 갱신한다.
    #   1) 기존 track마다 "자차가 그동안 전진(ego_speed*dt)하고 회전(yaw_rate*dt)한 만큼"
    #      보정해 "정지 장애물이라면 지금 있어야 할 위치"를 예측하고, 그 예측 위치와
    #      가장 가까운(LIDAR_MAX_JUMP_M 이내) 미사용 군집을 그 track의 새 위치로 매칭한다.
    #      실제 좌표와 예측 위치의 차이가 그 track의 절대(지면 기준) 속도(vx, vy)다.
    #   2) 절대속도 크기가 ABS_SPEED_ZERO_EPS_MPS(+ 회전 중이면 odom 회전각속도 추정
    #      오차를 보정하기 위해 거리·회전각속도에 비례해 넓어진 허용 오차) 미만이면
    #      "절대속도 0"(정지 물체)로 보고, 그 이상이면서 측면(y) 성분 |vy|가
    #      LATERAL_SPEED_MIN_MPS 이상이어야 "좌우로 움직임"으로 본다. 그 결과로 각
    #      track의 연속 moving_streak을 갱신한다(끊기면 0으로 리셋).
    #   3) 매칭되지 않은 track은 OBSTACLE_LOST_GRACE_SEC 동안만 유지하고(라이다 잡음/
    #      순간 가려짐으로 한두 프레임 놓쳐도 추적이 끊기지 않게), 그 시간이 지나면
    #      제거한다. 매칭되지 않은 군집(새 물체)은 새 track으로 추가한다.
    # -----------------------------------------
    def _update_tracks(self, clusters, now):
        unclaimed = list(clusters)
        updated_tracks = []

        for track in self._tracks:
            dt     = max(now - track['time'], 1e-3)
            dtheta = self.ego_yaw_rate_radps * dt
            ego_dx = self.ego_speed_mps * dt

            shifted_x = track['x'] - ego_dx
            shifted_y = track['y']
            predicted_x = shifted_x * math.cos(dtheta) + shifted_y * math.sin(dtheta)
            predicted_y = -shifted_x * math.sin(dtheta) + shifted_y * math.cos(dtheta)

            best_i, best_dist = None, None
            for i, c in enumerate(unclaimed):
                angle_rad = math.radians(c['angle_deg'])
                cx = c['distance'] * math.cos(angle_rad)
                cy = c['distance'] * math.sin(angle_rad)
                d = math.hypot(cx - predicted_x, cy - predicted_y)
                if d <= LIDAR_MAX_JUMP_M and (best_dist is None or d < best_dist):
                    best_i, best_dist = i, d

            if best_i is None:
                # 이번 프레임에 매칭되는 군집이 없음 -> 유예 시간 안이면 그대로 유지
                if now - track['last_seen'] < OBSTACLE_LOST_GRACE_SEC:
                    updated_tracks.append(track)
                continue

            c = unclaimed.pop(best_i)
            angle_rad = math.radians(c['angle_deg'])
            x = c['distance'] * math.cos(angle_rad)
            y = c['distance'] * math.sin(angle_rad)

            vx = (x - predicted_x) / dt
            vy = (y - predicted_y) / dt

            if math.hypot(vx, vy) > MAX_PLAUSIBLE_OBJECT_SPEED_MPS:
                # 사람이 낼 수 없는 속도 -> 추적 오류 스파이크로 보고 이번 프레임만
                # 무시한다(위치/시간은 갱신해 다음 프레임부터 정상적으로 이어간다).
                track['x'], track['y'], track['time'], track['last_seen'] = x, y, now, now
                updated_tracks.append(track)
                continue

            zero_eps = (ABS_SPEED_ZERO_EPS_MPS
                        + YAW_COMPENSATION_ERROR_RATIO * abs(self.ego_yaw_rate_radps) * c['distance'])
            is_moving = math.hypot(vx, vy) >= zero_eps and abs(vy) >= LATERAL_SPEED_MIN_MPS

            track['x'], track['y'], track['time'], track['last_seen'] = x, y, now, now
            track['moving_streak'] = track['moving_streak'] + 1 if is_moving else 0
            updated_tracks.append(track)

        for c in unclaimed:
            angle_rad = math.radians(c['angle_deg'])
            x = c['distance'] * math.cos(angle_rad)
            y = c['distance'] * math.sin(angle_rad)
            updated_tracks.append({
                'id': self._next_track_id,
                'x': x, 'y': y, 'time': now, 'last_seen': now,
                'moving_streak': 0,
            })
            self._next_track_id += 1

        self._tracks = updated_tracks

    def _find_track(self, track_id):
        for track in self._tracks:
            if track['id'] == track_id:
                return track
        return None

    # -----------------------------------------
    # main loop
    # -----------------------------------------
    def run(self):
        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.01)

            if self.scan is None:
                continue

            clusters = find_clusters_in_roi(
                self.scan, ROI_LATERAL_MIN_M, ROI_LATERAL_MAX_M,
                ROI_FORWARD_MIN_M, ROI_FORWARD_MAX_M)

            now = time.time()
            self._update_tracks(clusters, now)

            # ROI 안의 모든 track 중 "좌우로 움직임"이 연속 DETECT_CONFIRM_FRAMES 프레임
            # 확인된 track이 하나라도 있으면 즉시 정지한다(가장 가까운 것을 정지 사유로 기록).
            confirmed = [t for t in self._tracks if t['moving_streak'] >= DETECT_CONFIRM_FRAMES]

            if confirmed:
                target = min(confirmed, key=lambda t: math.hypot(t['x'], t['y']))
                self._stop_track_id = target['id']
                self._enter(STATE_STOP)
            elif self.state == STATE_STOP:
                stop_track = self._find_track(self._stop_track_id)
                if stop_track is None:
                    # 정지를 유발한 그 물체가 ROI에서 (유예 시간을 넘겨서까지) 완전히
                    # 사라짐 -> 그 순간만 짧게 가속해서 벗어난다.
                    self._enter(STATE_ESCAPE)
                    self._escape_start_time = now
            elif self.state == STATE_ESCAPE and (now - self._escape_start_time) >= ESCAPE_BURST_DURATION_SEC:
                # 탈출 가속 종료 -> 더 이상 모터를 건드리지 않고 track_drive.py에 맡긴다.
                self._enter(STATE_IDLE)

            if self.state == STATE_STOP:
                # 매 루프마다 직접 publish해 track_drive.py의 주행 명령을 덮어쓴다
                # (track_drive.py보다 빠른 주기로 우선권을 유지).
                self.pub_motor.publish(XycarMotor(angle=0.0, speed=0.0))
            elif self.state == STATE_ESCAPE:
                self.pub_motor.publish(XycarMotor(angle=0.0, speed=ESCAPE_MOTOR_SPEED))
            # STATE_IDLE: /xycar_motor를 publish하지 않음 -> track_drive.py가 그대로 주행

            state_msg = String()
            state_msg.data = self.state
            self.pub_state.publish(state_msg)

            # 디버그 시각화: 정지를 유발(중)인 track이 있으면 그것을, 없으면 표시 안 함.
            display_track = self._find_track(self._stop_track_id) if self.state != STATE_IDLE else None
            display_obstacle = None
            if display_track is not None:
                display_obstacle = {
                    'distance': math.hypot(display_track['x'], display_track['y']),
                    'angle_deg': math.degrees(math.atan2(display_track['y'], display_track['x'])),
                }
            self._update_debug_plot(display_obstacle)

    def destroy(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.destroy_node()


# =============================================
# main
# =============================================
def main(args=None):

    rclpy.init(args=args)
    node = PedestrianDetectionNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
