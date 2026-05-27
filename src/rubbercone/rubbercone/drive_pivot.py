#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라바콘 조향각 계산 노드 (ROS2)
- 라이다로 감지한 라바콘 위치 수신
- 왼쪽/오른쪽 분리 후 중간점 계산
- 선형회귀로 조향각 계산
- /xycar_motor_rubbercone 발행
- /rubbercone_active 발행 (모드 신호)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import XycarMotor
import numpy as np
import math
from scipy.stats import linregress


class Object:
    def __init__(self, centerX, centerY):
        self.centerX = centerX
        self.centerY = centerY

    def distance_to(self, other):
        return ((self.centerX - other.centerX)**2 + (self.centerY - other.centerY)**2) ** 0.5


class ConeArray:
    def __init__(self):
        self.cones = []
        self.size = 0


# pivot 리셋 영역
pivot_reset_area = [0.4, 0.0, 0.7, -0.7]


class DrivePivot(Node):
    def __init__(self):
        super().__init__('drive_pivot')

        # 구독
        self.create_subscription(Bool,      '/is_orange', self.orange_cb, 10)
        self.create_subscription(String,    '/mode',      self.mode_cb,   10)
        self.create_subscription(LaserScan, '/scan',      self.lidar_cb,  10)

        # 발행
        self.motor_pub  = self.create_publisher(XycarMotor, '/xycar_motor_rubbercone', 10)
        self.active_pub = self.create_publisher(Bool,       '/rubbercone_active',      10)

        # 상태 변수
        self.is_orange = False
        self.mode      = ''
        self.objects   = []

        # pivot 초기값
        self.leftPivot  = Object(0,  0.5)
        self.rightPivot = Object(0, -0.5)
        self.leftCones  = ConeArray()
        self.rightCones = ConeArray()

        # 속도 설정
        self.motor_speed = 30.0

        # 30Hz 루프
        self.timer = self.create_timer(1/30, self.timer_cb)

        self.get_logger().info('DrivePivot node started')

    def orange_cb(self, msg):
        self.is_orange = msg.data

    def mode_cb(self, msg):
        self.mode = msg.data

    def lidar_cb(self, msg):
        objects = []
        angle = msg.angle_min

        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                # ROI 필터링 (x: 0~0.75m, y: ±0.9m)
                if 0.0 < x < 0.75 and -0.9 < y < 0.9:
                    objects.append(Object(x, y))

            angle += msg.angle_increment

        origin = Object(0, 0)
        self.objects = sorted(objects, key=lambda o: o.distance_to(origin))

    def timer_cb(self):
        if self.mode == 'STATIC':
            return
        self.set_left_right_cone_info()
        self.set_waypoint_info()

    def set_left_right_cone_info(self):
        left_cones  = []
        right_cones = []

        left_point  = self.leftPivot
        right_point = self.rightPivot

        for cone in self.objects:
            d_left  = cone.distance_to(left_point)
            d_right = cone.distance_to(right_point)

            if d_left > 0.75 and d_right > 0.75:
                continue

            if d_left <= d_right:
                left_cones.append(cone)
                left_point = cone
            else:
                right_cones.append(cone)
                right_point = cone

        if not left_cones:
            left_cones = [self.leftPivot]
        else:
            self.leftPivot = left_cones[0]

        if not right_cones:
            right_cones = [self.rightPivot]
        else:
            self.rightPivot = right_cones[0]

        # pivot 리셋 조건
        if (self.leftPivot.centerX >= pivot_reset_area[0] or
            self.leftPivot.centerY <= pivot_reset_area[1] or
            self.leftPivot.centerY >= pivot_reset_area[2]):
            self.leftPivot = Object(0, 0.5)

        if (self.rightPivot.centerX >= pivot_reset_area[0] or
            self.rightPivot.centerY <= pivot_reset_area[3] or
            self.rightPivot.centerY >= pivot_reset_area[1]):
            self.rightPivot = Object(0, -0.5)

        self.leftCones.cones  = self._pad_to_5(left_cones)
        self.rightCones.cones = self._pad_to_5(right_cones)

        self.get_logger().info(
            f'left_cones={len(left_cones)} right_cones={len(right_cones)} ' \
            f'leftPivot=({self.leftPivot.centerX:.2f},{self.leftPivot.centerY:.2f}) ' \
            f'rightPivot=({self.rightPivot.centerX:.2f},{self.rightPivot.centerY:.2f})'
        )

    def _pad_to_5(self, cones):
        if len(cones) < 5:
            last = cones[-1]
            cones.extend([last] * (5 - len(cones)))
        return cones[:5]

    def set_waypoint_info(self):
        x_vals = []
        y_vals = []

        for i in range(5):
            l = self.leftCones.cones[i]
            r = self.rightCones.cones[i]
            mid_x = (l.centerX + r.centerX) / 2
            mid_y = (l.centerY + r.centerY) / 2
            x_vals.append(mid_x)
            y_vals.append(mid_y)

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        self.get_logger().info(
            f'x_vals={x_vals.tolist()} y_vals={y_vals.tolist()} ' \
            f'left_count={len(self.leftCones.cones)} right_count={len(self.rightCones.cones)}'
        )

        active    = False
        angle_deg = 0.0

        if len(set(x_vals)) == 1:
            self.get_logger().warn(
                'All mid_x values equal; likely insufficient distinct cones or duplicate padding.'
            )
            angle_deg = 0.0
            active    = False
        elif not self.is_orange:
            angle_deg = 0.0
            active    = False
        else:
            slope, _, _, _, _ = linregress(
                np.insert(x_vals, 0, 0),
                np.insert(y_vals, 0, 0)
            )
            angle_deg = -math.degrees(math.atan(slope))
            active    = True

        # 주행 명령 발행
        motor_msg = XycarMotor()
        motor_msg.speed = float(self.motor_speed)
        motor_msg.angle = float(angle_deg)
        self.motor_pub.publish(motor_msg)

        # 모드 신호 발행 (speed와 완전히 독립)
        active_msg = Bool()
        active_msg.data = active
        self.active_pub.publish(active_msg)

        self.get_logger().info(
            f'is_orange: {self.is_orange} | angle: {angle_deg:.1f} | active: {active}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DrivePivot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()