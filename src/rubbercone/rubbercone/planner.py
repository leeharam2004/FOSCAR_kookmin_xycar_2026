#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라바콘 플래너 노드 (ROS2)
- 모드 판별 (RUBBERCONE > STATIC > LANE)
- 최종 /xycar_motor 발행
"""
 
import rclpy
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from std_msgs.msg import String, Bool
 
 
class RubbercondPlanner(Node):
    def __init__(self):
        super().__init__('rubbercone_planner')
 
        # 구독
        self.create_subscription(XycarMotor, '/xycar_motor_rubbercone', self.rubbercone_cb,        10)
        self.create_subscription(XycarMotor, '/xycar_motor_lane',       self.lane_cb,              10)
        self.create_subscription(XycarMotor, '/xycar_motor_static',     self.static_cb,            10)
        self.create_subscription(Bool,       '/rubbercone_active',      self.rubbercone_active_cb, 10)
 
        # 발행
        self.motor_pub = self.create_publisher(XycarMotor, '/xycar_motor', 10)
        self.mode_pub  = self.create_publisher(String,     '/mode',        10)
 
        # 모터 메시지 초기화
        self.ctrl_rubbercone = XycarMotor()
        self.ctrl_lane       = XycarMotor()
        self.ctrl_static     = XycarMotor()
 
        # 모드 플래그
        self.rubbercone_mode_flag = False  # /rubbercone_active 토픽으로만 변경
        self.static_mode_flag     = False
        self.lane_mode_flag       = False
 
        self.mode = ''
 
        # 30Hz 루프
        self.timer = self.create_timer(1/30, self.timer_cb)
 
        self.get_logger().info('RubbercondPlanner node started')
 
    def rubbercone_cb(self, msg):
        # speed/angle 값만 저장 (모드 판별 X)
        self.ctrl_rubbercone = msg
 
    def rubbercone_active_cb(self, msg):
        # 모드 신호는 여기서만 판별
        self.rubbercone_mode_flag = msg.data
 
    def lane_cb(self, msg):
        self.ctrl_lane = msg
 
    def static_cb(self, msg):
        self.ctrl_static = msg
 
    def timer_cb(self):
        # ── MODE 판별 (우선순위: RUBBERCONE > STATIC > LANE) ──
        if self.rubbercone_mode_flag:
            self.mode = 'RUBBERCONE'
        elif self.static_mode_flag:
            self.mode = 'STATIC'
        else:
            self.mode = 'LANE'
 
        # ── MODE에 따른 motor/steer 설정 ──
        if self.mode == 'RUBBERCONE':
            motor = self.ctrl_rubbercone.speed
            steer = self.ctrl_rubbercone.angle
        elif self.mode == 'STATIC':
            motor = self.ctrl_static.speed
            steer = self.ctrl_static.angle
        else:  # LANE
            motor = self.ctrl_lane.speed
            steer = self.ctrl_lane.angle
 
        # ── 발행 ──
        cmd = XycarMotor()
        cmd.speed = float(motor)
        cmd.angle = float(steer)
        self.motor_pub.publish(cmd)
 
        mode_msg = String()
        mode_msg.data = self.mode
        self.mode_pub.publish(mode_msg)
 
        self.get_logger().info(f'MODE: {self.mode} | SPEED: {motor} | STEER: {steer}')
 
 
def main(args=None):
    rclpy.init(args=args)
    node = RubbercondPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()