import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from rclpy.qos import qos_profile_system_default #, DurabilityPolicy

import math
import numpy as np

class SpeedVisualizer():
    def __init__(self, logger: Node.get_logger):
        
        self.get_logger = logger

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_time_float = 0.0

        self.alpha = 0.5 #low pass filter 값
        self.filtered_speed = 0.0

        self.buffer = [] # moving_average용

    def speed_calculator(self, msg: Odometry) -> tuple[float, float, float]:
        """
        :return: (speed, distance, dt)
        """
        # 1. 헤더에서 현재 초(sec)와 나노초(nanosec) 추출
        curr_sec = msg.header.stamp.sec
        curr_nanosec = msg.header.stamp.nanosec
        
        # 2. 계산을 위해 하나의 실수(float) 초 단위로 합치기
        curr_time_float = curr_sec + (curr_nanosec / 1e9)
        
        # 3. 시간 차이(dt) 계산 (self.prev_time_float는 초기값 0.0으로 설정)
        dt = curr_time_float - self.prev_time_float
        
        if dt > 0:
            # 거리 계산 및 속도 계산 (피타고라스 정리)
            dist = math.sqrt(
                (msg.pose.pose.position.x - self.prev_x)**2 + 
                (msg.pose.pose.position.y - self.prev_y)**2
            )
            velocity = dist / dt

            # low pass filter
            self.filtered_speed = (self.alpha * velocity) + ((1.0 - self.alpha) * self.filtered_speed)

            # self.get_logger().info(f"Speed: {self.filtered_speed:.2f} m/s (dt: {dt:.4f})")
        
            # self.get_logger().info(f"dt_avr: {self.moving_average(dt)}")

        # 4. 다음 콜백을 위해 현재 값 저장
        self.prev_x = msg.pose.pose.position.x
        self.prev_y = msg.pose.pose.position.y
        self.prev_time_float = curr_time_float

        return velocity, dist, dt

    def moving_average(self, val): # 값 이동평균 보고싶을때 쓰기-지금은 안씀
        self.buffer.append(val)
        if len(self.buffer) > 50:
            self.buffer.pop(0)
            return sum(self.buffer) / len(self.buffer)
        else: return -1

        
class DeadReckoningNode(Node):
    def __init__(self):
        super().__init__("Dead_Reckoning")

        # 속도 계산용 Odom Topic 받아오기
        self.speed_calc = SpeedVisualizer(self.get_logger)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback,
            qos_profile_system_default
        )
        self.dist = 0.0

        # Heading(Theta) IMU에서 받아오기
        self.imu_sub = self.create_subscription(
            Imu, "/imu", self.imu_callback,
            qos_profile_system_default
        )
        self.heading = 0.0

        self.prev_x = 0.0
        self.prev_y = 0.0
    
    

    # def odom_calculator(dist, )

    def odom_callback(self, msg):
        _, self.dist, _ = self.speed_calc.speed_calculator(msg)

    def imu_callback(self, msg): # imu rate가 더 빠르므로 imu 기준 동기화

        # 쿼터니언 (x, y, z, w)
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # 쿼터니언을 Yaw(라디안)로 변환하는 정석 수식
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        raw_heading = math.atan2(siny_cosp, cosy_cosp)

        # 오른쪽으로 90도 틀어져 있다면, 1.57을 더해서 정면(0)으로 맞춤
        corrected_heading = raw_heading + 1.5708
        heading = math.atan2(math.sin(corrected_heading), math.cos(corrected_heading))
        
        self.get_logger().info(f"Heading: {heading}")
        


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
