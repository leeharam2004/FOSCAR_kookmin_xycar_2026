import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_system_default #, DurabilityPolicy
import math

class SpeedVisualizer(Node):
    def __init__(self):
        super().__init__("odom_speed_visualizer")

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_time_float = 0.0

        self.sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback,
            qos_profile_system_default
        )

        self.alpha = 0.5 #low pass filter 값
        self.filtered_speed = 0.0

        self.dt_buffer = [] # moving_average용

    def odom_callback(self, msg):
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

            self.get_logger().info(f"Speed: {self.filtered_speed:.2f} m/s (dt: {dt:.4f})")
        
            # self.get_logger().info(f"dt_avr: {self.moving_average(dt)}")

        # 4. 다음 콜백을 위해 현재 값 저장
        self.prev_x = msg.pose.pose.position.x
        self.prev_y = msg.pose.pose.position.y
        self.prev_time_float = curr_time_float

    def moving_average(self, val):
        self.dt_buffer.append(val)
        if len(self.dt_buffer) > 50:
            self.dt_buffer.pop(0)
            return sum(self.dt_buffer) / len(self.dt_buffer)
        else: return -1
        

def main(args=None):
    rclpy.init(args=args)
    node = SpeedVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
