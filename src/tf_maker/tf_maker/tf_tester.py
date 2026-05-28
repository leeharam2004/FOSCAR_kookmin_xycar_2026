import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, Quaternion

from xycar_msgs.msg import XycarMotor

from tf2_ros import TransformBroadcaster

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

            # 임시값!! 오차가 엄청 큼
            # 회전 시 누적 오차가 너무 커요 ㅠㅠㅠㅠ
            dist = dist / 2
            
            velocity = dist / dt

            # low pass filter
            self.filtered_speed = (self.alpha * velocity) + ((1.0 - self.alpha) * self.filtered_speed)

            self.get_logger().info(f"Got: spd{self.filtered_speed:.1f}")
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


class DeadReckoningNode(Node): #ros 노드인 메인 클래스
    def __init__(self):
        super().__init__("Dead_Reckoning")

        # 속도 계산용 Odom Topic 받아오기
        self.speed_calc = SpeedVisualizer(self.get_logger)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback,
            qos_profile_system_default
        )
        self.speed = 0.0
        self.dist = 0.0

        # Heading(Theta) IMU에서 받아오기
        self.imu_sub = self.create_subscription(
            Imu, "imu", self.imu_callback,
            qos_profile_system_default
        )
        self.raw_heading = None # 오른쪽으로 90도 틀어져있음
        self.heading = 0.0

        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_pub = self.create_publisher(
            Odometry, "ad/odom", qos_profile_system_default
        )

        #임시!!!!!! 모터 pub에서 방향 추정
        self.speed_sign_check = self.create_subscription(
            XycarMotor, "xycar_motor", self.motor_callback,
            qos_profile_system_default
        )
        self.speed_sign = 0

        self.x = 0.0
        self.y = 0.0
    
    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        """
        라디안 단위의 Yaw 값을 geometry_msgs/Quaternion 메시지로 변환합니다.
        (Roll과 Pitch는 0이라고 가정)
        """
        q = Quaternion()
        
        # 2D 평면 주행 로봇은 Z축 회전만 고려하므로 수식이 매우 간단합니다.
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        
        return q

    def odom_publisher(self):
        cur_time = self.get_clock().now().to_msg()

        # 임시-speed_sign이 방향추정을 위해 추가한 코드입니다
        # 이거때문에 시뮬레이터 키보드로는 안되고 kookmin9_viewer 통해서 조종해야합니다
        self.x = np.cos(self.heading) * self.dist * self.speed_sign + self.x
        self.y = np.sin(self.heading) * self.dist * self.speed_sign + self.y

        # --- 1. TF 발행 (odom -> base_link) ---
        # RViz에서 로봇이 실제로 움직이게 만드는 핵심 부분입니다.
        t = TransformStamped()
        t.header.stamp = cur_time
        t.header.frame_id = 'ad/odom'
        t.child_frame_id = 'ad/base_link'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        # Yaw를 다시 쿼터니언으로 변환해서 넣어야 합니다.
        q = self.yaw_to_quaternion(self.heading)
        t.transform.rotation = q

        self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = cur_time
        odom.header.frame_id = 'ad/odom'
        odom.child_frame_id = 'ad/base_link'

        # 위치 데이터
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = q
        # 속도 데이터 (Twist)
        odom.twist.twist.linear.x = self.speed  # 필터링된 속도
        odom.twist.twist.angular.z = 0.0    # 필요시 IMU의 angular velocity 입력

        self.odom_pub.publish(odom)

    def odom_callback(self, msg):
        self.speed, self.dist, _ = self.speed_calc.speed_calculator(msg)

    def imu_callback(self, msg): # 여기에서 최종 tf publish

        # 쿼터니언 (x, y, z, w)
        self.raw_heading = msg.orientation
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
        self.heading = math.atan2(math.sin(corrected_heading), math.cos(corrected_heading))
        
        # self.get_logger().info(f"Heading: {self.heading}")
        self.get_logger().info(f"Got:         hdg:{self.heading:.1f}")

        #드디어!!!!!!!!!!11 publisher랑 imu topic 받는거랑 동기식으로 묶었습니다
        self.odom_publisher()

    #임시!!!!!! 모터 pub에서 방향 추정
    def motor_callback(self, data):
        self.speed_sign = np.sign(data.speed)

        

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
