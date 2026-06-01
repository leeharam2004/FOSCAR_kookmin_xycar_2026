import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from xycar_msgs.msg import XycarMotor

class NavToXycarBridge(Node):
    def __init__(self):
        super().__init__('nav_to_xycar_bridge')
        
        # 파라미터 설정 (필요 시 수정)
        self.SPEED_MIN, self.SPEED_MAX = -50.0, 50.0
        self.ANGLE_MIN, self.ANGLE_MAX = -100.0, 100.0
        
        # Nav2는 보통 최대 속도를 1.0m/s 내외로 쏩니다. (가정치)
        # 만약 자이카가 너무 느리게 움직이면 이 SCALE 값을 키우세요.
        self.speed_scale = 30.0  
        self.angle_scale = 200.0

        self.sub_cmd_vel = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
            
        self.pub_motor = self.create_publisher(
            XycarMotor,
            'xycar_motor',
            10)

    def cmd_vel_callback(self, msg):
        motor_msg = XycarMotor()
        motor_msg.header.stamp = self.get_clock().now().to_msg()
        motor_msg.header.frame_id = 'base_link'

        # 1. Speed 변환 (선속도)
        # Nav2의 linear.x(m/s)를 자이카 단위로 환산
        target_speed = msg.linear.x * self.speed_scale
        motor_msg.speed = float(max(min(target_speed, self.SPEED_MAX), self.SPEED_MIN))
        
        # 2. Angle 변환 (각속도 -> 조향각)
        # Nav2의 angular.z(rad/s)를 자이카 조향각 단위로 환산
        # 보통 아커만 차량은 각속도를 조향비에 따라 매핑해야 합니다.
        target_angle = - msg.angular.z * self.angle_scale # 거꾸로 해주기
        motor_msg.angle = float(max(min(target_angle, self.ANGLE_MAX), self.ANGLE_MIN))
        
        self.pub_motor.publish(motor_msg)
        # 로그로 변환 값 확인
        self.get_logger().info(f'In: v={msg.linear.x:.2f}, w={msg.angular.z:.2f} -> Out: s={motor_msg.speed:.1f}, a={motor_msg.angle:.1f}')

def main(args=None):
    rclpy.init(args=args)
    node = NavToXycarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()