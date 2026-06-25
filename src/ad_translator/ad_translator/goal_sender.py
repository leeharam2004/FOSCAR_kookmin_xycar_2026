import math
import subprocess
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Int64MultiArray


# (x, y, ori_z, ori_w)
WAYPOINTS = [
    (19.552383422851562, 0.5542383193969727, 0.012713235474425032, 0.9999191835562371),   # intermediate
    (36.85664367675781, 12.282488822937012, 0.627079677806722, 0.7789551191704293),  # final goal
]

# matches nav2_params.yaml initial_pose
INITIAL_POSE = (0.39496, 0.0365, 0.007)  # x, y, yaw


def make_pose(node, x, y, ori_z, ori_w):
    p = PoseStamped()
    p.header.frame_id = 'ad/map'
    p.header.stamp = node.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = ori_z
    p.pose.orientation.w = ori_w
    return p


GREEN_PIXEL_THRESHOLD = 30


class GoalSenderNode(Node):
    def __init__(self):
        super().__init__('goal_sender')
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self._client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self._green_detected = False
        self.create_subscription(Int64MultiArray, '/traffic_light', self._traffic_light_callback, 10)
        self.get_logger().info('Waiting for follow_waypoints action server...')
        self._client.wait_for_server()
        time.sleep(26)
        self.get_logger().info('Waiting for green light...')
        # self._wait_for_green() #임시!!
        self._send_goal()
        self.get_logger().info('Waiting 1s before sending waypoints...')
        time.sleep(1)
        self._publish_initial_pose()

    def _traffic_light_callback(self, msg):
        if msg.data[1] >= GREEN_PIXEL_THRESHOLD:
            self._green_detected = True

    def _wait_for_green(self):
        while rclpy.ok() and not self._green_detected:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _publish_initial_pose(self):
        x, y, yaw = INITIAL_POSE
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'ad/map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # same covariance RViz2 uses for 2D Pose Estimate
        msg.pose.covariance[0] = 0.25                   # x ±0.5m
        msg.pose.covariance[7] = 0.25                   # y ±0.5m
        msg.pose.covariance[35] = 0.06853892326654787   # yaw ±~15°
        self._initial_pose_pub.publish(msg)
        self.get_logger().info('Published initial pose with wide covariance.')

    def _send_goal(self):
        goal = FollowWaypoints.Goal()
        goal.poses = [make_pose(self, x, y, oz, ow) for x, y, oz, ow in WAYPOINTS]
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(f'Sent {len(goal.poses)} waypoints.')

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2')
            return
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self.get_logger().info('All waypoints completed — killing motor_translator')

        # [현재: 방법 1] motor_translator 프로세스를 직접 종료 → count_publishers가 1로 떨어져 track_drive 활성화
        subprocess.Popen(['pkill', '-f', 'motor_translator'])
        # Nav2 컨테이너 전체 종료 (차선 주행 단계에서 불필요한 CPU 절감)
        # subprocess.Popen(['pkill', '-f', 'component_container_isolated'])

        # [방법 2로 바꾸려면 이 블록 대신]:
        # motor_translator는 그냥 놔두고, track_drive.py 쪽을 수정
        # track_drive.py의 drive() 또는 타이머 콜백에서:
        #   self.last_motor_msg_time = self.get_clock().now()  ← motor_translator 토픽 수신 시 갱신
        #   dt = (now - self.last_motor_msg_time).nanoseconds / 1e9
        #   if dt > 0.5: # 0.5초 이상 motor_translator 메시지 없으면 넘어감
        #       count_publishers 체크 스킵하고 track_drive 활성화


def main(args=None):
    rclpy.init(args=args)
    node = GoalSenderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
