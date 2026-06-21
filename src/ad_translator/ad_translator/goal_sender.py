import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


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


class GoalSenderNode(Node):
    def __init__(self):
        super().__init__('goal_sender')
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self._client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.get_logger().info('Waiting for follow_waypoints action server...')
        self._client.wait_for_server()
        time.sleep(7)
        self._send_goal()
        self.get_logger().info('Waiting 3s before sending waypoints...')
        time.sleep(1)
        self._publish_initial_pose()

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
        self._client.send_goal_async(goal)
        self.get_logger().info(f'Sent {len(goal.poses)} waypoints.')


def main(args=None):
    rclpy.init(args=args)
    node = GoalSenderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
