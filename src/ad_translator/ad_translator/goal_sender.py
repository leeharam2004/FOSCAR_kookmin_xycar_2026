import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped


# (x, y, ori_z, ori_w)
WAYPOINTS = [
    (19.552383422851562, 0.5542383193969727, 0.012713235474425032, 0.9999191835562371),   # intermediate
    (36.85664367675781, 12.282488822937012, 0.627079677806722, 0.7789551191704293),  # final goal
]


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
        self._client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.get_logger().info('Waiting for follow_waypoints action server...')
        self._client.wait_for_server()
        self.get_logger().info('Action server ready. Waiting 10s before sending waypoints...')
        time.sleep(10)
        self._send_goal()

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
