#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

# 전방 감지 범위
FRONT_ANGLE_DEG = 30.0   # 정면 기준 ±각도
MIN_DIST        = 0.3    # m — 이보다 가까우면 차체 자기 자신
MAX_DIST        = 3.0    # m — 감지 최대 거리

# 사람으로 인정하는 클러스터 폭
MIN_WIDTH = 0.2          # m
MAX_WIDTH = 1.0          # m

MIN_POINTS = 3           # 클러스터 최소 포인트 수
CLUSTER_GAP = 0.25       # m — 이 거리 이상 벌어지면 다른 물체


class PedestrianDetector(Node):

    def __init__(self):
        super().__init__('pedestrian_detector')
        self.create_subscription(LaserScan, '/ad/scan', self._scan_callback, 10)
        self._pub = self.create_publisher(Bool, '/pedestrian_detected', 10)
        self.get_logger().info('Pedestrian detector started')

    def _scan_callback(self, msg):
        points = self._front_points(msg)
        clusters = self._cluster(points)
        detected = any(
            len(c) >= MIN_POINTS and MIN_WIDTH <= self._width(c) <= MAX_WIDTH
            for c in clusters
        )
        out = Bool()
        out.data = detected
        self._pub.publish(out)

    def _front_points(self, msg):
        front_rad = math.radians(FRONT_ANGLE_DEG)
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            norm = math.atan2(math.sin(angle), math.cos(angle))
            if abs(norm) <= front_rad and MIN_DIST <= r <= min(msg.range_max, MAX_DIST):
                points.append((r * math.cos(angle), r * math.sin(angle)))
            angle += msg.angle_increment
        return points

    def _cluster(self, points):
        if not points:
            return []
        clusters, current = [], [points[0]]
        for pt in points[1:]:
            dx, dy = pt[0] - current[-1][0], pt[1] - current[-1][1]
            if math.sqrt(dx * dx + dy * dy) < CLUSTER_GAP:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]
        clusters.append(current)
        return clusters

    @staticmethod
    def _width(cluster):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
