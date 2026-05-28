#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from xycar_msgs.msg import Motor

WIDTH = 640
HEIGHT = 480

ROI_Y = 220
ROI_HEIGHT = 220

prev_angle = 0

class LaneDriver(Node):

```
def __init__(self):

    super().__init__('lane_driver')

    self.motor_pub = self.create_publisher(
        Motor,
        'xycar_motor',
        10
    )

    self.cap = cv2.VideoCapture(0)

    self.timer = self.create_timer(
        0.03,
        self.run
    )

def drive(self, angle, speed):

    msg = Motor()

    msg.angle = float(angle)
    msg.speed = float(speed)

    self.motor_pub.publish(msg)

def detect_lane(self, frame):

    global prev_angle

    roi = frame[ROI_Y:ROI_Y + ROI_HEIGHT, :]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        30,
        minLineLength=40,
        maxLineGap=20
    )

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.3:
                continue

            if slope < 0 and x2 < WIDTH // 2:
                left_lines.append(line[0])

            elif slope > 0 and x1 > WIDTH // 2:
                right_lines.append(line[0])

    left_x = None
    right_x = None

    if len(left_lines) > 0:

        x_sum = 0

        for line in left_lines:

            x1, y1, x2, y2 = line

            x_sum += (x1 + x2)

            cv2.line(
                roi,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

        left_x = x_sum // (len(left_lines) * 2)

    if len(right_lines) > 0:

        x_sum = 0

        for line in right_lines:

            x1, y1, x2, y2 = line

            x_sum += (x1 + x2)

            cv2.line(
                roi,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

        right_x = x_sum // (len(right_lines) * 2)

    if left_x is not None and right_x is not None:

        lane_center = (left_x + right_x) // 2

    elif right_x is not None:

        lane_center = right_x - 220

    elif left_x is not None:

        lane_center = left_x + 220

    else:

        return prev_angle, 15

    image_center = WIDTH // 2

    error = lane_center - image_center

    kp = 0.25

    if abs(error) > 60:
        kp = 0.4

    if abs(error) > 100:
        kp = 0.55

    if abs(error) > 140:
        kp = 0.7

    angle = kp * error

    angle = max(min(angle, 50), -50)

    angle = (
        prev_angle * 0.7
        + angle * 0.3
    )

    prev_angle = angle

    speed = 30

    if abs(angle) > 20:
        speed = 24

    if abs(angle) > 35:
        speed = 20

    if abs(angle) > 45:
        speed = 16

    cv2.line(
        roi,
        (image_center, 0),
        (image_center, ROI_HEIGHT),
        (0, 0, 255),
        2
    )

    cv2.circle(
        roi,
        (lane_center, ROI_HEIGHT // 2),
        8,
        (0, 255, 255),
        -1
    )

    cv2.imshow("Lane ROI", roi)
    cv2.imshow("Edges", edges)

    cv2.waitKey(1)

    return angle, speed

def run(self):

    ret, frame = self.cap.read()

    if not ret:
        return

    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT)
    )

    angle, speed = self.detect_lane(frame)

    self.drive(angle, speed)
```

def main(args=None):

```
rclpy.init(args=args)

node = LaneDriver()

rclpy.spin(node)

node.destroy_node()

rclpy.shutdown()
```

if **name** == '**main**':
main()
