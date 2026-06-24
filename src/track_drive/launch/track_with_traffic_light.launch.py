from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Nav2 없는 환경에서 traffic_Light.py의 대기를 해제한다.
        # 노드들이 구동되고 subscriber를 등록하기까지 2초 여유를 준다.
        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'topic', 'pub', '--once',
                        '/nav2_bypass', 'std_msgs/msg/Bool', 'data: true',
                    ],
                    output='screen',
                ),
            ],
        ),
        Node(
            package='track_drive',
            executable='track_drive',
            name='track_drive_2',
            output='screen',
        ),
        Node(
            package='track_drive',
            executable='traffic_light',
            name='traffic_detection',
            output='screen',
            parameters=[{
                'red_pixel_threshold': 50,
                'green_pixel_threshold': 50,
                'confirmation_frames': 3,
                'left_turn_angle': -100.0,
                'left_turn_speed': 3.0,
                'signal_left_turn_approach_sec': 1.5,
                'signal_wait_timeout_sec': 3.0,
                'no_signal_left_turn_hold_sec': 7.0,
                'lane_reacquire_frames': 3,
                'left_turn_timeout_frames': 150,
            }],
        ),
    ])
