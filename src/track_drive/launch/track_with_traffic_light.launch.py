from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='track_drive',
            executable='track_drive',
            name='track_drive_2',
            output='screen',
            remappings=[
                ('/xycar_motor', '/lane_motor_cmd'),
            ],
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
                'lane_reacquire_frames': 3,
                'left_turn_timeout_frames': 150,
            }],
        ),
    ])
