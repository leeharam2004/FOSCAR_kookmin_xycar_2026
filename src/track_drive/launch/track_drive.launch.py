from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='track_drive',
                executable='track_drive',
                name='track_drive',
                output='screen',
            ),
            Node(
                package='track_drive',
                executable='traffic_light',
                name='traffic_light',
                output='screen',
            ),
            Node(
                package='track_drive',
                executable='overtake_drive',
                name='overtake_drive',
                output='screen',
            ),
            Node(
                package='ad_tf_maker',
                executable='dead_reckoning',
                name='dead_reckoning',
                output='screen',
            ),
        ]
    )
