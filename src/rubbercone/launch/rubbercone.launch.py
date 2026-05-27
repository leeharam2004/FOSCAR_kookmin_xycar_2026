from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # 1. 주황색 라바콘 감지 (카메라 → /is_orange)
        Node(
            package='rubbercone',
            executable='orange_detection',
            name='orange_detection',
            output='screen',
        ),

        # 2. 라바콘 조향각 계산 (라이다 + /is_orange → /xycar_motor_rubbercone)
        Node(
            package='rubbercone',
            executable='drive_pivot',
            name='drive_pivot',
            output='screen',
        ),

        # 3. 플래너 (모드 판별 + 최종 /xycar_motor 발행)
        Node(
            package='rubbercone',
            executable='planner',
            name='rubbercone_planner',
            output='screen',
        ),

    ])
