import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. 경로 설정 (패키지 이름을 본인 것으로 수정하세요)
    pkg_name = 'track_drive'
    pkg_share = get_package_share_directory(pkg_name)
    
    # 2. TF 노드
    dead_reckoning_node = Node(
        package='ad_tf_maker',
        executable='dead_reckoning',  # setup.py에 등록된 이름
        output='screen',
        # ns 수정은 파일안에서!
        parameters=[]
    )

    # Lidar 더미값 제거 노드
    lidar_translate_node = Node(
        package='ad_translator',
        executable='lidar_translator',  # setup.py에 등록된 이름
        output='screen',
        # ns 수정은 파일안에서!
        parameters=[]
    )

    # 3. Static TF (base_link -> laser_frame)
    # 나중에 시뮬에서 주는 Static TF로 바꾸기
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.1', '0', '0', '0', 'ad/base_link', 'lidar_frame'],
        parameters=[]
    )

    # 4. SLAM Toolbox (기존 지도를 불러오도록 설정된 경우)
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')
        ]),
        launch_arguments={
            'slam_params_file': os.path.join(pkg_share, 'config', 'slam_toolbox_params.yaml')
        }.items()
    )

    # 5. Nav2 (가져온 지도 파일과 함께 실행)
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
        ]),
        launch_arguments={
            'params_file': os.path.join(pkg_share, 'config', 'nav2_params.yaml'),
            'slam': 'False',
            'use_sim_time': 'False',
            'map': os.path.join(pkg_share, 'config', 'my_map_edited.yaml'), # 이 부분이 중요!
        }.items()
    )

    return LaunchDescription([
        dead_reckoning_node,
        lidar_translate_node,
        static_tf_node,
        # slam_toolbox_launch,
        nav2_launch
    ])