# 🚀 자율주행 프로젝트 실행 가이드

이 저장소는 유니티(Unity) 시뮬레이션 환경에서 **SLAM Toolbox**를 이용한 지도 작성과 **Nav2**를 이용한 자율주행을 수행하기 위한 ROS2 패키지입니다.

## 📋 필수 요구 사항

팀원들의 컴퓨터에 다음 패키지들이 반드시 설치되어 있어야 합니다. (Humble 버전 기준)

```bash
sudo apt update
sudo apt install ros-humble-slam-toolbox \
                 ros-humble-navigation2 \
                 ros-humble-nav2-bringup \
                 ros-humble-tf2-tools

```

## 🛠️ 설치 및 빌드

워크스페이스에 패키지를 복사한 후 아래 명령어를 실행하세요.

```bash
# 워크스페이스 루트에서 실행
rm -r build/ install/ log/  # 기존 빌드 찌꺼기 제거
colcon build --symlink-install
source install/setup.bash

```

## 🏃 실행 방법

### 1. 전체 시스템 원클릭 실행 (Launch)

오도메트리, Static TF, SLAM, Nav2를 한 번에 실행합니다.

> **참고:** 유니티 시뮬레이터가 먼저 실행 중이어야 합니다.

```bash
ros2 launch track_drive bringup.launch.py
```

일단 한 반쯤은 되게 만들었어용

저 launch 한 후에
ad_translator 패키지의 motor_translator 실행시켜주고
rviz2에서 tf에다가 ad/map이라고 쓰고 기본 pose estimate 해준다음에 goal estimate 설정해주면 가려고 하는데
**아직 mapping이 제대로 안되는지 얼마 지나서 map의 좌표가 완전히 부정확해집니다 ㅠㅠ**
일단 저녁에 또 해볼예정!

### 2. 주요 설정 및 주의사항

* **중요!!!!!!!!!!!!!!!:** 시뮬레이터에서 차량을 움직일 때 `kookmin9_viewer`를 사용해야 합니다. 키보드로 움직이는 경우 Speed값을 코드가 받아올 수 없어 움직임을 인식하지 못합니다.
* **네임스페이스:** 모든 데이터는 `ad/` 네임스페이스를 사용하도록 설정되어 있습니다. (예: `/ad/scan`, `/ad/odom`)
* **Time Mode:** 현재 `/clock` 토픽이 없는 환경에 맞춰 `use_sim_time:=False`로 설정되어 있습니다. 시뮬레이션 렉이 심할 경우 주의가 필요합니다.
* **라이다 필터:** 로봇 차체 간섭을 피하기 위해 센서 위치가 조정되어 있습니다. (`static_tf_node` 확인)

## 📂 폴더 구조

* `misc_files/`: 개인 메모 및 참고 자료 (빌드에 포함되지 않음)
    * 미리 만들어놓은 my_map 파일 있어용
    * `rviz2` 명령어를 통해 SLAM 과정을 볼 수 있습니다. 미리 설정해둔 `mapping_setup.rviz` 파일을 열어주세요.

---

### 💡 팀원들에게 전하는 팁

1. AI 좋네요

## 사용하는 토픽 리스트

```
--사용하는 토픽 리스트--
/ad/odom
/ad/scan
/imu
/tf
/tf_static
--사용할 map 파일--
os.path.join(pkg_share, 'config', 'my_map.yaml')
--tf tree--
ad/map -> ad/odom -> ad/base_link -> lidar_frame
```
