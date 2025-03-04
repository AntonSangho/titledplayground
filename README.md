# RealSense & YDLidar 멀티센서 인물 추적 시스템

이 프로젝트는 Intel RealSense 카메라와 YDLidar, YOLOv8을 사용하여 실시간으로 사람과 물체를 감지하고 추적하는 시스템입니다.

## 주요 기능

- 실시간 인물 및 물체 감지/추적
- 멀티센서 퓨전 (RealSense + YDLidar)
- 3D 공간상의 위치 측정
- 개별 ID 할당 및 추적
- 시각화 도구 제공
  - 바운딩 박스
  - 2D 맵핑 뷰
  - 깊이 정보 표시
  - 라이다 스캔 데이터 시각화

## 파일 구조

- `1_camerainit_yolo8.py`: RealSense 카메라 초기화 및 기본 설정
- `2_humandetect_location.py`: RealSense 기반 인물 감지 및 추적
- `ydlidar-object-detection.py`: YDLidar 기반 물체 감지
- `lidar-realsense-fixedport.py`: RealSense와 YDLidar 통합 감지 시스템
- `yolo_circle.py`: 시각화 관련 유틸리티
- `.gitignore`: Git 버전 관리 제외 파일 설정

## 요구사항

- Python 3.8 이상
- Intel RealSense SDK 2.0
- YDLidar SDK
- 필요한 Python 패키지:
  - pyrealsense2
  - numpy
  - opencv-python
  - ultralytics (YOLOv8)
  - ydlidar-sdk

## 설치 방법

1. Intel RealSense SDK 2.0 설치
2. YDLidar SDK 설치:
   ```bash
   sudo apt-get install cmake
   git clone https://github.com/YDLIDAR/YDLidar-SDK.git
   cd YDLidar-SDK
   mkdir build
   cd build
   cmake ..
   make
   sudo make install
   ```
3. Python 패키지 설치

## 하드웨어 설정

- RealSense D435i
  - 해상도: 640x480
  - 프레임레이트: 30fps

- YDLidar
  - 포트: /dev/ttyUSB0
  - Baudrate: 128000
  - 스캔 주파수: 10Hz

## 키 조작

- 'q' 또는 'ESC': 프로그램 종료

## 출력 정보

- 실시간 디스플레이:
  - 카메라 뷰 (바운딩 박스 포함)
  - 2D 맵핑 뷰 (탑뷰)
  - 라이다 스캔 시각화
- 콘솔 출력:
  - 감지된 객체 수
  - 각 객체의 ID, 종류, 거리, 센서 소스
