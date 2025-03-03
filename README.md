# RealSense 인물 추적 시스템

이 프로젝트는 Intel RealSense 카메라와 YOLOv8을 사용하여 실시간으로 사람을 감지하고 추적하는 시스템입니다.

## 주요 기능

- 실시간 인물 감지 및 추적
- 3D 공간상의 위치 측정
- 개별 ID 할당 및 추적
- 시각화 도구 제공
  - 바운딩 박스
  - 2D 맵핑 뷰
  - 깊이 정보 표시

## 파일 구조

- `1_camerainit_yolo8.py`: RealSense 카메라 초기화 및 기본 설정
- `2_humandetect_location.py`: 메인 실행 파일 (인물 감지 및 추적)
- `yolo_circle.py`: 시각화 관련 유틸리티
- `.gitignore`: Git 버전 관리 제외 파일 설정

## 요구사항

- Python 3.8 이상
- Intel RealSense SDK 2.0
- 필요한 Python 패키지:
  - pyrealsense2
  - numpy
  - opencv-python
  - ultralytics (YOLOv8)

## 설치 방법

1. Intel RealSense SDK 2.0 설치
2. Python 패키지 설치:

## [2_humandetect_location.py](/2_humandetect_location.py) 파라미터 

- 카메라 설정:
  - 해상도: 640x480
  - 프레임레이트: 30fps
  
- 인물 감지 설정:
  - 모델: YOLOv8n
  - 신뢰도 임계값: 0.3
  - IoU 임계값: 0.45

- 추적 설정:
  - 최대 추적 거리: 100 픽셀
  - 미감지 임계값: 10 프레임

## 키 조작

- 'q': 프로그램 종료

## 출력 정보

- 실시간 디스플레이:
  - 왼쪽: 카메라 뷰 (바운딩 박스 포함)
  - 오른쪽: 2D 맵핑 뷰
- 콘솔 출력:
  - 1초마다 감지된 사람들의 3D 위치 정보
