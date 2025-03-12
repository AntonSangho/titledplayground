import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time

class PersonTracker:
    def __init__(self):
        self.persons = {}  # 사람들의 위치를 ID와 함께 저장하는 딕셔너리
        self.next_id = 1   # 다음에 부여할 ID 번호
        self.max_distance = 100  # 같은 사람으로 판단할 최대 거리 (픽셀)
        self.missing_threshold = 10  # 사람이 사라졌다고 판단할 프레임 수 (5에서 10으로 증가)
        self.person_history = {}  # 각 ID별 미감지 프레임 카운트

    def update_persons(self, current_positions):
        new_persons = {}
        used_positions = set()  # 이미 매칭된 현재 위치들을 추적

        # 1. 기존 ID와 현재 위치들을 매칭
        for old_id, old_pos in self.persons.items():
            best_distance = self.max_distance
            best_match = None

            for pos in current_positions:
                if pos in used_positions:
                    continue
                
                distance = np.linalg.norm(np.array(pos) - np.array(old_pos))
                if distance < best_distance:
                    best_distance = distance
                    best_match = pos

            if best_match is not None:
                new_persons[old_id] = best_match
                used_positions.add(best_match)
                self.person_history[old_id] = 0  # 감지 카운트 리셋
            else:
                # 매칭되지 않은 경우 카운트 증가
                self.person_history[old_id] = self.person_history.get(old_id, 0) + 1
                if self.person_history[old_id] < self.missing_threshold:
                    new_persons[old_id] = old_pos  # 이전 위치 유지

        # 2. 매칭되지 않은 새로운 위치들에 새 ID 할당
        for pos in current_positions:
            if pos not in used_positions:
                new_persons[self.next_id] = pos
                self.person_history[self.next_id] = 0
                self.next_id += 1

        self.persons = new_persons
        return self.persons

# RealSense 카메라 초기화 (1단계와 동일)
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    align = rs.align(rs.stream.color)
    
    return pipeline, align, depth_scale

# YOLO 모델 초기화
def initialize_model():
    model = YOLO("yolov8n.pt")  # 경량 모델 사용
    return model

# 맵 초기화 함수 수정
def initialize_map(width, height):
    # 흰색 배경 생성
    map_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 맵의 중심
    center = (width // 2, height // 2)
    
    # 맵의 최대 반지름 (6m 원을 표현)
    max_radius = min(width, height) // 2 - 20  # 여백 20px
    
    # 외부 경계 (6m)
    cv2.circle(map_frame, center, max_radius, (0, 0, 0), 2)
    
    # 내부 동심원 (1m부터 5m까지)
    for i in range(1, 6):
        # 6m 지름에서 각 거리에 해당하는 비율 계산
        radius = int(max_radius * i / 6)
        cv2.circle(map_frame, center, radius, (200, 200, 200), 1)
    
    return map_frame, center, max_radius

# 메인 함수
def main():
    # 초기화
    pipeline, align, depth_scale = initialize_camera()
    model = initialize_model()
    tracker = PersonTracker()
    
    # 감지 매개변수 초기화
    conf_threshold = 0.3
    iou_threshold = 0.45
    
    print("\n=== 객체 감지 설정 ===")
    print("[ + / - ] : Confidence 임계값 조절")
    print("[ [ / ] ] : IoU 임계값 조절")
    print("현재 설정: Conf={:.2f}, IoU={:.2f}".format(conf_threshold, iou_threshold))
    
    # 결과 시각화용 프레임 생성
    viz_width, viz_height = 640, 480
    
    # 맵 초기화
    map_frame, map_center, map_radius = initialize_map(viz_width, viz_height)
    
    try:
        while True:
            # 프레임 대기
            frames = pipeline.wait_for_frames()
            
            # 깊이 프레임을 컬러 프레임에 정렬
            aligned_frames = align.process(frames)
            
            # 컬러 이미지와 깊이 이미지 가져오기
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            # NumPy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 깊이 이미지 시각화 (디버깅용)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),  # 4m 환경에 맞게 조정
                cv2.COLORMAP_JET
            )
            
            # YOLO로 사람 감지 (verbose=False 추가)
            results = model(color_image, classes=[0], 
                          conf=conf_threshold, 
                          iou=iou_threshold,
                          verbose=False)  # 디버그 출력 제거
            
            # 맵 프레임 복사 (매 프레임마다 새로 그리기)
            viz_frame = map_frame.copy()
            
            # 감지된 사람들의 정보 저장 리스트
            current_positions = []

            # 결과 처리 - 먼저 모든 중심점 수집
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 중심점
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    current_positions.append((center_x, center_y))

            # 트래커 업데이트
            tracked_persons = tracker.update_persons(current_positions)

            # 감지된 사람들의 정보 저장 리스트
            people_info = []

            # 트래킹된 사람들 정보 처리
            for person_id, pos in tracked_persons.items():
                center_x, center_y = pos
                
                # 깊이 정보 (미터 단위)
                depth_value = depth_frame.get_distance(center_x, center_y)
                
                # 유효한 깊이 값이 있는 경우만 처리
                if depth_value > 0 and depth_value < 6.0:  # 6m 환경에 맞게 조정
                    # 3D 공간상의 위치 계산
                    depth_point = rs.rs2_deproject_pixel_to_point(
                        depth_frame.profile.as_video_stream_profile().intrinsics,
                        [center_x, center_y],
                        depth_value
                    )
                    
                    # 사람 정보 저장
                    person_info = {
                        "id": person_id,  # YOLO의 임시 ID 대신 트래커의 고유 ID 사용
                        "center": (center_x, center_y),
                        "depth": depth_value,
                        "3d_position": depth_point
                    }
                    people_info.append(person_info)
                    
                    # 바운딩 박스 그리기
                    box_half_width = 50
                    box_half_height = 100
                    x1, y1 = center_x - box_half_width, center_y - box_half_height
                    x2, y2 = center_x + box_half_width, center_y + box_half_height
                    
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 깊이 정보 텍스트 추가
                    text = f"ID:{person_id} Depth: {depth_value:.2f}m"
                    cv2.putText(color_image, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 위치를 맵에 표시
                    # 실제 3D 위치 기반 (x, z 좌표 사용, y는 높이)
                    x_pos = depth_point[0]  # 좌우 위치
                    z_pos = depth_point[2]  # 깊이
                    
                    # 6m 원 안에 표시하기 위한 스케일링 (최대 반지름을 6m로 가정)
                    distance = np.sqrt(x_pos**2 + z_pos**2)
                    
                    if distance < 6.0:  # 6m 이내에 있는 경우만 표시
                        # x, z 좌표를 맵의 좌표로 변환
                        # 카메라 기준 좌표를 맵 기준 좌표로 변환 (z축이 앞으로, x축이 오른쪽)
                        map_x = int(map_center[0] - (x_pos / 6.0) * map_radius)
                        map_y = int(map_center[1] - (z_pos / 6.0) * map_radius)
                        
                        # 감지된 사람 표시
                        cv2.circle(viz_frame, (map_x, map_y), 7, (0, 0, 255), -1)
                        cv2.putText(viz_frame, str(person_id), (map_x + 10, map_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 좌우 영상 합치기
            display = np.hstack((color_image, viz_frame))
            
            # 이미지 표시
            cv2.imshow('Person Detection', display)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(1.0, conf_threshold + 0.05)
                print(f"Confidence 임계값: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.1, conf_threshold - 0.05)
                print(f"Confidence 임계값: {conf_threshold:.2f}")
            elif key == ord(']'):
                iou_threshold = min(1.0, iou_threshold + 0.05)
                print(f"IoU 임계값: {iou_threshold:.2f}")
            elif key == ord('['):
                iou_threshold = max(0.1, iou_threshold - 0.05)
                print(f"IoU 임계값: {iou_threshold:.2f}")
            
            # 현재 설정을 화면에 표시
            cv2.putText(display, f"Conf: {conf_threshold:.2f}, IoU: {iou_threshold:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()