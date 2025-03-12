import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time

# 임포트 구문 아래, 함수 정의 전에 추가

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

# 메인 함수
def main():
    # 초기화
    pipeline, align, depth_scale = initialize_camera()
    model = initialize_model()
    tracker = PersonTracker() # PersonTracker 객체 초기화
    
    # 결과 시각화용 프레임 생성
    viz_width, viz_height = 640, 480
    circle_center = (viz_width // 2, viz_height // 2)
    circle_radius = min(viz_width, viz_height) // 3
    
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
                #cv2.convertScaleAbs(depth_image, alpha=0.1),  # 1m 환경에 맞게 조정
                cv2.convertScaleAbs(depth_image, alpha=0.03),  # 4m 환경에 맞게 조정
                cv2.COLORMAP_JET
            )
            
            # YOLO로 사람 감지
            #results = model(color_image, classes=[0], conf=0.5)  # 0: 사람 클래스만
            #results = model(color_image, classes=[0], conf=0.3)  # 더 낮은 신뢰도 
            results = model(color_image, classes=[0], conf=0.3, iou=0.45)  # IoU 임계값 조정 (추가옵션 ) 
            
            # 시각화 프레임 생성 (흰색 배경에 검은색 원)
            viz_frame = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
            cv2.circle(viz_frame, circle_center, circle_radius, (0, 0, 0), 2)
            
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
                if depth_value > 0 and depth_value < 5.0:  # 4m 환경에 맞게 조정
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
                    
                    # 바운딩 박스 그리기 (여기서는 중심점 기준으로 임의 크기 설정)
                    box_half_width = 50  # 임의의 상자 너비/2
                    box_half_height = 100  # 임의의 상자 높이/2
                    x1, y1 = center_x - box_half_width, center_y - box_half_height
                    x2, y2 = center_x + box_half_width, center_y + box_half_height
                    
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 깊이 정보 텍스트 추가
                    text = f"ID:{person_id} Depth: {depth_value:.2f}m"
                    cv2.putText(color_image, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 위치를 시각화 원 안에 표시
                    # x,y 좌표를 원 안의 좌표로 정규화
                    norm_x = (center_x / 640) * 2 - 1
                    norm_y = (center_y / 480) * 2 - 1

                    depth_factor = min(1.0, depth_value / 4.0)  # 깊이에 따른 가중치
                    viz_x = int(circle_center[0] + norm_x * circle_radius * depth_factor)
                    viz_y = int(circle_center[1] + norm_y * circle_radius * depth_factor)
                    
                    # 점 그리기
                    #cv2.circle(viz_frame, (viz_x, viz_y), 5, (0, 0, 255), -1)
                    #cv2.putText(viz_frame, str(person_id), (viz_x + 10, viz_y),
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 좌우 영상 합치기
            display = np.hstack((color_image, viz_frame))
            
            # 이미지 표시
            cv2.imshow('Person Detection', display)
            
            # 1초마다 감지된 사람들의 정보 출력 (Unity 전송 예시)
            current_time = time.time()
            if hasattr(main, 'last_log_time') and current_time - main.last_log_time >= 1.0:
                if people_info:
                    print("----- People Detected -----")
                    for person in people_info:
                        print(f"ID: {person['id']}, Position: X={person['3d_position'][0]:.2f}, Y={person['3d_position'][1]:.2f}, Z={person['3d_position'][2]:.2f}")
                main.last_log_time = current_time
            elif not hasattr(main, 'last_log_time'):
                main.last_log_time = current_time
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()