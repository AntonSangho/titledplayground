import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# PersonTracker 클래스: 감지된 사람들의 위치를 추적하고 고유 ID를 부여하는 클래스
class PersonTracker:
    def __init__(self):
        self.persons = {}  # 사람들의 위치를 ID와 함께 저장하는 딕셔너리
        self.next_id = 1   # 다음에 부여할 ID 번호
        self.max_distance = 100  # 같은 사람으로 판단할 최대 거리 (픽셀)
        self.missing_threshold = 5  # 사람이 사라졌다고 판단할 프레임 수
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

def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    align_to = rs.align(rs.stream.color)
    
    return pipeline, align_to

def create_visualization_frame(width=640, height=480):
    # 흰색 배경에 검은색 원을 그려서 시각화 프레임 생성
    # 원의 크기는 화면 크기의 1/3로 설정
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    cv2.circle(frame, center, radius, (0, 0, 0), 2)
    return frame, center, radius

def main():
    # 프로그램의 메인 루프
    # 1. YOLO 모델과 카메라를 초기화
    # 2. 프레임별로 다음 작업을 수행:
    #    - 카메라에서 RGB 및 깊이 프레임을 가져옴
    #    - YOLO로 사람을 감지
    #    - 감지된 사람들의 위치를 추적
    #    - 원형 시각화 프레임에 위치를 표시
    #    - 결과 화면을 표시
    # 초기화
    model = YOLO('yolov8n.pt')
    pipeline, align = initialize_camera()
    tracker = PersonTracker()
    
    # 시각화 프레임 생성
    viz_frame, viz_center, viz_radius = create_visualization_frame()
    
    try:
        while True:
            # 카메라 프레임 받기
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 이미지 변환
            color_image = np.asanyarray(color_frame.get_data())
            
            # YOLO 감지
            results = model(color_image)
            current_positions = []
            
            # 시각화 프레임 초기화
            viz_display = viz_frame.copy()
            
            # 감지 결과 처리
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 사람 클래스만 처리
                    class_name = result.names[int(box.cls[0])]
                    if class_name != 'person':
                        continue
                        
                    # 박스 좌표 계산
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    
                    # 현재 위치 저장
                    current_positions.append((x_center, y_center))
                    
                    # 깊이 값 계산
                    depth_value = depth_frame.get_distance(x_center, y_center)
                    
                    # 원본 이미지에 바운딩 박스 그리기
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Person Depth: {depth_value:.2f}m'
                    cv2.putText(color_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 트래커 업데이트
            tracked_persons = tracker.update_persons(current_positions)
            
            # 시각화 프레임에 위치 표시
            for person_id, pos in tracked_persons.items():
                # 좌표를 시각화 프레임의 원 안으로 변환
                norm_x = (pos[0] / 640) * viz_radius * 2 - viz_radius
                norm_y = (pos[1] / 480) * viz_radius * 2 - viz_radius
                viz_x = int(viz_center[0] + norm_x)
                viz_y = int(viz_center[1] + norm_y)
                
                # 점과 ID 표시
                cv2.circle(viz_display, (viz_x, viz_y), 5, (0, 0, 255), -1)
                cv2.putText(viz_display, str(person_id), (viz_x + 10, viz_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 좌우 화면 합치기
            display = np.hstack((color_image, viz_display))
            
            # 화면 표시
            cv2.namedWindow('RealSense Tracking', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Tracking', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()