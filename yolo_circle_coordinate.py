import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import json
import time
from datetime import datetime

class CoordinateLogger:
    def __init__(self):
        self.last_log_time = 0
        self.log_interval = 1.0  # 1초 간격
        self.log_file = "person_coordinates.json"
        # JSON 파일 초기화
        self.initialize_log_file()
        
    def initialize_log_file(self):
        # 파일이 없으면 빈 리스트로 초기화
        try:
            with open(self.log_file, 'r') as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
        
    def log_coordinates(self, coordinates):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "timestamp": timestamp,
                "persons": coordinates
            }
            
            # 기존 데이터 읽기
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                log_data = []
            
            # 새 데이터 추가
            log_data.append(data)
            
            # 파일에 저장
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            # 콘솔에도 출력
            print(json.dumps(data, indent=2))
            self.last_log_time = current_time

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
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    cv2.circle(frame, center, radius, (0, 0, 0), 2)
    return frame, center, radius

def main():
    model = YOLO('yolov8n.pt')
    pipeline, align = initialize_camera()
    coordinate_logger = CoordinateLogger()
    
    # 시각화 프레임 초기화
    viz_frame, viz_center, viz_radius = create_visualization_frame()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 시각화 프레임 초기화
            viz_display = viz_frame.copy()
            
            # 사람 감지 및 좌표 수집
            person_coordinates = []
            results = model(color_image, conf=0.5)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 클래스 확인
                    class_name = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    if class_name != 'person' or conf < 0.5:
                        continue
                        
                    # 좌표 계산
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    depth_value = depth_frame.get_distance(x_center, y_center)
                    
                    # 필터링
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if (box_width < 50 or box_height < 100 or 
                        box_width > 400 or box_height > 450 or
                        depth_value > 5.0 or depth_value < 0.5):
                        continue
                    
                    # 좌표 정규화 (0-1 범위)
                    normalized_x = x_center / 640
                    normalized_y = y_center / 480
                    
                    # 좌표 정보 저장
                    person_data = {
                        "id": len(person_coordinates) + 1,
                        "position": {
                            "x": normalized_x,
                            "y": normalized_y,
                            "depth": depth_value
                        },
                        "confidence": conf,
                        "pixel_position": {
                            "x": x_center,
                            "y": y_center
                        }
                    }
                    person_coordinates.append(person_data)
                    
                    # 원본 이미지에 바운딩 박스 표시
                    color = (0, int(255 * conf), 0)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    label = f'Person {person_data["id"]} ({x_center},{y_center}) {depth_value:.2f}m'
                    cv2.putText(color_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 시각화 프레임에 위치 표시
                    norm_x = (normalized_x * 2 - 1) * viz_radius
                    norm_y = (normalized_y * 2 - 1) * viz_radius
                    viz_x = int(viz_center[0] + norm_x)
                    viz_y = int(viz_center[1] + norm_y)
                    
                    # 점과 ID 표시
                    cv2.circle(viz_display, (viz_x, viz_y), 5, (0, 0, 255), -1)
                    cv2.putText(viz_display, str(person_data["id"]), 
                              (viz_x + 10, viz_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 좌표 로깅
            coordinate_logger.log_coordinates(person_coordinates)
            
            # 좌우 화면 합치기
            display = np.hstack((color_image, viz_display))
            
            # 화면 표시
            cv2.imshow('RealSense Tracking', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
