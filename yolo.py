import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def initialize_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    # Create an align object
    align_to = rs.align(rs.stream.color)
    
    return pipeline, align_to

def main():
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # load a pretrained model
    
    # Initialize Camera
    pipeline, align = initialize_camera()
    
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (주석 처리)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Run YOLO detection
            # conf=0.5: 신뢰도가 50% 이상인 객체만 감지
            results = model(color_image, conf=0.5)
            
            # 감지된 모든 객체에 대해 처리
            for result in results:
                boxes = result.boxes  # 감지된 객체들의 바운딩 박스 정보
                for box in boxes:
                    # 감지된 객체의 클래스(종류)와 신뢰도 확인
                    class_name = result.names[int(box.cls[0])]  # 객체의 클래스 이름
                    conf = float(box.conf[0])  # 감지 신뢰도 (0~1 사이 값)
                    
                    # 사람이 아니거나 신뢰도가 낮은 경우 건너뛰기
                    if class_name != 'person' or conf < 0.5:
                        continue
                    
                    # 바운딩 박스의 좌표 추출 (좌상단 x1,y1, 우하단 x2,y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 바운딩 박스 크기 검사
                    box_width = x2 - x1   # 박스 너비
                    box_height = y2 - y1  # 박스 높이
                    
                    # 너무 작거나 큰 객체는 필터링
                    if box_width < 50 or box_height < 100:  # 최소 크기 제한
                        continue
                    if box_width > 400 or box_height > 450:  # 최대 크기 제한
                        continue
                    
                    # 바운딩 박스의 중심점 계산
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    
                    # 깊이 카메라에서 중심점까지의 거리 측정
                    depth_value = depth_frame.get_distance(x_center, y_center)
                    
                    # 거리 기반 필터링 (너무 멀거나 가까운 객체 제외)
                    if depth_value > 5.0 or depth_value < 0.5:  # 단위: 미터
                        continue
                    
                    # 화면에 바운딩 박스 표시
                    # 신뢰도에 따라 색상 강도 조절 (신뢰도가 높을수록 진한 초록색)
                    color = (0, int(255 * conf), 0)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 객체 정보 텍스트 표시 (신뢰도와 거리 정보)
                    label = f'Person {conf:.2f} Depth: {depth_value:.2f}m'
                    cv2.putText(color_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Stack both images horizontally (이 부분도 수정)
            # images = np.hstack((color_image, depth_colormap))
            
            # Show images (depth map 없이 color_image만 표시)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)  # depth_colormap 제거
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()