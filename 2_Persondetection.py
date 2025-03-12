import pyrealsense2 as rs
import numpy as np
import cv2
import json
from ultralytics import YOLO

# RealSense 카메라 초기화
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 컬러와 깊이 스트림 활성화
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 카메라 시작
    profile = pipeline.start(config)
    
    # 깊이 센서 가져오기
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # HighDensityPreset.json 설정 로드 및 적용
    try:
        with open('HighDensityPreset.json', 'r') as f:
            preset = json.load(f)
            
        # 깊이 센서에 프리셋 설정 적용
        for key in preset:
            if depth_sensor.supports(key):
                depth_sensor.set_option(key, float(preset[key]))
    except Exception as e:
        print(f"프리셋 로드 중 오류 발생: {e}")
    
    # 컬러 프레임과 깊이 프레임 정렬 객체
    align = rs.align(rs.stream.color)
    
    return pipeline, align, depth_scale

# YOLO 모델 초기화
def initialize_model():
    model = YOLO("yolov8n.pt")  # 경량 모델 사용
    return model

def main():
    # 초기화
    pipeline, align, depth_scale = initialize_camera()
    model = initialize_model()
    
    # 감지 매개변수 초기화
    conf_threshold = 0.3
    iou_threshold = 0.45
    
    print("\n=== 객체 감지 설정 ===")
    print("[ + / - ] : Confidence 임계값 조절")
    print("[ [ / ] ] : IoU 임계값 조절")
    print("현재 설정: Conf={:.2f}, IoU={:.2f}".format(conf_threshold, iou_threshold))
    
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
            
            # YOLO로 사람 감지
            results = model(color_image, classes=[0], 
                          conf=conf_threshold, 
                          iou=iou_threshold,
                          verbose=False)
            
            # 결과 처리 - 바운딩 박스 그리기
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 바운딩 박스 그리기
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 이미지 표시
            cv2.imshow('Person Detection', color_image)
            
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
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 