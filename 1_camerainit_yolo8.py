import pyrealsense2 as rs
import numpy as np
import cv2
import json

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

# 메인 함수
def main():
    # 카메라 초기화
    pipeline, align, depth_scale = initialize_camera()
    
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
                #cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.convertScaleAbs(depth_image, alpha=0.1), 
                cv2.COLORMAP_JET
            )
            
            # 이미지 표시
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Depth Image', depth_colormap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()