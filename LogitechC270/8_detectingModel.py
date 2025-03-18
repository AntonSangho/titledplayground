import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 카메라 인덱스를 2번으로 고정
    camera_index = 2
    
    # 학습된 모델 로드
    model_path = "runs/detect/titledplayground2_model4/weights/best.pt"
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 {camera_index}를 열 수 없습니다.")
        return
    
    print(f"카메라 {camera_index} 선택됨")
    print(f"해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # 객체 인식 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 감지 수행
        results = model(frame, conf=0.25)  # 신뢰도 임계값 0.25로 설정
        
        # 원본 프레임 복사
        display_frame = frame.copy()
        
        # 감지된 객체가 있는지 확인
        detected = False
        
        # 감지된 객체 시각화
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 클래스 ID와 신뢰도
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 객체 감지됨
                detected = True
                
                # 바운딩 박스 그리기
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 객체 이름과 신뢰도 표시
                label = f"part: {conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 객체 중심점 계산 및 표시
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 화면 상단에 상태 표시
        status_text = "detected!" if detected else "no detection"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 화면에 표시
        cv2.imshow("Object Detection", display_frame)
        
        # ESC 키를 눌러 종료
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()