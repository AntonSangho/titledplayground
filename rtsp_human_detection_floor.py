import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # RTSP URL 설정
    rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"  # 실제 RTSP URL로 변경 필요
    
    # YOLO 모델 로드 (기본 YOLOv8n 모델 사용)
    print("YOLO 모델 로딩 중...")
    model = YOLO('yolov8n.pt')
    
    # RTSP 스트림 연결
    print(f"RTSP 스트림 연결 중: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    # RTSP 스트림 설정
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 버퍼 크기 최소화
    
    if not cap.isOpened():
        print("RTSP 스트림을 열 수 없습니다.")
        return
    
    print("스트림 연결 성공")
    print(f"해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # 바닥 그리드 크기 (실제 단위: cm)
    grid_size_cm = 30
    width_cm = grid_size_cm * 4  # 120cm
    height_cm = grid_size_cm * 3  # 90cm
    
    # 클릭한 점 저장 리스트
    clicked_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"{len(clicked_points)}번 점 클릭: ({x}, {y})")
    
    cv2.namedWindow('RTSP Human Tracking')
    cv2.setMouseCallback('RTSP Human Tracking', mouse_callback)
    
    print("캘리브레이션 모드: 바닥의 4개 꼭지점을 클릭하세요")
    print("1번 점: 왼쪽 위, 2번 점: 오른쪽 위, 3번 점: 오른쪽 아래, 4번 점: 왼쪽 아래")
    
    # 캘리브레이션 단계
    homography_matrix = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 재연결 시도...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        display_image = frame.copy()
        
        # 클릭한 점 표시
        for i, point in enumerate(clicked_points):
            cv2.circle(display_image, point, 5, (0, 255, 0), -1)
            cv2.putText(display_image, str(i+1), (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(clicked_points) == 4:
            cv2.putText(display_image, "All 4 points clicked. Press 'c' to continue", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('RTSP Human Tracking', display_image)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('c') and len(clicked_points) == 4:
            floor_points_image = np.array(clicked_points, dtype=np.float32)
            floor_points_real = np.array([
                [0, 0],
                [width_cm, 0],
                [width_cm, height_cm],
                [0, height_cm]
            ], dtype=np.float32)
            
            homography_matrix = cv2.findHomography(floor_points_image, floor_points_real)[0]
            print("호모그래피 행렬 계산 완료")
            break
    
    # 메인 트래킹 루프
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 재연결 시도...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        frame_count += 1
        # 3프레임마다 처리 (성능 최적화)
        if frame_count % 3 != 0:
            continue
            
        # YOLO로 사람 감지 (class 0 = person)
        results = model(frame, classes=[0], conf=0.3)
        
        display_image = frame.copy()
        
        # 바닥 경계 그리기
        for i in range(4):
            pt1 = clicked_points[i]
            pt2 = clicked_points[(i + 1) % 4]
            cv2.line(display_image, pt1, pt2, (0, 255, 0), 2)
        
        # 그리드 그리기
        for x in range(0, int(width_cm) + 1, grid_size_cm):
            for y in range(0, int(height_cm) + 1, grid_size_cm):
                real_point = np.array([[x, y]], dtype=np.float32)
                img_point = cv2.perspectiveTransform(real_point.reshape(-1, 1, 2), 
                                                   np.linalg.inv(homography_matrix))
                px, py = img_point[0][0]
                cv2.circle(display_image, (int(px), int(py)), 3, (0, 0, 255), -1)
        
        # 감지된 사람 처리
        detected = False
        people_count = 0
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                detected = True
                people_count += 1
                
                # 바운딩 박스 그리기
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 사람의 발 위치 (바운딩 박스 하단 중앙)
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(display_image, (foot_x, foot_y), 5, (0, 0, 255), -1)
                
                # 발 위치의 실제 좌표 계산
                img_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                try:
                    real_point = cv2.perspectiveTransform(img_point, homography_matrix)
                    real_x, real_y = real_point[0][0]
                    
                    # 위치 정보 표시
                    label = f"Person {people_count}: {conf:.2f}, Pos: ({real_x:.1f}, {real_y:.1f})cm"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    label = f"Person {people_count}: {conf:.2f}"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 상태 표시
        status_text = f"Persons detected: {people_count}" if detected else "No person detected"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(display_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow('RTSP Human Tracking', display_image)
        
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 