import cv2
import numpy as np
import socket
import json
import time
from ultralytics import YOLO

def main():
    # UDP 소켓 설정
    UDP_IP = "127.0.0.1"  # localhost
    UDP_PORT = 5065       # Unity에서 사용할 포트
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 카메라 열기 (USB 웹캠)
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
    
    cv2.namedWindow('Floor Tracking')
    cv2.setMouseCallback('Floor Tracking', mouse_callback)
    
    print("캘리브레이션 모드: 바닥의 4개 꼭지점을 클릭하세요")
    print("1번 점: 왼쪽 위, 2번 점: 오른쪽 위, 3번 점: 오른쪽 아래, 4번 점: 왼쪽 아래")
    
    # 캘리브레이션 단계
    homography_matrix = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_image = frame.copy()
        
        # 클릭한 점 표시
        for i, point in enumerate(clicked_points):
            cv2.circle(display_image, point, 5, (0, 255, 0), -1)
            cv2.putText(display_image, str(i+1), (point[0]+10, point[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(clicked_points) == 4:
            cv2.putText(display_image, "All 4 points clicked. Press 'c' to continue", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Floor Tracking', display_image)
        
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
    
    # 메인 객체 감지 및 데이터 전송 루프
    last_send_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 감지 수행
        results = model(frame, conf=0.25)  # 신뢰도 임계값 설정
        
        display_image = frame.copy()
        
        # 바닥 경계 그리기
        for i in range(4):
            pt1 = clicked_points[i]
            pt2 = clicked_points[(i + 1) % 4]
            cv2.line(display_image, pt1, pt2, (0, 255, 0), 2)
        
        # 그리드 그리기 (간소화를 위해 생략 가능)
        for x in range(0, int(width_cm) + 1, grid_size_cm):
            for y in range(0, int(height_cm) + 1, grid_size_cm):
                real_point = np.array([[x, y]], dtype=np.float32)
                img_point = cv2.perspectiveTransform(real_point.reshape(-1, 1, 2), 
                                                   np.linalg.inv(homography_matrix))
                px, py = img_point[0][0]
                cv2.circle(display_image, (int(px), int(py)), 3, (0, 0, 255), -1)
        
        # 감지된 객체 정보를 담을 리스트
        detected_objects = []
        
        # 현재 시간
        current_time = time.time()
        
        # 감지된 객체 처리
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 클래스 ID와 신뢰도
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 객체 중심점 계산
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # 객체의 실제 좌표(cm) 계산
                img_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                try:
                    real_point = cv2.perspectiveTransform(img_point, homography_matrix)
                    real_x, real_y = real_point[0][0]
                    
                    # 좌표 정규화 (0~1 사이 값으로)
                    norm_x = real_x / width_cm
                    norm_y = real_y / height_cm
                    
                    # 감지된 객체 정보 저장
                    detected_objects.append({
                        "id": cls_id,
                        "x": float(real_x),
                        "y": float(real_y),
                        "norm_x": float(norm_x),
                        "norm_y": float(norm_y),
                        "conf": float(conf)
                    })
                    
                    # 위치 정보 표시
                    label = f"ID:{cls_id} ({real_x:.1f}, {real_y:.1f})cm"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    # 변환 실패 시 이미지 좌표만 표시
                    label = f"ID:{cls_id} ({center_x}, {center_y})px"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # UDP를 통해 데이터 전송 (50ms마다 한 번)
        if current_time - last_send_time > 0.05 and detected_objects:
            # JSON 형식으로 변환
            data = json.dumps(detected_objects)
            sock.sendto(data.encode(), (UDP_IP, UDP_PORT))
            last_send_time = current_time
            
            # 콘솔에 전송 정보 출력
            print(f"데이터 전송: {len(detected_objects)}개 객체, {len(data)}바이트")
        
        # 상태 표시
        obj_count = len(detected_objects)
        status_text = f"Detected: {obj_count} objects"
        status_color = (0, 255, 0) if obj_count > 0 else (0, 0, 255)
        cv2.putText(display_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Unity 연결 상태 표시
        cv2.putText(display_image, f"Unity: {UDP_IP}:{UDP_PORT}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Floor Tracking', display_image)
        
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()