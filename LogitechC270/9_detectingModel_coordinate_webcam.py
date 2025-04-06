import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 카메라 열기 (필요한 경우 인덱스 조정)
    camera_index = 4   # 기본 카메라 인덱스
    
    
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
    grid_size_cm = 30  # 30cm 간격의 그리드
    width_cm = grid_size_cm * 4  # 바닥 가로 크기 (예: 120cm)
    height_cm = grid_size_cm * 3  # 바닥 세로 크기 (예: 90cm)
    
    # 클릭한 점 저장 리스트
    clicked_points = []
    
    # 마우스 클릭 이벤트 콜백 함수
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"{len(clicked_points)}번 점 클릭: ({x}, {y})")
    
    # 창 생성 및 마우스 콜백 설정
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
        
        # 4개 점이 모두 클릭되면 안내 메시지 표시
        if len(clicked_points) == 4:
            cv2.putText(display_image, "All 4 points clicked. Press 'c' to continue", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Floor Tracking', display_image)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('c') and len(clicked_points) == 4:
            # 이미지 상의 바닥 4개 점
            floor_points_image = np.array(clicked_points, dtype=np.float32)
            
            # 실제 바닥 좌표계에서의 4개 점 (단위: cm)
            floor_points_real = np.array([
                [0, 0],            # 왼쪽 위
                [width_cm, 0],     # 오른쪽 위
                [width_cm, height_cm], # 오른쪽 아래
                [0, height_cm]     # 왼쪽 아래
            ], dtype=np.float32)
            
            # 호모그래피 계산
            homography_matrix = cv2.findHomography(floor_points_image, floor_points_real)[0]
            print("호모그래피 행렬 계산 완료")
            break
    
    # 마우스 위치에 따른 실제 좌표 표시 함수 - 클로저 활용
    real_coords = [0, 0]
    
    def coord_mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # 이미지 좌표를 실제 좌표(cm)로 변환
            img_point = np.array([[[x, y]]], dtype=np.float32)
            try:
                real_point = cv2.perspectiveTransform(img_point, homography_matrix)
                real_coords[0], real_coords[1] = real_point[0][0]
            except:
                pass
    
    cv2.setMouseCallback('Floor Tracking', coord_mouse_callback)
    
    # 바닥 그리드 표시 및 객체 감지 단계
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 감지 수행
        results = model(frame, conf=0.25)  # 신뢰도 임계값 0.25로 설정
        
        display_image = frame.copy()
        
        # 원본 바닥 경계 그리기
        for i in range(4):
            pt1 = clicked_points[i]
            pt2 = clicked_points[(i + 1) % 4]
            cv2.line(display_image, pt1, pt2, (0, 255, 0), 2)
        
        # 그리드 그리기
        for x in range(0, int(width_cm) + 1, grid_size_cm):
            for y in range(0, int(height_cm) + 1, grid_size_cm):
                # 실제 좌표(cm)를 이미지 좌표로 변환
                real_point = np.array([[x, y]], dtype=np.float32)
                img_point = cv2.perspectiveTransform(real_point.reshape(-1, 1, 2), 
                                                 np.linalg.inv(homography_matrix))
                px, py = img_point[0][0]
                
                # 그리드 점 그리기
                cv2.circle(display_image, (int(px), int(py)), 3, (0, 0, 255), -1)
                
                # 좌표 텍스트 표시 (중요 지점만)
                if x % (grid_size_cm * 2) == 0 and y % (grid_size_cm * 2) == 0:
                    cv2.putText(display_image, f"({x}cm, {y}cm)", (int(px) + 5, int(py)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 그리드 선 그리기
        for x in range(0, int(width_cm) + 1, grid_size_cm):
            p1 = np.array([[x, 0]], dtype=np.float32)
            p2 = np.array([[x, height_cm]], dtype=np.float32)
            p1 = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), np.linalg.inv(homography_matrix))
            p2 = cv2.perspectiveTransform(p2.reshape(-1, 1, 2), np.linalg.inv(homography_matrix))
            cv2.line(display_image, (int(p1[0][0][0]), int(p1[0][0][1])), 
                   (int(p2[0][0][0]), int(p2[0][0][1])), (0, 255, 255), 1)
        
        for y in range(0, int(height_cm) + 1, grid_size_cm):
            p1 = np.array([[0, y]], dtype=np.float32)
            p2 = np.array([[width_cm, y]], dtype=np.float32)
            p1 = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), np.linalg.inv(homography_matrix))
            p2 = cv2.perspectiveTransform(p2.reshape(-1, 1, 2), np.linalg.inv(homography_matrix))
            cv2.line(display_image, (int(p1[0][0][0]), int(p1[0][0][1])), 
                   (int(p2[0][0][0]), int(p2[0][0][1])), (0, 255, 255), 1)
        
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
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 객체 중심점 계산 및 표시
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # 객체의 이미지 좌표를 실제 좌표(cm)로 변환
                img_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                try:
                    real_point = cv2.perspectiveTransform(img_point, homography_matrix)
                    real_x, real_y = real_point[0][0]
                    
                    # 객체 이름, 신뢰도 및 실제 좌표 표시
                    label = f"part: {conf:.2f}, Pos: ({real_x:.1f}, {real_y:.1f})cm"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    # 변환 실패 시 이미지 좌표만 표시
                    label = f"part: {conf:.2f}, Pos: ({center_x}, {center_y})px"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 화면 상단에 상태 표시
        status_text = "detected!" if detected else "no detection"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(display_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 마우스 위치의 실제 좌표 표시
        cv2.putText(display_image, f"Mouse coordinate: ({real_coords[0]:.1f}cm, {real_coords[1]:.1f}cm)", 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 화면에 표시
        cv2.imshow('Floor Tracking', display_image)
        
        # ESC 키를 눌러 종료
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
