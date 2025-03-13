import cv2
import numpy as np

def main():
    # 카메라 열기
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
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
    
    # 마우스 위치에 따른 실제 좌표 표시 함수
    def coord_mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # 이미지 좌표를 실제 좌표(cm)로 변환
            img_point = np.array([[[x, y]]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(img_point, homography_matrix)
            real_x, real_y = real_point[0][0]
            # 전역 변수에 저장 (디스플레이에 표시하기 위함)
            param['real_coords'] = (real_x, real_y)
    
    # 좌표 변환 단계
    cv2.setMouseCallback('Floor Tracking', coord_mouse_callback, {'real_coords': (0, 0)})
    
    # 변환된 좌표 표시
    real_coords = (0, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_image = frame.copy()
        
        # 원본 바닥 경계 그리기
        for i in range(4):
            pt1 = clicked_points[i]
            pt2 = clicked_points[(i + 1) % 4]
            cv2.line(display_image, pt1, pt2, (0, 255, 0), 2)
        
        # 마우스 위치의 실제 좌표 표시
        cv2.putText(display_image, f"Coordinate: ({real_coords[0]:.1f}cm, {real_coords[1]:.1f}cm)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Floor Tracking', display_image)
        
        if cv2.waitKey(1) == 27:  # ESC 키
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
