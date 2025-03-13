import cv2
import numpy as np

def main():
    # 카메라 열기
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    # 클릭한 점 저장 리스트
    clicked_points = []
    
    # 캘리브레이션 모드 플래그
    calibration_mode = True
    
    # 마우스 클릭 이벤트 콜백 함수
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and calibration_mode and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"{len(clicked_points)}번 점 클릭: ({x}, {y})")
    
    # 창 생성 및 마우스 콜백 설정
    cv2.namedWindow('Floor Tracking')
    cv2.setMouseCallback('Floor Tracking', mouse_callback)
    
    print("캘리브레이션 모드: 바닥의 4개 꼭지점을 클릭하세요")
    print("1번 점: 왼쪽 위, 2번 점: 오른쪽 위, 3번 점: 오른쪽 아래, 4번 점: 왼쪽 아래")
    print("4개 점을 모두 클릭한 후 'c' 키를 눌러 캘리브레이션을 완료하세요")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 작업용 이미지 복사
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
            
            # 화면에 표시
            cv2.imshow('Floor Tracking', display_image)
            
            # 키 입력 처리
            key = cv2.waitKey(1)
            if key == 27:  # ESC 키
                break
            elif key == ord('c') and len(clicked_points) == 4:
                calibration_mode = False
                print("캘리브레이션 준비 완료!")
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
