import cv2
import numpy as np

def main():
    # 카메라 열기
    cap = cv2.VideoCapture(2)  # 2은 Logitech C270
    
    # 카메라 설정 확인
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("카메라 해상도:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    try:
        while True:
            # 프레임 가져오기
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 화면에 표시
            cv2.imshow('Camera Feed', frame)
            
            # ESC 키를 누르면 종료
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        # 정리
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
