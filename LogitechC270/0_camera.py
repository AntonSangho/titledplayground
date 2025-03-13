import cv2

def list_available_cameras():
    """시스템에 연결된 모든 카메라 장치를 확인합니다."""
    available_cameras = []
    
    # 일반적으로 시스템에서는 카메라 인덱스를 0부터 9까지 확인합니다
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 카메라 정보 가져오기
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"카메라 {i}: {width}x{height} 해상도")
            
            # 카메라에서 프레임 한 장 가져오기
            ret, frame = cap.read()
            if ret:
                # 이미지 표시 (확인용)
                window_name = f"Camera {i}"
                cv2.imshow(window_name, frame)
                cv2.waitKey(1000)  # 1초 동안 이미지 표시
                cv2.destroyWindow(window_name)
            
            available_cameras.append(i)
        cap.release()
    
    return available_cameras

if __name__ == "__main__":
    print("연결된 카메라 확인 중...")
    cameras = list_available_cameras()
    
    if not cameras:
        print("연결된 카메라가 없습니다.")
    else:
        print(f"사용 가능한 카메라: {cameras}")
        
        # 사용자가 카메라를 선택하도록 함
        camera_index = int(input("사용할 카메라 번호를 입력하세요: "))
        
        # 선택한 카메라로 테스트
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"카메라 {camera_index} 선택됨")
            print(f"해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            
            # 간단한 뷰어로 테스트
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow(f"Camera {camera_index}", frame)
                
                # ESC 키를 눌러 종료
                if cv2.waitKey(1) == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"카메라 {camera_index}를 열 수 없습니다.")
