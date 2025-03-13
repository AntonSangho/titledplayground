import cv2
import os
import time
from datetime import datetime

# 이미지 저장 디렉토리 생성
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"디렉토리 생성됨: {save_dir}")

# 웹캠 설정 (2번 카메라 사용)
camera_index = 2  # 연결된 카메라가 2번 인덱스
cap = cv2.VideoCapture(camera_index)

# 카메라가 제대로 열렸는지 확인
if not cap.isOpened():
    print(f"오류: {camera_index}번 카메라를 열 수 없습니다.")
    print("다른 카메라 인덱스를 시도해보세요 (0, 1 등).")
    exit()

# 카메라 해상도 설정 (선택 사항)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 촬영 설정
num_images = 50    # 촬영할 이미지 수
delay = 2          # 이미지 사이의 지연 시간(초)
countdown = 3      # 시작 전 카운트다운(초)

print(f"카메라 준비 완료. {num_images}장의 이미지를 {delay}초 간격으로 촬영합니다.")
print(f"촬영 시작까지 {countdown}초 카운트다운...")

# 촬영 시작 전 카운트다운
for i in range(countdown, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("촬영 시작!")

image_count = 0
last_capture_time = time.time() - delay  # 첫 이미지 바로 촬영하기 위해

try:
    while image_count < num_images:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # 현재 시간 확인
        current_time = time.time()
        time_remaining = delay - (current_time - last_capture_time)
        
        # 화면에 정보 표시
        info_text = f"capture: {image_count}/{num_images}"
        if time_remaining > 0:
            info_text += f" | next shot: {time_remaining:.1f}sec"
        
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 미리보기 표시
        cv2.imshow('Camera Preview', frame)
        
        # 일정 간격으로 이미지 저장
        if current_time - last_capture_time >= delay:
            # 파일명 생성 (현재 날짜 및 시간 사용)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"image_{timestamp}_{image_count+1:03d}.jpg")
            
            # 이미지 저장
            cv2.imwrite(filename, frame)
            print(f"이미지 저장됨 ({image_count+1}/{num_images}): {filename}")
            
            # 카운터 및 시간 업데이트
            image_count += 1
            last_capture_time = current_time
            
            # 화면에 저장 표시
            saved_frame = frame.copy()
            cv2.putText(saved_frame, "saved!", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Camera Preview', saved_frame)
            cv2.waitKey(300)  # 저장 메시지 잠시 표시
        
        # ESC 키를 누르면 종료
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("사용자에 의해 중단됨")
            break
        
        # 스페이스바를 누르면 즉시 촬영
        if key == 32:  # 스페이스바
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"image_{timestamp}_{image_count+1:03d}.jpg")
            
            # 이미지 저장
            cv2.imwrite(filename, frame)
            print(f"이미지 수동 저장됨 ({image_count+1}/{num_images}): {filename}")
            
            # 카운터 및 시간 업데이트
            image_count += 1
            last_capture_time = current_time
            
            # 화면에 저장 표시
            saved_frame = frame.copy()
            cv2.putText(saved_frame, "saved!", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Camera Preview', saved_frame)
            cv2.waitKey(300)  # 저장 메시지 잠시 표시

except KeyboardInterrupt:
    print("프로그램이 중단되었습니다.")

finally:
    # 촬영 완료 메시지
    if image_count == num_images:
        print(f"모든 이미지({num_images}장) 촬영 완료!")
    else:
        print(f"총 {image_count}장의 이미지가 저장되었습니다.")
    
    print(f"이미지는 '{os.path.abspath(save_dir)}' 폴더에 저장되었습니다.")
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
