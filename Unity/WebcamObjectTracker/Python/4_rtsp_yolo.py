import cv2
import numpy as np
import socket
import json
import time
from ultralytics import YOLO
import 

class ObjectTracker:
    def __init__(self):
        self.objects = {}            # 객체 추적 딕셔너리
        self.next_id = 1             # 다음에 부여할 ID 번호
        self.max_distance = 50       # 같은 객체로 판단할 최대 거리 (픽셀)
        self.missing_threshold = 10  # 객체가 사라졌다고 판단할 프레임 수
        self.object_history = {}     # 각 ID별 미감지 프레임 카운트

    def update_objects(self, current_positions):
        """현재 감지된 객체 위치를 기반으로 객체 추적 업데이트"""
        new_objects = {}
        used_positions = set()  # 이미 매칭된 현재 위치들을 추적

        # 1. 기존 ID와 현재 위치들을 매칭
        for old_id, old_pos in self.objects.items():
            best_distance = self.max_distance
            best_match = None

            for pos_info in current_positions:
                pos = (pos_info['center_x'], pos_info['center_y'])
                if pos in used_positions:
                    continue
                
                # 2D 위치 기반으로 거리 계산
                distance = np.sqrt((old_pos[0] - pos[0])**2 + (old_pos[1] - pos[1])**2)
                if distance < best_distance:
                    best_distance = distance
                    best_match = pos_info

            if best_match is not None:
                new_objects[old_id] = (best_match['center_x'], best_match['center_y'])
                best_match['track_id'] = old_id  # 매치된 객체에 ID 할당
                used_positions.add((best_match['center_x'], best_match['center_y']))
                self.object_history[old_id] = 0  # 감지 카운트 리셋
            else:
                # 매칭되지 않은 경우 카운트 증가
                self.object_history[old_id] = self.object_history.get(old_id, 0) + 1
                if self.object_history[old_id] < self.missing_threshold:
                    new_objects[old_id] = old_pos  # 이전 위치 유지

        # 2. 매칭되지 않은 새로운 위치들에 새 ID 할당
        for pos_info in current_positions:
            pos = (pos_info['center_x'], pos_info['center_y'])
            if pos not in used_positions:
                new_objects[self.next_id] = pos
                pos_info['track_id'] = self.next_id
                self.object_history[self.next_id] = 0
                self.next_id += 1

        self.objects = new_objects
        return current_positions  # 트래킹 ID가 추가된 객체 정보 반환

def main():
    # UDP 소켓 설정
    UDP_IP = "127.0.0.1"  # localhost
    UDP_PORT = 5065       # Unity에서 사용할 포트
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 객체 트래커 초기화
    tracker = ObjectTracker()
    
    # RTSP URL 설정
    rtsp_url = "rtsp://admin:RELIQUUM0925@192.168.29.113:554/Preview_01_sub"
    
    # YOLOv8 모델 로드 (기본 모델 사용)
    print("YOLOv8 모델 로드 중...")
    #model.to(mps_device)
    model = YOLO("yolov8n.pt")  # 가벼운 YOLOv8 nano 모델 사용
    
    # 모델의 클래스 정보 출력
    if hasattr(model, 'names'):
        print(f"모델 클래스 정보: {model.names}")
        print(f"감지 가능한 클래스 수: {len(model.names)}")
    
    # RTSP 스트림 열기
    print(f"RTSP 스트림 연결 중: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    # RTSP 연결 옵션 설정 (버퍼 크기 조정)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
    
    if not cap.isOpened():
        print(f"RTSP 스트림을 열 수 없습니다: {rtsp_url}")
        return
    
    print("RTSP 스트림 연결 성공")
    print(f"해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # 바닥 그리드 크기 (실제 단위: cm)
    grid_size_cm = 3
    width_cm = grid_size_cm * 5  # 15cm
    height_cm = grid_size_cm * 5  # 15cm
    
    # 클릭한 점 저장 리스트
    clicked_points = []
    
    # 바닥 영역 내 포함 여부 확인 함수
    def is_point_inside_floor(x, y, width, height):
        """실제 좌표(cm)가 바닥 영역 내에 있는지 확인"""
        return 0 <= x <= width and 0 <= y <= height
    
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
            print("프레임을 읽을 수 없습니다. 다시 시도합니다.")
            # RTSP 연결 재시도
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
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
    
    # 마우스 실시간 좌표 표시를 위한 변수
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
    
    # 메인 객체 감지 및 데이터 전송 루프
    last_send_time = time.time()
    last_frame_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 다시 시도합니다.")
            # RTSP 연결 재시도
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        # FPS 계산
        current_time = time.time()
        frame_count += 1
        if current_time - last_frame_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_frame_time = current_time
        
        # 객체 감지 수행
        results = model(frame, conf=0.25)  # 기본 신뢰도 임계값 설정
        
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
                
                # 중요 지점에만 좌표 텍스트 표시
                if x % (grid_size_cm * 2) == 0 and y % (grid_size_cm * 2) == 0:
                    cv2.putText(display_image, f"({x}, {y})", (int(px) + 5, int(py)), 
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
        
        # 현재 감지된 객체 정보를 담을 리스트
        current_objects = []
        
        # 현재 시간
        current_time = time.time()
        
        # 바닥 영역 내 감지된 객체 카운트
        objects_in_floor = 0
        
        # 감지된 객체 정보 수집 - YOLOv8 결과 형식에 맞게 처리
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 클래스 ID와 신뢰도
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 사람 클래스(0)만 필터링 (선택적)
                if cls_id == 0:  # COCO 데이터셋에서 사람은 클래스 ID 0
                    # 객체 중심점 계산
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 객체의 실제 좌표(cm) 계산
                    img_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                    try:
                        real_point = cv2.perspectiveTransform(img_point, homography_matrix)
                        real_x, real_y = real_point[0][0]
                        
                        # 바닥 영역 내에 있는지 확인
                        if is_point_inside_floor(real_x, real_y, width_cm, height_cm):
                            # 좌표 정규화 (0~1 사이 값으로)
                            norm_x = real_x / width_cm
                            norm_y = real_y / height_cm
                            
                            # 객체 정보 저장 (트래킹 ID는 나중에 할당)
                            obj_info = {
                                'cls_id': cls_id,
                                'center_x': center_x,
                                'center_y': center_y,
                                'box': (x1, y1, x2, y2),
                                'real_x': real_x,
                                'real_y': real_y,
                                'norm_x': norm_x,
                                'norm_y': norm_y,
                                'conf': conf,
                                'track_id': -1  # 초기값, 트래커에서 할당될 예정
                            }
                            current_objects.append(obj_info)
                    except:
                        pass
        
        # 객체 트래킹 업데이트 (고유 ID 할당)
        if current_objects:
            tracked_objects = tracker.update_objects(current_objects)
            
            # 감지된 객체 정보를 담을 리스트 (Unity로 전송용)
            detected_objects = []
            
            # 트래킹된 객체 시각화 및 데이터 준비
            for obj_info in tracked_objects:
                if 'track_id' in obj_info and obj_info['track_id'] > 0:
                    objects_in_floor += 1
                    
                    # 객체 정보 추출
                    track_id = obj_info['track_id']
                    cls_id = obj_info['cls_id']
                    x1, y1, x2, y2 = obj_info['box']
                    center_x, center_y = obj_info['center_x'], obj_info['center_y']
                    real_x, real_y = obj_info['real_x'], obj_info['real_y']
                    norm_x, norm_y = obj_info['norm_x'], obj_info['norm_y']
                    conf = obj_info['conf']
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 객체 중심점 표시
                    cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # 클래스 이름 가져오기
                    class_name = model.names[cls_id] if hasattr(model, 'names') else f"class:{cls_id}"
                    
                    # 위치 정보 표시 (트래킹 ID 포함)
                    label = f"ID:{track_id} {class_name}: {conf:.2f}, ({real_x:.1f}, {real_y:.1f})cm"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Unity로 전송할 객체 정보 저장 (id 필드 사용)
                    detected_objects.append({
                        "id": int(track_id),        # 트래킹 ID를 Unity의 id 필드에 할당
                        "cls_id": int(cls_id),      # 클래스 ID (부가 정보)
                        "x": float(real_x),         # 실제 X 좌표 (cm)
                        "y": float(real_y),         # 실제 Y 좌표 (cm)
                        "norm_x": float(norm_x),    # 정규화된 X (0~1)
                        "norm_y": float(norm_y),    # 정규화된 Y (0~1)
                        "conf": float(conf)         # 신뢰도
                    })
            
            # UDP를 통해 데이터 전송 (50ms마다 한 번)
            if current_time - last_send_time > 0.05:
                detected_objects = [] if not detected_objects else detected_objects
                # JSON 형식으로 변환
                data = json.dumps(detected_objects)
                sock.sendto(data.encode(), (UDP_IP, UDP_PORT))
                last_send_time = current_time
                
                # 콘솔에 전송 정보 출력
                print(f"데이터 전송: 대상={UDP_IP}, 크기={len(data)}바이트")
                print(f"전송된 객체 수: {len(detected_objects)}")
        
        # 상태 표시
        status_text = f"Detected in floor: {objects_in_floor} objects | FPS: {fps}"
        status_color = (0, 255, 0) if objects_in_floor > 0 else (0, 0, 255)
        cv2.putText(display_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 마우스 위치 좌표 표시
        cv2.putText(display_image, f"Mouse: ({real_coords[0]:.1f}, {real_coords[1]:.1f})cm", 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Unity 연결 상태 표시
        cv2.putText(display_image, f"Unity: {UDP_IP}:{UDP_PORT}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Floor Tracking', display_image)
        
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()
