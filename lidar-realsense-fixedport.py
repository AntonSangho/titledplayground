import os
import ydlidar
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
from ultralytics import YOLO

class MultiSensorObjectDetector:
    def __init__(self, lidar_port="/dev/ttyUSB0", min_points=5, min_distance=0.5, max_distance=3.0):
        # 초기화 파라미터
        self.lidar_port = lidar_port  # 고정 포트 사용
        self.min_points = min_points
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.cluster_threshold = 0.2
        
        # 시각화 관련 설정
        self.frame_size = (800, 800)
        self.center = (self.frame_size[0] // 2, self.frame_size[1] // 2)
        self.scale_factor = 100  # 1m = 100px
        
        # YOLO 모델 초기화
        self.yolo_model = YOLO("yolov8n.pt")
        
        # LiDAR 초기화
        self.init_lidar()
        
        # RealSense 초기화
        self.init_realsense()
        
    def init_lidar(self):
        """LiDAR 초기화"""
        ydlidar.os_init()
        
        # 포트 확인만 출력하고 무시 (항상 고정 포트 사용)
        ports = ydlidar.lidarPortList()
        if ports:
            port_detected = next(iter(ports.values()))
            print(f"감지된 라이다 포트: {port_detected} (무시됨)")
        
        print(f"사용할 라이다 포트: {self.lidar_port}")
        
        self.laser = ydlidar.CYdLidar()
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, self.lidar_port)  # 고정 포트 사용
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
        self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        self.laser.setlidaropt(ydlidar.LidarPropMaxRange, 16.0)
        self.laser.setlidaropt(ydlidar.LidarPropMinRange, 0.08)
        self.laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
    
    def init_realsense(self):
        """RealSense 초기화"""
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.rs_depth_sensor = self.rs_profile.get_device().first_depth_sensor()
        self.rs_depth_scale = self.rs_depth_sensor.get_depth_scale()
        self.rs_align = rs.align(rs.stream.color)
        
    def connect(self):
        """센서 연결 시작"""
        print(f"YDLidar 연결 시도 중... (포트: {self.lidar_port})")
        lidar_connected = self.laser.initialize()
        if lidar_connected:
            print("YDLidar 연결 성공!")
            lidar_started = self.laser.turnOn()
            if lidar_started:
                print("YDLidar 스캔 시작...")
            else:
                print("YDLidar 스캔 시작 실패")
                lidar_connected = False
        else:
            print("YDLidar 연결 실패")
            print("다음을 확인하세요:")
            print("1. 라이다가 물리적으로 연결되어 있는지")
            print("2. 포트 권한: sudo chmod 666 " + self.lidar_port)
            print("3. 다른 포트 시도 (예: /dev/ttyUSB1, /dev/ttyACM0)")
        
        print("모든 센서 연결 상태:", "성공" if lidar_connected else "실패")
        return lidar_connected
    
    def disconnect(self):
        """센서 연결 종료"""
        print("센서 연결 해제 중...")
        
        # LiDAR 연결 해제
        self.laser.turnOff()
        self.laser.disconnecting()
        
        # RealSense 연결 해제
        self.rs_pipeline.stop()
        
        print("모든 센서 연결 해제 완료")
    
    def process_lidar_scan(self):
        """LiDAR 스캔 처리"""
        lidar_objects = []
        
        scan = ydlidar.LaserScan()
        ret = self.laser.doProcessSimple(scan)
        
        if ret and scan.config.scan_time != 0:
            # 스캔 데이터에서 포인트 추출
            points = []
            for i, point in enumerate(scan.points):
                distance = point.range  # 미터 단위
                angle = math.degrees(point.angle)  # 각도로 변환
                
                # 유효한 거리 범위 내 포인트만 저장
                if self.min_distance <= distance <= self.max_distance:
                    points.append((distance, angle))
            
            # 포인트 클러스터링하여 객체 감지
            clusters = self.cluster_lidar_points(points)
            
            # 클러스터에서 객체 정보 추출
            for i, cluster in enumerate(clusters):
                # 객체의 중심 계산
                distances = [p[0] for p in cluster]
                angles = [p[1] for p in cluster]
                avg_distance = sum(distances) / len(distances)
                avg_angle = sum(angles) / len(angles)
                
                # 직교 좌표계로 변환 (라이다 위치가 원점)
                x = avg_distance * math.cos(math.radians(avg_angle))
                y = avg_distance * math.sin(math.radians(avg_angle))
                
                lidar_objects.append({
                    'id': i+1,
                    'distance': avg_distance,
                    'angle': avg_angle,
                    'position': (x, y),
                    'type': 'unknown',
                    'source': 'lidar',
                    'size': len(cluster)
                })
        
        return lidar_objects
    
    def cluster_lidar_points(self, points):
        """LiDAR 포인트 클러스터링"""
        clusters = []
        current_cluster = []
        
        # 각도 기준 정렬
        sorted_points = sorted(points, key=lambda p: p[1])
        
        for i, point in enumerate(sorted_points):
            if not current_cluster:
                current_cluster.append(point)
            else:
                prev_point = current_cluster[-1]
                # 각도 차이와 거리 차이로 동일 객체 여부 판단
                angle_diff = abs(point[1] - prev_point[1])
                if angle_diff > 180:  # 각도 경계 처리
                    angle_diff = 360 - angle_diff
                
                # 극좌표계 거리 계산
                dist = math.sqrt(point[0]**2 + prev_point[0]**2 - 
                                2 * point[0] * prev_point[0] * math.cos(math.radians(angle_diff)))
                
                if dist < self.cluster_threshold:
                    current_cluster.append(point)
                else:
                    if len(current_cluster) >= self.min_points:
                        clusters.append(current_cluster)
                    current_cluster = [point]
        
        # 마지막 클러스터 처리
        if len(current_cluster) >= self.min_points:
            clusters.append(current_cluster)
        
        return clusters
    
    def process_realsense(self):
        """RealSense 처리 및 객체 감지"""
        rs_objects = []
        
        try:
            # 프레임 가져오기
            frames = self.rs_pipeline.wait_for_frames(1000)  # 1초 타임아웃
            aligned_frames = self.rs_align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return [], None, None
            
            # 이미지로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # YOLO 객체 감지
            results = self.yolo_model(color_image, classes=[0], conf=0.5)  # 사람만 검출
            
            # 이미지 내 객체 처리
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # 바운딩 박스
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 중심점
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 깊이 정보
                    depth_value = depth_frame.get_distance(center_x, center_y)
                    
                    # 유효한 깊이 값이 있는 경우만 처리
                    if depth_value > 0 and depth_value < self.max_distance:
                        # 3D 좌표 계산
                        depth_point = rs.rs2_deproject_pixel_to_point(
                            depth_frame.profile.as_video_stream_profile().intrinsics,
                            [center_x, center_y],
                            depth_value
                        )
                        
                        # D435i는 라이다 기준 위에 있으므로, Y 좌표에 조정이 필요함
                        # X 좌표는 뒤집어야 함 (카메라와 라이다의 좌표계 일치시키기)
                        rs_x = -depth_point[0]  # 좌우 반전 (라이다 좌표계에 맞추기)
                        rs_y = depth_point[2]  # Z축을 Y축으로 사용 (Top-view)
                        
                        rs_objects.append({
                            'id': i+100,  # 라이다와 구분하기 위해 ID 시작 값을 다르게 설정
                            'distance': depth_value,
                            'position': (rs_x, rs_y),
                            'type': 'person',
                            'source': 'realsense',
                            'confidence': float(box.conf[0]),
                            'bbox': (x1, y1, x2, y2)
                        })
            
            return rs_objects, color_image, depth_image
        except Exception as e:
            print(f"RealSense 처리 중 오류 발생: {e}")
            return [], None, None
    
    def merge_detections(self, lidar_objects, rs_objects):
        """라이다와 RealSense 감지 결과 병합"""
        merged_objects = []
        used_rs_indices = set()
        
        # 각 라이다 객체에 대해 가장 가까운 RealSense 객체 찾기
        for lidar_obj in lidar_objects:
            lidar_pos = lidar_obj['position']
            min_dist = float('inf')
            matching_rs_idx = -1
            
            for i, rs_obj in enumerate(rs_objects):
                if i in used_rs_indices:
                    continue
                
                rs_pos = rs_obj['position']
                dist = math.sqrt((lidar_pos[0] - rs_pos[0])**2 + (lidar_pos[1] - rs_pos[1])**2)
                
                if dist < min_dist and dist < 0.5:  # 0.5m 이내의 객체만 매칭
                    min_dist = dist
                    matching_rs_idx = i
            
            # 매칭되는 RealSense 객체가 있으면 정보 병합
            if matching_rs_idx >= 0:
                rs_obj = rs_objects[matching_rs_idx]
                used_rs_indices.add(matching_rs_idx)
                
                merged_obj = lidar_obj.copy()
                merged_obj['type'] = rs_obj['type']
                merged_obj['source'] = 'both'
                if 'confidence' in rs_obj:
                    merged_obj['confidence'] = rs_obj['confidence']
                if 'bbox' in rs_obj:
                    merged_obj['bbox'] = rs_obj['bbox']
                
                merged_objects.append(merged_obj)
            else:
                # 매칭되는 RealSense 객체가 없으면 라이다 객체 그대로 추가
                merged_objects.append(lidar_obj)
        
        # 매칭되지 않은 RealSense 객체 추가
        for i, rs_obj in enumerate(rs_objects):
            if i not in used_rs_indices:
                merged_objects.append(rs_obj)
        
        return merged_objects
    
    def create_visualization(self, objects, color_image=None):
        """객체 감지 결과 시각화"""
        # 탑뷰 맵 생성 (흰 배경)
        map_frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 255
        
        # 격자 그리기
        for i in range(1, 5):
            radius = int((i / 4) * min(self.center[0], self.center[1]))
            cv2.circle(map_frame, self.center, radius, (220, 220, 220), 1)
            # 거리 표시
            cv2.putText(map_frame, f"{i}m", 
                      (self.center[0] + radius, self.center[1]), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # 축 그리기
        cv2.line(map_frame, (self.center[0], 0), (self.center[0], self.frame_size[0]), (200, 200, 200), 1)
        cv2.line(map_frame, (0, self.center[1]), (self.frame_size[1], self.center[1]), (200, 200, 200), 1)
        
        # 라이다/카메라 위치 표시
        cv2.circle(map_frame, self.center, 10, (0, 0, 0), -1)
        cv2.putText(map_frame, "LIDAR", 
                  (self.center[0] + 15, self.center[1]), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 각 객체 표시
        for obj in objects:
            # 객체 위치를 맵 좌표로 변환
            x = int(self.center[0] + obj['position'][0] * self.scale_factor)
            y = int(self.center[1] - obj['position'][1] * self.scale_factor)
            
            # 화면 범위 내에 있는지 확인
            if 0 <= x < self.frame_size[0] and 0 <= y < self.frame_size[1]:
                # 센서 소스에 따라 색상 설정
                if obj['source'] == 'lidar':
                    color = (255, 0, 0)  # 파란색: 라이다만
                elif obj['source'] == 'realsense':
                    color = (0, 0, 255)  # 빨간색: RealSense만
                else:
                    color = (0, 255, 0)  # 녹색: 둘 다
                
                # 객체 유형에 따라 형태 설정
                if obj['type'] == 'person':
                    # 사람은 십자가 형태로 표시
                    cv2.line(map_frame, (x-10, y), (x+10, y), color, 2)
                    cv2.line(map_frame, (x, y-10), (x, y+10), color, 2)
                    size = 15
                else:
                    # 다른 객체는 원으로 표시
                    size = 10
                    cv2.circle(map_frame, (x, y), size, color, -1)
                
                # 객체 ID 표시
                cv2.putText(map_frame, f"#{obj['id']}", 
                          (x+size, y-size), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 거리 정보 표시
                cv2.putText(map_frame, f"{obj['distance']:.1f}m", 
                          (x+size, y+size), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 카메라 이미지가 있으면 표시
        if color_image is not None:
            # 객체 바운딩 박스 그리기
            for obj in objects:
                if 'bbox' in obj:
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # 센서 소스에 따라 색상 설정
                    if obj['source'] == 'realsense':
                        color = (0, 0, 255)  # 빨간색: RealSense만
                    else:
                        color = (0, 255, 0)  # 녹색: 병합됨
                    
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 객체 정보 표시
                    label = f"#{obj['id']}: {obj['type']}"
                    if 'confidence' in obj:
                        label += f" ({obj['confidence']:.2f})"
                    
                    cv2.putText(color_image, label, 
                              (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 이미지 크기 조정 (표시를 위해)
            color_image_resized = cv2.resize(color_image, (400, 300))
            
            # 이미지를 맵 위에 표시
            map_frame[20:320, 20:420] = color_image_resized
            
        return map_frame
    
    def run(self):
        """메인 실행 루프"""
        if not self.connect():
            return
        
        try:
            while True:
                # LiDAR 스캔 처리
                lidar_objects = self.process_lidar_scan()
                
                # RealSense 처리
                rs_objects, color_image, _ = self.process_realsense()
                
                # 감지 결과 병합
                merged_objects = self.merge_detections(lidar_objects, rs_objects)
                
                # 결과 출력
                if merged_objects:
                    print(f"감지된 객체 수: {len(merged_objects)}")
                    for obj in merged_objects:
                        print(f"객체 #{obj['id']} - 종류: {obj['type']}, 거리: {obj['distance']:.2f}m, 소스: {obj['source']}")
                
                # 시각화
                visualization = self.create_visualization(merged_objects, color_image)
                cv2.imshow("Multi-Sensor Object Detection", visualization)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
                    break
                
                # 약간의 대기 시간을 두어 CPU 사용률 조절
                time.sleep(0.05)
                
        finally:
            self.disconnect()
            cv2.destroyAllWindows()

def main():
    try:
        # 포트를 직접 지정하여 객체 생성
        detector = MultiSensorObjectDetector(lidar_port="/dev/ttyUSB0")
        detector.run()
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
