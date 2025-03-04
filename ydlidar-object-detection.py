import os
import ydlidar
import time
import numpy as np
import math
import cv2

class LidarObjectDetector:
    def __init__(self, port="/dev/ttyUSB0", min_points=5, min_distance=0.5, max_distance=3.0):
        # 초기화 파라미터
        self.port = port
        self.min_points = min_points  # 하나의 객체로 판단하기 위한 최소 포인트 수
        self.min_distance = min_distance  # 검출할 객체의 최소 거리 (미터)
        self.max_distance = max_distance  # 검출할 객체의 최대 거리 (미터)
        self.cluster_threshold = 0.2  # 같은 객체로 판단할 포인트 간 최대 거리 (미터)
        
        # LiDAR 초기화
        ydlidar.os_init()
        self.laser = ydlidar.CYdLidar()
        self.configure_lidar()
        
    def configure_lidar(self):
        """LiDAR 환경 설정"""
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, self.port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)  # 전방 180도만 스캔
        self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)  # 후방 180도도 스캔
        self.laser.setlidaropt(ydlidar.LidarPropMaxRange, 16.0)
        self.laser.setlidaropt(ydlidar.LidarPropMinRange, 0.08)
        self.laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
    
    def connect(self):
        """LiDAR 연결 및 초기화"""
        ret = self.laser.initialize()
        if ret:
            print("LiDAR 연결 성공!")
            ret = self.laser.turnOn()
            if ret:
                print("LiDAR 스캔 시작...")
                return True
            else:
                print("LiDAR 스캔 시작 실패")
        else:
            print("LiDAR 연결 실패")
        return False
    
    def disconnect(self):
        """LiDAR 연결 해제"""
        self.laser.turnOff()
        self.laser.disconnecting()
        print("LiDAR 연결 해제")
    
    def cluster_points(self, points):
        """포인트 클러스터링으로 객체 감지"""
        clusters = []
        current_cluster = []
        
        # 각도 기준으로 정렬
        sorted_points = sorted(points, key=lambda p: p[1])
        
        for i, point in enumerate(sorted_points):
            if not current_cluster:
                current_cluster.append(point)
            else:
                prev_point = current_cluster[-1]
                # 각도 차이와 거리 차이로 같은 객체인지 판단
                angle_diff = abs(point[1] - prev_point[1])
                if angle_diff > 180:  # 각도의 경계(360도 -> 0도) 처리
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
    
    def process_scan(self):
        """LiDAR 스캔 처리 및 객체 감지"""
        scan = ydlidar.LaserScan()
        ret = self.laser.doProcessSimple(scan)
        
        if ret and scan.config.scan_time != 0:
            points = []
            # 스캔 데이터에서 유효한 포인트 추출
            for i, point in enumerate(scan.points):
                distance = point.range  # 미터 단위
                angle = math.degrees(point.angle)  # 각도로 변환
                
                # 유효한 거리 범위에 있는 포인트만 저장
                if self.min_distance <= distance <= self.max_distance:
                    points.append((distance, angle))
            
            # 포인트 클러스터링하여 객체 감지
            clusters = self.cluster_points(points)
            
            # 객체 정보 추출
            objects = []
            for i, cluster in enumerate(clusters):
                # 객체의 중심 계산
                distances = [p[0] for p in cluster]
                angles = [p[1] for p in cluster]
                avg_distance = sum(distances) / len(distances)
                avg_angle = sum(angles) / len(angles)
                
                # 직교 좌표계로 변환 (라이다 위치가 원점)
                x = avg_distance * math.cos(math.radians(avg_angle))
                y = avg_distance * math.sin(math.radians(avg_angle))
                
                objects.append({
                    'id': i+1,
                    'distance': avg_distance,
                    'angle': avg_angle,
                    'position': (x, y),
                    'size': len(cluster)
                })
            
            return objects
        return []
    
    def visualize(self, objects, frame_size=(600, 600)):
        """객체 위치 시각화"""
        # 시각화 프레임 생성 (흰 배경)
        frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
        center = (frame_size[0] // 2, frame_size[1] // 2)
        
        # 격자 그리기
        for i in range(1, 5):
            radius = int((i / 4) * min(center[0], center[1]))
            cv2.circle(frame, center, radius, (220, 220, 220), 1)
        
        # 축 그리기
        cv2.line(frame, (center[0], 0), (center[0], frame_size[0]), (200, 200, 200), 1)
        cv2.line(frame, (0, center[1]), (frame_size[1], center[1]), (200, 200, 200), 1)
        
        # 라이다 위치 표시
        cv2.circle(frame, center, 10, (0, 0, 0), -1)
        
        # 각 객체 표시
        for obj in objects:
            # 직교 좌표를 화면 좌표로 변환
            x = int(center[0] + obj['position'][0] * 100)  # 100 픽셀 = 1m
            y = int(center[1] - obj['position'][1] * 100)  # y축은 아래로 양수
            
            # 객체 크기에 비례하는 원 그리기
            radius = max(5, min(20, int(obj['size'] / 2)))
            cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)
            
            # 객체 ID 표시
            cv2.putText(frame, f"#{obj['id']}", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 거리 정보 표시
            distance_text = f"{obj['distance']:.2f}m"
            cv2.putText(frame, distance_text, (x+10, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return frame

def main():
    # 사용 가능한 포트 확인
    ports = ydlidar.lidarPortList()
    if ports:
        port = next(iter(ports.values()))
        print("사용 가능한 포트:", port)
    else:
        port = "/dev/ttyUSB0"
        print(f"포트를 찾을 수 없어 기본값 사용: {port}")
    
    # LiDAR 객체 감지기 생성
    #detector = LidarObjectDetector(port=port)
    detector = LidarObjectDetector(port="/dev/ttyUSB0")
    
    if detector.connect():
        try:
            while ydlidar.os_isOk():
                # 객체 감지
                objects = detector.process_scan()
                
                if objects:
                    print(f"감지된 객체 수: {len(objects)}")
                    for obj in objects:
                        print(f"객체 #{obj['id']} - 거리: {obj['distance']:.2f}m, 각도: {obj['angle']:.1f}°, 크기: {obj['size']} 포인트")
                
                # 시각화
                frame = detector.visualize(objects)
                cv2.imshow("LiDAR Object Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
                    break
                
                time.sleep(0.1)
                
        finally:
            detector.disconnect()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
