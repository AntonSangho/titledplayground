import socket
import json

# UDP 설정 (함수 main() 시작 부분에 추가)
def main():
    # ... 기존 코드 ...
    
    # UDP 소켓 설정
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_address = ('127.0.0.1', 5065)  # Unity 앱이 실행 중인 IP와 포트
    
    # ... 기존 코드 ...
    
    # 객체 감지 루프 내에서 (for box in boxes: 루프 안에 추가)
    # 객체 중심점을 실제 좌표(cm)로 변환한 부분 바로 아래에 추가
    try:
        real_point = cv2.perspectiveTransform(img_point, homography_matrix)
        real_x, real_y = real_point[0][0]
        
        # Unity로 전송할 데이터 준비
        object_data = {
            "posX": float(real_x),
            "posY": float(real_y),
            "weight": 1.0  # 기본 가중치
        }
        
        # JSON으로 변환하여 전송
        json_data = json.dumps(object_data)
        udp_socket.sendto(json_data.encode(), unity_address)
        
        # ... 기존 시각화 코드 계속 ...
    except:
        # ... 기존 예외 처리 ...