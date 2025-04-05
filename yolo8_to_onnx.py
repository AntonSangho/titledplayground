from ultralytics import YOLO

# YOLOv8n 모델 로드
model = YOLO('yolov8n.pt')

# ONNX 형식으로 내보내기
# Unity Sentis 호환을 위한 설정 적용
model.export(format='onnx', 
             opset=13, 
             simplify=True, 
             dynamic=False)
