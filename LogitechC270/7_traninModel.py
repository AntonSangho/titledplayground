from ultralytics import YOLO
import os
import shutil
import yaml

def train_yolo_model():
    print("YOLOv8 모델 학습을 시작합니다...")
    
    # 데이터셋 경로 설정
    dataset_dir = "/home/anton/projects/titledplayground/Roboflow/titledplayground2.v1i.yolov8"
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    # data.yaml 파일 존재 확인 및 내용 검증
    if not os.path.exists(data_yaml_path):
        print(f"data.yaml 파일을 찾을 수 없습니다: {data_yaml_path}")
        # 현재 디렉토리에서 data.yaml 파일 찾기
        if os.path.exists("data.yaml"):
            data_yaml_path = "data.yaml"
            print(f"현재 디렉토리에서 data.yaml 파일을 찾았습니다.")
        else:
            # 현재 디렉토리에 data.yaml 파일이 없으면 직접 생성
            print("data.yaml 파일을 생성합니다...")
            data_content = {
                'train': os.path.join(dataset_dir, 'train/images'),
                'val': os.path.join(dataset_dir, 'valid/images'),
                'test': os.path.join(dataset_dir, 'test/images') if os.path.exists(os.path.join(dataset_dir, 'test/images')) else '',
                'nc': 1,
                'names': ['part']
            }
            
            with open('data.yaml', 'w') as f:
                yaml.dump(data_content, f)
            
            data_yaml_path = "data.yaml"
            print("data.yaml 파일이 생성되었습니다.")
    
    # data.yaml 파일 내용 확인 및 수정
    with open(data_yaml_path, 'r') as f:
        try:
            data_yaml = yaml.safe_load(f)
            print(f"data.yaml 내용: {data_yaml}")
            
            # 디렉토리 존재 확인
            train_path = data_yaml.get('train', '')
            valid_path = data_yaml.get('val', '')
            
            if not os.path.exists(train_path):
                print(f"경고: 훈련 이미지 경로를 찾을 수 없습니다: {train_path}")
                # 상대 경로 확인
                alt_train_path = os.path.join(dataset_dir, 'train/images')
                if os.path.exists(alt_train_path):
                    data_yaml['train'] = alt_train_path
                    print(f"훈련 이미지 경로를 {alt_train_path}로 수정했습니다.")
            
            if not os.path.exists(valid_path):
                print(f"경고: 검증 이미지 경로를 찾을 수 없습니다: {valid_path}")
                # 상대 경로 확인
                alt_valid_path = os.path.join(dataset_dir, 'valid/images')
                if os.path.exists(alt_valid_path):
                    data_yaml['val'] = alt_valid_path
                    print(f"검증 이미지 경로를 {alt_valid_path}로 수정했습니다.")
                else:
                    # 검증 세트 분리
                    print("검증 세트가 없습니다. 훈련 세트를 검증 세트로도 사용합니다.")
                    data_yaml['val'] = data_yaml['train']
            
            # 수정된 내용 저장
            with open('data.yaml', 'w') as f:
                yaml.dump(data_yaml, f)
                data_yaml_path = 'data.yaml'
                print("수정된 data.yaml 파일이 저장되었습니다.")
                
        except Exception as e:
            print(f"data.yaml 파일 읽기 또는 수정 중 오류 발생: {e}")
            return None
    
    # 사전 학습된 YOLOv8 모델 로드 (nano 크기)
    model = YOLO('yolov8n.pt')
    
    try:
        # 모델 학습 진행
        results = model.train(
            data=data_yaml_path,    # 데이터셋 설정 파일
            epochs=50,              # 학습 에폭 수
            imgsz=640,              # 이미지 크기
            #batch=16,               # 배치 크기
            batch=4,               # 배치 크기
            name='titledplayground2_model'  # 결과 저장 폴더명
        )
        
        # 모델 내보내기
        model_path = model.export(format='pt')  # 기본 PyTorch 형식으로 내보내기
        
        print(f"학습 완료! 모델이 저장된 경로: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    model_path = train_yolo_model()
    if model_path:
        print(f"이 경로를 객체 감지 스크립트에서 사용하세요: {model_path}")
    else:
        print("모델 학습에 실패했습니다.")