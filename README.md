## 구조

project_root/
│
├── sagemaker_train.py           # SageMaker 실행 스크립트
│
├── scripts/                     # SageMaker에 전달될 훈련 관련 파일들
│   ├── train.py                 # 실제 모델 훈련 스크립트
│   ├── tokenization_qwen2.py    # Qwen2 토크나이저 구현
│   ├── requirements.txt         # 필요한 패키지 목록 (스크립트가 자동 생성)
│   └── bootstrap.sh             # 환경 설정 스크립트 (스크립트가 자동 생성)
