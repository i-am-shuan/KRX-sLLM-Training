## 전체 구조

![image](https://github.com/user-attachments/assets/ab5e2c14-fe05-42b3-8851-9addb40e0e05)


**✅ Base Model: Qwen/Qwen2.5-7B-Instruct**

**✅ 각 파일의 역할:**

1. `sagemaker_train.py`:
    - SageMaker 훈련 작업 설정 및 시작
    - requirements.txt 및 [bootstrap.sh](http://bootstrap.sh/) 생성
    - 훈련 하이퍼파라미터 및 환경 설정
2. `scripts/train.py`:
    - 실제 모델 훈련 로직
    - 데이터 로딩 및 전처리
    - 모델 학습 및 저장
3. `scripts/tokenization_qwen2.py`:
    - Qwen2 모델의 토크나이저 구현
    - 현재는 최신 transformers를 사용하므로 필요하지 않을 수 있음
4. `scripts/requirements.txt`:
    - 훈련에 필요한 Python 패키지 목록
    - sagemaker_train.py에 의해 자동 생성됨
5. `scripts/bootstrap.sh`:
    - 컨테이너 시작 시 실행되는 스크립트
    - transformers 최신 버전 설치
    - sagemaker_train.py에 의해 자동 생성됨

**✅ 실행 순서:**

1. `sagemaker_train.py` 실행
2. `scripts/` 디렉토리 및 필요한 파일들 생성
3. SageMaker 훈련 작업 시작
4. 컨테이너에서 `bootstrap.sh` 실행
5. `train.py`로 실제 훈련 시작

주의사항:

- `scripts/` 디렉토리의 모든 파일은 SageMaker 훈련 작업에 포함됨
- `source_dir='./scripts'` 설정으로 인해 scripts 디렉토리의 내용이 훈련 컨테이너로 복사됨
- requirements.txt와 bootstrap.sh는 코드에 의해 자동 생성되므로 직접 생성할 필요 없음
