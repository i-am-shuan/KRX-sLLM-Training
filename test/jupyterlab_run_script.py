import sagemaker
from sagemaker.huggingface import HuggingFace
from datetime import datetime
import json
import os
import tempfile

def submit_model_test_job():
    # SageMaker 세션 설정
    role = sagemaker.get_execution_role()
    sess = sagemaker.Session()
    
    # 환경 변수 설정
    environment = {
        'MODEL_NAME': 'seong67360/Qwen2.5-7B-Instruct_v4',
        'MAX_LENGTH': '2048',
        'TEST_BATCH_SIZE': '4',
        'PIP_PACKAGES': 'git+https://github.com/huggingface/transformers.git'
    }
    
    # 테스트 메시지 준비
    test_messages = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "간단한 자기소개 해주세요."}],
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is quantum computing?"}],
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Write a short poem about AI."}]
    ]
    
    # 임시 파일에 테스트 데이터 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_messages, f, ensure_ascii=False, indent=2)
        temp_file_path = f.name
    
    try:
        # 테스트 데이터를 S3에 업로드
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_prefix = f'qwen-test/{timestamp}'
        s3_path = sess.upload_data(
            path=temp_file_path,
            bucket=sess.default_bucket(),
            key_prefix=s3_prefix
        )
        
        # requirements.txt 생성
        with open('source/requirements.txt', 'w') as f:
            f.write('git+https://github.com/huggingface/transformers.git\n')
            f.write('accelerate\n')
            f.write('scipy\n')
            f.write('scikit-learn\n')
            f.write('bitsandbytes\n')
            f.write('safetensors\n')
        
        # Hugging Face Estimator 설정
        huggingface_estimator = HuggingFace(
            entry_point='test_model.py',
            source_dir='./source',
            instance_type='ml.p5.48xlarge',
            instance_count=1,
            role=role,
            transformers_version='4.28',     # transformers_version 사용
            pytorch_version='2.0',
            py_version='py310',
            environment=environment,
            max_run=3600,
            base_job_name='qwen-model-test',
            disable_profiler=True,
            debugger_hook_config=False,
            hyperparameters={
                'test_data_path': f'{s3_path}/test_messages.json',
                'epochs': 1,
                'model_name': environment['MODEL_NAME']
            }
        )
        
        print("\nStarting training job with the following configuration:")
        print(f"Python version: {huggingface_estimator.py_version}")
        print(f"PyTorch version: {huggingface_estimator.pytorch_version}")
        print("Transformers version: 4.28 (will upgrade to GitHub version during startup)")
        print(f"Instance type: {huggingface_estimator.instance_type}")
        print(f"S3 test data path: {s3_path}")
        
        # Training Job 실행
        huggingface_estimator.fit()
        
        return huggingface_estimator
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    try:
        print(f"SageMaker SDK version: {sagemaker.__version__}")
        
        # source 디렉토리가 없으면 생성
        os.makedirs('source', exist_ok=True)
        
        estimator = submit_model_test_job()
        print(f"Training job completed: {estimator.latest_training_job.job_name}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())
