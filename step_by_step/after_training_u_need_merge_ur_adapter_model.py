import os
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role

def create_merge_script():
    """merge_script.py 파일 생성"""
    os.makedirs('scripts', exist_ok=True)
    
    merge_script = '''
import os
import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def download_from_s3(s3_uri, local_path):
    """S3에서 파일 다운로드"""
    print(f"Downloading {s3_uri} to {local_path}")
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])
    s3_client = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)

def merge_models():
    print("Starting model merge process...")

    ### adapter files S3 URIs ###
    config_s3_uri = "s3://sagemaker-us-east-1-104871657422/huggingface-pytorch-training-2024-11-07-03-40-25-985/output/adapter_config.json"
    model_s3_uri = "s3://sagemaker-us-east-1-104871657422/huggingface-pytorch-training-2024-11-07-03-40-25-985/output/adapter_model.safetensors"

    
    # 로컬 경로 설정
    adapter_path = "/opt/ml/input/data/adapter"
    os.makedirs(adapter_path, exist_ok=True)
    
    config_path = os.path.join(adapter_path, "adapter_config.json")
    model_path = os.path.join(adapter_path, "adapter_model.safetensors")
    
    # S3에서 파일 다운로드
    download_from_s3(config_s3_uri, config_path)
    download_from_s3(model_s3_uri, model_path)
    
    base_model_path = "Qwen/Qwen2.5-7B-Instruct"
    output_path = "/opt/ml/model"
    
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto"
    )
    
    print("Merging models...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed successfully!")

if __name__ == "__main__":
    try:
        merge_models()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
'''
    with open('scripts/merge_script.py', 'w') as f:
        f.write(merge_script)

def create_requirements():
    """requirements.txt 생성"""
    os.makedirs('scripts', exist_ok=True)
    requirements = [
        'transformers @ git+https://github.com/huggingface/transformers.git@main',
        'accelerate>=0.27.0',
        'torch>=2.1.0',
        'safetensors>=0.3.1',
        'peft>=0.6.0',
        'boto3'
    ]
    
    with open('scripts/requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))

def create_bootstrap():
    """bootstrap.sh 생성"""
    os.makedirs('scripts', exist_ok=True)
    bootstrap_script = '''#!/bin/bash
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
'''
    with open('scripts/bootstrap.sh', 'w') as f:
        f.write(bootstrap_script)
    os.chmod('scripts/bootstrap.sh', 0o755)

def setup_merge_job():
    """SageMaker merge job 설정"""
    region = "us-east-1"
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    
    environment = {
        'TORCH_HOME': '/tmp/torch',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'MAX_JOBS': '8',
        'OMP_NUM_THREADS': '8',
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_VERBOSITY': 'info',
        'TRUST_REMOTE_CODE': 'true',
        'HF_HOME': '/tmp'
    }
    
    merge_estimator = HuggingFace(
        entry_point='merge_script.py',
        source_dir='./scripts',
        instance_type='ml.g5.48xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.36.0',
        pytorch_version='2.1.0',
        py_version='py310',
        environment=environment,
        disable_profiler=True,
        debugger_hook_config=False,
        volume_size=256,
        max_retry_attempts=1,
        keep_alive_period_in_seconds=1800,
        sagemaker_session=session
    )
    
    return merge_estimator

def main():
    # 기존 scripts 디렉토리 삭제 (있다면)
    if os.path.exists('scripts'):
        import shutil
        shutil.rmtree('scripts')
    
    # 필요한 스크립트들 생성
    create_merge_script()
    create_requirements()
    create_bootstrap()
    
    # merge job 설정
    estimator = setup_merge_job()
    
    try:
        # merge job 실행 (입력 데이터 없이)
        estimator.fit(wait=True, logs="All")
        print(f"Merged model saved to: {estimator.model_data}")
    except Exception as e:
        print(f"Merge job failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
