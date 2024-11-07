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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmtools import merge_adapter

def merge_models():
    base_model_path = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "/opt/ml/input/data/adapter"
    output_path = "/opt/ml/model"
    
    print("Starting model merge process...")
    merge_adapter.merge(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_path=output_path,
        device="cuda"
    )
    print("Merge completed successfully!")

if __name__ == "__main__":
    merge_models()
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
        'llmtools',
        'peft>=0.6.0'
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
        'TRUST_REMOTE_CODE': 'true'
    }
    
    merge_estimator = HuggingFace(
        entry_point='merge_script.py',
        source_dir='./scripts',
        instance_type='ml.p5.48xlarge',
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
    # 필요한 스크립트들 생성
    create_merge_script()
    create_requirements()
    create_bootstrap()
    
    # merge job 설정
    estimator = setup_merge_job()
    
    # adapter 모델 경로
    adapter_s3_path = "s3://sagemaker-us-east-1-104871657422/huggingface-pytorch-training-2024-11-07-03-40-25-985/output/model.tar.gz"
    
    try:
        # merge job 실행
        estimator.fit({'adapter': adapter_s3_path}, wait=True, logs="All")
        print(f"Merged model saved to: {estimator.model_data}")
    except Exception as e:
        print(f"Merge job failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
