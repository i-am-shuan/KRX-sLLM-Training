import os
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import boto3

def create_requirements():
    """requirements.txt 생성"""
    os.makedirs('scripts', exist_ok=True)
    requirements = [
        'transformers @ git+https://github.com/huggingface/transformers.git@main',
        'accelerate>=0.27.0',
        'datasets>=2.10.0',
        'evaluate>=0.4.0',
        'torch>=2.1.0',
        'safetensors>=0.3.1',
        'sentencepiece>=0.1.99',
        'bitsandbytes>=0.41.0',
        'peft>=0.6.0',
        'regex'
    ]
    
    with open('scripts/requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))

def create_bootstrap():
    """bootstrap.sh 생성 - transformers 재설치를 위한 스크립트"""
    os.makedirs('scripts', exist_ok=True)
    bootstrap_script = '''#!/bin/bash
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
'''
    with open('scripts/bootstrap.sh', 'w') as f:
        f.write(bootstrap_script)
    os.chmod('scripts/bootstrap.sh', 0o755)

def setup_training():
    """SageMaker 훈련 설정"""
    region = "us-east-1"
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    huggingface_token = "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"

    os.makedirs('scripts', exist_ok=True)
    
    environment = {
        'MODEL_ID': 'Qwen/Qwen2.5-7B-Instruct',
        'HF_TOKEN': huggingface_token,
        'HUGGING_FACE_HUB_TOKEN': huggingface_token,
        'HF_HOME': '/tmp/huggingface',
        'TORCH_HOME': '/tmp/torch',
        
        # GPU 설정
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        
        # 메모리 최적화
        'MAX_JOBS': '8',
        'OMP_NUM_THREADS': '8',
        
        # 분산 학습 설정
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'efa',
        'FI_EFA_USE_DEVICE_RDMA': '1',
        'FI_PROVIDER': 'efa',
        'NCCL_MIN_NRINGS': '4',
        
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_VERBOSITY': 'info',
        'TRUST_REMOTE_CODE': 'true'
    }

    hyperparameters = {
        'epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1e-5,  # 이어학습을 위해 학습률 낮춤
        'max_steps': 1000,      # 더 많은 학습 스텝
        'bf16': True,
        'max_length': 2048,
        'gradient_checkpointing': True,
        'optim': 'adamw_torch',    # fused 옵션 제거
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 0.3
    }

    # HuggingFace Estimator 설정
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p5.48xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.36.0',
        pytorch_version='2.1.0',
        py_version='py310',
        hyperparameters=hyperparameters,
        environment=environment,
        disable_profiler=True,
        debugger_hook_config=False,
        volume_size=256,
        max_retry_attempts=1,
        keep_alive_period_in_seconds=1800,
        sagemaker_session=session,
        distribution={
            'torch_distributed': {
                'enabled': True
            }
        },
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
            {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'}
        ]
    )
    
    return huggingface_estimator

def main():
    create_requirements()
    create_bootstrap()  # bootstrap 스크립트 생성 추가
    estimator = setup_training()
    
    try:
        estimator.fit(wait=True, logs="All")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
