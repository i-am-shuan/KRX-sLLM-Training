import os
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
from sagemaker.network import NetworkConfig
import boto3

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

def create_requirements():
    """requirements.txt 생성"""
    os.makedirs('scripts', exist_ok=True)
    requirements = [
        'transformers @ git+https://github.com/huggingface/transformers.git@main',
        'accelerate>=0.27.0',
        'datasets>=2.10.0',
        'evaluate>=0.4.0',
        'torch==2.3.1',
        'safetensors>=0.3.1',
        'sentencepiece>=0.1.99',
        'bitsandbytes>=0.41.0',
        'peft>=0.6.0',
        'deepspeed>=0.10.0',
        'regex'
    ]
    
    with open('scripts/requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))

def setup_training():
    """SageMaker 훈련 설정"""
    region = "us-east-1"
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    huggingface_token = "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"

    # source_dir에 필요한 파일들 복사
    os.makedirs('scripts', exist_ok=True)
    
    # tokenization_qwen2.py 파일을 scripts 디렉토리로 복사
    import shutil
    if os.path.exists('tokenization_qwen2.py'):
        shutil.copy2('tokenization_qwen2.py', 'scripts/tokenization_qwen2.py')
    
    environment = {
        'MODEL_ID': 'Qwen/Qwen2.5-7B-Instruct',
        'HF_TOKEN': huggingface_token,
        'HUGGING_FACE_HUB_TOKEN': huggingface_token,
        'HF_HOME': '/tmp/huggingface',
        'TORCH_HOME': '/tmp/torch',
        
        # GPU 설정
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        
        # 분산 학습
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'efa',
        'FI_EFA_USE_DEVICE_RDMA': '1',
        'FI_PROVIDER': 'efa',
        'NCCL_MIN_NRINGS': '4',
        
        # 메모리 최적화
        'MAX_JOBS': '8',
        'OMP_NUM_THREADS': '8',
        
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_VERBOSITY': 'info',
        'TRUST_REMOTE_CODE': 'true'
    }

    hyperparameters = {
        "epochs": 3,
        "per_device_train_batch_size": 4, 
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "max_steps": 100,
        "bf16": True,
        "max_length": 2048,
        
        # DeepSpeed 설정 수정
        "deepspeed": {
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 8,
            "bf16": {
                "enabled": True
            },
            # 컴파일러 관련 설정 제거
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-5, 
                    "warmup_num_steps": 100
                }
            }
        }
    }

    # 분산 학습 설정
    distribution = {
        "torch_distributed": {
            "enabled": True,
            "backend": "nccl"
        }
    }

    # HuggingFace Estimator 설정
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p4de.24xlarge',
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
        distribution=distribution,
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
            {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'}
        ]
    )
    
    return huggingface_estimator

def main():
    create_requirements()
    create_bootstrap()  # bootstrap 스크립트도 생성
    estimator = setup_training()
    
    try:
        estimator.fit(wait=True, logs="All")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
