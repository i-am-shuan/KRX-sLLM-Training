import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
import argparse
from datetime import datetime
import psutil
import gc

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'/opt/ml/output/model_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def get_gpu_info():
    """GPU 정보 수집"""
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'total_memory': f"{props.total_memory / (1024**3):.1f}GB",
            'allocated': f"{allocated:.1f}GB",
            'cached': f"{cached:.1f}GB"
        })
    return gpu_info

def log_system_info(logger):
    """시스템 정보 로깅"""
    logger.info("=== System Information ===")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    logger.info(f"RAM Total: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    if torch.cuda.is_available():
        logger.info("=== GPU Information ===")
        for gpu in get_gpu_info():
            logger.info(f"GPU {gpu['id']} - {gpu['name']}:")
            logger.info(f"  Total Memory: {gpu['total_memory']}")
            logger.info(f"  Allocated: {gpu['allocated']}")
            logger.info(f"  Cached: {gpu['cached']}")

class ModelTester:
    def __init__(self, model_name, logger):
        self.model_name = model_name
        self.logger = logger
        self.setup_model()
    
    def setup_model(self):
        """모델 및 토크나이저 설정"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self, messages, max_length=2048):
        """응답 생성"""
        try:
            with torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    messages,
                    max_new_tokens=max_length,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.1
                )
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default=os.environ.get('MODEL_NAME'))
    parser.add_argument('--test-data-path', type=str, default=os.environ.get('TEST_DATA_PATH'))
    args, _ = parser.parse_known_args()
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("Starting model test...")
    
    # 시스템 정보 로깅
    log_system_info(logger)
    
    try:
        # 테스트 데이터 로드
        with open(args.test_data_path, 'r') as f:
            test_messages = json.load(f)
        
        # 모델 테스터 초기화
        tester = ModelTester(args.model_name, logger)
        
        # 각 테스트 케이스 실행
        results = []
        for i, messages in enumerate(test_messages, 1):
            logger.info(f"\nRunning test case {i}/{len(test_messages)}")
            logger.info(f"Input: {messages[-1]['content']}")
            
            response = tester.generate_response(messages)
            logger.info(f"Response: {response}")
            
            results.append({
                'test_case': i,
                'input': messages[-1]['content'],
                'response': response
            })
            
            # 메모리 상태 로깅
            log_system_info(logger)
        
        # 결과 저장
        output_path = '/opt/ml/output/test_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise
    finally:
        logger.info("Testing completed!")

if __name__ == "__main__":
    main()
