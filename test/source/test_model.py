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

def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    return args

class ModelTester:
    def __init__(self, model_name, logger):
        self.model_name = model_name
        self.logger = logger
        self.setup_model()
    
    def setup_model(self):
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
    # 로깅 설정
    logger = setup_logging()
    logger.info("Starting model test...")
    
    try:
        # 인자 파싱
        args = parse_args()
        logger.info(f"Test data path: {args.test_data_path}")
        logger.info(f"Model name: {args.model_name}")
        
        # 테스트 데이터 로드
        with open(args.test_data_path, 'r') as f:
            test_messages = json.load(f)
        
        # 모델 테스터 초기화
        tester = ModelTester(args.model_name, logger)
        
        # 결과 저장용 리스트
        results = []
        
        # 각 테스트 케이스 실행
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
        
        # 결과 저장
        output_path = '/opt/ml/output/test_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Test results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise
    finally:
        logger.info("Testing completed!")

if __name__ == "__main__":
    main()
