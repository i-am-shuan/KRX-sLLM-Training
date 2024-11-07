import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel
)
from datasets import load_dataset

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(tokenizer):
    """데이터셋 준비"""
    try:
        logger.info("Loading Sujet Finance dataset...")
        dataset = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")
        
        logger.info(f"Dataset loaded. Train size: {len(dataset['train'])} examples")

        def format_conversation(system_prompt, user_prompt, answer):
            """대화를 Qwen2 형식으로 포맷팅"""
            # 시스템 프롬프트와 사용자 프롬프트 결합
            full_prompt = f"{system_prompt}\n{user_prompt}" if user_prompt else system_prompt
            
            return f"<|im_start|>user\n{full_prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

        def preprocess_function(examples):
            """데이터셋 전처리"""
            formatted_texts = [
                format_conversation(
                    system_prompt=system_prompt if system_prompt else "",
                    user_prompt=user_prompt if user_prompt else "",
                    answer=answer if answer else ""
                )
                for system_prompt, user_prompt, answer in zip(
                    examples['system_prompt'],
                    examples['user_prompt'],
                    examples['answer']
                )
            ]
            
            tokenized_inputs = tokenizer(
                formatted_texts,
                truncation=True,
                max_length=2048,
                padding="max_length",
                return_tensors="pt"
            )
            
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            tokenized_inputs["labels"][tokenized_inputs["input_ids"] == tokenizer.pad_token_id] = -100
            
            return tokenized_inputs

        logger.info("Preprocessing dataset...")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
            num_proc=4
        )

        # 데이터셋 통계 출력
        input_lengths = [len(x) for x in tokenized_dataset['train']['input_ids']]
        logger.info("Dataset statistics:")
        logger.info(f"  Total examples: {len(tokenized_dataset['train'])}")
        logger.info(f"  Sequence length - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.2f}")

        return tokenized_dataset

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        raise

def load_model_and_tokenizer():
    """모델과 토크나이저 로드"""
    try:
        base_model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        peft_model_id = "seong67360/Qwen2.5-7B-Instruct_v2"  # 기존 학습된 LoRA 모델
        hf_token = os.environ.get("HF_TOKEN")
        
        logger.info("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            use_fast=False,
            token=hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info(f"Loading base model from {base_model_id}...")
        torch.cuda.empty_cache()

        # 4비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Device map 설정
        if torch.cuda.is_available():
            device_map = {'': torch.cuda.current_device()}
        else:
            device_map = "cpu"
            logger.warning("CUDA not available, using CPU")

        logger.info(f"Device map configuration: {device_map}")

        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )

        base_model = prepare_model_for_kbit_training(base_model)

        # 기존 LoRA 어댑터 로드
        logger.info(f"Loading LoRA adapter from {peft_model_id}...")
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_id,
            token=hf_token,
            is_trainable=True
        )
        
        model.print_trainable_parameters()
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {str(e)}")
        raise

def train_model(hyperparameters, model, tokenizer, dataset):
    """모델 훈련"""
    try:
        output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        logging_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyperparameters["epochs"],
            per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
            gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
            learning_rate=hyperparameters["learning_rate"],
            max_steps=hyperparameters["max_steps"],
            logging_dir=logging_dir,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            bf16=hyperparameters["bf16"],
            gradient_checkpointing=hyperparameters["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            report_to="none",
            optim=hyperparameters["optim"],
            lr_scheduler_type=hyperparameters["lr_scheduler_type"],
            warmup_ratio=hyperparameters["warmup_ratio"],
            weight_decay=hyperparameters["weight_decay"],
            max_grad_norm=hyperparameters["max_grad_norm"],
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=50,
            dataloader_pin_memory=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=default_data_collator
        )

        logger.info("Starting training...")
        train_result = trainer.train()
        
        logger.info(f"Training metrics: {train_result.metrics}")

        logger.info("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        if trainer.is_world_process_zero():
            with open(os.path.join(output_dir, "training_metrics.txt"), "w") as f:
                f.write(str(train_result.metrics))

    except Exception as e:
        logger.error(f"Failed during training: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting training pipeline...")
        
        # 환경 체크
        check_environment()
        
        hyperparameters = {
            "epochs": int(os.environ.get("SM_HP_EPOCHS", 3)),
            "per_device_train_batch_size": int(os.environ.get("SM_HP_PER_DEVICE_TRAIN_BATCH_SIZE", 4)),
            "gradient_accumulation_steps": int(os.environ.get("SM_HP_GRADIENT_ACCUMULATION_STEPS", 8)),
            "learning_rate": float(os.environ.get("SM_HP_LEARNING_RATE", 1e-5)),  # 이어학습을 위해 학습률 낮춤
            "max_steps": int(os.environ.get("SM_HP_MAX_STEPS", 1000)),  # 큰 데이터셋을 위해 스텝 수 증가
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 0.3
        }
        
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset(tokenizer)
        train_model(hyperparameters, model, tokenizer, dataset)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
