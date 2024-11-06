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
    get_peft_model,
)
from datasets import load_dataset

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """환경 설정 체크"""
    try:
        import torch
        import accelerate
        import transformers
        import bitsandbytes
        
        logger.info("=== Environment Information ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Accelerate version: {accelerate.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"Bitsandbytes version: {bitsandbytes.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i} Memory: {gpu_mem:.2f} GB")
                
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)
        logger.info(f"System Memory: {total_memory:.2f} GB")
        logger.info("============================")
        
    except Exception as e:
        logger.error(f"Error checking environment: {str(e)}")
        raise

def load_model_and_tokenizer():
    """모델과 토크나이저 로드"""
    try:
        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        hf_token = os.environ.get("HF_TOKEN")
        
        logger.info("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
            token=hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info(f"Loading model from {model_id}...")
        torch.cuda.empty_cache()

        # 4비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Device map 설정
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            if world_size > 1:
                # 분산 학습 환경
                device_map = {"": local_rank}
                logger.info(f"Using distributed training with device_map: {device_map}")
            else:
                # 단일 GPU 환경
                device_map = {"": 0}
                logger.info("Using single GPU training")
        else:
            device_map = "cpu"
            logger.warning("CUDA not available, using CPU")

        logger.info(f"device_map configuration: {device_map}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )

        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = prepare_model_for_kbit_training(model)

        # LoRA 설정
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, config)
        
        if torch.cuda.is_available():
            logger.info(f"Model is on device: {next(model.parameters()).device}")
            
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

def prepare_dataset(tokenizer):
    """데이터셋 준비"""
    try:
        logger.info("Loading KRX dataset...")
        dataset = load_dataset("amphora/krx-sample-instructions")
        
        logger.info(f"Dataset loaded. Train size: {len(dataset['train'])} examples")

        def format_conversation(prompt, response):
            """대화를 Qwen2 형식으로 포맷팅"""
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

        def preprocess_function(examples):
            """데이터셋 전처리"""
            formatted_texts = [
                format_conversation(prompt, response)
                for prompt, response in zip(examples['prompt'], examples['response'])
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

        return tokenized_dataset

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
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
            "learning_rate": float(os.environ.get("SM_HP_LEARNING_RATE", 2e-5)),
            "max_steps": int(os.environ.get("SM_HP_MAX_STEPS", 100)),
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
