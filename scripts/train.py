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

        # GPU 메모리 설정
        n_gpus = torch.cuda.device_count()
        max_memory = {i: "28GB" for i in range(n_gpus)}
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
            offload_folder="offload",
            load_in_8bit=False,  # 4비트 양자화만 사용
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,  # flash attention 비활성화
            use_cache=False
        )

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
            
            # 최적화 설정
            bf16=True,
            fp16=False,
            deepspeed={
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "allgather_partitions": True,
                    "reduce_scatter": True,
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "bf16": {
                    "enabled": True
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": hyperparameters["learning_rate"],
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    }
                }
            },
            
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            optim="adamw_torch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=default_data_collator,
            tokenizer=tokenizer,
        )

        logger.info("Starting training...")
        train_result = trainer.train()
        
        logger.info(f"Training metrics: {train_result.metrics}")

        logger.info("Saving model...")
        model.save_pretrained(output_dir)
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

        if len(tokenized_dataset['train']) > 0:
            seq_lens = [len(x) for x in tokenized_dataset['train']['input_ids']]
            logger.info("Dataset statistics:")
            logger.info(f"  Total examples: {len(tokenized_dataset['train'])}")
            logger.info(f"  Sequence length - Min: {min(seq_lens)}, Max: {max(seq_lens)}, Avg: {sum(seq_lens)/len(seq_lens):.2f}")
        
        return tokenized_dataset

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting training pipeline...")
        
        # 환경 변수 설정
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        hyperparameters = {
            "epochs": int(os.environ.get("SM_HP_EPOCHS", 3)),
            "per_device_train_batch_size": int(os.environ.get("SM_HP_PER_DEVICE_TRAIN_BATCH_SIZE", 1)),
            "gradient_accumulation_steps": int(os.environ.get("SM_HP_GRADIENT_ACCUMULATION_STEPS", 16)),
            "learning_rate": float(os.environ.get("SM_HP_LEARNING_RATE", 1e-5)),
            "max_steps": int(os.environ.get("SM_HP_MAX_STEPS", 100))
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
