import os
import torch

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Trainer, deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    )
from sft_utils import load_data, data_preprocess, SFTDataset, PATH_FOR_SFT_DATA

train_model_name_path = "Qwen/Qwen-7B-Chat"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=train_model_name_path)
    
@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default="model_cache")
    optim: str = field(default="adamw_torch")
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    save_path: str = field(default="./finetuning_out")
    
@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    
def main():
    parser = HfArgumentParser(
            (ModelArguments, TrainingArguments, LoraArguments)
        )
    
    model_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.max_length,
    )
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded.")
    # 开放生成
    tokenizer.pad_token_id = tokenizer.eod_id
    
    if training_args.use_lora:
        
        from peft import (
            TaskType,
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training)

        modules_to_save = ["wte", "lm_head"]
        # based on qwen docs
            
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    data = load_data(PATH_FOR_SFT_DATA)
    
    processed_data = data_preprocess(
        data,
        tokenizer,
        max_length=training_args.max_length,
    )
    dataset = SFTDataset(processed_data)
    
    trainer = Trainer(
        model=model, args=training_args, tokenizer=tokenizer, train_dataset=dataset
    )
    print("Trainer initialize done.")
    trainer.train()
    trainer.save_state()
    
    if not os.path.exists(training_args.save_path):
        os.mkdir(training_args.save_path)
    model.save_pretrained(training_args.save_path)
    
if __name__ == "__main__":
    main()