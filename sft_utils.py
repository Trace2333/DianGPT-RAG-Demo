import os
import json
import torch
from tqdm import tqdm
from typing import List
from transformers import (
    PreTrainedTokenizer,
    )
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


PATH_FOR_SFT_DATA = "./sft_data/instructions_10.json"

def load_data(path_for_data):
    if not os.path.exists(path_for_data):
        raise RuntimeError("given path no data")
    
    with open(path_for_data, "r", encoding="utf8") as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    
    return data


def data_preprocess(
    chat_logs: List,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    system_prompt: str = "You are a helpful assistant."
    ):
    """
        数据格式：
        [
            {
                "history": "你好",
                "response": "你好啊"
            },
            {
                "history": "你好",
                "response": "你好啊"
            }
        ]
        system_prompt 是预设的prompts，不用更改和传入 
    """
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    line_break = tokenizer('\n').input_ids
    
    _system = tokenizer('system').input_ids + line_break
    
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    # based on qwen finetuning.py
    roles_id = {"user": tokenizer("user\n").input_ids, "assistant": tokenizer("assistant\n").input_ids}
    
    input_ids, targets = [], []
    
    chat_logs = [r["conversations"] for r in chat_logs] # 暂时忽略id
    
    for round in tqdm(chat_logs):
        input_id, target = [], []
        # 每一个round只有一个预设的提示词，单独处理
        system = [im_start] + _system + tokenizer(system_prompt).input_ids + [im_end] + line_break
        input_id += system
        
        ignoring = [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + line_break
        target += ignoring
        assert len(input_id) == len(target)
        for sent in round:
            role = roles[sent["from"]]
            # 给定的问题直接tokenize
            _input_id = tokenizer(role).input_ids + line_break + \
                tokenizer(sent["value"]).input_ids + [im_end] + line_break
            input_id += _input_id
            # 回答格式遵循qwen标准
            if role == '<|im_start|>user':
                # 只贡献长度，不贡献内容
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + line_break
            elif role == '<|im_start|>assistant':
                # assistant的回答，贡献长度和内容
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + line_break
                    # :2 ---> im_end, line_break  +1 ---> line_break
            else:
                raise NotImplementedError
            target += _target
            assert len(input_id) == len(target)
            
        input_id += [tokenizer.pad_token_id] * (max_length - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_length - len(target))
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    
    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": torch.ne(input_ids, tokenizer.pad_token_id).int(),
    }

class SFTDataset(Dataset):
    def __init__(self, data_dict) -> None:
        super().__init__()        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_masks = data_dict["attention_mask"]
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_masks[idx],
        }
        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    data = load_data(PATH_FOR_SFT_DATA)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    processed_data = data_preprocess(
        data,
        tokenizer,
    )
    dataset = SFTDataset(processed_data)
    
    print(dataset)