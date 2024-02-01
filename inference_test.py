from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

save_dir = "./sft_out"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat", # path to the output directory
    device_map="auto",
    cache_dir="./model_cache",
    local_files_only=True, 
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        use_fast=False,
        trust_remote_code=True,
    )

model = PeftModel.from_pretrained(model, save_dir)

while(1):
    query = input("Query: ")
    response, history = model.chat(tokenizer, query, history=None)
    print(response)