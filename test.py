from transformers import AutoModelForCausalLM, AutoTokenizer


qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    use_fast=False,
    trust_remote_code=True)

input_ids = []
for _ in range(3):
    in_text = "user"
    ids = qwen_tokenizer(in_text).input_ids

qwen_tokenizer.pad(
    [ids],
    padding="max_length",
    max_length=40,
    )

print(qwen_tokenizer)