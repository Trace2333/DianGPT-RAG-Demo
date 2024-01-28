import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from database import create_and_add_ele, MyEmbeddingFunction, Query
from BCEmbedding import EmbeddingModel, RerankerModel

collection = create_and_add_ele()

qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).eval()
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    use_fast=False,
    trust_remote_code=True)

embedding = MyEmbeddingFunction()
rank_model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1", device="cuda:1")

querying = Query(collection, rank_model, embedding)

# response, history = qwen_model.chat(qwen_tokenizer, "你好", history=None)
history = None
while(1):
    query = input("query: ")
    
    query_out = querying.query_chroma(query)
    
    back_knowledge = query_out['rerank_passages'][0]
    
    query = "检索到的知识: " + back_knowledge + "回答：" + query
    
    response, history = qwen_model.chat(qwen_tokenizer, query, history=history)

    print("response: ", response)
