from transformers import AutoModelForCausalLM, AutoTokenizer
from BCEmbedding import EmbeddingModel
import torch


access_token = "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"
# list of sentences
# sentences = ['sentence_0', 'sentence_1']

# # init embedding model
# model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1", token=access_token)

# # extract embeddings
# embeddings = model.encode(sentences)

# from BCEmbedding import RerankerModel

# # your query and corresponding passages
# query = 'input_query'
# passages = ['passage_0', 'passage_1']

# # construct sentence pairs
# sentence_pairs = [[query, passage] for passage in passages]

# # init reranker model
# model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1", token=access_token)

# # method 0: calculate scores of sentence pairs
# scores = model.compute_score(sentence_pairs)

# # method 1: rerank passages
# rerank_results = model.rerank(query, passages)

# baichuan_model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
# baichuan_tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)

qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", use_fast=False, trust_remote_code=True)

