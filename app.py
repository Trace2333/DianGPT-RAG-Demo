import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from database import create_and_add_ele, MyEmbeddingFunction, Query, load_pdf
from BCEmbedding import EmbeddingModel, RerankerModel
from flask import Flask, request, jsonify

app = Flask(__name__)
app.json.ensure_ascii = False

collection = create_and_add_ele()
# collection = create_and_add_ele(load_pdf("indoor.pdf"), adding=True)

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

@app.route('/chat', methods=['POST'])
def chat():
    history = None
    try:
        data = request.get_json(force=False)
        query = data.get('query', '')

        query_out, search_dis = querying.query_chroma(query)
        back_knowledge = query_out['rerank_passages'][0]

        # if sum(query_out['rerank_scores'][:2]) > 1.0:
        if search_dis < 1.0 or sum(query_out['rerank_scores'][:2]) > 1.0:
            query = "检索到的知识: " + back_knowledge + "回答：" + query
        else:
            query = query
        response, history = qwen_model.chat(qwen_tokenizer, query, history=None)

        # query = "根据检索到的设计知识: " + back_knowledge + "回答：" + query
        # response, history = qwen_model.chat(qwen_tokenizer, query, history=None)
        
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(port=33357, debug=False, host="0.0.0.0")
