import chromadb
from tqdm import tqdm
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from BCEmbedding import EmbeddingModel, RerankerModel


embedding_model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1", device_map="cuda:1")
# rank_model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1", device="cuda:0")

def create_and_add_ele(documents=None, adding=False):
    client = chromadb.HttpClient(host='localhost', port=8000)
    # my_collection for diangpt
    collection = client.get_or_create_collection(name="my_collection", embedding_function=MyEmbeddingFunction())
    
    # interior_design for prompt generator
    # collection = client.get_or_create_collection(name="interior_design", embedding_function=MyEmbeddingFunction())
    
    if adding and documents is not None:
        documents_content = [doc.page_content.replace('\n', '').replace('\r', '').replace('\t', '') 
        for doc in documents]


        collection.add(
            documents=documents_content,
            metadatas=[doc.metadata for doc in documents],
            ids=[str(i) for i in range(len(documents))],
        )
    
    return collection

def load_docx(path_for_docx):
    loader = Docx2txtLoader(path_for_docx)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap  = 128,
    )
    documents = text_splitter.split_documents(documents)
    return documents

def load_pdf(path_for_pdf):
    loader = PyPDFLoader(path_for_pdf)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap  = 128,
    )
    documents = text_splitter.split_documents(documents)
    return documents

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = [embedding_model.encode(x).tolist()[0] for x in tqdm(input)]
        return embeddings
    

class Query():
    """访问对象"""
    def __init__(self, collection, rank_model, my_embed, ) -> None:
        self.rank_model = rank_model
        self.embed_func = my_embed
        self.collection = collection
        
    def query_chroma(self, query):
        out = self.collection.query(
            query_embeddings=self.embed_func([query]),
            n_results=16,
        )

        passages = out['documents'][0]
        min_search_distance = out['distances'][0][0]
        sentence_pairs = [[query, passage] for passage in passages]

        scores = self.rank_model.compute_score(sentence_pairs)

        rerank_results = self.rank_model.rerank(query, passages)

        return rerank_results, min_search_distance