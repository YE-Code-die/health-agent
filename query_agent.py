from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # or other LLM
from langchain_core.documents import Document
from langchain_community.llms import LlamaCpp  # For local models
import requests  # For Ollama API
from langchain_community.llms import HuggingFacePipeline  # For local HuggingFace models
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os

def load_vectorstore():
    # 假设你之前已保存到 local_db 文件夹
    return FAISS.load_local(
        folder_path="vector_db",
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True  # ✅ 显式允许 pickle 加载
    )

class OllamaAPI:
    """Simple wrapper for Ollama API"""
    def __init__(self, model="llama2", temperature=0.2):
        self.model = model
        self.temperature = temperature
        self.api_url = "http://localhost:11434/api/generate"
    
    def __call__(self, prompt):
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature
            }
        )
        return response.json()["response"]

def get_local_llm(model_type="huggingface", model_path=None):
    """
    获取本地部署的LLM模型
    
    参数:
        model_type: 模型类型，可选 "llama", "ollama", "huggingface"
        model_path: 模型路径，对于本地模型需要指定
    """
    if model_type == "llama":
        # 使用LlamaCpp加载本地模型
        return LlamaCpp(
            model_path=model_path,  # 例如: "models/deepseek-llm-7b-chat.gguf"
            temperature=0.2,
            max_tokens=512,
            n_ctx=2048,
            verbose=True
        )
    elif model_type == "ollama":
        # 使用Ollama API
        return OllamaAPI(model="llama2", temperature=0.2)
    elif model_type == "huggingface":
        # 使用HuggingFace本地模型
        model_id = model_path or "google/flan-t5-base"  # 使用更小的模型
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 创建模型缓存目录
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=cache_dir  # 指定权重卸载文件夹
        )
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.2
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        # 默认使用HuggingFaceHub
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.2, "max_length": 512}
        )

def ask_question(query: str, model_type="huggingface", model_path=None):
    db = load_vectorstore()

    # 获取本地LLM
    llm = get_local_llm(model_type=model_type, model_path=model_path)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    answer = qa_chain.invoke(query)["result"]  # 使用invoke替代run
    return answer

if __name__ == "__main__":
    # 设置模型类型和路径
    MODEL_TYPE = "huggingface"  # 改为huggingface，因为Ollama需要先拉取模型
    MODEL_PATH = "google/flan-t5-base"  # 使用更小的模型
    
    print(f"使用模型类型: {MODEL_TYPE}")
    if MODEL_PATH:
        print(f"模型路径: {MODEL_PATH}")
    
    while True:
        question = input("❓ 请输入你的问题（输入 exit 退出）：\n> ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = ask_question(question, model_type=MODEL_TYPE, model_path=MODEL_PATH)
        print(f"\n💬 回答：{answer}\n")
