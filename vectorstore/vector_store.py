from langchain_community.vectorstores import FAISS
import os

def create_vector_store(split_docs, embedder, save_path="vector_db/"):
    """
    使用嵌入器构建向量数据库，并保存到本地。
    """
    db = FAISS.from_documents(split_docs, embedder)
    db.save_local(save_path)
    return db

def load_vector_store(embedder, save_path="vector_db/"):
    """
    加载已有向量数据库（用于检索）
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"{save_path} 不存在，请先构建数据库！")
    return FAISS.load_local(save_path, embedder)
