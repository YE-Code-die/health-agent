from embeddings.embedder import get_embedder
from utils.text_splitter import split_documents
from vectorstore.vector_store import create_vector_store
from data.load_data import load_medquad_xml  # ✅ 用新脚本

def run_pipeline():
    print("📥 从 XML 加载 MedQuAD 数据...")
    documents = load_medquad_xml("MedQuAD")

    print(f"✅ 加载完毕，共 {len(documents)} 条问答对")

    print("🔪 开始分块...")
    split_docs = split_documents(documents)

    print("🧠 加载嵌入模型...")
    embedder = get_embedder()

    print("📦 嵌入并存入向量数据库...")
    create_vector_store(split_docs, embedder)

    print("🎉 成功构建 MedQuAD 向量数据库！")

if __name__ == "__main__":
    run_pipeline()
