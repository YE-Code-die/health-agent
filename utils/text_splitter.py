from langchain.text_splitter import CharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    将长文档按字符进行切割，适合传入到向量数据库中。
    """
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
