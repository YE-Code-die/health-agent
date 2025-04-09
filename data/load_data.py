import os
import xml.etree.ElementTree as ET
from langchain.docstore.document import Document

def load_medquad_xml(root_path="MedQuAD"):
    documents = []
    xml_count = 0
    qa_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".xml"):
                xml_count += 1
                full_path = os.path.join(dirpath, file)
                try:
                    tree = ET.parse(full_path)
                    root = tree.getroot()

                    # 查找所有 QAPair 节点
                    for qapair in root.findall(".//QAPair"):
                        question = qapair.findtext("Question")
                        answer = qapair.findtext("Answer")
                        if question and answer:
                            text = f"Q: {question.strip()}\nA: {answer.strip()}"
                            documents.append(Document(page_content=text, metadata={"source": file}))
                            qa_count += 1
                except Exception as e:
                    print(f"⚠️ 解析失败: {full_path} | 原因: {e}")

    print(f"🔍 共扫描到 {xml_count} 个 XML 文件，成功提取 {qa_count} 条问答对。")
    return documents