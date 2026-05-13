# rag/indexer.py
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 本地 FAISS 索引保存目录（持久化到磁盘，便于下次直接加载）
INDEX_DIR = Path("indexes/faiss_index")


def get_embedding_model():
    # 多语言 embedding：把中文/日文/英文都映射到同一向量空间
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
## embedding 不行。。。

def load_or_create_index(docs: List[Document] | None = None) -> FAISS:
    # 创建 embedding 模型实例（用于向量化文档和查询）
    embed = get_embedding_model()

    # 如果索引目录存在：先加载已有索引
    if INDEX_DIR.exists():
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embed,
            allow_dangerous_deserialization=True,
        )

        # 若传入了新增文档，则追加入向量库并重新保存
        if docs:
            vectorstore.add_documents(docs)
            vectorstore.save_local(str(INDEX_DIR))
        return vectorstore
    else:
        # 索引不存在：必须提供初始文档来创建（否则没有数据可建索引）
        if not docs:
            raise ValueError("索引不存在，且未提供初始文档。")
        vectorstore = FAISS.from_documents(docs, embed)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        return vectorstore