# rag/loader.py
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 文档切分器：把长文本拆成更小的 chunks（用于向量化检索）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


def load_pdf(path: Path) -> List[Document]:
    # 读取 PDF 文档（按页面/段落组织后再切分）
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        # 为每个文档块补充来源信息，便于回答时做溯源/标注
        d.metadata["source_type"] = "pdf"
        d.metadata["source"] = str(path.name)
    return text_splitter.split_documents(docs)


def load_text(path: Path) -> List[Document]:
    # 读取纯文本文件（UTF-8）
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        # 同样写入来源元数据
        d.metadata["source_type"] = "text"
        d.metadata["source"] = str(path.name)
    return text_splitter.split_documents(docs)


def load_urls(urls: list[str]) -> List[Document]:
    # 过滤空行/空 URL
    urls = [u for u in urls if u.strip()]
    if not urls:
        return []

    # 使用 UnstructuredURLLoader 抓取网页并抽取文本
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    for d in docs:
        # 标注来源类型为 web；尽量保留 loader 给出的 source
        d.metadata["source_type"] = "web"
        d.metadata["source"] = d.metadata.get("source", "")
    return text_splitter.split_documents(docs)