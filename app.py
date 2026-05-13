# app.py
# 这是 Streamlit 前端入口（RAG 搜索 + 问答）。 
import os
from pathlib import Path

import streamlit as st

from rag.loader import load_pdf, load_text, load_urls
from rag.indexer import load_or_create_index
from rag.chat import answer_with_rag


# 页面配置：标题/图标/布局
st.set_page_config(page_title="RAG LangChain App", page_icon="📚", layout="wide")

# 页面标题
st.title("📚 本地 RAG 问答 Demo (PDF / Web / 多文档)")


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """
    启动时尝试加载本地 FAISS 索引。

    如果索引不存在（且调用方未提供初始文档），则返回 `None`，让用户先在侧边栏上传/添加 URL。
    """
    try:
        return load_or_create_index(docs=None)
    except Exception:
        return None


# 初次渲染时的向量库（可能为 None）
vectorstore = get_vectorstore()

# 左侧：知识库管理区（上传文件 / 添加 URL / 更新向量索引）
st.sidebar.header("📥 知识库管理")

# 上传文件（PDF / TXT），用户点击“添加到知识库”后会被写入 `uploads/` 并加载为文档块
uploaded_files = st.sidebar.file_uploader(
    "上传 PDF 或 文本文件",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

# 输入 URL（每行一个），点击“添加到知识库”后会被加载为网页文档块
url_input = st.sidebar.text_area(
    "输入要加入知识库的 URL（每行一个）",
    height=100,
)

# 点击后：把上传文件/URL 转为文档块（chunks），再写入/更新 FAISS 索引
if st.sidebar.button("添加到知识库"):
    new_docs = []
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    # 保存上传到 `uploads/`，并根据文件类型用 loader 切分为文档块（chunks）
    for f in uploaded_files or []:
        save_path = upload_dir / f.name
        with open(save_path, "wb") as out:
            out.write(f.read())
        if f.name.lower().endswith(".pdf"):
            new_docs.extend(load_pdf(save_path))
        else:
            new_docs.extend(load_text(save_path))

    # 加载 URL：抓取网页内容，并切分为文档块（chunks）
    urls = [u.strip() for u in (url_input or "").splitlines() if u.strip()]
    if urls:
        new_docs.extend(load_urls(urls))

    # 没有可用文档时给出提示；否则更新/创建向量索引
    if not new_docs:
        st.sidebar.warning("没有有效的文档或 URL。")
    else:
        with st.spinner("正在更新向量索引…"):
            # 将新文档写入（追加）本地 FAISS 索引，并返回向量库实例
            vectorstore = load_or_create_index(new_docs)
        st.sidebar.success(f"已加入文档片段：{len(new_docs)} 条。")


# ----------------------------
# 主区域：提问与生成回答
# ----------------------------
st.divider()
st.subheader("💬 提问")

# 输入用户问题（支持中文/日文）
question = st.text_input("请输入你的问题（支持中文、日文）：", "")

# 点击后：检索相关片段并生成最终回答
if st.button("发送") and question.strip():
    if vectorstore is None:
        # 索引尚未创建/加载成功，无法进行 RAG 检索
        st.error("当前还没有可用的索引，请先在左侧上传文档或添加 URL。")
    else:
        with st.spinner("生成回答中…"):
            answer = answer_with_rag(vectorstore, question.strip())
        st.markdown("### 回答")
        st.write(answer)