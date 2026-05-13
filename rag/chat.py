# rag/chat.py
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

# 当最小的向量距离（distance）小于等于该阈值时，认为检索片段“足够相关”
RETRIEVAL_MAX_DISTANCE = 1

SYSTEM_PROMPT = (
    "你是助手。下面可能会提供一些文档片段，它们有可能与问题无关。\n"
    "规则：\n"
    "1) 若片段内容与问题直接相关，优先依据片段回答，并可说明信息来自这些文档。\n"
    "2) 若片段不相关或信息不足，请用你的通用知识正常回答；"
    "不要仅以「文档未提及」为由拒绝回答一般性问题。"
)


def get_llm() -> ChatOllama:
    # 使用 Ollama 本地模型（qwen2.5:3b 可在 Ollama 中自行拉取）
    return ChatOllama(model="qwen2.5:3b", temperature=0)


def answer_with_rag(vectorstore: FAISS, question: str) -> str:
    # 1) 初始化 LLM
    llm = get_llm()

    # 2) 组装向量检索查询：这里用一个固定前缀（兼容部分 embedding 的习惯用法）
    query = f"query: {question}"

    # 3) 在向量库中做相似度检索（带 score/distance）
    scored = vectorstore.similarity_search_with_score(query, k=4)

    if not scored:
        context = ""
    else:
        # 以检索结果的最小距离来判断“是否值得把片段塞给模型”
        min_distance = min(score for _, score in scored)
        print(min_distance)
        if min_distance <= RETRIEVAL_MAX_DISTANCE:
            parts: List[str] = []
            for doc, score in scored:
                # 从 metadata 中取出来源信息（如：pdf/text/web + 文件名/URL）
                src = doc.metadata.get("source", "")
                stype = doc.metadata.get("source_type", "")
                head = f"[{stype}] {src} (score={score:.3f})"
                parts.append(head + "\n" + doc.page_content.strip())
            context = "\n\n---\n\n".join(parts)
            print(context)
        else:
            context = ""

    if context:
        # 把检索到的片段和用户问题一起喂给模型
        user_block = f"文档片段：\n{context}\n\n用户问题：{question}"
    else:
        # 如果没有足够相关片段，提示模型“本次检索不够相关”，但仍允许正常回答
        user_block = f"（本次检索无足够相关内容）\n\n用户问题：{question}"

    # 4) 通过 system/human prompt 调用模型
    msg = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_block)])
    return msg.content