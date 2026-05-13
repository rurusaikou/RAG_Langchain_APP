# RAG_Langchain_App

## 项目简介（中文）
这是一个使用 **Streamlit + LangChain + FAISS + Ollama** 的本地 RAG 问答 Demo。
支持两类知识来源：
- 上传 **PDF / TXT** 文件
- 提供 **URL**（每行一个），抓取网页内容作为知识

系统会把文档切分成 chunks，生成向量并构建/更新本地 FAISS 索引，然后用本地大模型根据检索到的片段生成回答。

## プロジェクト概要（日本語）
これは **Streamlit + LangChain + FAISS + Ollama** を使ったローカル RAG（検索拡張生成）デモです。
知識ソースは次の2種類に対応します：
- **PDF / TXT** のファイルをアップロード
- **URL**（1行に1つ）を入力して、Webページの内容を知識として取り込み

文書は chunks に分割され、ベクトル化してローカルの FAISS インデックスを作成/更新し、取得した断片に基づいてローカルLLMが回答します。


## 功能点
- PDF / TXT / Web URL -> 统一抽取为 `Document` chunks
- 本地保存 FAISS 索引：`indexes/faiss_index`
- 通过相似度阈值控制“检索片段是否足够相关”


## 运行方法（中文）
### 1. 准备
- 安装并启动 `Ollama`
- 拉取模型（与代码一致）：`qwen2.5:3b`
  - `ollama pull qwen2.5:3b`

### 2. 安装依赖
建议在项目虚拟环境中执行（注意 `unstructured[html]` 需要引号，避免 zsh 的通配符报错）：
```bash
python -m pip install -U \
  streamlit \
  langchain langchain-community langchain-text-splitters \
  langchain-huggingface langchain-ollama sentence-transformers \
  faiss-cpu \
  pypdf \
  "unstructured[html]"
```

### 3. 启动
```bash
streamlit run app.py
```

### 4. 使用
1. 左侧“知识库管理”上传 `pdf/txt`，或在 URL 输入框中粘贴链接（每行一个）
2. 点击“添加到知识库”构建/更新索引
3. 在“提问”输入问题，点击“发送”


## 実行手順（日本語）
### 1. 準備
- `Ollama` をインストールして起動
- コードと同じモデルを用意：`qwen2.5:3b`
  - `ollama pull qwen2.5:3b`

### 2. 依存関係のインストール
仮想環境で実行してください（`unstructured[html]` は zsh のグロブ回避のため引用符が必要です）：
```bash
python -m pip install -U \
  streamlit \
  langchain langchain-community langchain-text-splitters \
  langchain-huggingface langchain-ollama sentence-transformers \
  faiss-cpu \
  pypdf \
  "unstructured[html]"
```

### 3. 起動
```bash
streamlit run app.py
```

### 4. 使い方
1. 左の「知識ベース管理」で `pdf/txt` をアップロード、または URL を入力（1行に1つ）
2. 「知識ベースに追加」を押してインデックスを作成/更新
3. 「質問」に入力して「送信」を押す


## 注意 / 補足
- 本地索引は `indexes/faiss_index`，上传文件会先写入 `uploads/`。
- LLM 型号は `rag/chat.py` の `get_llm()` で配置（默认 `qwen2.5:3b`）。

## 模型使用与切换（Ollama）
### 中文
本项目使用的 LLM 由 `rag/chat.py` 的 `get_llm()` 决定（当前硬编码为 `qwen2.5:3b`）。

1. 拉取并确认模型可用
```bash
ollama pull qwen2.5:3b
ollama list
```
2. 如果要换成别的 Ollama 模型
- 打开 `rag/chat.py`
- 修改 `get_llm()` 里的 `model="qwen2.5:3b"` 为你自己的模型名（例如 `llama3:8b`）
- 重新启动 Streamlit

例如把下面这一行换掉即可：
```python
return ChatOllama(model="qwen2.5:3b", temperature=0)
```

### 日本語
このプロジェクトで使う LLM は `rag/chat.py` の `get_llm()` で決まっています（現在は `qwen2.5:3b` にハードコード）。

1. モデルを用意して確認
```bash
ollama pull qwen2.5:3b
ollama list
```
2. 別の Ollama モデルに変更したい場合
- `rag/chat.py` を開く
- `get_llm()` の `model="qwen2.5:3b"` を変更する（例: `llama3:8b`）
- Streamlit を再起動する

置き換える行はこの部分です：
```python
return ChatOllama(model="qwen2.5:3b", temperature=0)
```
