import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json, faiss
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
import numpy as np


class ChatRequest(BaseModel):
    message: str
    history: list[str] | None = None


def build_index(
    data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    index_save_path: str,
    json_save_path: str,
    embeded_model: SentenceTransformer,
) -> tuple:
    docs = SimpleDirectoryReader(input_dir=data_dir).load_data()
    nodes = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        include_metadata=True,
    ).get_nodes_from_documents(docs)

    chunks = []
    texts = []
    for idx, node in enumerate(nodes):
        file_name = node.metadata.get("file_name", "unknown")
        chunks.append({"idx": idx, "text": node.text, "file": file_name})
        texts.append(node.text)

    document_embeddings = embeded_model.encode(
        texts, normalize_embeddings=True, batch_size=32
    ).astype("float32")

    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    index.add(document_embeddings)

    # Save the index to disk
    with open(index_save_path, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())
    # Save the json to disk
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return index, chunks


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embeded_model: SentenceTransformer,
    top_k: int = 5,
) -> list[tuple[dict, float]]:
    # Encode the query
    query_embedding = embeded_model.encode(
        [f"为这个句子生成表示以用于检索相关文章：{query}"], normalize_embeddings=True
    ).astype("float32")

    # Search the index
    scores, indices = index.search(query_embedding, top_k)

    # Retrieve the top-k documents
    results = [(chunks[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

    return results


def answer(
    model: BaseChatModel,
    req: ChatRequest,
    index: faiss.Index,
    chunks: list[dict],
    embeded_model: SentenceTransformer,
    top_k: int = 5,
) -> dict:

    results_retrieved = retrieve(req.message, index, chunks, embeded_model, top_k)

    formatted_docs = []
    for i, (chunk, score) in enumerate(results_retrieved, start=1):
        formatted_docs.append(
            f"[Doc {i}] (来源: {chunk.get('file', '未知')})\n{chunk['text']}"
        )
    ref_docs_text = "\n\n".join(formatted_docs)

    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的问答助手。请根据下方提供的参考文档回答用户的问题。\n\n"
        "## 要求：\n"
        "1. 答案必须严格基于参考文档，不得捏造。\n"
        "2. 在引用具体内容时，请在句末标注对应的引用来源编号（例如 [Doc 1]）。\n"
        "3. 如果参考文档不足以回答问题，请直接说明“资料中未提及”。\n\n"
        "## 参考文档：\n"
        "{ref_docs}\n\n"
        "## 用户问题：\n"
        "{question}"
    )

    chain = prompt | model | StrOutputParser()

    return {
        "message": chain.invoke({"question": req.message, "ref_docs": ref_docs_text})
    }


if __name__ == "__main__":

    DATA_DIR = r"D:\Study\AgentGuide\learning\Daily\Day-03-RAG加载与分割\data"
    WORK_DIR = r"D:\Study\AgentGuide\learning\Daily\Day-05-手撕NaiveRAG系统"
    INDEX_PATH = os.path.join(WORK_DIR, "data.index")
    JSON_PATH = os.path.join(WORK_DIR, "data.json")

    embeded_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(JSON_PATH):
        print("索引不存在，正在构建...")
        index, chunks = build_index(
            data_dir=DATA_DIR,
            chunk_size=256,
            chunk_overlap=16,
            embeded_model=embeded_model,
            index_save_path=INDEX_PATH,
            json_save_path=JSON_PATH,
        )
    else:
        print("发现已有本地索引与元数据，跳过重新切片与编码，直接读取！")
        # 启动时常驻加载一次，避免循环中重复读盘
        index = faiss.deserialize_index(
            np.frombuffer(open(INDEX_PATH, "rb").read(), dtype="uint8")
        )
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    queries = [
        "Claude Code 的记忆是怎么分层实现的",  # 与原文用词高度重合
        "Claude Code 的记忆分层机制是怎么设计的",  # 换个说法
        "这套系统部署需要多大的 Docker 镜像",  # 文中根本没有
    ]

    model = init_chat_model(
        model="qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
        model_provider="openai",
        base_url="http://127.0.0.1:9527/v1",
        api_key="not-needed",
        temperature=0.3,
    )

    for q in queries:
        req = ChatRequest(message=q)
        message = answer(model, req, index, chunks, embeded_model, top_k=5)
        print(f"\n{'='*40}")
        print(f"Q: {q}")
        print(f"A:\n{message['message']}")
