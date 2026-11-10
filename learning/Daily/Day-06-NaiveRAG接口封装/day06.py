import sys
from pathlib import Path

DAY05_DIR = Path(__file__).resolve().parent.parent / "Day-05-手撕NaiveRAG系统"
sys.path.append(str(DAY05_DIR))

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import json, faiss
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR = r"D:\Study\AgentGuide\learning\Daily\Day-06-NaiveRAG接口封装\data"
INDEX_PATH = r"D:\Study\AgentGuide\learning\Daily\Day-06-NaiveRAG接口封装\data.index"
JSON_PATH = r"D:\Study\AgentGuide\learning\Daily\Day-06-NaiveRAG接口封装\data.json"


class IngestResponse(BaseModel):
    built: bool = Field(..., description="是否调用embedded模型重建")
    chunks: int = Field(..., description="分块数量")
    seconds: float = Field(..., description="耗时/s", ge=0.0)


class AskRequest(BaseModel):
    message: str = Field(..., description="LLM的当前轮输入")
    # history: list[str] | None = Field(default_factory=list, description="LLM的历史记录")


class Citation(BaseModel):
    doc_id: int = Field(..., description="引用文档的序号从1开始")
    file: str = Field(..., description="命中文档的文件名")
    score: float = Field(..., description="余弦相似度分数", ge=0.0, le=1.0)
    snippet: str = Field(..., description="命中块的前80字摘要")


class AskResponse(BaseModel):
    answer: str = Field(..., description="LLM 生成的最终回答")
    citations: list[Citation] = Field(
        default_factory=list, description="检索命中的引用凭据列表"
    )
    hit: bool = Field(..., description="是否成功命中知识库（是否过阈值）")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is Starting Up...")

    print("[Init] LLModel is Loading...")
    model = init_chat_model(
        model="qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
        model_provider="openai",
        base_url="http://127.0.0.1:9527/v1",
        api_key="not-needed",
        temperature=0.3,
    )
    print("[Init] LLModel Loaded.")

    print("[Init] Embeded Model is Loading...")
    embeded_model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5", device="cpu", local_files_only=True
    )
    print("[Init] Embeded Model Loaded.")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(JSON_PATH):
        print("[Init] Data Index or Json Not Exists, Rebuilding...")
        index, chunks = build_index(
            data_dir=DATA_DIR,
            chunk_size=512,
            chunk_overlap=128,
            index_save_path=INDEX_PATH,
            json_save_path=JSON_PATH,
            embeded_model=embeded_model,
        )
    else:
        print("[Init] Data Index and Json Exist, Loading...")
        # 启动时常驻加载一次，避免循环中重复读盘
        index = faiss.deserialize_index(
            np.frombuffer(open(INDEX_PATH, "rb").read(), dtype="uint8")
        )
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    print("[Init] Data Index and Json Loaded.")

    # yield {
    #     "model": model,
    #     "embeded_model": embeded_model,
    #     "index": index,
    #     "chunks": chunks,
    # }

    app.state.model = model
    app.state.embeded_model = embeded_model
    app.state.index = index
    app.state.chunks = chunks

    yield

    print("App Closed")


app = FastAPI(lifespan=lifespan)


@app.post("/ingest")
def ingest(
    request: Request,
    force: bool = False,
) -> IngestResponse:

    t0 = time.time()

    if force:
        print("-- Data Index or Json Not Exists, Rebuilding...")
        index, chunks = build_index(
            data_dir=DATA_DIR,
            chunk_size=64,
            chunk_overlap=32,
            index_save_path=INDEX_PATH,
            json_save_path=JSON_PATH,
            embeded_model=request.app.state.embeded_model,
        )
        request.app.state.index = index
        request.app.state.chunks = chunks
    else:
        print("-- Data Index and Json Exist, ReLoading...")
        request.app.state.index = faiss.deserialize_index(
            np.frombuffer(open(INDEX_PATH, "rb").read(), dtype="uint8")
        )
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            request.app.state.chunks = json.load(f)
    print("-- Data Index and Json Loaded.")

    t1 = time.time()

    return {"built": force, "chunks": len(chunks), "seconds": t1 - t0}


@app.post("/ask")
def ask(
    req: AskRequest, request: Request, top_k: int = 5, hit_thresh: float = 0.5
) -> AskResponse:

    # === Retrieve ===
    query_embedding = request.app.state.embeded_model.encode(
        [f"为这个句子生成表示以用于检索相关文章：{req.message}"],
        normalize_embeddings=True,
    ).astype("float32")
    scores, indices = request.app.state.index.search(query_embedding, top_k)
    results_retrieved = []
    for j, i in enumerate(indices[0]):
        if i < 0 or i >= len(request.app.state.chunks):
            continue
        results_retrieved.append((request.app.state.chunks[i], float(scores[0][j])))

    citations = list()
    for i, (chunk, score) in enumerate(results_retrieved, start=1):
        citations.append(
            Citation(
                doc_id=chunk["idx"],
                file=chunk["file"],
                snippet=chunk["text"][:80],
                score=score,
            )
        )

    if citations[0].score <= hit_thresh:
        hit = False
        prompt = ChatPromptTemplate.from_template(
            "你是一个专业的问答助手。请回答用户的问题。\n\n"
            "## 用户问题：\n"
            "{question}"
        )
        answer = "资料中未提及(No LLM)"
        citations = []
    else:
        hit = True
        formatted_docs = []
        for i, (chunk, score) in enumerate(results_retrieved, start=1):
            formatted_docs.append(
                f"[Doc {chunk.get('idx', '未知')}] (来源: {chunk.get('file', '未知')})\n{chunk['text']}"
            )
        ref_docs_text = "\n\n".join(formatted_docs)
        prompt = ChatPromptTemplate.from_template(
            "你是一个专业的问答助手。请根据下方提供的参考文档回答用户的问题。\n\n"
            "## 要求：\n"
            "1. 答案必须严格基于参考文档，不得捏造。\n"
            "2. 在引用具体内容时，请在句末标注对应的引用来源。\n"
            "3. 如果参考文档不足以回答问题时，请直接说明“资料中未提及”。\n\n"
            "## 参考文档\n"
            "{ref_docs}\n\n"
            "## 用户问题：\n"
            "{question}"
        )
        chain = prompt | request.app.state.model | StrOutputParser()
        answer = chain.invoke({"question": req.message, "ref_docs": ref_docs_text})

    return {
        "answer": answer,
        "citations": citations,
        "hit": hit,
    }
