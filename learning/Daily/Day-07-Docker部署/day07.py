import sys
from pathlib import Path
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_PATH = str(BASE_DIR / "data.index")
JSON_PATH = str(BASE_DIR / "data.json")
META_PATH = str(BASE_DIR / "meta.json")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:9527/v1")
EMBED_LOCAL_ONLY = os.getenv("EMBED_LOCAL_ONLY", "0") == "1"
CHUNK_SIZE, CHUNK_OVERLAP = 512, 128


class IngestResponse(BaseModel):
    built: bool = Field(..., description="是否调用embedded模型重建")
    chunks: int = Field(..., description="分块数量")
    seconds: float = Field(..., description="耗时/s", ge=0.0)


class AskRequest(BaseModel):
    message: str = Field(..., description="LLM的当前轮输入")


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


def get_files_mtime(data_dir: str) -> dict[str, float]:
    """获取 data 目录下所有文件的相对路径和修改时间字典 {相对路径: 修改时间戳}"""
    files = {}
    for p in Path(data_dir).glob("**/*"):
        if p.is_file():
            rel_path = str(p.relative_to(data_dir))
            files[rel_path] = p.stat().st_mtime
    return files


def is_index_fresh(
    meta_path: str, data_dir: str, chunk_size: int, chunk_overlap: int
) -> bool:
    """比对 meta.json 与当前参数及文件状态，判断索引是否最新"""
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 1. 校验分块参数
        if (
            meta.get("chunk_size") != chunk_size
            or meta.get("chunk_overlap") != chunk_overlap
        ):
            return False
        # 2. 校验文件修改时间
        if meta.get("files") != get_files_mtime(data_dir):
            return False
        return True
    except Exception:
        return False


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
    # 保存元数据指纹
    meta = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": get_files_mtime(data_dir),
        "n_chunks": len(chunks),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return index, chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is Starting Up...")

    print("[Init] LLModel is Loading...")
    model = init_chat_model(
        model="qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
        model_provider="openai",
        base_url=LLM_BASE_URL,
        api_key="not-needed",
        temperature=0.3,
    )
    print("[Init] LLModel Loaded.")

    print(f"[Init] Embeded Model is Loading (local_files_only={EMBED_LOCAL_ONLY})...")
    embeded_model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5", device="cpu", local_files_only=EMBED_LOCAL_ONLY
    )
    print("[Init] Embeded Model Loaded.")

    if (
        not os.path.exists(INDEX_PATH)
        or not os.path.exists(JSON_PATH)
        or not is_index_fresh(META_PATH, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    ):
        print("[Init] Data Index or Json Not Exists or Old, Rebuilding...")
        index, chunks = build_index(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
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
    print(f"[Init] Data Index and Json Loaded ({len(chunks)} chunks).")

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
    rebuilt = force or not is_index_fresh(META_PATH, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    if rebuilt:
        print("-- Data Index or Json Not Exists or Stale, Rebuilding...")
        index, chunks = build_index(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
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

    return {"built": rebuilt, "chunks": len(request.app.state.chunks), "seconds": t1 - t0}


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
                doc_id=chunk["idx"] + 1,
                file=chunk["file"],
                snippet=chunk["text"][:80],
                score=score,
            )
        )

    if not citations or citations[0].score <= hit_thresh:
        hit = False
        answer = "资料中未提及(No LLM)"
        # 保留 citations，便于调试与阈值校准观察真实得分
    else:
        hit = True
        formatted_docs = []
        for i, (chunk, score) in enumerate(results_retrieved, start=1):
            formatted_docs.append(
                f"[Doc {chunk.get('idx', '未知') + 1}] (来源: {chunk.get('file', '未知')})\n{chunk['text']}"
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
