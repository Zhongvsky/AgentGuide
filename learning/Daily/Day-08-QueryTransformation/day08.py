"""
Day 08: Advanced RAG - Query Transformation (Multi-Query & HyDE)
================================================================
本脚本为 Day 08 的实战代码骨架。
请根据其中的契约（Type Hints & Pydantic）与清晰的 # TODO 引导，
实现查询改写（Multi-Query）与假设性文档嵌入（HyDE），并通过 A/B 对照实验量化验证检索召回提升。
"""

import os
import sys
import json
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re

# 设置镜像加速与环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_PATH = str(BASE_DIR / "data.index")
JSON_PATH = str(BASE_DIR / "data.json")

# 本地模型配置（兼容 OpenAI 协议的本地端点）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:9527/v1")
CHUNK_SIZE, CHUNK_OVERLAP = 512, 128


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class RetrievalResult(BaseModel):
    """单个召回分块的结果契约"""

    chunk_id: int = Field(..., description="文本块 ID")
    file: str = Field(..., description="来源文件名")
    score: float = Field(..., description="余弦相似度得分 (0.0~1.0)")
    text: str = Field(..., description="文本内容")
    snippet: str = Field(..., description="前 80 字摘要")


class MultiQueryOutput(BaseModel):
    """Multi-Query 改写输出契约"""

    original_query: str
    expanded_queries: List[str] = Field(
        ..., description="改写扩展后的 Query 列表（通常 3~4 个）"
    )


class HyDEOutput(BaseModel):
    """HyDE 假想文档输出契约"""

    original_query: str
    hypothetical_document: str = Field(..., description="假想的回答/政策文档文段")


# ==========================================
# 2. 向量库索引脚手架 (无需重复实现，开箱即用)
# ==========================================


def get_or_build_index(
    data_dir: str, index_path: str, json_path: str, embed_model: SentenceTransformer
) -> tuple[faiss.Index, list[dict]]:
    """加载或构建本地 FAISS 索引"""
    if os.path.exists(index_path) and os.path.exists(json_path):
        print(f"[*] 发现已有索引，直接加载: {index_path}")
        index = faiss.read_index(index_path)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    print(f"[*] 未检测到索引，开始对 {data_dir} 进行文本解析与向量化...")
    docs = SimpleDirectoryReader(input_dir=data_dir).load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)

    chunks = []
    texts = []
    for i, node in enumerate(nodes):
        meta = node.metadata or {}
        chunks.append(
            {"id": i, "text": node.text, "file": meta.get("file_name", "unknown")}
        )
        texts.append(node.text)

    # 向量化并归一化 (Cosine Similarity via Inner Product)
    embeddings = embed_model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 写入文件
    faiss.write_index(index, index_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[+] 索引构建完成！共 {len(chunks)} 个分块。")
    return index, chunks


def naive_retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """基础 Naive 检索：直接计算 Query Embedding 并检索 Top-K"""
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        snippet = chunk["text"][:80].replace("\n", " ") + "..."
        results.append(
            RetrievalResult(
                chunk_id=chunk["id"],
                file=chunk["file"],
                score=float(score),
                text=chunk["text"],
                snippet=snippet,
            )
        )
    return results


# ==========================================
# 3. 核心任务手撕区：Query Transformation (闯关实战区)
# ==========================================


def build_multi_query_chain(llm):
    """
    【关卡 1】：构建 Multi-Query 改写链 (LCEL)

    目标：
    设计一个 Prompt，要求 LLM 将用户的原始问题从不同角度或词汇习惯扩展为 3 个不同的检索式 Query。

    实现提示：
    1. 提示词设计：要求输出 3 行改写后的查询，每行一条，严禁输出序号（如 1. 2.）或客套话；
    2. LCEL 组装：PromptTemplate | llm | StrOutputParser()
    """
    # TODO 1: 请在此处实现 Multi-Query 提示词与 LCEL 链

    PromptTemplate = ChatPromptTemplate.from_template(
        'You are a helpful assistant. Based on the original question: "{query}" \n'
        "output 3 rewritten queries, one per line, and do not include numbers (like 1. 2.) or polite phrases."
    )

    multi_query_chain = PromptTemplate | llm | StrOutputParser()

    return multi_query_chain


def generate_multi_queries(query: str, llm) -> MultiQueryOutput:
    """
    【关卡 2】：执行 Multi-Query 链并提取干净的 Query 列表

    实现提示：
    1. 调用 chain.invoke({"question": query}) 获得原始字符串；
    2. 按行切分 splitlines()，过滤空行并去除首尾空白；
    3. 清洗可能残留的序号前缀（如 "1. "、"- "）；
    4. 务必将原始 query 也保留在检索候选中！
    """
    # TODO 2: 请在此处实现调用与文本清洗提取逻辑

    multi_query_chain = build_multi_query_chain(llm)
    queries_raw = multi_query_chain.invoke({"query": query})

    # queries = [query]
    queries = []
    for line in queries_raw.splitlines():
        if line.strip():
            line = re.sub(r"^(?:\d+[.)、]\s*|[-*+]\s*)", "", line.strip())
            queries.append(line)

    return MultiQueryOutput(original_query=query, expanded_queries=queries)


def build_hyde_chain(llm):
    """
    【关卡 3】：构建 HyDE (Hypothetical Document Embeddings) 假设文档生成链

    目标：
    让模型针对问题“假想”一段权威、严谨的规范性陈述文档（100~150 字），击穿疑问句与陈述句在向量空间的鸿沟。

    实现提示：
    1. System Prompt：设定角色为制度专家，要求直接输出正式管理制度条文口吻的正文；
    2. LCEL 组装：prompt | llm | StrOutputParser()
    """
    # TODO 3: 请在此处实现 HyDE 假想文档生成链

    PromptTemplate = ChatPromptTemplate.from_template(
        "You are a professional administrative officer, skilled in drafting and formulating relevant regulations.\n"
        'Based on the user\'s question: "{query}", generate a speculative/hypothetical text in the tone of a formal management regulation, '
        "without any unnecessary extra words.\n"
        "Keep the same language as the user's question."
    )

    hyde_chain = PromptTemplate | llm | StrOutputParser()

    return hyde_chain


def generate_hyde_doc(query: str, llm) -> HyDEOutput:
    """
    【关卡 4】：执行 HyDE 链生成假想文档
    """
    # TODO 4: 请在此处实现 HyDE 假想文档调用逻辑

    hyde_chain = build_hyde_chain(llm)
    hyde_doc = hyde_chain.invoke({"query": query})

    return HyDEOutput(original_query=query, hypothetical_document=hyde_doc.strip())


def multi_query_retrieve(
    queries: List[str],
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    【关卡 5】：多路 Query 并行检索与去重合并 (Union & Max-Score Deduplication)

    实现提示：
    1. 对 queries 列表中的每个 query 进行向量化检索（可调用 embed_model.encode + index.search）；
    2. 聚合去重：如果同一个分块 (chunk_id) 被多个 query 召回，保留最高的相似度分数；
    3. 按最终分数降序排序，截取 Top-K 返回为 List[RetrievalResult]。
    """
    # TODO 5: 请在此处实现多 Query 检索合并与去重逻辑

    map_idx_result = {}
    for query in queries:
        q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_emb, top_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = chunks[idx]
            snippet = chunk["text"][:80].replace("\n", " ") + "..."
            if chunk["id"] not in map_idx_result:
                map_idx_result[chunk["id"]] = RetrievalResult(
                    chunk_id=chunk["id"],
                    file=chunk["file"],
                    score=float(score),
                    text=chunk["text"],
                    snippet=snippet,
                )
            elif score > map_idx_result[chunk["id"]].score:
                map_idx_result[chunk["id"]].score = float(score)

    map_idx_result_sorted = dict(
        sorted(map_idx_result.items(), key=lambda x: x[1].score, reverse=True)
    )

    results = []
    for result in map_idx_result_sorted.values():
        results.append(result)

    return results[:top_k]


def hyde_retrieve(
    hypo_doc: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    【关卡 6】：基于假想文档的 HyDE 检索

    实现提示：
    直接将假想文档 hypo_doc 作为检索文本输入，调用 naive_retrieve 获得检索结果。
    """
    # TODO 6: 请在此处实现 HyDE 检索逻辑

    return naive_retrieve(hypo_doc, index, chunks, embed_model, top_k)


# =======================================================================
# 4. 参考实现（折叠区）
# 规则：请务必先独立挑战上方 1~6 关卡，卡壳思考 20 分钟以上或全部写完后再展开对照！
# =======================================================================
# region 🙈 官方参考实现 (VS Code 点击左侧行号旁的下箭头可折叠/展开)
"""
def _ref_build_multi_query_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个专业的智能搜索助手。你的任务是将用户输入的问题，"
            "从不同角度、使用不同的行业术语或近义表达，重写为 3 个更适合向量数据库检索的查询语句。\n"
            "要求：\n"
            "1. 每行输出一个改写后的问题，共 3 行；\n"
            "2. 绝对不要包含序号（如 1.、2.）、前缀或任何多余解释；\n"
            "3. 保持原意不变，侧重补充关键术语与陈述式语义。"
        )),
        ("human", "原始问题：{question}")
    ])
    return prompt | llm | StrOutputParser()

def _ref_generate_multi_queries(query: str, llm) -> MultiQueryOutput:
    chain = _ref_build_multi_query_chain(llm)
    raw_output = chain.invoke({"question": query})
    import re
    expanded = []
    for line in raw_output.strip().splitlines():
        line = line.strip()
        cleaned = re.sub(r"^[0-9]+[\.\、\)\s\-]+", "", line).strip()
        if cleaned and cleaned not in expanded:
            expanded.append(cleaned)
    return MultiQueryOutput(original_query=query, expanded_queries=expanded)

def _ref_build_hyde_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一名严谨的政策合规专家。请针对用户提出的问题，写一段模拟的、正式规范的制度管理条例回答。\n"
            "要求：\n"
            "1. 使用严谨的公文/规章制度口吻（如'按照规定...'、'应当...'、'报销标准...'）；\n"
            "2. 篇幅在 100~150 字以内，直接输出条例正文，严禁任何客套话。"
        )),
        ("human", "问题：{question}")
    ])
    return prompt | llm | StrOutputParser()

def _ref_generate_hyde_doc(query: str, llm) -> HyDEOutput:
    chain = _ref_build_hyde_chain(llm)
    return HyDEOutput(original_query=query, hypothetical_document=chain.invoke({"question": query}).strip())

def _ref_multi_query_retrieve(queries, index, chunks, embed_model, top_k=3):
    q_embs = embed_model.encode(queries, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_embs, top_k)
    best_candidates = {}
    for q_idx in range(len(queries)):
        for score, c_idx in zip(scores[q_idx], indices[q_idx]):
            if c_idx == -1:
                continue
            s = float(score)
            if c_idx not in best_candidates or s > best_candidates[c_idx][0]:
                best_candidates[c_idx] = (s, chunks[c_idx])
    sorted_candidates = sorted(best_candidates.values(), key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in sorted_candidates[:top_k]:
        snippet = chunk["text"][:80].replace("\\n", " ") + "..."
        results.append(RetrievalResult(
            chunk_id=chunk["id"], file=chunk["file"], score=score, text=chunk["text"], snippet=snippet
        ))
    return results

def _ref_hyde_retrieve(hypo_doc, index, chunks, embed_model, top_k=3):
    return naive_retrieve(hypo_doc, index, chunks, embed_model, top_k=top_k)
"""
# endregion


# ==========================================
# 4. A/B 对照实验评测套件
# ==========================================


def run_ab_comparison(
    test_queries: List[str],
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    llm,
):
    """运行 A/B 对照评测，横向对比 Naive vs Multi-Query vs HyDE"""
    print("\n" + "=" * 75)
    print("           Advanced RAG: Query Transformation A/B 评测实验           ")
    print("=" * 75)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 测试原始 Query: 「{q}」")
        print("-" * 75)

        # 1. Naive RAG
        t0 = time.perf_counter()
        naive_res = naive_retrieve(q, index, chunks, embed_model, top_k=3)
        t_naive = (time.perf_counter() - t0) * 1000

        # 2. Multi-Query
        t0 = time.perf_counter()
        mq_out = generate_multi_queries(q, llm)
        all_queries = [q] + mq_out.expanded_queries
        mq_res = multi_query_retrieve(all_queries, index, chunks, embed_model, top_k=3)
        t_mq = (time.perf_counter() - t0) * 1000

        # 3. HyDE
        t0 = time.perf_counter()
        hyde_out = generate_hyde_doc(q, llm)
        hyde_res = hyde_retrieve(
            hyde_out.hypothetical_document, index, chunks, embed_model, top_k=3
        )
        t_hyde = (time.perf_counter() - t0) * 1000

        # 打印改写生成细节
        print(f"  [Multi-Query 衍生 ({len(mq_out.expanded_queries)} 条)]:")
        for eq in mq_out.expanded_queries:
            print(f"    -> {eq}")
        print(f"  [HyDE 假想文档摘要]: {hyde_out.hypothetical_document[:90]}...")
        print()

        # 打印横向对比表
        print(
            f"  {'策略':<12} | {'Top-1 分数':<10} | {'耗时 (ms)':<10} | {'Top-1 命中片段'}"
        )
        print("  " + "-" * 70)

        n_top = naive_res[0] if naive_res else None
        mq_top = mq_res[0] if mq_res else None
        h_top = hyde_res[0] if hyde_res else None

        if n_top:
            print(
                f"  {'Naive':<12} | {n_top.score:<10.4f} | {t_naive:<10.1f} | {n_top.snippet[:30]}..."
            )
        if mq_top:
            print(
                f"  {'Multi-Query':<12} | {mq_top.score:<10.4f} | {t_mq:<10.1f} | {mq_top.snippet[:30]}..."
            )
        if h_top:
            print(
                f"  {'HyDE':<12} | {h_top.score:<10.4f} | {t_hyde:<10.1f} | {h_top.snippet[:30]}..."
            )

    print("\n" + "=" * 75)
    print("[+] A/B 评测完成！观察不同场景下 Multi-Query 与 HyDE 的召回能力与延迟表现。")


# ==========================================
# 5. 主入口
# ==========================================

if __name__ == "__main__":
    print("[*] 正在初始化 Embedding 模型 (BAAI/bge-small-zh-v1.5)...")
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

    print(f"[*] 正在连接本地 LLM 端点 ({LLM_BASE_URL})...")
    llm = init_chat_model(
        model="Qwen/Qwen2.5-7B-Instruct",  # 或本地实际加载的模型名称
        model_provider="openai",
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.7,
    )

    # 1. 准备/加载向量索引
    index, chunks = get_or_build_index(
        data_dir=DATA_DIR,
        index_path=INDEX_PATH,
        json_path=JSON_PATH,
        embed_model=embed_model,
    )

    # 2. 定义测试用例（覆盖典型的口语化与语义鸿沟场景）
    test_queries = [
        "普通员工出差去北京，坐高铁能买一等座吗？",
        "去外地出差，每天伙食费能给补贴多少钱？",
    ]

    # 3. 运行评测
    run_ab_comparison(test_queries, index, chunks, embed_model, llm)
