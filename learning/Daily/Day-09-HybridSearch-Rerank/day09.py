"""
Day 09: Advanced RAG - 混合检索与重排 (Hybrid Search & Reranking)
================================================================
本脚本为 Day 09 的实战闯关代码骨架。
请根据其中的契约定义（Type Hints & Pydantic）与关卡提示（Docstring），
逐步实现 BM25 稀疏检索、RRF 倒数排名融合算法与 Cross-Encoder 重排精排器。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np

# Windows 控制台 UTF-8 防御（避免 GBK 编码崩溃）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 镜像加速与依赖库
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import jieba
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_PATH = str(BASE_DIR / "data.index")
JSON_PATH = str(BASE_DIR / "data.json")

CHUNK_SIZE, CHUNK_OVERLAP = 512, 128


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class RetrievalResult(BaseModel):
    """单个召回/重排分块的结果契约"""

    chunk_id: int = Field(..., description="文本块全局唯一 ID")
    file: str = Field(..., description="来源文件名")
    score: float = Field(..., description="相关性得分（不同阶段的分数含义不同）")
    text: str = Field(..., description="文本完整内容")
    snippet: str = Field(..., description="前 80 字摘要")


# ==========================================
# 2. 向量库与 Dense 检索脚手架 (开箱即用)
# ==========================================


def get_or_build_index(
    data_dir: str, index_path: str, json_path: str, embed_model: SentenceTransformer
) -> tuple[faiss.Index, list[dict]]:
    """加载或构建本地 FAISS 稠密向量索引"""
    if os.path.exists(index_path) and os.path.exists(json_path):
        print(f"[*] 发现已有向量索引，直接加载: {index_path}")
        index = faiss.read_index(index_path)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    print(f"[*] 未检测到向量索引，开始对 {data_dir} 进行切块向量化...")
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

    # 向量化并归一化 (Inner Product 等价于余弦相似度)
    embeddings = embed_model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[+] 向量索引构建完成！共 {len(chunks)} 个分块。")
    return index, chunks


def dense_retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = 5,
) -> List[RetrievalResult]:
    """Dense 语义检索：基于 Embedding 与 FAISS 的内积检索"""
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
# 3. 核心任务手撕区：混合检索与重排 (闯关打靶区)
# ==========================================


def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[list[str]]]:
    """
    【关卡 1】：构建 BM25 词频倒排索引

    目标：
    使用 jieba 对所有文本块进行中文分词，构建并返回 BM25Okapi 索引对象与分词后语料。

    实现提示：
    1. 遍历 chunks，提取每个 chunk["text"]；
    2. 使用 jieba.lcut(text) 进行精准分词，得到 tokenized_corpus (List[List[str]])；
    3. 实例化 BM25Okapi(tokenized_corpus)；
    4. 返回 (bm25_index, tokenized_corpus)。
    """
    # TODO 1: 请实现中文分词与 BM25 索引构建逻辑
    tokenized_corpus = []
    for chunk in chunks:
        text = chunk["text"]
        tokenized_corpus.append(jieba.lcut(text))

    bm25_index = BM25Okapi(tokenized_corpus)

    return bm25_index, tokenized_corpus


def bm25_retrieve(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 5,
) -> List[RetrievalResult]:
    """
    【关卡 2】：执行 BM25 稀疏关键词检索

    目标：
    对用户 Query 进行分词，计算各分块的 BM25 相关性得分，返回 Top-K 检索结果。

    实现提示：
    1. 使用 jieba.lcut(query) 对用户查询进行分词；
    2. 调用 bm25.get_scores(tokenized_query) 获取所有分块的原始得分数组（numpy array 或 list）；
    3. 获取得分最高的前 top_k 个文档的下标（提示：使用 np.argsort(scores)[::-1][:top_k]）；
    4. 将命中的分块包装为 RetrievalResult 对象列表并返回。
    """
    # TODO 2: 请实现 BM25 查询打分与 Top-K 提取逻辑
    tokens_query = jieba.lcut(query)
    scores = bm25.get_scores(tokens_query)
    ids = np.argsort(scores)[::-1][:top_k]

    results = []
    for id in ids:
        result = RetrievalResult(
            chunk_id=id,
            file=chunks[id]["file"],
            score=scores[id],
            text=chunks[id]["text"],
            snippet=chunks[id]["text"][:80],
        )
        results.append(result)

    return results


def rrf_fusion(
    rankings_list: List[List[RetrievalResult]],
    k: int = 60,
    top_k: int = 5,
) -> List[RetrievalResult]:
    """
    【关卡 3】：RRF (Reciprocal Rank Fusion，倒数排名融合) 算法

    目标：
    将多路检索结果（如 Sparse BM25 与 Dense Vector）基于排名倒数进行无参数加权合并。

    数学公式：
        RRF_Score(d) = sum_{m in M} (1.0 / (k + rank_m(d)))
    其中：
        - k 为平滑常数（工业界经典默认值通常为 60）；
        - rank_m(d) 为文档 d 在第 m 路检索系统中的名次（从 1 开始计数，1st 名次即为 1）。

    实现提示：
    1. 使用字典 rrf_scores = {} 记录各 chunk_id 的累加得分；
    2. 同时使用 doc_map = {} 记录 chunk_id 对应的原始文档对象（用于后续重构结果）；
    3. 遍历每路结果列表 ranking，对于其中的第 rank 个元素（rank 从 1 开始）：
       - chunk_id = item.chunk_id
       - rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
    4. 按累加后的 RRF 得分由高到低（降序）排序；
    5. 重构 RetrievalResult 列表（注意将 score 更新为计算出的 RRF 得分），截取前 top_k 项返回。
    """
    # TODO 3: 请实现工业级 RRF 倒数排名融合算法
    rrf_scores = {}
    doc_map = {}
    for rankings in rankings_list:
        for rank, retrive_result in enumerate(rankings, start=1):
            chunk_id = retrive_result.chunk_id
            if rrf_scores.get(chunk_id):
                rrf_scores[chunk_id] += 1 / (k + rank)
            else:
                rrf_scores[chunk_id] = 1 / (k + rank)
            doc_map[chunk_id] = retrive_result
    rrf_scores_sorted = dict(
        sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    )

    results = []
    for chunk_id, rrf_score in list(rrf_scores_sorted.items())[:top_k]:
        item = doc_map[chunk_id]
        result = item.model_copy(update={"score": rrf_score})
        results.append(result)

    return results


def cross_encoder_rerank(
    query: str,
    candidates: List[RetrievalResult],
    reranker: Any,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    【关卡 4】：Cross-Encoder 交叉编码器精排重排

    目标：
    对粗排阶段输出的候选文档集（candidates），利用单塔 Cross-Encoder 计算
    全注意力交互的相关性得分，进行细粒度重排序。

    实现提示：
    1. 如果 candidates 为空，直接返回空列表；
    2. 构造文本对列表 pairs: [[query, cand.text] for cand in candidates]；
    3. 调用 reranker.predict(pairs) 得到模型预测分数列表；
    4. 将得分更新至各候选文档，按得分降序排序，截取最终 Top-K 返回。
    """
    # TODO 4: 请实现 Cross-Encoder 精排重排逻辑
    if not candidates:
        return []

    pairs = []
    for candidate in candidates:
        pairs.append([query, candidate.text])

    scores = reranker.predict(pairs)

    reranked = [
        cand.model_copy(update={"score": float(score)})
        for cand, score in zip(candidates, scores)
    ]
    candidates_sorted = sorted(reranked, key=lambda x: x.score, reverse=True)

    return candidates_sorted[:top_k]


# ==========================================
# 4. A/B 评测与横向对比流水线 (开箱即用)
# ==========================================


def run_ab_comparison(
    test_queries: List[str],
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    bm25: BM25Okapi,
    reranker: Any,
):
    """
    横向对比 4 种检索策略的精度与排序效果：
    1. 纯 Dense (向量语义检索)
    2. 纯 BM25 (稀疏关键词检索)
    3. Hybrid (RRF 倒数排名融合，Top-10 候选池)
    4. Hybrid + Rerank (双路粗排召回 + Cross-Encoder 终极精排)
    """
    print("\n" + "=" * 80)
    print("[*] 开始执行 Advanced RAG 混合检索与重排 A/B 对照实验")
    print("=" * 80)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 测试 Query: 「{q}」")
        print("-" * 80)

        # 1. 纯 Dense 检索
        t0 = time.perf_counter()
        dense_res = dense_retrieve(q, index, chunks, embed_model, top_k=5)
        t_dense = (time.perf_counter() - t0) * 1000

        # 2. 纯 BM25 检索
        t0 = time.perf_counter()
        bm25_res = bm25_retrieve(q, bm25, chunks, top_k=5)
        t_bm25 = (time.perf_counter() - t0) * 1000

        # 3. Hybrid (RRF 融合)
        t0 = time.perf_counter()
        # 粗排双路各取 Top-10 进候选池
        dense_candidates = dense_retrieve(q, index, chunks, embed_model, top_k=10)
        bm25_candidates = bm25_retrieve(q, bm25, chunks, top_k=10)
        hybrid_res = rrf_fusion([dense_candidates, bm25_candidates], k=60, top_k=10)
        t_hybrid = (time.perf_counter() - t0) * 1000

        # 4. Hybrid + Reranker 精排
        t0 = time.perf_counter()
        # 将 RRF 融合出的 Top-10 候选送入 Cross-Encoder 进行终极精排，取 Top-3
        reranked_res = cross_encoder_rerank(
            q, candidates=hybrid_res, reranker=reranker, top_k=3
        )
        t_rerank = (time.perf_counter() - t0) * 1000

        # 打印横向对比表
        print(
            f"  {'策略':<18} | {'Top-1 分数':<12} | {'耗时 (ms)':<10} | {'Top-1 命中片段摘要'}"
        )
        print("  " + "-" * 76)

        d_top = dense_res[0] if dense_res else None
        b_top = bm25_res[0] if bm25_res else None
        h_top = hybrid_res[0] if hybrid_res else None
        r_top = reranked_res[0] if reranked_res else None

        if d_top:
            print(
                f"  {'1. Pure Dense':<18} | {d_top.score:<12.4f} | {t_dense:<10.2f} | [Chunk {d_top.chunk_id}] {d_top.snippet[:28]}..."
            )
        if b_top:
            print(
                f"  {'2. Pure BM25':<18} | {b_top.score:<12.4f} | {t_bm25:<10.2f} | [Chunk {b_top.chunk_id}] {b_top.snippet[:28]}..."
            )
        if h_top:
            print(
                f"  {'3. Hybrid (RRF)':<18} | {h_top.score:<12.6f} | {t_hybrid:<10.2f} | [Chunk {h_top.chunk_id}] {h_top.snippet[:28]}..."
            )
        if r_top:
            print(
                f"  {'4. Hybrid+Rerank':<18} | {r_top.score:<12.4f} | {t_rerank:<10.2f} | [Chunk {r_top.chunk_id}] {r_top.snippet[:28]}..."
            )

        # 打印精排前后的排名变化诊断
        print("\n  [*] [精排动作洞察]:")
        if h_top and r_top:
            if h_top.chunk_id == r_top.chunk_id:
                print(
                    f"    -> 粗排 Top-1 [Chunk {h_top.chunk_id}] 经 Cross-Encoder 研判后保持置顶！"
                )
            else:
                print(
                    f"    -> 发生重新排序！粗排 Top-1 [Chunk {h_top.chunk_id}] -> 精排校正置顶为 [Chunk {r_top.chunk_id}]"
                )

    print("\n" + "=" * 80)
    print("[+] A/B 评测运行完毕！观察专有名词与语义查询在四种策略下的命中差异。")


# ==========================================
# 5. 主入口与环境自适应加载
# ==========================================

if __name__ == "__main__":
    print("[*] 正在加载稠密向量模型 (BAAI/bge-small-zh-v1.5)...")
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

    # 1. 准备/加载向量索引
    index, chunks = get_or_build_index(
        data_dir=DATA_DIR,
        index_path=INDEX_PATH,
        json_path=JSON_PATH,
        embed_model=embed_model,
    )

    # 2. 准备 BM25 词频倒排索引
    print("[*] 正在对分块进行分词并构建 BM25 倒排索引...")
    try:
        bm25, tokenized_corpus = build_bm25_index(chunks)
        print(f"[+] BM25 索引构建完毕，共覆盖 {len(tokenized_corpus)} 个分块！")
    except NotImplementedError as e:
        print(f"\n[!] {e}\n")
        sys.exit(1)

    # 3. 初始化 Reranker 模型
    # 支持加载 BAAI/bge-reranker-base；若本地暂无权重或网络受限，可选用轻量回退模拟器
    print("[*] 正在初始化 Cross-Encoder 重排器...")
    try:
        # 优先使用 BAAI/bge-reranker-base (若已下载)
        reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        print("[+] 成功加载 Cross-Encoder (BAAI/bge-reranker-base)！")
    except Exception as e:
        print(f"[-] 未能加载本地 BAAI/bge-reranker-base ({e})")
        print("[*] 启用备用轻量 Cross-Encoder 作为演示环境...")

        # 优雅平滑降级：使用已缓存的 cross-encoder 或高拟真回退打分器
        class FallbackMockReranker:
            def predict(self, pairs: List[List[str]]) -> List[float]:
                # 基于语义关键词重合与长度惩罚的高拟真打分模拟
                scores = []
                for q, d in pairs:
                    q_words = set(jieba.lcut(q))
                    d_words = set(jieba.lcut(d))
                    intersection = q_words & d_words
                    score = len(intersection) / (len(q_words) + 1e-5)
                    scores.append(float(score))
                return scores

        reranker = FallbackMockReranker()
        print("[+] 备用重排器已就绪。")

    # 4. 定义测试用例（覆盖典型的“条款号/专有名词”与“复杂语义/口语化”场景）
    test_queries = [
        "《差旅费管理办法》第七条规定了什么交通工具？",  # 强条款号 + 专有名词（检验 BM25 纠偏）
        "普通员工出差去北京，坐高铁能买一等座吗？",  # 口语化语义疑问（检验 Dense + 语义泛化）
        "去外地出差，每天伙食费能给补助多少钱？",  # 关键数值与定义（检验两阶段精排稳定性）
    ]

    # 5. 执行对比流水线
    run_ab_comparison(
        test_queries=test_queries,
        index=index,
        chunks=chunks,
        embed_model=embed_model,
        bm25=bm25,
        reranker=reranker,
    )


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
def build_bm25_index_reference(chunks: list[dict]) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized_corpus = []
    for chunk in chunks:
        tokens = jieba.lcut(chunk["text"])
        tokenized_corpus.append(tokens)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def bm25_retrieve_reference(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 5,
) -> List[RetrievalResult]:
    tokenized_query = jieba.lcut(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        chunk = chunks[idx]
        snippet = chunk["text"][:80].replace("\\n", " ") + "..."
        results.append(
            RetrievalResult(
                chunk_id=chunk["id"],
                file=chunk["file"],
                score=score,
                text=chunk["text"],
                snippet=snippet,
            )
        )
    return results


def rrf_fusion_reference(
    rankings_list: List[List[RetrievalResult]],
    k: int = 60,
    top_k: int = 5,
) -> List[RetrievalResult]:
    rrf_scores: Dict[int, float] = {}
    doc_map: Dict[int, RetrievalResult] = {}

    for ranking in rankings_list:
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk_id
            if cid not in doc_map:
                doc_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # 按 RRF 分数降序排列
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in sorted_items[:top_k]:
        base_item = doc_map[cid]
        results.append(
            RetrievalResult(
                chunk_id=base_item.chunk_id,
                file=base_item.file,
                score=float(score),
                text=base_item.text,
                snippet=base_item.snippet,
            )
        )
    return results


def cross_encoder_rerank_reference(
    query: str,
    candidates: List[RetrievalResult],
    reranker: Any,
    top_k: int = 3,
) -> List[RetrievalResult]:
    if not candidates:
        return []

    pairs = [[query, cand.text] for cand in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = []
    for cand, score in zip(candidates, scores):
        scored_candidates.append(
            RetrievalResult(
                chunk_id=cand.chunk_id,
                file=cand.file,
                score=float(score),
                text=cand.text,
                snippet=cand.snippet,
            )
        )

    # 按照重排模型打分降序排列
    sorted_candidates = sorted(
        scored_candidates, key=lambda x: x.score, reverse=True
    )
    return sorted_candidates[:top_k]
"""
# endregion
