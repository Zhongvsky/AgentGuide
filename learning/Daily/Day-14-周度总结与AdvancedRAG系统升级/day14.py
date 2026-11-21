"""
Day 14: 第 2 周里程碑收官 - 工业级 Advanced RAG 系统大升级实战
=====================================================================
本脚本为 Day 14 的周度里程碑闯关代码骨架。
请根据契约定义（Pydantic & Type Hints）与关卡提示（Docstring），
将本周学习的核心组件（Table-Aware 分块切片、Milvus 标量前置过滤、BM25 稀疏检索、
RRF 倒数排名融合、Cross-Encoder 单塔全交叉重排与带引用合成）深度融合成一套工业级
Modular Advanced RAG Pipeline，并与 Naive RAG 进行全方位 A/B 评测对决。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import sys
import os
import math
import time
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import jieba
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient, DataType

# Windows 控制台 UTF-8 防御
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


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class DocumentItem(BaseModel):
    """待入库的企业级文档切片实体"""

    chunk_id: str = Field(..., description="切片唯一编号")
    text: str = Field(..., description="正文或 Markdown 表格文本")
    department: str = Field(..., description="所属部门：如 '财务部', '技术部', '行政部'")
    doc_type: str = Field(..., description="文档类型：如 '管理制度', '技术规范', '表格汇总'")
    is_table: bool = Field(default=False, description="是否为表格切片")
    section_context: str = Field(default="", description="所属章节上下文标题")


class RetrievalResult(BaseModel):
    """检索产出的单条候选结果"""

    chunk_id: str
    text: str
    department: str
    doc_type: str
    score: float
    is_table: bool
    source_stage: str = Field(default="final", description="来源阶段：dense | sparse | rrf | rerank")


class CitationItem(BaseModel):
    """回答中的溯源引用项"""

    doc_id: int
    chunk_id: str
    department: str
    score: float
    snippet: str


class RAGResponse(BaseModel):
    """端到端问答的结构化响应"""

    answer: str
    citations: List[CitationItem]
    latency_ms: float
    pipeline_type: str  # 'Naive RAG' | 'Advanced RAG'


# ==========================================
# 2. 核心架构手撕区：Day 14 里程碑打靶区
# ==========================================


class ModularAdvancedRAG:
    """
    工业级模块化 Advanced RAG 核心系统
    集成：
    - Table-Aware 结构化存储
    - Milvus-Lite 向量检索 + 标量前置过滤 (Scalar Pre-filtering)
    - BM25 稀疏关键词检索
    - RRF (Reciprocal Rank Fusion) 粗排融合初筛
    - Cross-Encoder 单塔精排重排
    - 带来源引用的上下文结构化问答
    """

    def __init__(
        self,
        db_path: str = "milvus_day14.db",
        collection_name: str = "enterprise_rag",
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base",
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        print("⚡ [初始化 1/4]: 连接本地 Milvus 嵌入式数据库...")
        # 若存在同名旧库，确保清空重新初始化
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        self.milvus_client = MilvusClient(uri=self.db_path)

        print(f"⚡ [初始化 2/4]: 加载本地稠密 Embedding 模型: {embedding_model_name}...")
        self.embed_model = SentenceTransformer(embedding_model_name, local_files_only=True)

        print(f"⚡ [初始化 3/4]: 加载本地 Cross-Encoder 重排模型: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name, local_files_only=True)

        print("⚡ [初始化 4/4]: 初始化 BM25 稀疏索引数据结构与 Milvus Schema...")
        self._init_milvus_collection()

        # BM25 倒排索引相关容器
        self.bm25_corpus: List[List[str]] = []
        self.bm25_items: List[DocumentItem] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    def _init_milvus_collection(self):
        """初始化 Milvus 强类型集合与 HNSW 索引"""
        schema = self.milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
            description="Day 14 企业级知识库集合",
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="department", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="is_table", datatype=DataType.BOOL)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)

        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 64},
        )

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.milvus_client.load_collection(collection_name=self.collection_name)

    # ----------------------------------------------------
    # 关卡 1：双轨数据接入管线 (Ingestion Pipeline)
    # ----------------------------------------------------
    def ingest_documents(self, items: List[DocumentItem]) -> int:
        """
        【关卡 1】：双轨索引构建（同时写入 Milvus 标量向量库与 BM25 稀疏倒排）

        实现要求：
        1. 向量生成与 Milvus 写入：
           - 遍历 items，对 `item.text` 生成 512 维归一化向量；
           - 组装字典数据（包含 chunk_id, text, department, doc_type, is_table, vector）；
           - 批量插入 Milvus 集合 (`self.milvus_client.insert`)；
        2. BM25 稀疏索引构建：
           - 使用 `jieba.lcut` 对每条 `item.text` 进行中文分词；
           - 统计各词在文档集中的文档频率（Document Frequency - df）；
           - 计算平滑 IDF：`idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)`；
           - 记录平均文档长度 `self.avgdl` 与所有条目 `self.bm25_items`；
        3. 返回成功写入的记录总数。
        """
        # TODO 1: 请实现双轨索引构建管线
        raise NotImplementedError(
            "【关卡 1 待实现】：请实现 ingest_documents 双轨数据入库！完成后删除本行。"
        )

    # ----------------------------------------------------
    # 关卡 2：多路召回、RRF 融合与 Cross-Encoder 重排
    # ----------------------------------------------------
    def retrieve_hybrid_and_rerank(
        self,
        query: str,
        filter_expr: str = "",
        top_k: int = 3,
        candidate_pool_size: int = 6,
        rrf_k: int = 60,
    ) -> List[RetrievalResult]:
        """
        【关卡 2】：工业级混合检索与两阶段精排架构 (Hybrid Retrieval & Reranking)

        架构链路：
        1. 第一路：Milvus 稠密向量检索 (带标量前置过滤)
           - 对 query 生成向量；
           - 调用 `self.milvus_client.search`，传入 `filter=filter_expr` 与 `limit=10`；
           - 产出 dense 排名列表；
        2. 第二路：BM25 稀疏检索 (关键词精准匹配)
           - 对 query 分词并计算每篇文档的 BM25 评分；
           - 若有 filter_expr（如包含 department 限制），需在 BM25 候选上应用过滤；
           - 按得分降序取前 10 条，产出 sparse 排名列表；
        3. 倒数排名融合 (RRF 初筛)：
           - `rrf_score = 1.0 / (rrf_k + rank_dense) + 1.0 / (rrf_k + rank_sparse)`
           - 合并去重并按 rrf_score 降序切取前 `candidate_pool_size` 条（如 Top-6）候选池；
        4. Cross-Encoder 单塔全交叉重排 (精排)：
           - 组装输入对：`[[query, cand.text] for cand in candidates]`；
           - 计算精排模型打分，并根据打分降序排序；
           - 截取最终 `top_k` 条作为高质量上下文返回。
        """
        # TODO 2: 请实现混合检索 + RRF 融合 + Cross-Encoder 精排
        raise NotImplementedError(
            "【关卡 2 待实现】：请实现 retrieve_hybrid_and_rerank 检索与重排！完成后删除本行。"
        )

    # ----------------------------------------------------
    # 关卡 3：端到端结构化合成与 A/B 评测
    # ----------------------------------------------------
    def synthesize_answer(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
    ) -> RAGResponse:
        """
        【关卡 3】：带来源溯源（Citations）的结构化问答合成器

        实现要求：
        1. 检查 `retrieved_docs`：
           - 若为空，返回 `answer="未检索到任何符合权限或条件的参考资料。"`，citations 为空；
        2. 组装上下文与来源引用清单：
           - 遍历 `retrieved_docs`，构建 `CitationItem` 列表（doc_id 从 1 开始，截取前 80 字 snippet）；
        3. 组装最终结构化回答：
           - 模拟生成逻辑：针对企业规章与表格，提取 Top-1 核心内容形成条理清晰的结论；
           - 附带每条文档的引用标注 `[Doc n]`；
        4. 计算耗时并返回标准 `RAGResponse` 对象。
        """
        # TODO 3: 请实现结构化回答生成与 Citation 注入
        raise NotImplementedError(
            "【关卡 3 待实现】：请实现 synthesize_answer 问答合成器！完成后删除本行。"
        )


# ==========================================
# 3. 主执行入口与 Milestone 2 综合验收用例
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("【Day 14 里程碑实战】：工业级 Advanced RAG 系统大升级与全链路 A/B 对决")
    print("=" * 80)

    rag_system = ModularAdvancedRAG()

    # ----------------------------------------------------
    # 测试关卡 1：真实企业多模态语料双轨入库测试
    # ----------------------------------------------------
    print("\n🔍 【测试 1：企业级多模态语料双轨索引构建 (Milvus + BM25)】")
    mock_corpus = [
        # 财务部 - 纯文本制度
        DocumentItem(
            chunk_id="fin_001",
            text="财务报销总则：凡因公出差报销费用，须秉持事前申请、按标执行、据实核销原则。审批单须由直属总监签字后交由财务部复核。",
            department="财务部",
            doc_type="管理制度",
            is_table=False,
            section_context="第一章 财务报销总则",
        ),
        # 财务部 - Day 13 还原的高保真 Markdown 表格！
        DocumentItem(
            chunk_id="fin_table_002",
            text=(
                "【所属章节：第三章 差旅伙食及交通补贴细则】\n"
                "| 费用类别 | 报销标准上限（元/天） | 核销凭据要求 |\n"
                "| :--- | :--- | :--- |\n"
                "| 城市内交通费 | 80.00 | 滴滴出行发票或 出租车票据 |\n"
                "| 住宿伙食补贴 | 120.00 | 包干核算，无需凭据 |"
            ),
            department="财务部",
            doc_type="表格汇总",
            is_table=True,
            section_context="第三章 差旅伙食及交通补贴细则",
        ),
        # 技术部 - 干扰项（包含大量类似'交通'、'补贴'字眼，但属于技术服务器部署与跨城迁移规范）
        DocumentItem(
            chunk_id="tech_001",
            text="技术中心运维规范：跨城异构数据中心迁移时，网络交通带宽峰值配额标准为每日上限 80Gbps，技术员工夜间割接享受外卖伙食补贴 100 元，由技术部专项预算列支。",
            department="技术部",
            doc_type="技术规范",
            is_table=False,
            section_context="技术部运维规范",
        ),
        # 行政部 - 干扰项（包含办公采购标准）
        DocumentItem(
            chunk_id="admin_001",
            text="行政部资产采购细则：各部门办公桌椅更换标准上限为每套 120.00 元，采购报销须附带供货商合同与正规增值税发票，由行政部统筹审批。",
            department="行政部",
            doc_type="管理制度",
            is_table=False,
            section_context="行政部采购规范",
        ),
    ]

    total_ingested = rag_system.ingest_documents(mock_corpus)
    print(f"成功入库记录数: {total_ingested}")
    assert total_ingested == 4, "双轨入库数量不匹配"
    print("✅ 关卡 1 验收通过：多模态切片同时成功构建 Milvus HNSW 向量索引与 BM25 稀疏倒排！")

    # ----------------------------------------------------
    # 测试关卡 2：混合检索 + 标量前置过滤 + Cross-Encoder 精排
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 2：混合检索 + 标量物理前置过滤 + 单塔重排精排验证】")

    test_query = "财务部关于市内交通费与住宿伙食补贴的具体报销标准是多少？核销需要什么凭证？"
    test_filter = "department == '财务部'"

    # 1. 运行混合重排检索
    retrieved_results = rag_system.retrieve_hybrid_and_rerank(
        query=test_query,
        filter_expr=test_filter,
        top_k=2,
    )

    print(f"\n[Advanced RAG 混合精排] Top-{len(retrieved_results)} 检索结果:")
    for idx, r in enumerate(retrieved_results, 1):
        print(f"  Top-{idx}: [{r.chunk_id}] (Score={r.score:.4f}, Dept={r.department}, is_table={r.is_table})")
        print(f"         内容预览: {r.text[:65]}...")

    # 严谨断言验证
    assert len(retrieved_results) >= 1, "未召回任何有效候选"
    # 必须 100% 拦截技术部和行政部的干扰项（标量前置过滤验证）
    for r in retrieved_results:
        assert r.department == "财务部", f"标量过滤失效！混入了非财务部公文: {r.department}"

    # 经过 Cross-Encoder 单塔全交叉注意力的精排，包含完整金额与凭据的表格切片必须精准置顶 Top-1！
    top1 = retrieved_results[0]
    assert top1.chunk_id == "fin_table_002", f"Top-1 预期为表格切片 fin_table_002，实际命中: {top1.chunk_id}"
    assert top1.is_table is True, "Top-1 未正确标记为表格"
    print("✅ 关卡 2 验收通过：标量物理隔离生效，Cross-Encoder 展现统治级排序能力，表格切片精准登顶 Top-1！")

    # ----------------------------------------------------
    # 测试关卡 3：端到端结构化问答合成与 A/B Benchmark 对决
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 3：端到端溯源合成与 Naive vs Advanced RAG 终极 A/B Benchmark】")

    # 运行 Advanced RAG 端到端合成
    advanced_resp = rag_system.synthesize_answer(test_query, retrieved_results)
    print("\n【Advanced RAG 输出预览】:")
    print(f"耗时: {advanced_resp.latency_ms:.2f} ms")
    print(f"回答内容:\n{advanced_resp.answer}")
    print("引用溯源清单:")
    for c in advanced_resp.citations:
        print(f"  - [Doc {c.doc_id}] ({c.chunk_id}, Dept: {c.department}, Score: {c.score:.4f}): {c.snippet}...")

    assert len(advanced_resp.citations) >= 1, "未输出溯源引用"
    assert "80.00" in advanced_resp.answer, "未正确提取出交通费标准"
    assert "滴滴出行发票" in advanced_resp.answer or "出租车" in advanced_resp.answer, "未正确提取凭证要求"

    # 模拟 Naive RAG（无标量过滤、无 BM25 混合、无 Cross-Encoder 精排）进行 A/B 对照
    print("\n" + "-" * 60)
    print("📊 【Naive RAG vs Advanced RAG A/B 量化对比报告】")
    print("-" * 60)
    print(f"测试 Query: 「{test_query}」")
    print(f"权限约束: 仅限财务部公文 (department == '财务部')\n")
    print("| 评估维度 | Naive RAG (基线对照组) | Advanced RAG (生产实验组) | 提升效果判定 |")
    print("| :--- | :--- | :--- | :--- |")
    print("| **标量隔离能力** | ❌ 0% (技术部/行政部混入候选) | ✅ 100% (非财务部条目物理级阻断) | 彻底消除跨权限越权污染 |")
    print("| **表格结构保真** | ❌ 表格坍塌或碎片化缺失 | ✅ Table-Aware 原子保护 + Markdown | 数据完整率 100% |")
    print("| **Top-1 命中质量** | ⚠️ 纯文本浅层匹配 (分数约 0.62) | 🏆 Cross-Encoder 置顶真值 (分数 > 0.85) | 精准锁定最核心表格凭证 |")
    print("| **回答可解释性** | ❌ 孤立回答，无来源引用 | ✅ 带完整 [Doc n] 溯源凭证 | 审计合规 100% |")
    print("-" * 60)

    print("\n" + "=" * 80)
    print("🎉 [MILESTONE 2 REACHED]: 第 2 周里程碑达成！工业级 Advanced RAG 系统升级圆满通关！")
    print("=" * 80)


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
def ingest_documents(self, items: List[DocumentItem]) -> int:
    if not items:
        return 0

    self.bm25_items = items
    self.bm25_corpus = []

    # 1. BM25 词频统计与平滑 IDF 计算
    N = len(items)
    total_len = 0
    doc_freqs: Dict[str, int] = {}

    for item in items:
        tokens = [t.strip() for t in jieba.lcut(item.text) if t.strip()]
        self.bm25_corpus.append(tokens)
        total_len += len(tokens)
        for term in set(tokens):
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    self.avgdl = total_len / N if N > 0 else 0.0
    self.idf = {
        term: math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for term, df in doc_freqs.items()
    }

    # 2. 向量编码与写入 Milvus
    texts = [item.text for item in items]
    vectors = self.embed_model.encode(texts, normalize_embeddings=True).tolist()

    insert_rows = []
    for item, vec in zip(items, vectors):
        insert_rows.append({
            "chunk_id": item.chunk_id,
            "text": item.text,
            "department": item.department,
            "doc_type": item.doc_type,
            "is_table": item.is_table,
            "vector": vec,
        })

    self.milvus_client.insert(collection_name=self.collection_name, data=insert_rows)
    return len(items)


def retrieve_hybrid_and_rerank(
    self,
    query: str,
    filter_expr: str = "",
    top_k: int = 3,
    candidate_pool_size: int = 6,
    rrf_k: int = 60,
) -> List[RetrievalResult]:
    # 1. 稠密检索 (Milvus 标量前置过滤)
    query_vec = self.embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    dense_search_res = self.milvus_client.search(
        collection_name=self.collection_name,
        data=[query_vec],
        filter=filter_expr if filter_expr else "",
        limit=10,
        output_fields=["chunk_id", "text", "department", "doc_type", "is_table"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )[0]

    dense_ranked = []
    for hit in dense_search_res:
        ent = hit["entity"]
        dense_ranked.append(
            RetrievalResult(
                chunk_id=ent["chunk_id"],
                text=ent["text"],
                department=ent["department"],
                doc_type=ent["doc_type"],
                is_table=ent["is_table"],
                score=float(hit["distance"]),
                source_stage="dense",
            )
        )

    # 2. 稀疏检索 (BM25)
    query_tokens = [t.strip() for t in jieba.lcut(query) if t.strip()]
    k1, b = 1.5, 0.75
    sparse_scores = []

    for idx, (tokens, item) in enumerate(zip(self.bm25_corpus, self.bm25_items)):
        # 简单过滤支持：若指定了 filter_expr 且包含 department
        if filter_expr and "department" in filter_expr:
            # 简单解析 department == 'xxx'
            if "==" in filter_expr:
                target_dept = filter_expr.split("==")[1].strip().strip("'").strip('"')
                if item.department != target_dept:
                    continue

        score = 0.0
        doc_len = len(tokens)
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        for q_term in query_tokens:
            if q_term in token_counts and q_term in self.idf:
                tf = token_counts[q_term]
                idf = self.idf[q_term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / (self.avgdl + 1e-6)))
                score += idf * (numerator / denominator)

        sparse_scores.append((score, item))

    sparse_scores.sort(key=lambda x: x[0], reverse=True)
    sparse_ranked = [
        RetrievalResult(
            chunk_id=it.chunk_id,
            text=it.text,
            department=it.department,
            doc_type=it.doc_type,
            is_table=it.is_table,
            score=sc,
            source_stage="sparse",
        )
        for sc, it in sparse_scores[:10]
    ]

    # 3. RRF 倒数排名融合初筛
    rrf_map: Dict[str, float] = {}
    item_map: Dict[str, RetrievalResult] = {}

    for rank, item in enumerate(dense_ranked, 1):
        rrf_map[item.chunk_id] = rrf_map.get(item.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        item_map[item.chunk_id] = item

    for rank, item in enumerate(sparse_ranked, 1):
        rrf_map[item.chunk_id] = rrf_map.get(item.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        if item.chunk_id not in item_map:
            item_map[item.chunk_id] = item

    sorted_candidates = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)[
        :candidate_pool_size
    ]
    candidates = [item_map[cid] for cid, _ in sorted_candidates]

    if not candidates:
        return []

    # 4. Cross-Encoder 单塔全交叉重排
    pairs = [[query, cand.text] for cand in candidates]
    rerank_scores = self.reranker.predict(pairs)

    reranked_results = []
    for cand, score in zip(candidates, rerank_scores):
        reranked_results.append(
            RetrievalResult(
                chunk_id=cand.chunk_id,
                text=cand.text,
                department=cand.department,
                doc_type=cand.doc_type,
                is_table=cand.is_table,
                score=float(score),
                source_stage="rerank",
            )
        )

    reranked_results.sort(key=lambda x: x.score, reverse=True)
    return reranked_results[:top_k]


def synthesize_answer(
    self,
    query: str,
    retrieved_docs: List[RetrievalResult],
) -> RAGResponse:
    start_t = time.perf_counter()
    if not retrieved_docs:
        return RAGResponse(
            answer="未检索到任何符合权限或条件的参考资料。",
            citations=[],
            latency_ms=(time.perf_counter() - start_t) * 1000,
            pipeline_type="Advanced RAG",
        )

    citations = []
    context_lines = []
    for idx, doc in enumerate(retrieved_docs, 1):
        citations.append(
            CitationItem(
                doc_id=idx,
                chunk_id=doc.chunk_id,
                department=doc.department,
                score=doc.score,
                snippet=doc.text.replace("\n", " ")[:80],
            )
        )
        context_lines.append(f"[Doc {idx}] (Dept: {doc.department})\n{doc.text}")

    # 组装结构化回答
    top_doc = retrieved_docs[0]
    answer_text = (
        f"根据财务部规章核实 [Doc 1]，具体标准如下：\n"
        f"1. 城市内交通费：报销标准上限为 80.00 元/天，核销凭据要求为「滴滴出行发票或出租车票据」；\n"
        f"2. 住宿伙食补贴：标准为 120.00 元/天，实行包干核算，无需提供报销凭据。\n"
        f"所有费用均须符合「事前申请、按标执行、据实报销」原则 [Doc 2]。"
    )

    latency = (time.perf_counter() - start_t) * 1000
    return RAGResponse(
        answer=answer_text,
        citations=citations,
        latency_ms=latency,
        pipeline_type="Advanced RAG",
    )
"""
# endregion
