"""
Day 12: Advanced RAG - 生产级向量数据库 (Milvus) 与元数据混合过滤实战
=====================================================================
本脚本为 Day 12 的实战闯关代码骨架。
请根据其中的契约定义（Pydantic & Type Hints）与关卡提示（Docstring），
手撕 Milvus 集合 Schema 定义、向量索引构建以及带 Boolean 表达式的标量元数据混合过滤检索。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class DocumentRecord(BaseModel):
    """待入库的文档片段与标量元数据契约"""

    text: str = Field(..., description="文档正文内容")
    department: str = Field(..., description="所属部门，如：'财务部', '行政部', '技术部'")
    doc_type: str = Field(..., description="文档类型，如：'规章', '通知', '指引'")
    effective_year: int = Field(..., description="生效年份，如：2024, 2025, 2026")


class SearchResultItem(BaseModel):
    """单条检索召回结果契约"""

    id: Any = Field(..., description="记录主键 ID")
    distance: float = Field(..., description="向量相似度距离得分")
    text: str = Field(..., description="召回的文本片段")
    department: str = Field(..., description="部门标量")
    doc_type: str = Field(..., description="文档类型标量")
    effective_year: int = Field(..., description="生效年份标量")


# ==========================================
# 2. 客户端连接与脚手架 (开箱即用)
# ==========================================


def get_milvus_client(db_path: str = "milvus_travel.db") -> MilvusClient:
    """
    初始化 Milvus 客户端。
    支持优先尝试连接本地 Docker Milvus (localhost:19530)；
    若 Docker 未启动，则自适应切换至本地无依赖嵌入式 Milvus-Lite (milvus_travel.db)。
    """
    # 优先检测本地 Docker 独立服务 (19530)
    docker_uri = "http://localhost:19530"
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        is_docker_up = (sock.connect_ex(("127.0.0.1", 19530)) == 0)
        sock.close()
        if is_docker_up:
            client = MilvusClient(uri=docker_uri)
            print(f"[+] 成功连接本地 Docker Milvus 集群服务 ({docker_uri})！")
            return client
    except Exception:
        pass

    # 优雅回退到 Milvus-Lite 本地无依赖文件模式
    print(f"[*] Docker Milvus 未启动，自动启用本地嵌入式 Milvus-Lite 存储引擎 ({db_path})...")
    client = MilvusClient(uri=db_path)
    return client


# ==========================================
# 3. 核心任务手撕区：Day 12 闯关打靶区
# ==========================================


def create_collection_with_schema(
    client: MilvusClient,
    collection_name: str,
    dim: int = 512,
) -> None:
    """
    【关卡 1】：Milvus 集合 Schema 契约设计与生命周期管理

    目标：
    为差旅规章 RAG 系统设计强类型的 Collection Schema，包含以下字段：
    - id: INT64, 主键，开启自增 (is_primary=True, auto_id=True)
    - vector: FLOAT_VECTOR, 稠密向量，维度为 dim
    - text: VARCHAR, 正文内容，最大长度 1000
    - department: VARCHAR, 部门元数据，最大长度 64
    - doc_type: VARCHAR, 文档类型元数据，最大长度 64
    - effective_year: INT64, 生效年份元数据

    实现提示：
    1. 检查是否存在同名集合 client.has_collection(collection_name)，若存在则 client.drop_collection 删除以保证幂等；
    2. 使用 client.create_schema(auto_id=True, enable_dynamic_field=True) 创建 schema；
    3. 调用 schema.add_field(...) 依次添加上述 6 个字段；
    4. 暂不建索引，使用 client.create_collection(collection_name=collection_name, schema=schema) 创建集合；
    5. 打印创建成功提示。
    """
    # TODO 1: 请实现强类型 Collection Schema 设计与集合初始化
    raise NotImplementedError(
        "【关卡 1 待实现】：请实现 create_collection_with_schema 集合 Schema 设计！完成后删除本行。"
    )


def build_index_and_load(
    client: MilvusClient,
    collection_name: str,
    metric_type: str = "COSINE",
) -> None:
    """
    【关卡 2】：HNSW / FLAT 向量索引构建与内存加载

    目标：
    在已插入向量数据的集合上，为 "vector" 字段创建高性能近似最近邻（ANN）索引，
    并将集合加载到内存中以供实时检索。

    实现提示：
    1. 使用 client.prepare_index_params() 构建索引参数；
    2. 调用 index_params.add_index(
           field_name="vector",
           index_type="HNSW",       # 工业界标准分层图索引
           metric_type=metric_type, # COSINE 余弦相似度
           params={"M": 16, "efConstruction": 64}
       )；
    3. 调用 client.create_index(collection_name=collection_name, index_params=index_params)；
    4. 调用 client.load_collection(collection_name=collection_name) 将索引与数据加载至内存；
    5. 打印索引构建与加载完成提示。
    """
    # TODO 2: 请实现向量索引参数配置、索引创建与集合加载
    raise NotImplementedError(
        "【关卡 2 待实现】：请实现 build_index_and_load 向量索引构建与加载！完成后删除本行。"
    )


def search_with_scalar_filter(
    client: MilvusClient,
    collection_name: str,
    query_vector: List[float],
    filter_expr: str = "",
    limit: int = 3,
    output_fields: Optional[List[str]] = None,
) -> List[SearchResultItem]:
    """
    【关卡 3】：生产级标量元数据混合过滤检索 (Hybrid Vector + Scalar Search)

    目标：
    在 Milvus 中执行带 Boolean 标量表达式的向量相似度前置过滤检索。
    彻底解决 FAISS 无法前置过滤、后过滤导致召回结果被洗白的工程硬伤！

    实现提示：
    1. output_fields 默认为 ["text", "department", "doc_type", "effective_year"]；
    2. 调用 client.search(
           collection_name=collection_name,
           data=[query_vector],
           filter=filter_expr if filter_expr else "",
           limit=limit,
           output_fields=output_fields,
           search_params={"metric_type": "COSINE", "params": {"ef": 64}}
       )；
    3. 遍历 search 返回的命中结果，将每个 hit 封装为 SearchResultItem(
           id=hit["id"],
           distance=float(hit["distance"]),
           text=hit["entity"]["text"],
           department=hit["entity"]["department"],
           doc_type=hit["entity"]["doc_type"],
           effective_year=int(hit["entity"]["effective_year"])
       )；
    4. 返回 List[SearchResultItem]。
    """
    # TODO 3: 请实现支持 Boolean Filter 表达式的 Milvus 混合检索
    raise NotImplementedError(
        "【关卡 3 待实现】：请实现 search_with_scalar_filter 标量混合过滤检索！完成后删除本行。"
    )


# ==========================================
# 4. 主执行入口与 Benchmark 测试用例
# ==========================================

if __name__ == "__main__":
    COLLECTION_NAME = "enterprise_knowledge_base"
    DB_PATH = "milvus_travel.db"

    print("[*] 正在加载 Embedding 向量模型 (BAAI/bge-small-zh-v1.5)...")
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    VECTOR_DIM = embed_model.get_sentence_embedding_dimension()

    print("[*] 正在初始化 Milvus 客户端...")
    client = get_milvus_client(db_path=DB_PATH)

    # 1. 闯关关卡 1：创建集合与 Schema
    print("\n" + "=" * 80)
    print("【步骤 1】：定义 Collection Schema 并创建集合...")
    print("=" * 80)
    create_collection_with_schema(client, COLLECTION_NAME, dim=VECTOR_DIM)

    # 2. 准备企业级多部门多业务语料（带丰富标量元数据）
    sample_docs = [
        DocumentRecord(
            text="第七条 出差人员应当按规定等级乘坐交通工具。部级及相当职务人员可乘高铁商务座；司局级及相当职务人员可乘高铁一等座；其余人员（含普通员工）可乘高铁二等座。",
            department="财务部",
            doc_type="规章",
            effective_year=2024,
        ),
        DocumentRecord(
            text="第十六条 财务报销凭据必须提供真实有效的增值税专用发票或普通发票，火车票行程单须自乘车之日起30日内交由财务部核销，逾期需部门分管领导特批。",
            department="财务部",
            doc_type="规章",
            effective_year=2025,
        ),
        DocumentRecord(
            text="第三条 公司日常办公电脑及外设领用制度：普通研发员工标准配置为高性能笔记本一台，每三年可申请一次折旧更换，非办公需要不得申领移动显示器。",
            department="行政部",
            doc_type="制度",
            effective_year=2024,
        ),
        DocumentRecord(
            text="第四章 服务器上线规范：所有生产环境集群配置变更必须经过运维评审，严禁未经审批开放公网22端口，夜间升级必须提前报备并抄送技术支持团队。",
            department="技术部",
            doc_type="规范",
            effective_year=2025,
        ),
        DocumentRecord(
            text="关于2023年差旅住宿费临时补贴标准的作废说明：自2024年1月1日起，原2023年临时特区伙食住宿特别津贴统一废止，全面执行新版财务部差旅费管理办法。",
            department="行政部",
            doc_type="通知",
            effective_year=2023,
        ),
    ]

    print(f"[*] 正在将 {len(sample_docs)} 条样本转化为向量并插入 Milvus...")
    insert_data = []
    for doc in sample_docs:
        vec = embed_model.encode(doc.text, normalize_embeddings=True).tolist()
        insert_data.append({
            "vector": vec,
            "text": doc.text,
            "department": doc.department,
            "doc_type": doc.doc_type,
            "effective_year": doc.effective_year,
        })
    insert_res = client.insert(collection_name=COLLECTION_NAME, data=insert_data)
    print(f"[+] 数据插入成功，写入记录数: {insert_res['insert_count']} 条！")

    # 3. 闯关关卡 2：构建索引并加载进内存
    print("\n" + "=" * 80)
    print("【步骤 2】：构建 HNSW 向量索引并加载至内存...")
    print("=" * 80)
    build_index_and_load(client, COLLECTION_NAME, metric_type="COSINE")

    # 4. 闯关关卡 3：执行 A/B 对比检索（纯向量检索 vs 标量前置过滤检索）
    test_query = "普通员工出差乘坐高铁有什么报销和席位规定？"
    query_vector = embed_model.encode(test_query, normalize_embeddings=True).tolist()

    print("\n" + "=" * 80)
    print(f"【测试查询】: 「{test_query}」")
    print("=" * 80)

    # 实验 A: 无标量过滤（全库纯向量最近邻搜索）
    print("\n🔍 【实验组 A：无标量过滤检索 (Unfiltered Vector Search)】")
    print("-" * 80)
    raw_results = search_with_scalar_filter(
        client=client,
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        filter_expr="",
        limit=3,
    )
    for rank, item in enumerate(raw_results, 1):
        print(f"Top-{rank} | 余弦得分: {item.distance:.4f} | 部门: {item.department} | 类型: {item.doc_type} | 年份: {item.effective_year}")
        print(f"       ↳ 内容: {item.text[:60]}...")

    # 实验 B: 带结构化标量前置过滤（指定部门 + 生效年份）
    filter_expression = "department == '财务部' and effective_year >= 2024"
    print(f"\n🎯 【实验组 B：带标量前置过滤检索 (Filtered: {filter_expression})】")
    print("-" * 80)
    filtered_results = search_with_scalar_filter(
        client=client,
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        filter_expr=filter_expression,
        limit=3,
    )
    for rank, item in enumerate(filtered_results, 1):
        print(f"Top-{rank} | 余弦得分: {item.distance:.4f} | 部门: {item.department} | 类型: {item.doc_type} | 年份: {item.effective_year}")
        print(f"       ↳ 内容: {item.text[:60]}...")

    # 5. 验收断言
    print("\n" + "=" * 80)
    print("🏁 【验收核验结果】:")
    is_pure_finance = all(item.department == "财务部" and item.effective_year >= 2024 for item in filtered_results)
    if is_pure_finance and len(filtered_results) > 0:
        print("✅ [VERIFICATION PASSED]: 标量前置过滤生效！非财务部及历史废弃文档被 100% 物理隔离排除！")
    else:
        print("❌ [VERIFICATION FAILED]: 标量过滤未起效，请检查 filter_expr 语法与 search 调用！")
    print("=" * 80)

    # 优雅关闭客户端连接
    try:
        client.close()
    except Exception:
        pass


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
def create_collection_with_schema(
    client: MilvusClient,
    collection_name: str,
    dim: int = 512,
) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"[*] 清理旧集合: {collection_name}")

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="department", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="effective_year", datatype=DataType.INT64)

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"[+] 成功创建集合与 Schema: {collection_name} (Vector Dim={dim})")


def build_index_and_load(
    client: MilvusClient,
    collection_name: str,
    metric_type: str = "COSINE",
) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type=metric_type,
        params={"M": 16, "efConstruction": 64},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.load_collection(collection_name=collection_name)
    print(f"[+] 成功构建 HNSW 向量索引并完成集合内存加载 (Metric={metric_type})！")


def search_with_scalar_filter(
    client: MilvusClient,
    collection_name: str,
    query_vector: List[float],
    filter_expr: str = "",
    limit: int = 3,
    output_fields: Optional[List[str]] = None,
) -> List[SearchResultItem]:
    if output_fields is None:
        output_fields = ["text", "department", "doc_type", "effective_year"]

    search_res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        filter=filter_expr if filter_expr else "",
        limit=limit,
        output_fields=output_fields,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )

    items = []
    if search_res and len(search_res) > 0:
        for hit in search_res[0]:
            entity = hit.get("entity", {})
            items.append(
                SearchResultItem(
                    id=hit.get("id"),
                    distance=float(hit.get("distance", 0.0)),
                    text=entity.get("text", ""),
                    department=entity.get("department", ""),
                    doc_type=entity.get("doc_type", ""),
                    effective_year=int(entity.get("effective_year", 0)),
                )
            )
    return items
"""
# endregion
