# 向量检索与嵌入核心技术学习笔记

本学习指南针对以下三份官方文档与核心目标制定，帮助你在最短时间内掌握向量检索流水线的核心基石：
1. **Sentence Transformers**：理解文本到稠密向量的表示与主流选型。
2. **FAISS**：掌握底层暴力精确索引（Flat Index）的构建、检索、存盘与加载。
3. **Chroma**：建立对轻量级向量数据库的认知定位，为 Day 12 对比分布式向量数据库 Milvus 打下基础。

---

## 一、Sentence Transformers：文本向量化引擎

- 官方文档：[Sentence Transformers Documentation](https://www.sbert.net/)

### 1. 核心定位与 Bi-Encoder 架构
传统的 BERT 模型通过 Cross-Encoder（双句拼接输入）计算相似度，复杂度为 $O(N^2)$，在大规模知识库检索时速度极慢。  
Sentence Transformers 采用 **Bi-Encoder（双塔结构）**：将每个句子独立编码为固定维度的稠密向量（Dense Embedding），之后只需计算向量间的余弦相似度或内积，复杂度降为 $O(1)$，适合大规模离线预建索引。

### 2. 核心 API 用法

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 加载模型（首次运行会自动从 Hugging Face 下载并缓存）
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. 待编码文本
sentences = [
    "人工智能与深度学习正在重塑现代技术体系。",
    "向量检索与相似度搜索是 RAG 系统的核心底座。",
    "今天天气晴朗，适合户外徒步运动。"
]

# 3. 编码生成嵌入向量
# 注意：务必设置 normalize_embeddings=True
embeddings = model.encode(sentences, normalize_embeddings=True)

print(f"数据类型: {type(embeddings)}")        # <class 'numpy.ndarray'>
print(f"向量形状: {embeddings.shape}")       # (3, 384) -> (文本数量, 向量维度)
print(f"元素精度: {embeddings.dtype}")       # float32
```

### 3. 模型选型核心考量指标

在实际工程中选择 Embedding 模型时，需权衡以下四个核心维度：
1. **向量维度（Dimension）**：
   - 常见维度：`384`、`768`、`1024`。
   - 影响：维度越高，语义表达上限越强，但显存/内存消耗呈线性增长，FAISS 检索计算延迟也会相应增加。
2. **最大上下文长度（Max Sequence Length）**：
   - 常见限制：256 / 512 tokens（部分长文本模型可达 8192）。超出部分会被截断，切分 Chunk 时必须小于该阈值。
3. **语言分布（Monolingual vs Multilingual）**：
   - 英文专用模型（如 `all-MiniLM-L6-v2`）在纯英文任务上表现优异且小巧。
   - 中文或多语言场景需选用针对多语言预训练的模型（如 `bge`、`multilingual-e5`）。
4. **MTEB（Massive Text Embedding Benchmark）综合得分**：
   - 检索（Retrieval）、重排（Rerank）、分类、聚类综合评测指标。

### 4. 常见开源模型快速选型参考

| 模型名称 | 向量维度 | 最大长度 | 适合语言 | 特点与适用场景 |
| :--- | :---: | :---: | :---: | :--- |
| `all-MiniLM-L6-v2` | 384 | 256 | 英文 | 极速、轻量（约 80MB），原型开发与英文测试首选 |
| `BAAI/bge-small-zh-v1.5` | 512 | 512 | 中文/中英 | 性能与尺寸兼备，资源受限或轻量中文 RAG 首选 |
| `BAAI/bge-base-zh-v1.5` | 768 | 512 | 中文/中英 | 中文场景标准基线，检索精度与推理开销平衡极佳 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | 50+ 语言 | 跨语言语义对齐快速测试 |
| `intfloat/multilingual-e5-base` | 768 | 512 | 多语言 | 通用性强，查询需加前缀 `query: ` 与 `passage: ` |

> **关键工程技巧**：
> 调用 `model.encode(..., normalize_embeddings=True)` 对向量进行 L2 归一化后，**余弦相似度（Cosine Similarity）与点积（Inner Product）在数学上完全等价**。这使得底层检索无需计算复杂的除模长运算，直接用内积即可完成余弦匹配。

---

## 二、FAISS：高性能向量相似度检索库

- 官方文档：[FAISS Official Website](https://faiss.ai/)
- 官方 Wiki：[FAISS GitHub Wiki - Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

### 1. FAISS 的本质与边界
FAISS（Facebook AI Similarity Search）是一个用 C++ 编写（提供 Python 绑定的）**向量计算与索引算法库**。
- **FAISS 是什么**：纯粹的数学引擎，专注于高维向量（`float32` 矩阵）的邻近搜索。
- **FAISS 不是什么**：它**不是**数据库！它默认**不存储原始文本**，**不存储 Metadata 标量属性**，也**不支持 SQL/属性过滤条件**。原始文本与 Metadata 的对应关系必须由外部程序自行维护。

### 2. 核心索引类型：IndexFlatL2 与 IndexFlatIP

今天仅需聚焦暴力检索索引（Flat Index），无需接触复杂的聚类倒排（IVF）或量化（PQ）：

1. **`faiss.IndexFlatL2(d)`**：
   - 算法：计算欧氏距离（L2 距离）平方 $\sum (x_i - y_i)^2$。
   - 规则：**距离越小越相似**（0 表示完全相同）。
2. **`faiss.IndexFlatIP(d)`**：
   - 算法：计算点积（Inner Product）$\sum x_i y_i$。
   - 规则：**分数越高越相似**。
   - 最佳实践：当向量经过 L2 归一化（模长为 1）时，点积即为**余弦相似度**。推荐作为语义检索的首选。

### 3. 从建索引到存盘加载的标准流程

```python
import faiss
import numpy as np

# 设定维度
d = 384

# 1. 创建索引（Flat 索引无需 train，直接可用）
index = faiss.IndexFlatIP(d)
print(f"索引是否已训练: {index.is_trained}")  # True
print(f"初始向量总数: {index.ntotal}")        # 0

# 2. 准备向量库数据（必须为 float32 类型的连续 numpy 数组）
# 模拟 1000 条 384 维度的向量
xb = np.random.random((1000, d)).astype('float32')
faiss.normalize_L2(xb)  # 原地归一化

# 3. 添加向量到索引
index.add(xb)
print(f"添加后向量总数: {index.ntotal}")       # 1000

# 4. 执行 Top-K 相似度检索
k = 3
xq = np.random.random((1, d)).astype('float32')
faiss.normalize_L2(xq)

distances, indices = index.search(xq, k)
print("检索得分 (Top 3):", distances)        # shape: (1, 3)
print("检索索引 ID (Top 3):", indices)       # shape: (1, 3)，对应 xb 中的行号

# 5. 索引存盘持久化
index_file_path = "vector_store.index"
faiss.write_index(index, index_file_path)
print(f"索引已保存至: {index_file_path}")

# 6. 从磁盘重新加载索引
loaded_index = faiss.read_index(index_file_path)
print(f"加载后索引中的向量总数: {loaded_index.ntotal}")
```

### 4. 关键注意事项与避坑指南
1. **数据类型严格限制**：FAISS 只接收 `numpy.float32`，传入 `float64` 会直接触发 C++ 运行时异常。
2. **C-Contiguous 内存布局**：切片或转置后的数组需要使用 `np.ascontiguousarray()` 确保内存连续。
3. **ID 管理**：`IndexFlat` 默认分配从 `0` 到 `N-1` 的自增整数索引。你必须在外部用字典或列表记录 `id_to_text = {0: "doc0_content", ...}`。

---

## 三、Chroma：一站式轻量级向量数据库认知

- 官方文档：[Chroma Documentation](https://docs.trychroma.com/)

### 1. 为什么需要 Chroma？
在用 FAISS 构建系统时，你会发现必须自己编写大量胶水代码：
- 手动调用 Sentence Transformers 编码。
- 维护从 FAISS 内部 ID 到原始文本的映射。
- 维护文章时间、作者、分类等元数据（Metadata）。
- 想要按 `category == 'tech'` 过滤时，FAISS 无法原生支持。
- 手动管理 `.index` 文件与元数据 SQLite/JSON 文件的同步保存。

**Chroma 的核心价值**：把 **「向量计算 + 原始文本 + 元数据存储与过滤 + 本地持久化」** 一揽子打包管起来。

### 2. Chroma 的核心特性
- **开箱即用（In-Process & Zero DevOps）**：轻量级嵌入式设计，无需启动独立的服务端进程，本地通过 DuckDB/SQLite 自动管理持久化。
- **自动嵌入**：向集合中插入纯文本列表时，Chroma 可内置自动调用默认或自定义 Embedding 模型生成向量。
- **标量元数据混合过滤**：支持基于属性的过滤查询（例如 `where={"source": "wiki", "year": {"$gte": 2024}}`）。

### 3. FAISS vs Chroma vs Milvus 对比全景图

| 维度 | FAISS | Chroma | Milvus (Day 12 预告) |
| :--- | :--- | :--- | :--- |
| **形态定位** | 底层向量计算与索引算法库 | 嵌入式/单机轻量级向量数据库 | 企业级云原生分布式向量数据库 |
| **部署方式** | Python 库（内存/单机文件） | In-Memory / 本地目录持久化 | 分布式微服务集群（K8s / Docker） |
| **元数据存储** | ❌ 不支持（需外挂数据库） | ✅ 原生支持（KV 属性） | ✅ 原生支持丰富标量类型与强类型 Schema |
| **元数据过滤** | ❌ 需自研前置/后置过滤 | ✅ 支持丰富条件过滤 | ✅ 高性能标量与向量混合联合查询 |
| **数据规模** | 百万至千万（受单机内存限制） | 百万级以内（适合原型、个人项目） | 百亿级（分布式水平伸缩、冷热分层） |
| **生产角色** | 算法研究底座 / 定制引擎核心 | 个人知识库、轻量 RAG 快速原型 | 企业级生产系统、高并发大规模搜索 |

---

## 四、今日代码实战：Sentence Transformers + FAISS 极简端到端

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 模拟文本库
documents = [
    "大语言模型（LLM）基于 Transformer 架构构建。",
    "FAISS 提供了高效的高维向量稠密检索能力。",
    "Sentence Transformers 能够方便地生成句子级向量表征。",
    "卷积神经网络（CNN）广泛应用于计算机视觉领域。",
    "Chroma 是一种便于本地开发的轻量级向量存储方案。"
]

# 2. 向量化
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, normalize_embeddings=True)
doc_embeddings = np.ascontiguousarray(doc_embeddings, dtype=np.float32)

# 3. 构建 IndexFlatIP 索引
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

# 4. 模拟用户查询
query = "如何对文本进行向量相似度搜索？"
query_vec = embedder.encode([query], normalize_embeddings=True)
query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

# 5. 检索最匹配的前 2 条结果
top_k = 2
distances, indices = index.search(query_vec, top_k)

print(f"用户查询: {query}\n")
for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"Top {rank} (匹配度得分: {score:.4f}):")
    print(f"  内容: {documents[idx]}\n")
```
