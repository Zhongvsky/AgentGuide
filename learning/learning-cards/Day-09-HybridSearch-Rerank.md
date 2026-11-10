# Day 09 · Advanced RAG：混合检索与重排 (Hybrid Search & Reranking)

> 2026-11-11（周三 · 由 11-10 顺延）｜第 9 / 60 个学习日｜第 2 周 · Advanced RAG 与生产级向量数据库
> **周度里程碑倒计时**：距离第 2 周里程碑（Day 14 · 混合检索+Reranker+Milvus 升级）还差 **5 个学习日**。

---

## 一、 承前启后与痛点剖析

在 Day 8 中，我们手撕了 **Query Transformation（Multi-Query 与 HyDE）**，通过大模型重写与假想文档，有效弥合了用户口语化提问与规章陈述句之间的“文体与句式鸿沟”。

但在工业级生产场景中，单纯依赖稠密向量（Dense Embedding）检索依然有着致命的**阿喀琉斯之踵**：
1. **实体与数值盲区**：Dense 向量由高维语义空间降维聚合而成，擅长宏观语义泛化，却对**专有名词、规章条款编号、精确数值与代码缩写**（如“第七条”、“司局级”、“800元”、“BGE”）极不敏感。用户检索“第二十三条规定了什么”，向量检索往往返回第十六条或第十四条，因为它们的公文句式特征几乎完全相同。
2. **粗排语义粗糙（双塔 Bi-Encoder 的固有缺陷）**：Embedding 检索采用双塔架构（Query 与 Document 分别独立编码为固定长度向量后计算余弦相似度），缺乏 Query 与 Document 中各 Token 之间的全交叉注意力（Cross-Attention）交互，无法准确捕捉复杂的逻辑否定、条件约束与长上下文匹配。

**今天 Day 09 攻克 Advanced RAG 的核心重武器 —— 混合检索（Hybrid Search）与重排（Rerank）**：
- **Sparse（BM25 词频检索）+ Dense（向量语义检索）**：双路并行召回，取长补短。BM25 负责精准拦截专有名词与条款号，Embedding 负责兜底语义近义泛化。
- **RRF (Reciprocal Rank Fusion，倒数排名融合)**：工业界最优雅的无参数融合算法，抹平不同检索器分数尺度的差异，合并生成高质量粗排候选集。
- **Cross-Encoder Reranker（精排器）**：采用单塔全交互模型对候选分块打分，将最精准的证据块推向 Top-1。

---

## 二、 今日主题与学习目标

- **主题**：混合检索与重排 (Hybrid Search & Reranking)
- **目标**：
  1. 理解 Sparse 与 Dense 检索的底层数学原理与互补优势；
  2. 手撕基于 `jieba` 中文分词与 `rank_bm25` 的稀疏检索器；
  3. 手撕经典工业级 **RRF (Reciprocal Rank Fusion)** 倒数排名融合算法；
  4. 集成 **Cross-Encoder Reranker**，构建生产级两阶段（粗排过滤 -> 精排重排）检索漏斗；
  5. 运行 A/B 对照实验，量化验证纯 Dense、纯 BM25、Hybrid (RRF) 与 Hybrid + Rerank 四种模式在真实公文上的检索命中率与分数差异。

---

## 三、 精选参考资源（≤3 条）

1. **Modular RAG 经典综述**：《Modular RAG for Computing the Unseen: An In-Depth Survey》（Gao et al., arXiv:2407.21059 / arXiv:2312.10997，系统阐述 Retrieval-Reranking 两阶段解耦架构）：<https://arxiv.org/abs/2312.10997>
2. **LlamaIndex 生产级指南 · Hybrid Search & Rerank**（深入解析 Sparse/Dense 混合召回与 Cohere/BGE 重排器的落地实践）：<https://docs.llamaindex.ai/en/stable/examples/low_level/oss_hybrid_search/>
3. **BAAI FlagEmbedding 官方文档**（理解 Bi-Encoder 向量模型与 Cross-Encoder Reranker 交叉注意力重排器的算力/精度权衡）：<https://github.com/FlagOpen/FlagEmbedding>

---

## 四、 时间规划建议（约 2~2.5 小时）

| 时段 | 内容 | 目标与类型 |
|---|---|---|
| **0:00 - 0:35（35m）** | **架构心智模型**：精读 Sparse vs Dense 特征空间差异、RRF 算法数学推导、双塔与单塔计算复杂度对比 | 读懂即可 |
| **0:35 - 1:35（60m）** | **代码闯关手撕**：在 `day09.py` 中依次闯关手撕 BM25 检索器、RRF 融合算法、Cross-Encoder 精排器 | 动手实操 |
| **1:35 - 2:05（30m）** | **A/B 量化实测**：运行专有名词/条款号典型测试用例，打印四路检索对比表格，分析命中排序 | 动手实操 |
| **2:05 - 2:20（15m）** | **技术沉淀归档**：提炼工业级踩坑点、延迟优化策略与面试核心考点至 `Gemini-Study.md` | 总结归档 |

---

## 五、 今日动手清单（Checklist）

- [ ] **动手 · BM25 稀疏检索器**：使用 `jieba` 分词构建分块倒排语料库，实现 `bm25_retrieve` 函数，支持根据精确关键词快速检索 Top-K。
- [ ] **动手 · RRF 倒数排名融合算法**：实现公式 $RRF\_Score(d) = \sum_{m} \frac{1}{k + rank_m(d)}$（$k=60$），将 BM25 结果列表与 Dense 向量结果列表融合去重并重新排序。
- [ ] **动手 · Cross-Encoder Reranker 精排器**：实现基于单塔交叉编码器的重排逻辑，接收两阶段粗排候选集（Top-K），计算 Query-Doc 联合匹配分数并截取最终 Top-N。
- [ ] **动手 · A/B 实验量化评测**：在《中央和国家机关差旅费管理办法》真实语料上，对比纯 Dense、纯 BM25、Hybrid (RRF)、Hybrid + Reranker 四组策略的表现。

---

## 六、 量化验收标准

执行 `uv run python day09.py`：
1. 终端正常无报错运行，成功构建/加载 BM25 词频倒排索引与 FAISS 向量索引；
2. 针对包含特定实体/条款号的 Query（如 `“差旅费管理办法第七条规定的交通工具有哪些？”` 或 `“伙食补助费是按每天多少钱补助？”`），输出结构化对照表：
   - **纯 Dense 检索**：展示 Top-1 命中片段及得分（暴露条款号误匹配缺陷）；
   - **纯 BM25 检索**：展示精确命中“第七条”的 Top-1 片段及 BM25 分数；
   - **Hybrid (RRF)**：展示双路加权融合后的 Top-1 片段及 RRF 分数；
   - **Hybrid + Rerank**：展示经过 Cross-Encoder 语义精排后的最终分数与精准排序。
3. 证明混合检索 + Rerank 能够在保留语义泛化的同时，彻底纠正单纯 Dense 对专有名词/条款号的漂移。

---

## 七、 周度里程碑倒计时

- **本周目标**：第 2 周 · Advanced RAG 与生产级向量数据库
- **里程碑节点**：Day 14（2026-11-23 周一）· 混合检索 + Reranker + Milvus 升级完成
- **今日积淀**：混合检索与 Reranker 是目前几乎所有头部大模型厂商（OpenAI、Anthropic、智谱、百川）生产级知识库的标准基建。今天打磨好的 BM25 + Dense + Rerank 模块，将在 Day 12~14 直接迁移至 Milvus 生产集群！
