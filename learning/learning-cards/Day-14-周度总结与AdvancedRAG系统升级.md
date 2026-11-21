# Day 14 · 第 2 周里程碑收官：工业级 Advanced RAG 系统大升级与端到端闭环

> 2026-11-23（周一）｜第 14 / 60 个学习日｜第 2 周 · Advanced RAG 与生产级向量数据库  
> **周度里程碑日**：今天正式达成**第 2 周核心里程碑——工业级混合检索 + Reranker + Milvus 生产级 RAG 系统全面升级完成！**

---

## 一、 承前启后与痛点剖析

在第 2 周的 Day 8 到 Day 13 中，我们分别攻克了 Advanced RAG 的几大核心离散组件：
- **Day 8**：通过 Query Transformation（Multi-Query 与 HyDE）解决了用户短 Prompt 表达语义贫瘠的痛点；
- **Day 9**：通过 BM25 + Dense 稠密向量混合检索与 Cross-Encoder 重排，攻克了单一语义向量对关键词与编号识别不足的痛点；
- **Day 10~11**：手撕 RAG Triad 评估体系（Faithfulness, Answer Relevance, Context Relevance），建立了科学的量化度量基准；
- **Day 12**：引入分布式云原生向量数据库 Milvus，通过 Schema 契约与 HNSW 标量前置过滤，消除了 FAISS 无法动态过滤与元数据丢失的弊端；
- **Day 13**：手撕版面重构与表格感知分块（Table-Aware Chunking），确保复杂文档与财报表格不再被腰斩。

**但在工业界真实生产中，这些能力绝不能是孤立的代码脚本！**
如果各组件没有被解耦并统一封装为高内聚、模块化（Modular RAG）的端到端管线（Pipeline），企业面对海量异构请求时，就会陷入“召回漏标量、精排卡并发、表格丢表头、答案难溯源”的灾难中。

今天 Day 14 作为**第 2 周里程碑收官日**，我们将所有进阶模块全面合体：**手撕一套可直接上线投产的工业级 `AdvancedRAGPipeline`，并与首周的 Naive RAG 进行全方位 A/B Benchmark 终极对决！**

---

## 二、 今日主题与学习目标

- **主题**：工业级 Advanced RAG 全链路工程化大升级与 A/B 量化验证
- **目标**：
  1. 掌握 **Modular RAG（模块化 RAG）** 生产级架构拓扑：数据接入层、双路检索层、重排融合层、端到端合成层；
  2. 手撕 **统一数据摄取与索引管线（Pipeline Ingestion）**：表格感知分块后，同时构建 BM25 词频倒排与 Milvus 强类型 HNSW 向量索引；
  3. 手撕 **工业级多路召回与单塔重排检索器（Modular Hybrid Retriever）**：Milvus 标量前置过滤 + 稠密检索与 BM25 稀疏检索双路并发，经 RRF 倒数排名融合后送入 Cross-Encoder 进行全交叉精排；
  4. 手撕 **带来源引用的结构化端到端问答器（Synthesizer with Citations）** 与全自动 A/B 量化评测，见证 Advanced RAG 对 Naive RAG 的全方位性能压制。

---

## 三、 精选参考资源（≤3 条）

1. **Modular RAG: Transforming RAG Systems into LEGO-like Architecture (论文与技术博客)**：<https://arxiv.org/abs/2312.10997>
2. **Milvus 官方最佳实践：Hybrid Search & Scalar Filtering**：<https://milvus.io/docs/multi-vector-search.md>
3. **LlamaIndex Advanced RAG Pipeline 指南**：<https://docs.llamaindex.ai/en/stable/optimizing/production_rag/>

---

## 四、 时间规划建议（约 2~2.5 小时）

- **架构心智模型与系统全景设计（30% · 40分钟）**：
  - 梳理 Modular RAG 的模块边界与各组件交互拓扑；
  - 搞懂为什么多路召回要先做 RRF 候选池初筛（Top-10 -> Top-6），再送入 Cross-Encoder 进行精排以平衡延迟（P99）与准确率。
- **核心代码动手实现（50% · 65分钟）**：
  - 【关卡 1】：手撕 `ModularAdvancedRAG.ingest`（文档流经过 Table-Aware 切片，双轨刷入 Milvus 与 BM25）；
  - 【关卡 2】：手撕 `ModularAdvancedRAG.retrieve`（Milvus 标量前置过滤向量检索 + BM25 稀疏检索 + RRF 融合初筛 + Cross-Encoder 精排）；
  - 【关卡 3】：手撕 `ModularAdvancedRAG.synthesize_and_benchmark`（上下文组装、带 Citations 引用回答与 Naive vs Advanced A/B 量化对比）。
- **实测对比与复盘（20% · 25分钟）**：
  - 运行 `uv run python learning/Daily/Day-14-周度总结与AdvancedRAG系统升级/day14.py`；
  - 见证在面对复杂含表公文与跨部门干扰场景下，Advanced RAG 展现的碾压级准确度与信噪比；
  - 庆祝第 2 周里程碑圆满收官！

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：在 `day14.py` 中手撕将切片同时写入 Milvus（带标量字段）和 BM25 稀疏索引的双轨摄取管线；
- [ ] **必做**：实现严格支持布尔过滤表达式（`filter_expr`）的混合检索（Hybrid Search）与 Cross-Encoder 二次重排逻辑；
- [ ] **必做**：运行端到端 A/B 测试用例，输出带有精确得分、命中判定与引用片段的响应；
- [ ] **进阶（读懂即可）**：评估生产环境下引入 Redis 缓存高频 Query 的精排结果，进一步压低 P99 延迟的设计思路。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-14-周度总结与AdvancedRAG系统升级/day14.py
```
**判定标准**：
1. 脚本无报错跑通，三大关卡全绿灯通过；
2. 关卡 1 成功将包含正文与表格的多模态文档切为结构化 Chunks 并同时入库 Milvus 与 BM25；
3. 关卡 2 在带有强标量过滤（如 `department == '财务部'`）时，精准拦截无关部门干扰项，且经 Cross-Encoder 精排后真实答案 Chunk 稳居 Top-1；
4. 关卡 3 成功输出带 Markdown 格式的完整问答和溯源引用，且 A/B 对照实验中 Advanced RAG 相比 Naive RAG 在相关度与精准度指标上实现显著提升，终端打印：
   `🎉 [MILESTONE 2 REACHED]: 第 2 周里程碑达成！工业级 Advanced RAG 系统升级圆满通关！`

---

## 七、 周度里程碑收官宣言

- **今天是第 2 周的核心收官日！**
- 完成今天的内容后，你不仅掌握了 RAG 的基础原理，更亲手构建过改写、混合检索、单塔精排、科学评测、Milvus 生产级存储、复杂版面解析六大核心技术；
- 下周我们将跨入全新篇章：**第 3 周 · Agent 开发与 Tool Calling（从检索走向自主行动）**！