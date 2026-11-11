# Day 10 · Advanced RAG：RAG 科学评估体系与 Ragas 落地实战 (RAG Evaluation & Ragas)

> 2026-11-16（周一）｜第 10 / 60 个学习日｜第 2 周 · Advanced RAG 与生产级向量数据库  
> **周度里程碑倒计时**：距离第 2 周里程碑（Day 14 · 混合检索+Reranker+Milvus 升级）还差 **4 个学习日**。

---

## 一、 承前启后与痛点剖析

在前面的学习中，我们先后手撕了：
- **Day 05~06**：Naive RAG（向量检索 + LLM 拼装回答）；
- **Day 08**：Query Transformation（Multi-Query + HyDE 查询改写）；
- **Day 09**：混合检索与重排（BM25 词频倒排 + RRF 倒数排名融合 + Cross-Encoder 精排）。

**但是，在工业级生产落地中，算法工程师面临着一个最严峻的拷问**：
> “你凭什么说加了 Reranker 比没加更好？好多少？准确率从百分之几提升到了百分之几？回答有没有幻觉（Hallucination）？”

如果没有科学的量化评测体系：
1. **“感觉大法”不可靠**：凭肉眼看 2~3 个 Case 觉得回答顺畅，但实际上模型可能在胡编乱造，或者引入了严重的负向漂移（Negative Transfer）；
2. **传统 NLP 指标（BLEU / ROUGE）失效**：BLEU/ROUGE 基于字词 n-gram 重合度。若真实答案是“不可乘坐一等座”，模型回答“仅限乘坐二等座及以下席位”，字面上几乎零重合，BLEU 得分为 0，但语义完全正确！

**今天 Day 10 我们攻克 RAG 工业化落地的定海神针 —— RAG 科学评估体系（RAG Triad）与 Ragas 核心原理**：
- **解耦评估两阶段**：将 RAG 拆为**检索端（Retrieval）**与**生成端（Generation）**独立打分；
- **LLM-as-a-Judge（大模型裁判员）**：手撕工业级无参考（Reference-Free）与有参考评估链，通过命题原子化（Atomic Claims）拆解计算 **Faithfulness（忠实度）** 与 **Answer Relevance（回答相关度）**；
- **全流程打分流水线**：为 RAG 系统建立数字化的 Benchmark 评测台账。

---

## 二、 今日主题与学习目标

- **主题**：RAG 科学评估体系与 Ragas 落地实战 (RAG Evaluation & Ragas)
- **目标**：
  1. 深入理解 **RAG Triad（三元评估心智模型）**：Context Relevance、Faithfulness、Answer Relevance 的数学与工程定义；
  2. 理解 **LLM-as-a-Judge** 的工作原理与 Prompt 评测设计；
  3. 手撕 **Faithfulness（真实度/抗幻觉度）** 评估链：将回答拆解为原子陈述句，由裁判模型判定各句是否能由检索上下文严密推导；
  4. 手撕 **Answer Relevancy（回答相关性）** 评估链：基于逆向问题生成（Reverse Question Generation）与余弦相似度计算相关度；
  5. 运行完整评估流水线，对 Naive RAG 与 Advanced RAG 的输出进行多维度量化对比打分。

---

## 三、 精选参考资源（≤3 条）

1. **Ragas 官方权威文档**（生产级 RAG 自动化评估体系与评估指标权威定义）：<https://docs.ragas.io/>
2. **Ragas 经典论文**：《Ragas: Automated Evaluation of Retrieval Augmented Generation》（Es et al., arXiv:2309.15217，详细阐述 Faithfulness 原子化拆解数学模型）：<https://arxiv.org/abs/2309.15217>
3. **TruLens · RAG Triad 原理指南**（解析 Context Relevance、Groundedness、Answer Relevance 三元核心）：<https://www.trulens.org/getting_started/core_concepts/rag_triad/>

---

## 四、 时间规划建议（约 2~2.5 小时）

| 时段 | 内容 | 目标与类型 |
|---|---|---|
| **0:00 - 0:35（35m）** | **架构心智模型**：精读 RAG Triad 架构、原子命题拆解机制、LLM-as-a-Judge 提示词工程规范 | 读懂即可 |
| **0:35 - 1:35（60m）** | **代码闯关手撕**：在 `day10.py` 中依次闯关手撕原子句提取、Faithfulness 校验链与 Answer Relevancy 评估器 | 动手实操 |
| **1:35 - 2:05（30m）** | **量化评测对比**：运行 Benchmark 基准集，输出包含检索与生成维度的综合评测对比表 | 动手实操 |
| **2:05 - 2:20（15m）** | **技术沉淀归档**：提炼评估指标优化、裁判模型偏差防御策略至 `Gemini-Study.md` | 总结归档 |

---

## 五、 今日动手清单（Checklist）

- [ ] **动手 · 评估样本数据契约**：基于 Pydantic 规范定义 `EvaluationSample`（含 query, context, answer, ground_truth）与 `MetricScore`。
- [ ] **动手 · 原子化命题拆解与 Faithfulness 评估**：设计 Prompt 让裁判模型将生成的长回答拆分为原子事实陈述（Atomic Statements），逐条校验其在 Context 中的支撑性，计算 $\frac{\text{有效支撑数}}{\text{总陈述数}}$。
- [ ] **动手 · 逆向问题生成与 Answer Relevancy 评估**：让裁判模型依据生成的回答逆向反推 $N$ 个可能的问题，利用 Embedding 模型计算反推问题与原始 Query 的余弦相似度均值。
- [ ] **动手 · 自动化评估流水线运行**：对不同版本的 RAG 回答进行打分，输出直观的雷达型量化对比报告。

---

## 六、 量化验收标准

执行 `uv run python day10.py`：
1. 终端正常无报错运行，成功加载本地评估裁判模型与 Embedding 向量模型；
2. 针对包含真实答案与幻觉答案的不同测试样本，输出结构化评测指标表：
   - **忠实回答样本**：Faithfulness 达到 **0.90 ~ 1.00**，Answer Relevancy 达到 **0.85+**；
   - **典型幻觉样本（篡改金额/无中生有）**：Faithfulness 发生断崖式下跌（**< 0.40**），准确识别出未被 Context 支撑的幻觉陈述；
3. 输出完整的量化评测对比总结。

---

## 七、 周度里程碑倒计时

- **本周目标**：第 2 周 · Advanced RAG 与生产级向量数据库
- **里程碑节点**：Day 14（2026-11-23 周一）· 混合检索 + Reranker + Milvus 升级完成
- **今日积淀**：今天实现的 RAG 科学评估器，是后续 Day 12~14 系统重构验证的度量衡标尺，也是大厂高频考察的系统级工程能力！
