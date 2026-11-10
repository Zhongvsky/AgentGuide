# Day 08 · Advanced RAG：Query Transformation（查询变换与重写）

> 2026-11-09（周一）｜第 8 / 60 个学习日｜第 2 周 · Advanced RAG 与生产级向量数据库
> **周度里程碑倒计时**：距离第 2 周里程碑（Day 14 · 混合检索+Reranker+Milvus 升级）还差 **6 个学习日**。

---

## 一、 承前启后与背景

在第 1 周中，我们完成了 Naive RAG 的全链路闭环——从文本加载切块、Embedding 向量化、FAISS 本地索引，到 FastAPI 接口封装与 Docker 容器化打包。
然而，在真实生产场景中，Naive RAG 有一个致命瓶颈：**直接拿用户的原始输入去检索向量库**。
用户的问题通常非常简短（如“坐高铁能买一等座吗？”），且以疑问句为主；而知识库内的法规条文全是以高度规范的专业陈述句撰写（如“第七条 乘坐交通工具的等级...司局级及相当职务人员...高铁一等座”）。这种**句式与词汇失配（Vocabulary & Style Mismatch）**会导致向量相似度严重下降甚至漏检。

今天正式开启第 2 周 **Advanced RAG** 进阶篇，核心攻克 **Query Transformation（查询变换）**：
1. **Multi-Query（多路查询重写与召回）**：利用 LLM 将原始 Query 从不同角度、不同关键词重写为 3~4 个衍生查询，多路并行检索后合并去重，解决提问视角单一与关键词漏检问题。
2. **HyDE（Hypothetical Document Embeddings，假设性文档嵌入）**：让 LLM 先“假想”一段回答文档，利用“文档搜文档”代替“疑问句搜文档”，击穿疑问句与陈述句之间的语义鸿沟。

---

## 二、 今日主题与学习目标

- **主题**：Query Transformation（查询改写与假设性文档生成）
- **目标**：基于 LangChain LCEL 与本地 Qwen 模型，实现 Multi-Query 与 HyDE 查询变换策略，并构建 A/B 对照实验，量化验证检索召回效果。

---

## 三、 精选参考资源（≤3 条）

1. **HyDE 原始论文**：《Precise Zero-Shot Dense Retrieval without Relevance Labels》（Gao et al., arXiv:2212.10496，深入理解为何“错误事实的假想文档”依然能在向量空间提供优异的邻近引导）：<https://arxiv.org/abs/2212.10496>
2. **LangChain 官方指南 · Multi-Query Retriever**（理解多角度 Query 扩展与输出解析的设计模式）：<https://python.langchain.com/docs/how_to/MultiQueryRetriever/>
3. **LlamaIndex 生产级 RAG 指南 · Query Transformations**（系统对比 Sub-Question、Rewrite-Retrieve-Read 与 HyDE 的适用场景）：<https://docs.llamaindex.ai/en/stable/optimizing/production_rag/#query-transformations>

---

## 四、 时间规划建议（约 2~2.5 小时）

| 时段 | 内容 | 目标与类型 |
|---|---|---|
| **0:00 - 0:35（35m）** | **原理破壁与架构心智**：精读 HyDE 与 Multi-Query 数据流，理解语义空间投影差异 | 读懂即可 |
| **0:35 - 1:35（60m）** | **代码动手**：在 `day08.py` 中用 LCEL 编写 Multi-Query 生成器、HyDE 生成器及检索合并器 | 动手实操 |
| **1:35 - 2:05（30m）** | **A/B 评测实跑**：用 2 组典型问题实测 Naive vs Multi-Query vs HyDE，打印对比表 | 动手实操 |
| **2:05 - 2:20（15m）** | **技术沉淀**：记录实验数据、延迟开销分析与面试思考题至 `Gemini-Study.md` | 总结归档 |

---

## 五、 今日动手清单（Checklist）

- [x] **动手 · Multi-Query 改写链**：编写 Prompt 驱动本地 Qwen 模型，将 1 个原始问题改写为 3 个不同视角的陈述/检索式 Query，并完成格式解析与合并去重。
- [x] **动手 · HyDE 假设性文档链**：编写假设生成 Prompt，生成一段 100~200 字的政策风格假想文段，提取其 Embedding 去检索向量库。
- [x] **动手 · A/B 实验评测**：设计评测函数，对比三种策略在《差旅费管理办法》数据集上的 Top-3 召回内容与相似度得分，输出量化日志。

---

## 六、 量化验收标准

执行 `uv run python day08.py`：
1. 终端成功打印出改写后的衍生 Query（3 条）与 HyDE 假想文档片段。
2. 针对指定语义鸿沟案例（如 `“出差坐高铁能不能报销一等座？”`），清晰输出三种策略的检索对比结果表格：
   - **Naive 检索**：Top-1 得分及对应文段片段。
   - **Multi-Query 检索**：去重合并后的候选集大小与命中精度。
   - **HyDE 检索**：Top-1 得分与命中文段。
3. 证明至少在 1 个复杂/口语化问题上，Multi-Query 或 HyDE 的 Top-1 匹配度较 Naive 方案有显著提升。

---

## 七、 周度里程碑倒计时

- **本周目标**：第 2 周 · Advanced RAG 与生产级向量数据库
- **里程碑节点**：Day 14（2026-11-23 周一）· 混合检索 + Reranker + Milvus 升级完成
- **今日积淀**：Query Transformation 是 Advanced RAG 的第一道关卡，今天产出的改写模块将直接作为 Day 9 混合检索的前置预处理器！

---

## 八、 今日实操验收与归档记录（已完成）

- [x] **验收状态**：已全流程通过实测验收（2026-11-09 21:28）。
- [x] **量化 A/B 实测成果**：
  - 测试用例「普通员工出差去北京，坐高铁能买一等座吗？」：
    - Naive 得分：`0.5786`
    - Multi-Query 得分：`0.6151`（+6.3%）
    - HyDE 得分：`0.7448`（+28.7% 语义跃迁显著提升）
  - 测试用例「去外地出差，每天伙食费能给补贴多少钱？」：
    - Naive 得分：`0.7699`
    - Multi-Query 得分：`0.7699`（正确去重保住最高分）
    - HyDE 得分：`0.7577`（高位吻合核心条款）
- [x] **排错与沉淀**：
  - 修复 `sorted(..., reverse=True)` 逆向排序缺陷；
  - 修复字典去重中“低分覆盖高分”的分支漏洞；
  - 完整实验日志与深度架构解析已沉淀至 `learning/Daily/Day-08-QueryTransformation/Gemini-Study.md`。
