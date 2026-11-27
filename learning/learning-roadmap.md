# AI Agent 开发工程师 · 60 天学习路线图（Day 18–60 夜间实战优化版）

> 来源基线：<https://github.com/adongwanai/AgentGuide/blob/main/docs/05-roadmaps/learning-roadmap-development.md>  
> 结构：8 周 / 60 个学习日（每周一、二、三夜间推进，约 20 周完成）。  
> **Day 18–60 节奏重构说明（2026-11-27 生效）**：  
> - 将剩余 43 个单元（Day 18–60）定义为 **43 个“2 小时夜间里程碑单元”**（扣除打开环境与收尾，**有效编码+调试时间约 90 分钟**）。  
> - 白天通过学习任务卡提前完成概念理解与 Schema/架构设计；晚上**零思考阻力直接手撕代码**，每晚只攻克 **1 个单一、可验证的闭环模块**（通常 2–3 个紧凑关卡）。  
> - 跳过重型本地运维（本地部署 vLLM/SGLang/TensorRT-LLM/LMDeploy、自托管 Langfuse+Grafana+ELK）及过时框架（旧版 LangChain Memory、AutoGen、CrewAI 多天 Demo），聚焦 **Context Engineering + MCP + LangGraph + Prompt/Redis 缓存 + SSE + Langfuse Cloud + VLM 多模态 + Agent Eval**，并将节省的 6 晚全部投入两大计算摄影/ISP 工业级项目实战。

## 阶段总览

| 周次 | 学习日范围 | 阶段主题 | 核心内容与技术栈 |
|---|---|---|---|
| 第1周 | Day 1–7（已完成） | 大模型应用开发基础 + 手撕 Naive RAG | FastAPI、LangChain LCEL、FAISS、Docker |
| 第2周 | Day 8–14（已完成） | Advanced RAG 与生产级向量数据库 | HyDE/Multi-Query、BM25+Rerank、RAGAs、Milvus、版面解析 |
| 第3周 | Day 15–21（4 晚收尾） | Agent 开发、Context Engineering 与 MCP | ReAct、Tool Calling、Text-to-SQL、Function Calling、Context Engineering、运行时熔断、MCP Server |
| 第4–5周 | Day 22–29（压缩为 8 晚） | 工程性能、流式交互、可观测与部署 | cProfile/py-spy、Prompt Caching + Redis、Async/SSE、Batching、Langfuse Cloud + Prometheus + structlog、Docker Compose + Locust |
| 第6周 | Day 30–37（重构为 8 晚） | LangGraph + VLM 多模态 + Agent Eval | LangGraph（状态路由/ToolNode/Checkpointer+HITL/Subgraph）、VLM 定性初筛+Python算子定量复核（MCP）、Golden Dataset + 自动化离线评测 |
| 第7–8周 | Day 38–60（扩容为 23 晚） | 两大工业级代表作与面试冲刺 | 项目一（8晚：ISP 故障排查专家 RAG）、项目二（11晚：画质故障自动诊断 Agent）、简历量化 + 架构图 + 系统设计 + Mock 面试（4晚） |

---

## 一、每日计划表（Day 1–60）

### 第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG（已完成）

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 1 | FastAPI 快速入门 | FastAPI Official Tutorial (https://fastapi.tiangolo.com/tutorial/) | 掌握 FastAPI 基础，能创建路由、处理请求 |
| Day 2 | LangChain 核心概念 | LangChain Quickstart (https://python.langchain.com/docs/tutorials/)、吴恩达 LangChain 课程、Building Systems with the ChatGPT API | 理解 LangChain 六大核心模块，熟练使用 LCEL |
| Day 3 | RAG Part 1：加载与分割 | LlamaIndex Loaders (https://docs.llamaindex.ai/)、Unstructured.io、MinerU、Docling | 掌握 PDF/MD 等格式文档的加载与文本分块策略 |
| Day 4 | RAG Part 2：向量化与存储 | FAISS Intro (https://faiss.ai/)、Sentence Transformers | 理解 Embedding 原理，用 FAISS/Chroma 构建本地向量索引 |
| Day 5-6 | 手撕 Naive RAG 系统 | RAG from Scratch、LLM Powered Autonomous Agents、动手学大模型应用开发、LLM 入门教程 | 整合 FastAPI + LangChain，完成端到端文档问答 API |
| Day 7 | 周度总结与项目部署 | — | 将本周 RAG 项目用 Docker 打包并成功运行 |

**本周手撕清单**
- [x] FastAPI 搭建 "Hello, World" API 服务
- [x] LangChain LCEL 编写第一个 LLM Chain
- [x] 30 分钟手撕一个完整的 Naive RAG 应用

### 第 2 周 · Advanced RAG 与生产级向量数据库（已完成）

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 8 | Query Transformation | LlamaIndex Query Transformations | 实现 HyDE、Multi-Query 等查询改写策略 |
| Day 9 | 混合检索与重排 (Rerank) | LlamaIndex Cohere Rerank、论文 Modular RAG | 实现 BM25 + Embedding 混合检索并集成 Reranker |
| Day 10-11 | RAG 评估体系 | RAGAs (https://docs.ragas.io/)、FlashRAG、DeepEval、Lighteval | 掌握 RAG 核心评估指标，用 RAGAs 对比优化前后性能 |
| Day 12 | 生产级向量数据库 (Milvus) | Milvus Quick Start (https://milvus.io/docs)、Infinity、Qdrant | Docker 部署 Milvus，掌握其 Python SDK |
| Day 13 | 高级数据处理 | Unstructured.io、MinerU、PDF-Extract-Kit、Docling、GOT-OCR2.0 | 用 Unstructured/MinerU 解析含表格、图片的复杂 PDF |
| Day 14 | 周度总结与系统升级 | — | 将 RAG 系统升级，集成混合检索、Reranker 和 Milvus |

**本周手撕清单**
- [x] 实现 BM25 + 向量的混合检索
- [x] 引入 Cross-Encoder Rerank 提升检索精度
- [x] 用 RAGAs 评估 Faithfulness 与 Answer Relevancy
- [x] Milvus-Lite 部署并完成 Schema 与标量过滤检索

### 第 3 周 · Agent 开发、Context Engineering 与 MCP（Day 15–17 已完成，Day 18–21 共 4 晚收尾）

| 单元 | 学习主题 | 核心资源 | 今晚单一目标与交付边界（~2h，有效编码 ~90min） |
|---|---|---|---|
| **Day 15** | Agent 核心概念 | LLM Powered Autonomous Agents、论文 ReAct | （已完成）手撕 ToolRegistry、ReActParser 与 ReActAgentLoop 闭环 |
| **Day 16** | 自定义工具开发 | LangChain Custom Tools、MCP 协议概念 | （已完成）手撕 `@tool` 与 `BaseTool` 双模式强类型工具及并发调度 |
| **Day 17** | SQL & 数据库工具 | SQLite DDL 探查、只读连接隔离 | （已完成）手撕安全只读 Text-to-SQL Agent 多表联查闭环 |
| **Day 18** | Function Calling 实战 | OpenAI Function Calling & Structured Outputs | 关卡 1–3：Strict Schema 抽取 `ISPBugTicket` + 原生 `tool_calls` 循环与分发表 + 两段式分诊（若调试超时，可将关卡 3 顺延至次日 30min 收尾） |
| **Day 19** | Context Engineering 与记忆 | Context Engineering for Agents、SQLite/JSON 状态持久化 | 实现上下文窗口裁剪/摘要、大工具输出截断策略，以及轻量会话状态持久化（跳过旧版 `LangChain Memory`） |
| **Day 20** | 容错与运行时熔断 | `tenacity` Docs、Agent Guardrails 实践 | 为原生 Agent 循环接入 `tenacity` 指数退避重试 + `max_steps` / Token 预算熔断护栏 + 工具异常结构化降级 |
| **Day 21** | ISP 研究助手 + MCP 初探 | Model Context Protocol Python SDK (`mcp`) | 搭建集成 RAG/检索的「ISP 研究助手」，并将 1 个本地查询工具封装成最小可运行 **MCP Server** 跑通调用 |

**本周手撕清单**
- [x] 实现 ReAct 循环、自定义工具与只读 Text-to-SQL Agent
- [ ] 用原生 Function Calling 实现 `ISPBugTicket` 抽取与两段式分诊
- [ ] 实现上下文裁剪/摘要、大工具输出截断与轻量持久化
- [ ] 接入 `tenacity` 重试与 `max_steps`/Token 预算熔断护栏
- [ ] 封装 1 个最小可运行 MCP Server 并接入研究助手

---

### 第 4–5 周 · 工程性能、流式交互、可观测与部署（Day 22–29，压缩为 8 晚）

> **架构精简说明**：跳过重型本地运维（不本地部署 vLLM/SGLang/TensorRT-LLM/LMDeploy，不自托管 Langfuse+Grafana+ELK），聚焦高 ROI 云端/轻量方案。  
> **周末前置准备提示**：建议在 **Day 27 前的周末**提前注册 **Langfuse Cloud（免费版）** 并获取 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`，避免占用工作日夜间编码时间。

| 单元 | 主题 | 核心资源 | 今晚单一目标与交付边界（~2h，有效编码 ~90min） |
|---|---|---|---|
| **Day 22** | 性能 Profiling | Python `cProfile` / `py-spy` | 用 `cProfile` / `py-spy` 分析现有链路耗时占比（拆清 LLM RTT vs 检索 vs 本地解析） |
| **Day 23** | 分层缓存优化 | API Prompt Caching、Redis Python (`redis.asyncio`) | 验证 API 级 **Prompt Caching** 命中条件（固定 System/Tools 前缀）；用 Redis 缓存高频 RAG 查询与确定性工具结果 |
| **Day 24** | FastAPI 异步化 | FastAPI Async、`AsyncOpenAI` / `httpx` | 将同步 HTTP / OpenAI 调用迁移至 `AsyncOpenAI` / `httpx` + `asyncio.gather` 并发执行独立工具 |
| **Day 25** | SSE 流式输出 | FastAPI `StreamingResponse`、SSE 协议 | 基于 FastAPI `StreamingResponse` 实现 SSE，实时向前端推送 Agent 推理步骤进度与最终流式输出 |
| **Day 26** | Batching 与推理原理 | SentenceTransformers Batching、KV Cache / PagedAttention | 实现 Embedding / Rerank 批处理；精读 KV Cache、PagedAttention、Prefix Caching 与 TTFT/TPOT（不折腾本地多引擎部署） |
| **Day 27** | 可观测性（Langfuse + Prometheus） | Langfuse Cloud、`prometheus-fastapi-instrumentator` | 接入 **Langfuse Cloud 免费版** 追踪调用链与 Token 成本 + 用 `prometheus-fastapi-instrumentator` 暴露 `/metrics` |
| **Day 28** | 结构化日志与单机镜像 | `structlog`、Docker Multi-stage Build | 集成 `structlog` 输出 JSON 日志 + 为 FastAPI 编写多阶段 `Dockerfile` 并本地构建启动成功 |
| **Day 29** | Docker Compose 与压测 | Docker Compose、`locust` | 编写 `docker-compose.yml` 一键拉起 `FastAPI + Redis + Milvus` + 使用 Locust 压测并记录基准 QPS/P99 |

**本阶段手撕清单**
- [ ] 完成链路 Profiling 耗时拆解（LLM RTT vs 检索 vs 本地解析）
- [ ] 验证 Prompt Caching 前缀命中，并接入 Redis 缓存高频 RAG/工具结果
- [ ] 改造 FastAPI 异步并发与 SSE 步骤流式推送
- [ ] 接入 Langfuse Cloud Trace + Prometheus `/metrics` + `structlog` JSON 日志
- [ ] Docker Compose 一键拉起全栈并用 Locust 记录基准 QPS/P99

---

### 第 6 周 · LangGraph + VLM 多模态 + Agent Eval（Day 30–37，重构为 8 晚）

> **架构重构说明**：砍掉已归档/冗余的 AutoGen 与 CrewAI 多天 Demo；8 晚 100% 为项目二（画质故障自动诊断 Agent）复用服务。  
> **周末前置准备提示**：建议在 **Day 34 前的周末**准备好 3–5 张模拟/脱敏画质测试图（含局部 ROI），在 **Day 36 前的周末**梳理好 Golden Dataset 的初步输入输出样例。

| 单元 | 主题 | 核心资源 | 今晚单一目标与交付边界（~2h，有效编码 ~90min） |
|---|---|---|---|
| **Day 30** | LangGraph ①：状态与路由 | LangGraph Docs (`StateGraph`) | 掌握 `StateGraph`、Pydantic/TypedDict State、Node 与条件边（Conditional Edges），跑通基础路由闭环 |
| **Day 31** | LangGraph ②：循环纠错闭环 | LangGraph `ToolNode` & Edges | 实现 `ToolNode` 与报错自动回溯重试环，并与 Day 18 手写 `while` 循环做结构对比 |
| **Day 32** | LangGraph ③：断点恢复与 HITL | LangGraph Persistence & `interrupt` | 接入 `Checkpointer`（SQLite/Memory）实现会话断点续跑，跑通 `interrupt` **Human-in-the-Loop** 人工审批流程 |
| **Day 33** | LangGraph ④：Subgraph 与协作模式 | LangGraph Multi-Agent / Supervisor | 用 Subgraph / Supervisor 模式实现多角色分工（覆盖 Multi-Agent 概念），并整理 LangGraph vs CrewAI/MAF 选型对比 |
| **Day 34** | VLM 多模态 ①：图文输入与初筛 | OpenAI Vision / 多模态消息格式、Pillow | 实现图像缩放/Base64 传参、控制多图 Token 成本，用 VLM 输出结构化画质缺陷初筛结果（缺陷类型 + ROI 区域） |
| **Day 35** | VLM 多模态 ②：视觉+算子协同 | MCP Server + Python 图像算子 | 跑通“VLM 定性初筛 → 调用确定性 Python 图像工具定量复核”闭环，并将视觉工具封装为 MCP Server |
| **Day 36** | Agent Eval ①：黄金集与轨迹采集 | Golden Dataset 设计、轨迹日志规范 | 构建 15–20 条带预期工具与结论的 Golden Dataset（含边界 Bad Case），编写批量运行与轨迹日志采集脚本 |
| **Day 37** | Agent Eval ②：自动化指标评测 | 离线 Eval 评分器 & LLM-as-a-Judge | 实现离线评分脚本（Tool 选择命中率、参数 Schema 合规率、轨迹收敛步数、LLM-as-a-Judge 诊断准确率） |

**本阶段手撕清单**
- [ ] 用 LangGraph 跑通 StateGraph 路由、ToolNode 纠错循环、Checkpointer + HITL、Subgraph/Supervisor
- [ ] 跑通 VLM 多模态结构化初筛 + 确定性 Python 工具复核（MCP Server 封装）
- [ ] 建成 15–20 条 Golden Dataset 与自动化离线评测脚本

---

### 第 7–8 周 · 两大工业级代表作与面试冲刺（Day 38–60，共 23 晚）

> **周末前置准备提示**：
> - **Day 38 前周末**：整理好脱敏/公开 ISP Spec 文档、排障 SOP 与 C++ 参数配置字典素材；
> - **Day 48 前周末**：整理好模拟 Bad Case 图片与脱敏 XML/JSON 配置文件，以及用于 Day 55–56 的 20–30 条 Agent Eval 黄金样本，避免占用工作日晚上 2 小时编码时间。

| 单元 | 主题 | 今晚单一目标与交付边界（~2h，有效编码 ~90min） |
|---|---|---|
| **Day 38** | 项目一 ①：语料与元数据规划 | 基于周末备好的脱敏/公开 ISP 文档与配置字典，完成结构化切分与 `module/chip/defect` 标签提取 |
| **Day 39** | 项目一 ②：Milvus Schema 与入库 | 定义 Milvus 强类型 Schema 与布尔标量过滤索引，完成批量向量化入库与前置过滤查询验证 |
| **Day 40** | 项目一 ③：混合检索与重排 | 实现 BM25 + BGE 向量双路召回 + Cross-Encoder Rerank 融合管线 |
| **Day 41** | 项目一 ④：RAGAs 评测调优 | 构建 15–20 条领域 QA 测试集，用 RAGAs 评测并调优 TopK/Chunk，目标 Faithfulness > 0.90 |
| **Day 42** | 项目一 ⑤：FastAPI 异步与缓存 | 封装异步问答 API，接入 Prompt Caching 与 Redis 高频检索缓存 |
| **Day 43** | 项目一 ⑥：可观测与日志接入 | 接入 Langfuse Trace、Prometheus `/metrics` 与 `structlog` 结构化日志 |
| **Day 44** | 项目一 ⑦：容器化与压测达标 | Docker Compose 一键部署项目一全栈，Locust 压测并优化至 P99 < 300ms |
| **Day 45** | 项目一 ⑧：收尾归档 | 整理实测性能报告、架构图、README 与复盘记录 |
| **Day 46** | 项目二 ①：核心算子 `calc_halo_sdf` | 实现基于边缘法向环带 SDF 的 Halo 量化算子 + 单元测试 |
| **Day 47** | 项目二 ②：核心算子 `calc_noise_envelope` | 实现暗部局部方差与信噪比包络量化算子 + 单元测试 |
| **Day 48** | 项目二 ③：轻量工具集与测试样本 | 实现 `calc_color_cast`、`calc_ghost_metric`、`parse_isp_config`（轻量确定性版本）+ 整理模拟 Bad Case 图与 XML |
| **Day 49** | 项目二 ④：MCP Server 封装联调 | 将 5 个诊断工具统一封装为标准 **MCP Server** 并验证工具发现与调用 |
| **Day 50–51** | 项目二 ⑤–⑥：LangGraph 主闭环 | 用 LangGraph 编排：`VLM 初筛 → MCP 视觉算子 → 联动项目一 Milvus 根因检索 → 推导 C++/XML 调参建议`（2 晚完成闭环与联调） |
| **Day 52** | 项目二 ⑦：业务护栏与 HITL | 加入 **参数名白名单/安全边界校验护栏**（防幻觉编造宏）、Pydantic 报告输出及高危改参 HITL 确认节点 |
| **Day 53** | 项目二 ⑧：SSE 流式交互接口 | FastAPI 封装项目二接口并支持 SSE 实时推送诊断步骤进度 |
| **Day 54** | 项目二 ⑨：全栈容器化联调 | Docker Compose 联调项目一（知识库服务）+ 项目二（诊断 Agent + MCP Server） |
| **Day 55–56** | 项目二 ⑩–⑪：Agent Eval 与稳定性调优 | 用 20–30 条构造 Bad Case 跑完离线 Agent Eval，补全降级容错与异常重试，记录实测成功率与耗时对比（2 晚完成） |
| **Day 57** | 简历量化打磨 | 按算法/AI 工程复合岗要求，用 STAR 法则写入两项目量化指标 |
| **Day 58** | 架构图与项目讲述 | 绘制两项目架构/数据流图，打磨 3 分钟项目亮点口述稿 |
| **Day 59** | 系统设计高频演练 | 重点演练：VLM vs 确定性算子边界、LangGraph vs Multi-Agent 选型、参数防幻觉护栏、Prompt Caching 优化 |
| **Day 60** | 1v1 模拟面试 | 对照两项目开展全真深挖问答与查漏补缺 |

---

## 二、项目详细要求

### 项目一：工业级计算摄影与 ISP 故障排查专家 RAG 系统（Day 38–45，8 晚）
- **业务场景**：解决相机、手机与车载 ISP 影像算法研发中的高难度画质缺陷排查难、知识碎片化痛点（如 FGF 边缘光晕 Halo、暗部噪声塌陷、高光截断偏色、去鬼影连通域双轮廓/漏检、频闪）。
- **数据源**：脱敏/公开 ISP Pipeline 架构规范 + 光学/滤波算法原理（FGF/TMC/Deghosting/时空去噪） + 真实画质缺陷排障 SOP 手册 + C++ 模块参数配置字典（Milvus + SQLite/JSON）。
- **核心技术**：
  - Milvus 强类型 Schema 与布尔标量物理前置过滤（按模块 `module=='TMC'`、平台 `chip=='SA8155'`、缺陷类型 `defect=='Halo'` 精准隔离）；
  - BM25 + BGE 向量混合检索与 Cross-Encoder 重排；
  - RAGAs 体系量化评估 Faithfulness 真实度（目标 > 0.90）与 Context Relevance 信噪比。
- **性能与可观测部署**：
  - FastAPI 异步高并发接口 + API 级 Prompt Caching + Redis 高频检索缓存；
  - Langfuse Cloud Trace + Prometheus `/metrics` + `structlog` JSON 日志；
  - Docker Compose 一键部署，Locust 压测优化至 P99 < 300ms。
- **简历亮点**：打造行业首个 ISP 画质领域专家知识检索系统；长篇复杂 Spec 与宏配置检索从人工翻阅升级为秒级精准定位；P99 检索延迟 < 300ms，事实忠实度得分 > 0.90。

### 项目二：基于 LangGraph + VLM + MCP 工具链的画质故障自动诊断 Agent（Day 46–56，11 晚）
- **业务场景**：输入用户上报的画质 Bad Case 图像（或 ROI 区域）+ 当前 ISP Pipeline XML/JSON 配置文件，通过 **“VLM 定性初筛 + 确定性 Python 算子定量复核”** 范式，联动项目一知识库逆向推导根因，并输出针对特定 C++ 模块的调参建议。
- **核心架构与编排（LangGraph + MCP）**：
  1. **VLM 定性初筛**：缩放传图与 Token 控制，初步识别缺陷类型与疑似 ROI 区域；
  2. **MCP 确定性工具层**（5 个工具统一封装为标准 **MCP Server**）：
     - **2 个核心物理算子做深**（含单元测试）：
       1. `calc_halo_sdf`：基于边缘法向环带符号距离场（SDF）计算人像边缘 Halo 溢出度与 P95 分位数；
       2. `calc_noise_envelope`：计算暗部局部方差与信噪比包络，评估噪声塌陷程度；
     - **3 个轻量干净确定性工具**（避免开发耗时失控）：
       3. `calc_color_cast`：计算高光/暗部 RGB-YUV 色差与偏色角分布；
       4. `calc_ghost_metric`：多帧对齐差分与移动物体置信度校验；
       5. `parse_isp_config`：解析并提取当前模块滤波与增益配置参数字典；
  3. **知识推理与业务护栏**：
     - 联动项目一 Milvus 知识库，结合量化异常指标检索物理机理；
     - **参数名白名单/安全边界校验护栏**（防幻觉编造不存在的 XML/C++ 宏与越界值）；
     - Pydantic 结构化报告输出 + 高危改参 **HITL（Human-in-the-Loop）** 人工确认节点；
  4. **流式交付与全栈容器化**：FastAPI + **SSE** 实时推送诊断步骤进度，Docker Compose 联调项目一与项目二。
- **离线评测与稳定性**：`tenacity` 指数退避重试 + 运行时步数/Token 熔断 + 降级容错；通过 20–30 条构造 Bad Case 的 **Agent Eval 黄金集**实测工具选择命中率、Schema 合规率、轨迹收敛步数与诊断准确率。
- **简历亮点**：行业罕有的“计算摄影底层 + VLM + MCP + LangGraph”复合落地代表作；将资深算法工程师排查复杂画质 Bad Case 的耗时从“天级”缩短至“分钟级”；兼顾物理算子确定性与 Agent 闭环防幻觉工程落地。

---

## 三、核心资源与技术栈

- **编排与协议**：LangGraph（`StateGraph` / `ToolNode` / `Checkpointer` / `Subgraph`）、MCP (Model Context Protocol)、OpenAI Function Calling & Structured Outputs（选型对比参考 CrewAI / Microsoft Agent Framework）
- **检索与数据**：Milvus（Milvus-Lite / Docker）、BM25、BGE Embedding、Cross-Encoder Rerank、PyMuPDF/MinerU
- **性能与服务**：FastAPI（Async + SSE `StreamingResponse`）、Redis、API Prompt Caching、Locust、Docker Compose
- **可观测与评估**：Langfuse Cloud、Prometheus (`/metrics`)、`structlog`、RAGAs、Agent Eval（Golden Dataset + LLM-as-a-Judge）
- **多模态与算子**：OpenAI Vision / 多模态 VLM、NumPy / OpenCV / Pillow 确定性图像量化算子

## 四、简历三件套（第 57–60 天用）

1. **2 套端到端可运行、Docker Compose 全栈联调的专属领域项目**：ISP 故障排查专家 RAG + 画质故障自动诊断 Agent
2. **真实可验证的量化指标**：RAG P99 < 300ms、RAGAs Faithfulness > 0.90、Agent Eval 黄金集工具命中率 / Schema 合规率 / 诊断成功率实测数据、故障定位效率从天级缩短至分钟级
3. **复合生产级技术栈**：FastAPI (Async + SSE) + LangGraph + MCP + VLM + Milvus + Redis + Prompt Caching + Langfuse Cloud + Prometheus + `structlog` + Docker Compose + Python 计算摄影视觉算子库
