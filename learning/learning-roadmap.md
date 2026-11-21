# AI Agent 开发工程师 · 60 天学习路线图（工程落地版）

> 来源：https://github.com/adongwanai/AgentGuide/blob/main/docs/05-roadmaps/learning-roadmap-development.md
> 结构：8 周 / 60 天。每周 = 学习内容 + 手撕打卡清单 + 解锁技能 + 每日计划。
> 第 7、14、21、28、35、42 天为周度总结日；第 43 天起进入项目实战。

## 阶段总览

| 周次 | 阶段主题 | 核心内容 |
|---|---|---|
| 第1周 | 大模型应用开发基础 + 手撕 Naive RAG | FastAPI、LangChain、Naive RAG |
| 第2周 | Advanced RAG 与生产级向量数据库 | Query 改写、混合检索、RAGAs 评估、Milvus |
| 第3周 | Agent 开发与 Tool Calling | ReAct、Function Calling、Memory、错误处理 |
| 第4周 | 系统性能优化 | Redis 缓存、异步、批处理、vLLM |
| 第5周 | 监控、可观测性与部署 | LangSmith、Prometheus、Grafana、Docker |
| 第6周 | Multi-Agent 系统开发 | AutoGen、CrewAI |
| 第7-8周 | 工业级项目实战与面试准备 | 两大专属代表作（计算摄影/ISP 故障排查 RAG + 画质诊断 Agent）+ 简历 + 面试 Mock |

---

## 一、每日计划表（Day 1-60）

### 第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 1 | FastAPI 快速入门 | FastAPI Official Tutorial (https://fastapi.tiangolo.com/tutorial/) | 掌握 FastAPI 基础，能创建路由、处理请求 |
| Day 2 | LangChain 核心概念 | LangChain Quickstart (https://python.langchain.com/docs/tutorials/)、吴恩达 LangChain 课程、Building Systems with the ChatGPT API | 理解 LangChain 六大核心模块，熟练使用 LCEL |
| Day 3 | RAG Part 1：加载与分割 | LlamaIndex Loaders (https://docs.llamaindex.ai/)、Unstructured.io、MinerU、Docling | 掌握 PDF/MD 等格式文档的加载与文本分块策略 |
| Day 4 | RAG Part 2：向量化与存储 | FAISS Intro (https://faiss.ai/)、Sentence Transformers | 理解 Embedding 原理，用 FAISS/Chroma 构建本地向量索引 |
| Day 5-6 | 手撕 Naive RAG 系统 | RAG from Scratch、LLM Powered Autonomous Agents、动手学大模型应用开发、LLM 入门教程 | 整合 FastAPI + LangChain，完成端到端文档问答 API |
| Day 7 | 周度总结与项目部署 | — | 将本周 RAG 项目用 Docker 打包并成功运行 |

**本周手撕清单**
- [ ] FastAPI 搭建 "Hello, World" API 服务
- [ ] LangChain LCEL 编写第一个 LLM Chain
- [ ] 30 分钟手撕一个完整的 Naive RAG 应用

### 第 2 周 · Advanced RAG 与生产级向量数据库

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 8 | Query Transformation | LlamaIndex Query Transformations | 实现 HyDE、Multi-Query 等查询改写策略 |
| Day 9 | 混合检索与重排 (Rerank) | LlamaIndex Cohere Rerank、论文 Modular RAG | 实现 BM25 + Embedding 混合检索并集成 Reranker |
| Day 10-11 | RAG 评估体系 | RAGAs (https://docs.ragas.io/)、FlashRAG、DeepEval、Lighteval | 掌握 RAG 核心评估指标，用 RAGAs 对比优化前后性能 |
| Day 12 | 生产级向量数据库 (Milvus) | Milvus Quick Start (https://milvus.io/docs)、Infinity、Qdrant | Docker 部署 Milvus，掌握其 Python SDK |
| Day 13 | 高级数据处理 | Unstructured.io、MinerU、PDF-Extract-Kit、Docling、GOT-OCR2.0 | 用 Unstructured/MinerU 解析含表格、图片的复杂 PDF |
| Day 14 | 周度总结与系统升级 | — | 将 RAG 系统升级，集成混合检索、Reranker 和 Milvus |

**本周手撕清单**
- [ ] 实现 BM25 + 向量的混合检索
- [ ] 引入 Cohere Rerank 提升检索精度
- [ ] 用 RAGAs 评估 Faithfulness 与 Answer Relevancy
- [ ] Docker 部署 Milvus 并完成增删改查

### 第 3 周 · Agent 开发与 Tool Calling

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 15 | Agent 核心概念 | LLM Powered Autonomous Agents、LangChain Agents、论文 ReAct | 理解 ReAct 框架，跑通一个官方 Agent 示例 |
| Day 16 | 自定义工具开发 | LangChain Custom Tools、MCP 协议、MCP 教程 | 编写天气查询自定义工具并集成进 Agent |
| Day 17 | SQL & 数据库工具 | LangChain SQL Agent | 构建自然语言查询数据库的 SQL Agent |
| Day 18 | Function Calling 实战 | OpenAI Function Calling、GPT Best Practices | 用 OpenAI API 实现按用户问题调用函数的 Agent |
| Day 19 | Agent Memory | LangChain Memory、Mem0、MemoryScope | 为 Agent 添加对话历史记忆 |
| Day 20 | Agent 错误处理 | Error Handling in Agents | 为工具调用加重试 (tenacity) 与降级策略 |
| Day 21 | 周度总结与项目构建 | — | 构建集成 RAG + Web 搜索的「ISP 算法与代码研究助手」Agent |

**本周手撕清单**
- [ ] 实现 3 个自定义工具（天气、SQL 查询、API 调用）
- [ ] 基于 LangChain 构建可链式调用工具的 Agent
- [ ] 用 OpenAI Function Calling 实现结构化数据提取

### 第 4 周 · 系统性能优化

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 22 | 性能瓶颈分析 | py-spy、Scalene | 用 cProfile/py-spy 定位现有系统性能瓶颈 |
| Day 23 | 缓存优化 (Redis) | FastAPI with Redis、LiteLLM Caching | 为 Agent 系统加 Redis 缓存，缓存 LLM 响应 |
| Day 24-25 | 异步处理 (Async) | FastAPI Async、LangChain Async | 将 I/O 密集操作改造为异步 |
| Day 26 | 批处理优化 (Batching) | Batch Processing | 实现 Embedding/Reranker 批处理，提升吞吐 |
| Day 27 | 高性能推理 (vLLM) | vLLM Quickstart (https://docs.vllm.ai/)、SGLang、TensorRT-LLM、LMDeploy | 用 vLLM 部署开源模型并测吞吐 |
| Day 28 | 周度总结与性能压测 | locust、jmeter | 压测并记录 QPS、P99 等指标 |

**本周手撕清单**
- [ ] 为 RAG 系统引入 Redis 缓存，对比优化前后性能
- [ ] 将 FastAPI 同步接口改造为异步接口
- [ ] 部署 vLLM 并通过 API 推理

### 第 5 周 · 监控、可观测性与部署

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 29 | 链路追踪 (LangSmith) | LangSmith、OpenTelemetry、LangFuse | 集成 LangSmith，分析调用链路与延迟 |
| Day 30 | 指标监控 (Prometheus) | Prometheus Python Client、FastAPI Instrumentator | 暴露 QPS、延迟、错误率等核心指标 |
| Day 31 | 可视化 (Grafana) | Grafana Dashboard | 安装 Grafana，创建监控大盘 |
| Day 32 | 容器化 (Docker) | Docker for FastAPI、Docker Best Practices | 为 FastAPI 写 Dockerfile 并成功构建镜像 |
| Day 33 | 服务编排 (Docker Compose) | Docker Compose、Full Stack FastAPI | 编写 docker-compose.yml 一键启动应用栈 |
| Day 34 | 日志系统 | Python Logging、Loguru、structlog | 日志以 JSON 输出，为接入 ELK 做准备 |
| Day 35 | 周度总结与生产环境模拟 | — | 模拟线上故障，用本周工具链定位问题 |

**本周手撕清单**
- [ ] 集成 LangSmith，追踪每一步调用与延迟
- [ ] 用 Prometheus 暴露自定义指标（Token 消耗、缓存命中率）
- [ ] Docker Compose 一键部署 FastAPI + Milvus + Redis

### 第 6 周 · Multi-Agent 系统开发

| 天数 | 学习主题 | 资源 | 目标 |
|---|---|---|---|
| Day 36-37 | AutoGen 核心概念 | AutoGen Tutorial、论文 AutoGen Framework | 掌握 ConversableAgent、GroupChat 等概念 |
| Day 38 | AutoGen 实战 | AutoGen Examples | 实现「研究员-程序员-测试员」多智能体系统 |
| Day 39-40 | CrewAI 核心概念 | CrewAI Docs、CrewAI Quickstart | 掌握 Agent/Task/Crew/Process 概念 |
| Day 41 | CrewAI 实战 | CrewAI Examples | 实现「旅行规划师-本地向导-预订专员」系统 |
| Day 42 | 框架对比与总结 | agentUniverse、AgentScope、Qwen-Agent、Lagent、PraisonAI | 对比 AutoGen 与 CrewAI 的设计哲学与适用场景 |

**本周手撕清单**
- [ ] 用 AutoGen 构建「研究员-程序员-测试员」协作团队
- [ ] 用 CrewAI 构建「旅行规划师-本地向导-预订专员」

### 第 7-8 周 · 工业级项目实战与面试准备

| 天数 | 学习主题 | 目标 |
|---|---|---|
| Day 43-47 | 项目一：工业级计算摄影与 ISP 故障排查专家 RAG | 需求分析、架构设计、ISP Spec/SOP 知识库构建与核心检索开发 |
| Day 48-51 | 项目一：性能优化与容器化部署 | Milvus 标量过滤、混合检索 Rerank、RAGAs 忠实度评估、FastAPI 异步封装、Docker 部署 |
| Day 52-56 | 项目二：基于 ReAct + 多模态工具链的画质故障自动诊断 Agent | 需求分析、自研 Python 图像诊断工具集开发、ReAct 闭环、根因推导与 C++ 调参建议生成 |
| Day 57-58 | 简历撰写与项目总结 | 按算法/开发复合岗量化写入两大专属代表作项目 |
| Day 59-60 | 系统设计与面试 Mock | 高频 Agent/RAG + 计算摄影系统设计题 + 1v1 模拟面试 |

---

## 二、项目详细要求

### 项目一：工业级计算摄影与 ISP 故障排查专家 RAG 系统
- 业务场景：解决相机、手机与车载 ISP 影像算法研发中的高难度画质缺陷排查难、知识碎片化痛点（如 FGF 边缘光晕 Halo、暗部噪声塌陷、高光截断偏色、去鬼影连通域双轮廓/漏检、频闪）。
- 数据源：ISP Pipeline 架构规范 + 光学/滤波算法原理（FGF/TMC/Deghosting/时空去噪） + 真实画质缺陷排障 SOP 手册 + C++ 模块参数配置字典（Milvus + SQLite/JSON）。
- 核心技术：
  - Milvus 强类型 Schema 与布尔标量物理前置过滤（按模块 `module=='TMC'`, 平台 `chip=='SA8155'`, 缺陷类型 `defect=='Halo'` 精准隔离）；
  - BM25 + BGE 向量混合检索与 Cross-Encoder 重排；
  - RAGAs 体系量化评估 Faithfulness 真实度与 Context Relevance 信噪比。
- 性能与部署：FastAPI 异步高并发接口、Redis 缓存高频提问、Docker Compose 容器化一键部署、Prometheus + Grafana 可观测性大盘。
- 简历亮点：打造行业首个 ISP 画质领域专家知识检索系统；长篇复杂 Spec 与宏配置检索从人工翻阅升级为秒级精准定位；P99 检索延迟 < 300ms，事实忠实度得分 > 0.90。

### 项目二：基于 ReAct + 多模态视觉工具链的画质故障自动诊断 Agent
- 业务场景：输入用户上报的画质 Bad Case 图像（或 ROI 区域）+ 当前 ISP Pipeline XML/JSON 配置文件，自动量化分析缺陷特征、匹配知识库定位根因，并输出针对特定 C++ 模块的调参建议。
- ReAct 决策与协作架构：
  - 感知分析层：自主调用自研 Python 图像量化分析工具包；
  - 知识推理层：结合项目一的 Milvus 知识库，根据量化异常指标逆向推导物理机理（如判定是否为 FGF 8x 上采样在负斜率边缘协方差为负）；
  - 方案输出层：输出结构化 Markdown 排障报告、精准定位出问题的模块，给出建议调整的 XML 配置参数或 C++ 宏分支修复建议。
- 核心工具集（Tool Calling ≥ 5 个）：
  1. `calc_halo_sdf`：基于边缘法向环带符号距离场（SDF）计算人像边缘 Halo 溢出度与 P95 分位数；
  2. `calc_noise_envelope`：计算暗部局部方差与信噪比包络，评估噪声塌陷程度；
  3. `calc_color_cast`：计算高光/暗部 RGB-YUV 色差与偏色角分布；
  4. `parse_isp_config`：解析并提取当前模块滤波与增益配置参数字典；
  5. `calc_ghost_metric`：多帧对齐差分与移动物体置信度校验。
- 稳定性保障：Tenacity 异常自动重试、Pydantic 结构化输出强约束、模型幻觉防范与降级容错，复杂诊断任务成功率 > 95%。
- 简历亮点：行业罕有的“计算摄影底层 + AI Agent 工具链”复合落地代表作；将资深算法工程师排查复杂画质 Bad Case 的耗时从“天级”缩短至“分钟级”；兼具深厚数理功底与前沿 Agent 架构落地能力。

---

## 三、核心资源

- 框架：LangChain、LlamaIndex、Dify ｜ Agent：AutoGen、CrewAI、AgentScope
- 向量库：Milvus、Qdrant、Chroma ｜ 推理：vLLM、SGLang、Ollama
- 评估：RAGAs、DeepEval、LangSmith
- 教程：Hello-Agents（Datawhale）、All-in-RAG（Datawhale）、动手学大模型应用开发、nanoGPT

## 四、简历三件套（第 57-58 天用）

1. ≥ 2 个端到端可运行、有容器化部署与实测指标的专属领域项目（ISP 知识库 RAG + 画质自动诊断 Agent）
2. 量化业务指标：故障定位效率提升 90%（天级 -> 分钟级）、P99 检索延迟 < 300ms、RAGAs 忠实度 0.90+、并发吞吐提升
3. 生产级技术栈：FastAPI + LangChain + Milvus + Redis + Docker + Prometheus + Python 视觉分析算子库
