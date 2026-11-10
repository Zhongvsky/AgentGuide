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
| 第7-8周 | 工业级项目实战与面试准备 | 两大项目 + 简历 + 面试 Mock |

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
| Day 21 | 周度总结与项目构建 | — | 构建集成 RAG + Web 搜索的「研究助手」Agent |

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
| Day 43-47 | 项目一：智能客服 RAG | 需求分析、架构设计、核心功能开发 |
| Day 48-51 | 项目一：优化与部署 | 性能优化、监控集成、Docker 部署、项目文档 |
| Day 52-56 | 项目二：自动化投研 Agent | 需求分析、Agent 设计、工具开发、工作流实现 |
| Day 57-58 | 简历撰写与项目总结 | 按开发岗模板量化写入两个项目 |
| Day 59-60 | 系统设计与面试 Mock | 高频系统设计题 + 1v1 模拟面试 |

---

## 二、项目详细要求

### 项目一：企业级智能客服 RAG 系统
- 业务场景：电商客服，自动回答 80% 重复问题（订单状态、物流、退款）
- 数据源：FAQ 文档 + 商品数据库 (PostgreSQL)
- 核心：混合检索 RAG，优先数据库精确查询，未命中再走文档模糊检索
- 性能目标：QPS > 200，P99 延迟 < 500ms
- 监控：LangSmith + Prometheus + Grafana 完整体系
- 部署：Docker Compose
- 简历亮点：高并发、低延迟、生产级监控、节省人力成本

### 项目二：Agent 驱动的自动化投研系统
- 业务场景：输入公司名，自动完成信息搜集、分析、报告撰写
- Multi-Agent：CrewAI 构建「信息搜集 Agent / 财报分析 Agent / 报告撰写 Agent」
- 工具集：Google Search、SEC API、文件读写等 ≥ 5 个工具
- 稳定性：异常处理 + 重试，任务成功率 > 95%
- 工作流：顺序工作流，记录每步中间产出
- 简历亮点：Multi-Agent 协作、复杂工作流自动化、效率提升量化

---

## 三、核心资源

- 框架：LangChain、LlamaIndex、Dify ｜ Agent：AutoGen、CrewAI、AgentScope
- 向量库：Milvus、Qdrant、Chroma ｜ 推理：vLLM、SGLang、Ollama
- 评估：RAGAs、DeepEval、LangSmith
- 教程：Hello-Agents（Datawhale）、All-in-RAG（Datawhale）、动手学大模型应用开发、nanoGPT

## 四、简历三件套（第 57-58 天用）

1. ≥ 2 个端到端可运行、有线上部署经验的系统项目
2. 量化业务指标：QPS +100%、P99 延迟 -80%、成本 -50% 一类
3. 生产级技术栈：LangChain + FastAPI + Milvus + Redis + Docker + Prometheus
