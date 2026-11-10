# Langchain Core Concepts And Lcel Guide

# LangChain 核心概念、六大核心模块与 LCEL 实战学习指南

本指南以官方 [Home - Docs by LangChain](https://docs.langchain.com/ "null") 与 [LangChain Python Overview](https://docs.langchain.com/oss/python/langchain/overview "null") 为基准，结合 [LangChain Core API Reference](https://reference.langchain.com/python/langchain-core "null") 与 [LangChain 概念指南](https://www.langchain.com.cn/docs/concepts/ "null")，系统梳理 LangChain 的最新生态架构、经典六大核心模块演进映射，以及 LangChain 表达式语言（LCEL）从基础语法到生产级流式、并发、容错的完整实战用法。

## 一、LangChain 核心概念与最新生态架构

根据 [Home - Docs by LangChain](https://docs.langchain.com/ "null") 的最新定位，LangChain 已经从早期的“提示词拼接与链式封装库”演进为完整的**开源智能体工程生态（Open Agent Engineering Ecosystem）**，覆盖构建（Build）、测试（Test）、部署（Deploy）与监控（Monitor）全生命周期。

### 1.1 分层包架构体系

在 [LangChain 概念指南](https://www.langchain.com.cn/docs/concepts/ "null") 与 [langchain-core 参考文档](https://reference.langchain.com/python/langchain-core "null") 中，框架采用严格解耦的分层架构，避免单包依赖膨胀并保障接口稳定性：

|                                                                                     |                                                                                                                                                                          |                                                        |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **层级包名**                                                                            | **核心职责与包含内容**                                                                                                                                                            | **适用场景与安装建议**                                          |
| [**langchain-core**](https://reference.langchain.com/python/langchain-core "null")  | 定义生态系统的底层抽象接口（`Runnable` 协议、消息类 `BaseMessage`、提示模板、输出解析器、检索器接口、回调机制），零第三方重依赖                                                                                             | 所有基于 LangChain 组件开发的基础底座（`pip install langchain-core`） |
| [**langchain**](https://docs.langchain.com/oss/python/langchain/overview "null")    | 提供构成应用“认知架构”的跨厂商通用编排能力，包括 `create_agent` 智能体底座、`init_chat_model` 统一初始化接口及内置中间件（[Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview "null")） | 构建标准化智能体与通用应用流程（`pip install langchain`）               |
| **合作伙伴独立包 (`langchain-{provider}`)**                                                | 主流大模型与向量库的独立官方维护包（如 `langchain-openai`、`langchain-anthropic`、`langchain-google-genai`、`langchain-pinecone`）                                                              | 按需安装特定模型供应商集成（如 `pip install -U "langchain[openai]"`）  |
| [**langchain-community**](https://www.langchain.com.cn/docs/concepts/ "null")       | 由开源社区维护的长尾第三方集成集合（涵盖 200+ 文档加载器、50+ 向量存储及外部工具包），所有依赖均为可选                                                                                                                 | 接入特定本地文档加载器、小众数据库或社区工具                                 |
| [**LangGraph**](https://docs.langchain.com/oss/python/langgraph/overview "null")    | 底层图状态编排框架，将工作流建模为状态（State）、节点（Node）和边（Edge），支持持久化断点续传（Checkpointer）、循环推理与人在回路（HITL）                                                                                      | 构建有状态、多步骤循环、多智能体协同的生产级复杂工作流                            |
| [**Deep Agents**](https://docs.langchain.com/oss/python/deepagents/overview "null") | 构建在 LangChain 智能体之上的“开箱即用（Batteries-Included）”高级智能体框架，内置自动上下文压缩、虚拟文件系统与子智能体派生（Subagent-spawning）                                                                         | 快速构建复杂数据分析、深度研究（Deep Research）等自主长程任务智能体               |
| [**LangSmith**](https://docs.langchain.com/ "null")                                 | 框架无关的智能体全生命周期开发者平台，提供全链路追踪（Tracing）、数据集评估（Evaluation）、Prompt 调优与生产环境监控                                                                                                   | 研发调试、质量评测、延迟与 Token 成本监控（设置 `LANGSMITH_TRACING=true`）  |

### 1.2 四大核心基石概念

1. **统一模型抽象（Standard Model Interface）**：通过 [Models 模块](https://docs.langchain.com/oss/python/langchain/models "null") 的 `init_chat_model` 接口，用统一的方法签名调用 OpenAI、Anthropic、Google Gemini、AWS Bedrock、Ollama 等不同模型提供商，支持零代码改造切换底层模型。

2. **标准消息单元（Standard Messages & Content Blocks）**：在 [Messages 模块](https://docs.langchain.com/oss/python/langchain/messages "null") 中，消息（`SystemMessage`、`HumanMessage`、`AIMessage`、`ToolMessage`）是模型上下文的基本单元。新版统一引入 `.content_blocks` 属性，将不同厂商的思考推理过程（`reasoning`）、文本（`text`）、工具调用（`tool_call`）及多模态输入规范化为跨平台一致的类型安全结构。

3. **可组合执行协议（Runnable Interface）**：在 [Runnables 参考文档](https://reference.langchain.com/python/langchain-core/runnables "null") 中，`Runnable` 是整个框架的原子工作单元，统一暴露同步/异步的单次调用（`invoke`/`ainvoke`）、流式输出（`stream`/`astream`）、批量并发（`batch`/`abatch`）及事件流（`astream_events`）。

4. **智能体底座公式（Agent = Model + Harness）**：根据 [Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents "null")，现代 LangChain 将智能体定义为“在循环中调用工具的模型（Model）”加上围绕该循环的“控制底座（Harness）”。通过 `create_agent` 函数，开发者可将模型、工具（[Tools](https://docs.langchain.com/oss/python/langchain/tools "null")）、系统提示词（System Prompt）与中间件（[Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview "null")）灵活组装，并在底层自动依托 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview "null") 实现状态持久化与中断恢复。


## 二、LangChain 六大核心模块深度解析与演进对照

在 [LangChain 六大核心模块指南](https://juejin.cn/post/7561706231033757737 "null") 与 [LangChain 六链构建 AI 应用骨架](https://cloud.baidu.com/article/3364149 "null") 中，框架的能力版图被划分为 **Model I/O、Retrieval、Chains、Memory、Agents、Callbacks** 六大核心模块。理解这六大模块的功能边界及其在最新 [Docs by LangChain](https://docs.langchain.com/oss/python/langchain/overview "null") 中的演进方式，是掌握 LangChain 的关键。

### 2.1 六大核心模块全景对比

|                                       |                                          |                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **核心模块**                              | **解决的核心痛点**                              | **核心子组件与关键类**                                                                                                                                                                                                                                                                                                                                             | **最新官方文档演进与生产最佳实践**                                                                                                                                                                                                                                                                                                                                                                                      |
| **1. Model I/O（模型输入/输出）**             | 屏蔽不同厂商 API 差异，实现输入提示词参数化与输出数据强类型结构化      | `PromptTemplate`、`ChatPromptTemplate`、`MessagesPlaceholder`；[Chat Models](https://docs.langchain.com/oss/python/langchain/models "null") (`init_chat_model`)；[Output Parsers](https://reference.langchain.com/python/langchain-core/output_parsers "null") (`StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`)                                | 推荐使用 `init_chat_model` 统一初始化；结构化输出优先使用原生 `.with_structured_output()` 或 `create_agent` 的 [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output "null")（`ProviderStrategy` / `ToolStrategy`）                                                                                                                                                                               |
| **2. Retrieval（检索增强/外部数据）**           | 突破大模型知识截止时间限制、私有数据盲区与上下文窗口长度瓶颈           | `DocumentLoader`（如 `PyPDFLoader`, `WebBaseLoader`）、`TextSplitter`（如 `RecursiveCharacterTextSplitter`）、`Embeddings`、`VectorStore`（如 `FAISS`, `Chroma`, `Qdrant`）、`Retriever`                                                                                                                                                                               | 在 [Retrieval 指南](https://docs.langchain.com/oss/python/langchain/retrieval "null") 中演进为三大架构：**2-Step RAG**（固定先检索后生成）、**Agentic RAG**（由 Agent 按需调用检索工具）与 **Hybrid RAG**（带查询重写与质量校验）                                                                                                                                                                                                                       |
| **3. Chains（链式编排管道）**                 | 将孤立的提示词、模型、检索器与自定义函数串联或并联为端到端业务流水线       | [LCEL](https://langchain.nodejs.cn/docs/concepts/lcel "null") 管道操作符 `\|`、`RunnableSequence`、`RunnableParallel`、`RunnablePassthrough`、`RunnableLambda`、`RunnableBranch`                                                                                                                                                                                    | 全面废弃遗留的 `LLMChain`、`SequentialChain` 等黑盒子类；简单线性或并行数据流使用 **LCEL**，复杂循环、有状态分支流程使用 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview "null")                                                                                                                                                                                                                                             |
| **4. Memory（记忆与状态管理）**                | 赋予无状态的大模型跨轮次上下文保持能力与跨会话长期个性化记忆           | 传统：`ConversationBufferMemory`、`ConversationSummaryBufferMemory`；现代：`RunnableWithMessageHistory`、`AgentState`、`InMemorySaver`、`PostgresSaver`、`InMemoryStore`                                                                                                                                                                                              | 在 [Memory 概念指南](https://docs.langchain.com/oss/python/concepts/memory "null") 中拆分为两层：[Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory "null")（通过 Checkpointer 与 `thread_id` 管理会话内状态，配合 `@before_model` 裁剪或 `SummarizationMiddleware` 摘要）与 [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory "null")（通过 `Store` 跨会话持久化语义、情景与程序性记忆） |
| **5. Agents（自主智能体与工具）**               | 让模型作为推理引擎自主规划步骤、动态选择并调用外部工具直至完成任务        | [`@tool` 装饰器](https://docs.langchain.com/oss/python/langchain/tools "null")、`ToolRuntime`、`bind_tools`、[`create_agent`](https://docs.langchain.com/oss/python/langchain/agents "null")、`AgentState`、[Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview "null")（`@before_model`, `@after_model`, `HumanInTheLoopMiddleware`） | 弃用旧版 `AgentExecutor`；统一使用基于 LangGraph 构建的 `create_agent` 函数，通过 `ToolRuntime` 安全访问 `state`、`context` 和 `store`，并用 Middleware 实现护栏与人工审批                                                                                                                                                                                                                                                                    |
| **6. Callbacks & Streaming（回调与流式观测）** | 解决长耗时推理的黑盒问题与首字延迟（TTFT），支持日志审计、监控与实时流式交互 | `BaseCallbackHandler`、`AsyncCallbackHandler`（监听 `on_llm_start`、`on_llm_new_token`、`on_tool_start` 等事件）；[Streaming](https://docs.langchain.com/oss/python/langchain/streaming "null")（`stream`、`astream_events`）                                                                                                                                           | 流式交互推荐使用 `stream(stream_mode=["messages", "updates", "custom"])` 或 `stream_events(version="v3")`；生产级可观测性直接开启 [LangSmith](https://docs.langchain.com/ "null") 自动全链路追踪                                                                                                                                                                                                                                     |

### 2.2 模块一：Model I/O（模型输入、调用与输出解析）

Model I/O 将大模型交互拆解为 **Format（输入格式化） → Predict（模型推理） → Parse（输出解析）** 三个标准阶段：

1. **提示词模板（Prompts）**：根据 [LangChain 概念指南](https://www.langchain.com.cn/docs/concepts/ "null")，通过 `ChatPromptTemplate.from_messages()` 组合系统指令（`system`）、用户输入（`user`）和动态历史占位符（`MessagesPlaceholder` 或 `("placeholder", "{msgs}")`），调用后生成标准化的 `PromptValue`。

2. **聊天模型（Chat Models）**：在 [Models 文档](https://docs.langchain.com/oss/python/langchain/models "null") 中，现代应用统一使用以消息列表为输入、返回 `AIMessage` 的 Chat Models（而非旧版纯文本补全 LLM）。内置自动指数退避重试（默认 `max_retries=6`，自动重试网络错误、429 限流与 5xx 服务端错误）。

3. **输出解析与结构化输出（Output Parsers & Structured Output）**：

    - 当需要将 `AIMessage` 转为纯文本流时，使用 [output_parsers 模块](https://reference.langchain.com/python/langchain-core/output_parsers "null") 中的 `StrOutputParser()`。

    - 当需要提取 JSON 或强类型 Pydantic 对象时，现代模型首选 `model.with_structured_output(PydanticSchema)`，或在智能体中使用 [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output "null") 的 `response_format` 参数（支持原生 `ProviderStrategy` 与基于工具调用加自动纠错重试的 `ToolStrategy`）。


```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 1. 统一初始化模型（支持 openai:gpt-5.5, anthropic:claude-sonnet-4-6, google_genai:gemini-2.5-flash-lite 等）
model = init_chat_model(
    "openai:gpt-5.5",
    temperature=0.3,
    timeout=30,
    max_retries=6,
)

# 2. 构建多角色对话提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深的 {domain} 架构师，请用简洁专业的中文回答。"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])

# 3. 方式 A：使用 StrOutputParser 提取纯文本（天然支持流式输出）
text_chain = prompt | model | StrOutputParser()

# 4. 方式 B：使用 with_structured_output 获取强类型 Pydantic 结构化数据
class ArchitectureAdvice(BaseModel):
    """架构评估建议结构"""
    core_pattern: str = Field(description="推荐的核心架构模式")
    pros: list[str] = Field(description="核心优势列表")
    risks: list[str] = Field(description="潜在风险与注意事项")

structured_chain = prompt | model.with_structured_output(ArchitectureAdvice)
```

### 2.3 模块二：Retrieval（检索增强生成 RAG 体系）

在 [Retrieval 官方文档](https://docs.langchain.com/oss/python/langchain/retrieval "null") 与 [LangChain 六大核心模块指南](https://juejin.cn/post/7561706231033757737 "null") 中，Retrieval 模块通过五大积木构建外部知识库，并提供三种主流 RAG 架构：

1. **五大核心积木**：

    - **Document Loaders（文档加载器）**：从 PDF（`PyPDFLoader`）、网页（`WebBaseLoader`）、数据库等 200+ 数据源加载数据，统一返回包含 `page_content: str` 与 `metadata: dict` 的 `Document` 对象。

    - **Text Splitters（文本分割器）**：首选 `RecursiveCharacterTextSplitter`（按 `["\n\n", "\n", " ", ""]` 递归切分并保留 `chunk_overlap` 上下文重叠），或按标题切分的 `MarkdownHeaderTextSplitter`、按语义突变切分的 `SemanticChunker`。

    - **Embedding Models（嵌入模型）**：通过 `embed_documents(texts)`（批量文档向量化）和 `embed_query(text)`（单条查询向量化）将文本映射为高维语义向量。

    - **Vector Stores（向量存储）**：负责存储向量与元数据并执行相似度检索（`similarity_search`）或最大边际相关性检索（`max_marginal_relevance_search`，即 MMR，兼顾相关性与多样性）。

    - **Retrievers（检索器接口）**：接收字符串查询并返回 `list[Document]` 的标准 `Runnable` 接口（如 `vector_store.as_retriever()`、`MultiQueryRetriever`、`ContextualCompressionRetriever`、`ParentDocumentRetriever`）。

2. **三大 RAG 架构演进**（[Retrieval 指南](https://docs.langchain.com/oss/python/langchain/retrieval "null")）：

    - **2-Step RAG（两步式 RAG）**：固定先执行检索、再将文档注入 Prompt 生成回答，控制力高、延迟可预测，适合知识库问答与 FAQ 机器人。

    - **Agentic RAG（智能体 RAG）**：将检索器封装为 `@tool` 交给智能体，由 LLM 在推理循环中自主决定“是否需要检索、检索什么关键词、结果是否足够”，灵活性极高。

    - **Hybrid RAG（混合式 RAG）**：结合查询重写（Query Enhancement，如多视角重写、HyDE 假设文档嵌入）、检索质量校验与生成后自纠错（Self-RAG / Corrective RAG）。

### 2.4 模块三：Chains（从 Legacy Chains 到 LCEL 与 LangGraph）

在 [LCEL 概念指南](https://langchain.nodejs.cn/docs/concepts/lcel "null") 中，LangChain 官方给出了清晰的编排选型准则：

- **单次模型调用**：无需构建 Chain，直接调用 `model.invoke()`。

- **确定性流水线与简单编排**（如 `Prompt + Model + Parser`、两步式 RAG、多路并行提取）：使用 **LCEL** 声明式组合原语。

- **复杂状态管理、条件循环与多智能体**：使用 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview "null") 构建图工作流（在单个图节点内部依然可以使用 LCEL）。


### 2.5 模块四：Memory（短期会话状态与长期跨会话记忆）

根据 [Memory 概念总览](https://docs.langchain.com/oss/python/concepts/memory "null")、[Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory "null") 与 [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory "null")，现代 LangChain 的记忆体系分为两个维度：

1. **短期记忆（Short-term Memory / Thread-scoped State）**：

    - **作用域**：单个会话线程（`thread_id`）内的多轮对话历史与自定义上下文状态。

    - **实现方式**：在 LCEL 链中使用 `RunnableWithMessageHistory`；在智能体中通过 `create_agent(..., checkpointer=InMemorySaver())`（生产环境使用 `PostgresSaver`）配合 `config={"configurable": {"thread_id": "thread_1"}}` 自动持久化 `AgentState`。

    - **上下文窗口治理**：长对话容易超出 Token 限制或导致模型注意力分散。官方推荐三种中间件模式：
  
        - **裁剪消息（Trim messages）**：使用 `@before_model` 中间件配合 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 仅保留系统提示与最近 N 条消息。

        - **删除特定消息（Delete messages）**：在 `@after_model` 中返回 `RemoveMessage(id=m.id)` 清理过期或敏感消息。

        - **自动摘要（Summarize messages）**：直接使用内置的 `SummarizationMiddleware(model="gpt-5.4-mini", trigger=("tokens", 4000), keep=("messages", 20))`。

2. **长期记忆（Long-term Memory / Cross-thread Store）**：

    - **作用域**：跨会话、跨线程持久保存的用户画像或领域知识，按自定义的 `(namespace, key)` 层级存储 JSON 文档并支持向量语义检索。

    - **三大认知记忆分类**（[Memory 概念总览](https://docs.langchain.com/oss/python/concepts/memory "null")）：

        - **语义记忆（Semantic Memory）**：记住关于用户的事实与偏好（可通过持续更新的 JSON Profile 或文档 Collection 管理）。

        - **情景记忆（Episodic Memory）**：记住过去的成功经验与操作轨迹（通常作为动态 Few-shot 示例注入提示词）。

        - **程序性记忆（Procedural Memory）**：记住执行任务的规则与技能（通过 Reflection 元提示动态迭代优化 Agent 的 System Prompt 并存入 Store）。


### 2.6 模块五：Agents（智能体底座、工具运行时与中间件）

在 [Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents "null")、[Tools 官方文档](https://docs.langchain.com/oss/python/langchain/tools "null") 与 [Middleware 官方文档](https://docs.langchain.com/oss/python/langchain/middleware/overview "null") 中，现代智能体开发围绕 `create_agent` 展开：

1. **工具定义与 `ToolRuntime` 上下文注入**：
 
    - 使用 `@tool` 装饰器定义函数，类型注解（Type Hints）或 Pydantic `args_schema` 自动生成工具 JSON Schema，函数文档字符串（Docstring）作为模型决策依据（工具命名推荐使用 `snake_case`）。

    - 在工具参数中声明 `runtime: ToolRuntime`（该参数对大模型隐藏，由框架在运行期自动注入），可在工具内部访问：
  
        - `runtime.state`：读取当前会话的短期记忆（如 `messages` 或自定义状态字段），并可通过返回 `Command(update={...})` 修改状态。

        - `runtime.context`：读取调用时传入的不可变配置（如 `user_id`、租户权限信息）。

        - `runtime.store`：读写跨会话的长期记忆（`store.get(namespace, key)` / `store.put(namespace, key, value)`）。

        - `runtime.stream_writer`：在长耗时工具执行期间实时向前端推送自定义进度流。

2. **中间件（Middleware）精细化控制**：

    - 通过向 `create_agent(..., middleware=[...])` 传入中间件，可在模型调用前后（`@before_model`、`@after_model`、`@after_agent`）或动态提示词阶段（`@dynamic_prompt`）介入执行流。

    - 官方内置中间件（[Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in "null")）包括：`SummarizationMiddleware`（长对话自动摘要）、`HumanInTheLoopMiddleware(interrupt_on={"tool_name": True})`（高危工具执行前挂起并等待人工审批确认）等。


```
from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

@dataclass
class UserContext:
    user_id: str

class CustomState(AgentState):
    call_count: int

@tool
def query_user_preference(runtime: ToolRuntime[UserContext, CustomState]) -> str:
    """查询当前用户的长期偏好设置。"""
    user_id = runtime.context.user_id
    pref = runtime.store.get(("users",), user_id)
    return str(pref.value) if pref else "未找到偏好设置"

# 组装包含短期记忆(Checkpointer)、长期记忆(Store)与中间件(Middleware)的生产级智能体
agent = create_agent(
    model="openai:gpt-5.5",
    tools=[query_user_preference],
    system_prompt="你是一名贴心的智能助手，回答前请先查询用户偏好。",
    state_schema=CustomState,
    context_schema=UserContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
    middleware=[
        SummarizationMiddleware(model="openai:gpt-5.4-mini", trigger=("tokens", 4000), keep=("messages", 20)),
    ],
)
```

### 2.7 模块六：Callbacks & Streaming（回调事件机制与实时流式系统）

1. **回调处理器（Callbacks）**：在 [LangChain 概念指南](https://www.langchain.com.cn/docs/concepts/ "null") 中，回调机制提供全生命周期钩子（`on_chat_model_start`、`on_llm_new_token`、`on_llm_end`、`on_chain_start`、`on_tool_start`、`on_retriever_end`、`on_retry` 等）。
    
      
    - **请求时回调（Request-time callbacks）**：通过 `invoke(input, config={"callbacks": [handler]})` 传入，**会自动向下传播给所有子组件**（推荐方式）。
        
          
        
    - **构造函数回调（Constructor callbacks）**：在组件初始化时传入 `callbacks=[handler]`，**仅作用于当前对象本身，不会继承给子组件**。
        
          
        
2. **多模式流式系统（Streaming）**：在 [Streaming 官方文档](https://docs.langchain.com/oss/python/langchain/streaming "null") 中，LangChain 支持三种核心 `stream_mode`（可组合传入列表如 `stream_mode=["messages", "updates", "custom"]`）：
    
      
    - `"updates"`：每完成一个智能体节点步骤（如模型生成工具调用、工具返回结果），流式推送一次状态更新。
        
          
        
    - `"messages"`：实时流式推送 LLM 生成的增量 Token 块（`AIMessageChunk`，包含文本流 `.text`、思考流 `.reasoning` 与工具参数块 `.tool_call_chunks`）及节点元数据。
        
          
        
    - `"custom"`：流式接收工具内部通过 `runtime.stream_writer` 或 `get_stream_writer()` 发送的自定义业务进度信号。
        
          
        

## 三、熟练使用 LCEL（LangChain Expression Language）

根据 [LCEL 官方概念指南](https://langchain.nodejs.cn/docs/concepts/lcel "null") 与 [langchain_core.runnables 参考文档](https://reference.langchain.com/python/langchain-core/runnables "null")，LCEL 是一种基于 `Runnable` 协议的**声明式（Declarative）组合语言**——你只需声明“数据流经哪些组件”，LangChain 会在底层自动完成并行化调度、首字流式直通（TTFT 优化）、异步并发以及 [LangSmith](https://docs.langchain.com/ "null") 全链路追踪。

### 3.1 Runnable 标准执行接口速查

每一个实现了 `Runnable` 接口的组件或 LCEL 组合链，都具备以下完全一致的调用方法（均接收可选的 `config: RunnableConfig` 参数，用于传递 `callbacks`、`tags`、`metadata`、`max_concurrency`、`configurable` 等运行时配置）：

|   |   |   |   |   |
|---|---|---|---|---|
|**同步方法**|**异步方法**|**输入类型**|**输出类型**|**核心用途与底层机制**|
|`invoke(input, config)`|`await ainvoke(input, config)`|单个输入对象|单个最终输出|单次完整推理调用，等待所有步骤执行完毕后返回最终结果|
|`stream(input, config)`|`async for chunk in astream(input, config)`|单个输入对象|`Iterator[OutputChunk]`|增量流式传输；只要上下游组件支持流式（如 `Model -> StrOutputParser/JsonOutputParser`），便能实时逐块产出|
|`batch(inputs, config)`|`await abatch(inputs, config)`|`list[Input]`|`list[Output]`|批量并发执行；同步模式下默认基于线程池并发，异步模式下基于 `asyncio.gather` 高并发执行（可通过 `config={"max_concurrency": N}` 限流）|
|`stream_events(...)`|`async for ev in astream_events(input, version="v2")`|单个输入对象|事件字典迭代器|细粒度事件流；不仅能获取最终输出流，还能实时捕获链内部每个中间步骤的 `on_chat_model_stream`、`on_retriever_end` 等事件|

此外，任何 LCEL 链都支持通过 `chain.input_schema.model_json_schema()` 和 `chain.output_schema.model_json_schema()` 自动推断输入与输出的 Pydantic Schema，或通过 `chain.get_graph().print_ascii()` 可视化打印管道拓扑结构。

### 3.2 LCEL 六大核心组合原语与语法糖

在 [Runnables 参考文档](https://reference.langchain.com/python/langchain-core/runnables "null") 与 [LangChain 概念指南](https://www.langchain.com.cn/docs/concepts/ "null") 中，LCEL 通过两大基础组合原语（`RunnableSequence`、`RunnableParallel`）及其变体构建任意数据管道，并提供极简的隐式类型转换语法糖：

|                           |                                                                                  |                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **组合原语类**                 | **简写语法糖 / 隐式转换**                                                                 | **核心作用与数据流变换逻辑**                                                                                   |
| **`RunnableSequence`**    | `runnable1 \| runnable2` 或 `runnable1.pipe(runnable2)`                           | **顺序串联**：前一个组件的输入/输出依次传递，即 `output = r2.invoke(r1.invoke(x))`                                      |
| **`RunnableParallel`**    | 管道中的 Python 字典字面量 `{"k1": r1, "k2": r2}`                                         | **并行扇出（Fan-out）**：将同一个输入同时分发给字典中的多个 `Runnable` 并行执行，返回具有相同键名的结果字典                                  |
| **`RunnablePassthrough`** | `RunnablePassthrough()` / `RunnablePassthrough.assign(k=r)`                      | **透传与字段追加**：`RunnablePassthrough()` 原样透传输入；`.assign(new_key=runnable)` 在保留原输入字典所有键值的基础上，并行计算并追加新字段 |
| **`RunnableLambda`**      | 管道中的普通 Python 函数或 `lambda x: ...`                                                | **自定义函数包装**：将任意同步/异步 Python 函数转换为标准 `Runnable`，自动接入流式与追踪链路（还可用 `@chain` 装饰器）                       |
| **`RunnableBranch`**      | `RunnableBranch((cond1, r1), (cond2, r2), default_r)`                            | **条件路由分支**：按顺序评估 `(条件函数, 目标Runnable)` 元组，执行第一个返回 `True` 的分支；若均不满足则执行末尾默认分支                         |
| **`RunnableBinding`**     | `.bind(...)` / `.with_Config(...)` / `.with_fallbacks(...)` / `.with_retry(...)` | **运行时装饰器绑定**：在不改动原组件代码的前提下，为 `Runnable` 绑定额外调用参数、工具、指数退避重试或备用降级链                                   |

### 3.3 LCEL 四大高频实战模式（可直接运行代码模板）

#### 模式 1：基于 `RunnableParallel` 与 `RunnablePassthrough.assign` 的标准 RAG 管道（含来源返回）

在 RAG 场景中，我们不仅需要把用户问题传给检索器获取文档，还需要保留原始问题传给 Prompt，甚至在最终结果中同时返回“生成的回答”和“引用的原始文档列表”。使用 `RunnablePassthrough.assign` 可以优雅地逐步扩充上下文字典：

```python
from operator import itemgetter
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# 模拟检索器（实际项目中替换为 vector_store.as_retriever()）
def fake_retriever(query: str) -> list[Document]:
    return [
        Document(page_content="LCEL 支持流式、异步、批处理和自动并行优化。", metadata={"source": "docs.langchain.com/lcel"}),
        Document(page_content="RunnableParallel 可以同时执行多个子链并合并字典输出。", metadata={"source": "reference.langchain.com/runnables"}),
    ]

retriever = RunnableLambda(fake_retriever)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)

prompt = ChatPromptTemplate.from_template(
    "请仅根据以下参考资料回答问题：\n{context}\n\n问题：{question}"
)
model = init_chat_model("openai:gpt-5.5")

# 子链：从包含 docs 和 question 的字典生成文本回答
answer_subchain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
    | prompt
    | model
    | StrOutputParser()
)

# 完整 RAG 链：先检索出 docs，再将 answer 追加到字典中，最终同时返回 question、docs 和 answer
rag_chain_with_sources = (
    RunnableParallel({"docs": itemgetter("question") | retriever, "question": itemgetter("question")})
    .assign(answer=answer_subchain)
)

# 调用示例：
# result = rag_chain_with_sources.invoke({"question": "LCEL 有哪些核心优势？"})
# print(result["answer"])
# print(result["docs"])
```

#### 模式 2：使用 `RunnableBranch` 或语义路由构建动态多分支链

当应用需要根据用户输入特征（或上游分类模型的判断结果）分发给不同专业领域的子链时，可组合分类器与 `RunnableBranch`：

```
from typing import Literal
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

model = init_chat_model("openai:gpt-5.5")

class RouteIntent(BaseModel):
    category: Literal["coding", "math", "general"]

classifier_chain = (
    ChatPromptTemplate.from_template("请判断以下用户问题的所属类别：{query}")
    | model.with_structured_output(RouteIntent)
)

coding_chain = ChatPromptTemplate.from_template("作为资深架构师，请给出代码级解答：{query}") | model | StrOutputParser()
math_chain = ChatPromptTemplate.from_template("作为数学家，请分步推导以下问题：{query}") | model | StrOutputParser()
general_chain = ChatPromptTemplate.from_template("请简明回答以下问题：{query}") | model | StrOutputParser()

# 使用 RunnableBranch 根据上游分类结果路由
routing_chain = (
    RunnablePassthrough.assign(intent=classifier_chain)
    | RunnableBranch(
        (lambda x: x["intent"].category == "coding", coding_chain),
        (lambda x: x["intent"].category == "math", math_chain),
        general_chain,  # 默认兜底分支
    )
)
```

#### 模式 3：生产级高可用保障——重试（`.with_retry()`）与降级回退（`.with_fallbacks()`）

在生产环境中，单一模型接口可能遇到速率限制或瞬时故障。LCEL 允许在任意组件或整条子链上叠加重试策略与跨厂商备用模型回退：

```
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("请总结以下技术趋势：{topic}")

# 主模型配置自动重试（最多尝试 3 次，指数退避抖动）
primary_model = init_chat_model("openai:gpt-5.5").with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# 备用模型（跨厂商容灾：当 OpenAI 服务不可用时自动无缝切换到 Anthropic Claude）
fallback_model = init_chat_model("anthropic:claude-sonnet-4-6")

resilient_model = primary_model.with_fallbacks([fallback_model])

production_chain = prompt | resilient_model | StrOutputParser()
```

#### 模式 4：使用 `RunnableWithMessageHistory` 为 LCEL 链挂载多会话记忆

对于不涉及复杂多步工具循环的对话式 LCEL 链，使用 `RunnableWithMessageHistory` 即可按 `session_id` 自动读写历史消息：

```
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名精通 LangChain 的技术导师。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

base_chain = prompt | init_chat_model("openai:gpt-5.5") | StrOutputParser()

# 会话存储字典（生产环境可替换为 RedisChatMessageHistory 或 SQLChatMessageHistory）
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_lcel = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 调用时通过 configurable 传入 session_id
# res1 = conversational_lcel.invoke(
#     {"question": "我正在学习 LCEL 的 RunnableParallel。"},
#     config={"configurable": {"session_id": "user_001"}},
# )
```

## 四、循序渐进的四阶段学习路径建议

1. **第一阶段：掌握底座抽象与 Model I/O（1–2 天）**
    
      
    - 阅读 [LangChain Overview](https://docs.langchain.com/oss/python/langchain/overview "null")、[Models](https://docs.langchain.com/oss/python/langchain/models "null") 与 [Messages](https://docs.langchain.com/oss/python/langchain/messages "null")。
        
          
        
    - 动手练习使用 `init_chat_model` 初始化模型，对比 `StrOutputParser` 与 `.with_structured_output(PydanticModel)` 的输出差异。
        
          
        
2. **第二阶段：熟练编写 LCEL 声明式管道（2–3 天）**
    
      
    - 阅读 [LCEL 概念指南](https://langchain.nodejs.cn/docs/concepts/lcel "null") 与 [Runnables API 参考](https://reference.langchain.com/python/langchain-core/runnables "null")。
        
          
        
    - 重点练习 `|`（`RunnableSequence`）、字典字面量并行（`RunnableParallel`）、`RunnablePassthrough.assign()` 与 `.with_fallbacks()`，并用 `stream()` 和 `astream_events()` 观察实时数据流。
        
          
        
3. **第三阶段：构建端到端 RAG 检索增强系统（2–3 天）**
    
      
    - 阅读 [Retrieval 官方文档](https://docs.langchain.com/oss/python/langchain/retrieval "null") 与 [Learn 教程专区](https://docs.langchain.com/oss/python/learn "null") 中的 Semantic Search 教程。
        
          
        
    - 走通 `PyPDFLoader -> RecursiveCharacterTextSplitter -> Embeddings -> VectorStore -> as_retriever()` 全链路，并分别用 LCEL（2-Step RAG）和 `@tool`（Agentic RAG）实现文档问答。
        
          
        
4. **第四阶段：进阶现代 Agent、记忆管理与生产观测（3–5 天）**

    - 阅读 [Agents](https://docs.langchain.com/oss/python/langchain/agents "null")、[Tools](https://docs.langchain.com/oss/python/langchain/tools "null")、[Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory "null")、[Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory "null") 与 [Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview "null")。

    - 使用 `create_agent` 结合 `InMemorySaver`、`InMemoryStore` 与 `SummarizationMiddleware` 搭建带记忆的智能体，并开启 [LangSmith](https://docs.langchain.com/ "null")（`LANGSMITH_TRACING=true`）检视每一步推理与 Token 消耗。
