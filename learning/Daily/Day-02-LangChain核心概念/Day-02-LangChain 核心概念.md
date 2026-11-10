
> 2026-10-21（周三）｜第 2 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

## 今日主题 + 学习目标

**主题**：LangChain 核心概念 **目标**：理解 LangChain 六大核心模块，熟练使用 LCEL

递进关系：Day 1 搭好的 FastAPI 是**壳**（收请求、返响应），今天学的 LangChain 是壳里的**引擎**（把 prompt → 模型 → 解析串起来）。Day 3-4 的加载/向量化是喂给引擎的燃料，Day 5-6 把三者组装成 Naive RAG。所以今天的 `/chat` 接口要从 mock 升级成真链路——这是本周唯一一次「同一条接口改造两次」，别跳过。

## 资源清单（3 条）

1. LangChain Quickstart（官方，今天主读）：[https://python.langchain.com/docs/tutorials/](https://python.langchain.com/docs/tutorials/)

2. LCEL 概念页（只读 Runnable / `|` 管道 / invoke 三节就够）：[https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)

3. 吴恩达《LangChain for LLM Application Development》短课（选看，模型/记忆/链三节）：[https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)

> **待确认**：你打算用哪家模型 API（OpenAI / DeepSeek / 本地 Ollama）？没定也别卡住——下面的打卡项可以用 `FakeListChatModel` 或 Ollama 先跑通链路，换模型只改一行 `init_chat_model`。

## 时间安排建议（约 2.5 小时 · 40% 读 / 40% 写 / 20% 记录）

|   |   |   |
|---|---|---|
|时段|内容|类型|
|0:00-1:00|Quickstart + LCEL 概念页；边读边把六大模块在纸上画一遍|读懂即可|
|1:00-2:00|敲 LCEL 最小链 → 接进 Day 1 的 `/chat`|动手|
|2:00-2:30|跑通验收 + 填笔记区（重点记 invoke / stream 的输出类型差异）|动手|

## 今日手撕打卡项（来自第 1 周手撕清单）

- **动手**：用 LCEL 写第一个 LLM Chain —— `prompt | model | StrOutputParser()` 三段，`.invoke({"topic": "..."})` 能打印出字符串结果

- **动手**：把 Day 1 的 `POST /chat` 从 mock 改成调这条 chain，`/docs` 里请求一次拿到真实回答

参考骨架（自己敲，注意三段各自的输入输出类型）：

三个必须弄懂的点：

- 六大模块记住一句话版：**Model I/O**（模型进出的统一封装）、**Prompts**（模板化输入）、**Chains/LCEL**（把组件用 `|` 串成 Runnable）、**Memory**（对话历史，Day 19 细讲）、**Retrieval**（检索，Day 3-4 细讲）、**Agents**（让模型自己选工具，第 3 周）

- LCEL 的本质：每个组件都是 Runnable，`A | B` 就是「A 的输出喂给 B」，所以前一段的输出类型必须对得上后一段的输入

- `invoke`（一次拿全量）/ `stream`（逐 token 出）/ `batch`（批量）三个调用方式共用同一条 chain，这是第 4 周异步与批处理的伏笔

## 验收标准

`/chat` 返回的不再是「收到问题：xxx」这种拼接字符串，而是一句模型真实生成的回答；并且你能说清 `prompt | model | parser` 这三段各自的输入和输出分别是什么类型。

## 昨日回顾（Day 1 · FastAPI 快速入门）

打卡表里 Day 1 的两项都是**待完成**，没有收到你的结果反馈。这不是问题——但今天的 `/chat` 改造要建在它上面，所以先花 15 分钟补：

1. （5 分钟）把 Day 1 卡片里的骨架敲进 `week1-rag/main.py`，`uvicorn main:app --reload` 起来

2. （5 分钟）浏览器打开 `/docs` → `POST /chat` → 传 `{"question": "hi"}`，确认返回 200

3. （5 分钟）故意传 `{"question": 123}`，看到 422 就说明 Pydantic 校验生效，Day 1 就算过了

补完直接在对话里说一声，我把 Day 1 标成已完成。

## 进度提示

- 距第 7 个学习日（周度总结，2026-10-29 周四）还剩 **5 个学习日**

- Day 7 要交的是「Naive RAG 项目 Docker 化跑通」，现在要开始攒的产出：能跑的 `week1-rag/` 目录（今天完成后，FastAPI 壳 + LangChain 引擎都有了）、`requirements.txt`、一份能喂进去的样例 PDF（Day 3 用）

- 下一个里程碑之后依次是 Day 14（混合检索 + Reranker + Milvus）、Day 21（研究助手 Agent），都不用现在操心，先把 Day 7 的 Dockerfile 素材攒齐

---

## 笔记区

**今日卡点**

**关键代码片段**

**待补知识**