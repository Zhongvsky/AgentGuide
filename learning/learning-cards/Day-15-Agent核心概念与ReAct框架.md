# Day 15 · 第 3 周启航：Agent 核心概念与 ReAct 框架手撕实战

> 2026-11-24（周二）｜第 15 / 60 个学习日｜第 3 周 · Agent 开发与 Tool Calling  
> **阶段跨越**：告别单纯被动检索系统，正式迈向具备自主思考、工具调用与多步推理能力的 **AI Agent（智能体）** 全新纪元！

---

## 一、 承前启后与痛点剖析

在第 1~2 周中，我们从 Naive RAG 一路手撕到集成了混合检索、Cross-Encoder 单塔重排与 Milvus 标量过滤的生产级 Modular RAG 系统。

但即便最顶级的 RAG 系统，本质上也是**被动单轮管道（One-Shot Passive Pipeline）**：
- 用户输入一个 Query $\rightarrow$ 检索 Top-K $\rightarrow$ 喂给 LLM $\rightarrow$ 产生答案。
- 如果用户提出复合型工程问题（例如：“*帮我分析这块图像区域的暗部噪声方差 12.5 和均值 80.0，计算其实际 SNR 分贝值；并检索知识库判断该噪声水平是否超标？如果超标给出降噪模块配置建议。*”）：
  - 纯 RAG 会束手无策：大模型无法精准进行浮点对数数学计算（算错 SNR），也无法先算完再根据中间结果去动态决定下一步该查什么知识库。

**要解决复杂多步问题，必须赋予大模型“手和脚”以及“自主决策闭环”！**  
今天 Day 15 作为第 3 周的开篇，我们将深入攻克智能体领域最经典的认知范式——**ReAct（Reasoning + Acting）**，从零手撕一个完整的 ReAct 状态机循环引擎！

---

## 二、 今日主题与学习目标

- **主题**：Agent 认知心智模型拆解与 ReAct 状态机执行引擎手撕
- **目标**：
  1. 深入理解 **ReAct（Reasoning + Acting）** 认知范式（Yao et al., 2022），掌握 `Thought -> Action -> Action Input -> Observation -> Thought ... -> Final Answer` 循环流转机理；
  2. 手撕 **强类型工具注册表（Tool Registry with Pydantic）**：实现工具元数据提取、强类型入参校验与容错分发；
  3. 手撕 **ReAct 状态机输出解析器（ReAct Output Parser）**：掌握正则与流式状态机原理，精准解耦模型的内在推理（Thought）与行动指令（Action / Action Input / Final Answer）；
  4. 手撕 **Agent 核心执行循环（Agent Execution Loop）**：构建多轮动态 Prompt 上下文（Scratchpad）、单步调度、环境反馈注入与最大步数熔断控制。

---

## 三、 精选参考资源（≤3 条）

1. **ReAct: Synergizing Reasoning and Acting in Language Models (ICLR 2023)**：<https://arxiv.org/abs/2210.03629>
2. **LLM Powered Autonomous Agents (Lilian Weng 经典综述)**：<https://lilianweng.github.io/posts/2023-06-23-agent/>
3. **LangChain 官方 Agents 概念模型解析**：<https://python.langchain.com/docs/concepts/agents/>

---

## 四、 时间规划建议（约 2~2.5 小时）

- **认知破壁与架构心智模型（30% · 40分钟）**：
  - 搞懂 Plan-and-Solve、Chain-of-Thought（CoT）与 ReAct 的核心区别；
  - 弄懂为什么 ReAct 能通过引入环境反馈（Observation）从根本上抑制幻觉并打破模型计算瓶颈。
- **核心代码动手实现（50% · 65分钟）**：
  - 【关卡 1】：手撕 `ToolRegistry`（基于 Pydantic 契约的工具注册、Schema 序列化与异常防御）；
  - 【关卡 2】：手撕 `ReActParser`（利用正则表达式严谨解析 Thought、Action、Action Input 与 Final Answer）；
  - 【关卡 3】：手撕 `ReActAgentLoop`（装配 Scratchpad 观察历史，实现多轮环境自迭代直到 Final Answer）。
- **实测验证与复盘总结（20% · 25分钟）**：
  - 运行 `uv run python learning/Daily/Day-15-Agent核心概念与ReAct框架/day15.py`；
  - 观察 Agent 如何先自主调用数学计算算子，根据计算得到的实际数值，再主动调用领域专家知识库，最后输出严谨决策报告；
  - 体验智能体自主推理的魅力！

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：在 `day15.py` 中手撕 `ToolRegistry.register` 与 `ToolRegistry.execute`，支持 Pydantic 强类型参数注入；
- [ ] **必做**：手撕 `ReActParser.parse`，能够对模型生成的文本进行状态判定，返回 `AgentAction` 或 `AgentFinish`；
- [ ] **必做**：手撕 `ReActAgentLoop.run` 执行循环，能够将 `Observation` 实时追加回 Scratchpad 中供模型下一轮思考；
- [ ] **进阶（读懂即可）**：评估当工具报错时，如何将带有堆栈简报的错误信息包装为 Observation 回传模型，实现 Agent 的“自主纠错（Self-Correction）”。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-15-Agent核心概念与ReAct框架/day15.py
```
**判定标准**：
1. 脚本无报错跑通，三大关卡全绿灯通过；
2. 关卡 1 成功注册数学计算工具与 ISP 缺陷知识工具，非法入参时能够安全拦截；
3. 关卡 2 精准解析正常 Action 输出与 Final Answer 终止输出；
4. 关卡 3 Agent 成功在 3 轮思考以内自主串联“噪声指标计算 $\rightarrow$ 知识库检索 $\rightarrow$ 输出最终诊断”，终端打印：
   `🎉 [PASSED]: Day 15 关卡全通！ReAct 认知闭环引擎手撕成功！`

---

## 七、 周度里程碑倒计时

- **第 3 周里程碑**：构建集成 RAG + Web 搜索的「ISP 算法与代码研究助手」Agent（Day 21）；
- **距离本周里程碑日还差**：**6 个学习日**；
- **本周前置储备**：从今日的 ReAct 底层闭环，到后续自定义 Tool 封装、SQL Agent 交互、OpenAI Function Calling 与短长记忆集成。
