# Day 16 · 自定义工具开发与 Tool Calling 原生协议实战

> 2026-11-25（周三）｜第 16 / 60 个学习日｜第 3 周 · Agent 开发与 Tool Calling  
> **范式演进**：告别脆弱的 Prompt 正则文本解析，全面跨入现代大模型原生 **Tool Calling（函数调用）** 与工具协议时代！

---

## 一、 承前启后与痛点剖析

在昨天的 Day 15 中，我们从零手撕了经典 ReAct 状态机循环。在跑通测试的过程中，我们深切体会到了纯文本 ReAct 的两个致命痛点：
1. **格式漂移与解析脆弱**：大模型本质是自回归文本生成器，输出格式可能稍微缩进、少个换行、把 JSON 包在 Markdown 代码块里，或者漏掉 `Thought:` 标识，极易引发解析崩溃；
2. **缺乏协议标准化**：对于多个并发工具调用、唯一调用 ID 追踪以及异步流式执行，基于纯 Prompt 的字符串切割显得捉襟见肘。

为了解决这个问题，现代主流模型（GPT-4、Claude 3.5、Qwen-2.5、混元等）在底层权重中原生预训练了 **Tool Calling（Function Calling）** 能力。  
今天我们将深入攻克 LangChain 的工具封装体系与原生 Tool Calling 机制，让工具开发具备工业级的强类型约束、错误自愈与标准化协议支持！

---

## 二、 今日主题与学习目标

- **主题**：LangChain 自定义工具封装体系与原生 Tool Calling 闭环
- **目标**：
  1. 掌握 LangChain 工具构建两大主流模式：轻量级 `@tool` 装饰器与工业级继承 `BaseTool` 强类型类模式；
  2. 深入理解 **Tool Calling 底层数据协议**：掌握 `tools` / `tool_choice` 参数格式、`AIMessage.tool_calls` 结构体（`name`, `args`, `id`）与 `ToolMessage` 回传生命周期；
  3. 掌握工具执行的**容错防御与自愈机制**（`handle_tool_error` 与参数重试）；
  4. 了解开放工具通信标准 **MCP（Model Context Protocol）** 的核心架构与发展脉络；
  5. 手撕端到端原生 Tool Calling 智能体分发引擎，替代传统 ReAct 正则解析。

---

## 三、 精选参考资源（≤3 条）

1. **LangChain How-to: Create Custom Tools**：<https://python.langchain.com/docs/how_to/custom_tools/>
2. **Anthropic / OpenAI Tool Use 规范解析**：<https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
3. **Model Context Protocol (MCP) 官方架构指南**：<https://modelcontextprotocol.io/introduction>

---

## 四、 时间规划建议（约 2~2.5 小时）

- **认知破壁与协议原理（30% · 40分钟）**：
  - 弄懂为什么“原生 Tool Calling”比“纯 Prompt 正则匹配”在工业生产中可靠性提升一个数量级；
  - 搞清 `HumanMessage` -> `AIMessage(tool_calls=[...])` -> 执行工具 -> `ToolMessage(content=..., tool_call_id=...)` -> `AIMessage(Final)` 的标准通信时序。
- **核心代码动手实现（50% · 65分钟）**：
  - 【关卡 1】：使用 `@tool` 装饰器配合 Pydantic v2 `args_schema` 封装计算摄影导向滤波光晕过冲计算器；
  - 【关卡 2】：继承 `BaseTool` 打造工业级 ISP 多帧降噪曝光增益评估器（内置 `handle_tool_error` 容错自愈与类型约束）；
  - 【关卡 3】：使用 `llm.bind_tools` 绑定工具，手撕原生 Tool Calling 执行循环，完成真实大模型端到端调用闭环。
- **实测验证与复盘总结（20% · 25分钟）**：
  - 运行 `uv run python learning/Daily/Day-16-自定义工具开发与ToolCalling/day16.py`；
  - 观察模型如何原生吐出结构化 `tool_calls`，体验强类型工具调用的丝滑与稳健。

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：在 `day16.py` 中使用 `@tool` 装饰器完成 `calc_halo_overshoot` 工具封装；
- [ ] **必做**：继承 `BaseTool` 实现 `ISPExposureGainTool` 类，覆盖 `_run` 并开启异常自愈拦截；
- [ ] **必做**：实现 `run_native_tool_agent` 函数，动态分发 `tool_calls`，构造 `ToolMessage` 并回传给模型产出终极回答；
- [ ] **进阶（读懂即可）**：对比分析 Anthropic MCP 协议与 LangChain Tool 的映射转换关系。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-16-自定义工具开发与ToolCalling/day16.py
```
**判定标准**：
1. 脚本无报错跑通，三大关卡全绿灯通过；
2. 关卡 1 工具元数据提取出完整的 JSON Schema，非法负数入参被 Pydantic 安全拦截；
3. 关卡 2 工具继承 `BaseTool` 并成功拦截非法异常，返回自定义自愈诊断提示；
4. 关卡 3 模型成功触发原生 `tool_calls`，携带标准 `id` 与 `args`，完成工具执行与 `ToolMessage` 回传，输出完整画质诊断报告，终端打印：
   `🎉 [PASSED]: Day 16 关卡全通！LangChain 自定义工具与原生 Tool Calling 闭环手撕成功！`

---

## 七、 周度里程碑倒计时

- **第 3 周里程碑**：构建集成 RAG + Web 搜索的「ISP 算法与代码研究助手」Agent（Day 21）；
- **距离本周里程碑日还差**：**5 个学习日**；
- **后续节奏**：明天打通原生 Tool Calling 后，Day 17 将进阶到 SQL 数据库工具，Day 18 深入复杂参数抽取，步步逼近研究助手 Agent！
