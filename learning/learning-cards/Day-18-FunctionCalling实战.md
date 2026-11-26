# Day 18 · Function Calling 实战（结构化抽取与原生协议闭环）

> 2026-12-01（周二）｜第 18 / 60 个学习日｜第 3 周 · Agent 开发与 Tool Calling  
> **范式演进**：从“LangChain 封装调用”下沉到“OpenAI 原生 Function Calling 协议与 Pydantic 强类型结构化抽取”，彻底掌握工具调用底层 Wire Protocol！

---

## 一、 承前启后与痛点剖析

在 Day 16 与 Day 17 中，我们借助 LangChain 的 `@tool` 与 `BaseTool` 快速跑通了工具调用与 Text-to-SQL Agent。但在生产环境中，仅会用上层框架封装往往不够：

1. **非结构化文本入库难**：前端、邮件或工单系统传来的是一段杂乱的自然语言故障描述（如“SA8155 昨晚夜景 FGF 模块出现严重 Halo，guide_radius 设了 8.0...”），如何稳定抽取出严格符合 Pydantic Schema 的 JSON 对象，而不是靠正则硬抠？
2. **框架黑盒感**：LangChain 在底层帮我们把 `@tool` 转成了 JSON Schema、把 `ToolMessage` 转成了 `role: "tool"` 消息。如果脱离框架，直接用原生 `openai` SDK 发请求，`tools`、`tool_choice` 与 `tool_calls` 协议到底长什么样？
3. **复杂业务链路组合**：真实线上系统往往是“**先结构化抽取实体（Structured Output） -> 再按实体路由触发工具调用（Function Calling）**”的两段式流水线。

---

## 二、 今日主题与核心目标

- **主题**：OpenAI 原生 Function Calling 协议与 Pydantic 结构化抽取实战
- **核心目标**：
  1. 掌握基于 Pydantic v2 Schema 的 **Structured Output（结构化抽取）**，从非结构化 ISP 故障报告中零误差提取强类型工单对象；
  2. 剥离框架封装，使用原生 `openai.OpenAI` SDK 完成 `tools` JSON Schema 定义、`tool_choice` 控制、工具分发与 `role="tool"` 回传闭环；
  3. 串联“结构化抽取 + 原生 Function Calling”构建端到端 **ISP 故障自动分诊流水线**。

---

## 三、 精选参考资源（≤3 条）

1. **OpenAI API Guide: Function Calling**：<https://platform.openai.com/docs/guides/function-calling>
2. **OpenAI API Guide: Structured Outputs**：<https://platform.openai.com/docs/guides/structured-outputs>
3. **LangChain Docs: Structured outputs**：<https://python.langchain.com/docs/how_to/structured_output/>

---

## 四、 时间规划建议（约 2~2.5 小时）

- **原理理解（30% · 40 分钟）**：
  - 对比 `Prompt 要求输出 JSON` vs `Structured Outputs / Function Calling` 在 Schema 约束力与容错率上的本质差异；
  - 厘清原生 OpenAI Chat Completions 协议中 `assistant.tool_calls` 与 `role: "tool"` + `tool_call_id` 的请求/响应报文字段。
- **代码动手（50% · 65 分钟）**：
  - 【关卡 1】：实现 `extract_isp_bug_ticket`，从非结构化文本抽取强类型 `ISPBugTicket`；
  - 【关卡 2】：实现 `run_native_openai_fc`，使用原生 `openai.OpenAI` SDK 完成多工具调度闭环；
  - 【关卡 3】：实现 `run_isp_triage_pipeline`，串联结构化抽取与工具分诊闭环。
- **实测对比与复盘（20% · 25 分钟）**：
  - 运行 `uv run python learning/Daily/Day-18-FunctionCalling实战/day18.py`；
  - 复盘原生 SDK 报文格式与框架封装之间的映射关系。

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：实现 `extract_isp_bug_ticket`，能从非结构化工单准确抽取出 `ISPBugTicket` 各字段；
- [ ] **必做**：实现 `run_native_openai_fc`，不依赖 LangChain Agent 封装，用原生 `openai.OpenAI` 跑通 `tools` 调用与结果回传；
- [ ] **必做**：实现 `run_isp_triage_pipeline`，完成“先抽取后工具诊断”的端到端分诊闭环；
- [ ] **进阶（读懂即可）**：理解 `tool_choice="auto"`、`tool_choice="required"` 与指定特定函数名的适用场景。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-18-FunctionCalling实战/day18.py
```
**判定标准**：
1. 关卡 1 成功将两份非结构化 ISP 故障文本抽取为合法的 `ISPBugTicket` 实例，关键字段（芯片平台、模块、缺陷类型、严重度）100% 准确；
2. 关卡 2 原生 `openai.OpenAI` 客户端成功触发工具调用并回传 `role="tool"` 消息，最终输出完整回复；
3. 关卡 3 端到端分诊流水线跑通，终端打印：
   `🎉 [PASSED]: Day 18 关卡全通！Function Calling 与结构化抽取闭环通关！`

---

## 七、 周度里程碑倒计时

- **第 3 周里程碑**：构建集成 RAG + Web 搜索的「ISP 算法与代码研究助手」Agent（Day 21）；
- **距离本周里程碑日还差**：**3 个学习日**；
- **后续节奏**：Day 18 夯实 Function Calling 与结构化抽取后，Day 19 将进入 Agent Memory（对话记忆与持久化）。
