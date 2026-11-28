# Day 18 复盘笔记 · Function Calling 实战与结构化抽取

> **日期**：2026-11-28（周六）｜**进度**：18 / 60  
> **核心代码**：[day18.py](file:///d:/Study/AgentGuide/learning/Daily/Day-18-FunctionCalling实战/day18.py)

---

## 一、 核心架构与知识点沉淀

1. **Structured Outputs 两种底层策略对比**：
   - `method="json_schema"`（`langchain-openai` 1.7.x 默认）：通过 HTTP 请求体的 `response_format` 下发 JSON Schema，依赖推理引擎服务端做约束解码；第三方兼容/免费代理端点若未实现服务端约束，模型会自由编造字段名（实测吐出 `ticket_id` 导致 Pydantic `ValidationError`）。
   - `method="function_calling"`：将 Pydantic Schema 编译为虚拟工具放进 `tools` 并绑定 `tool_choice`，模型在 `tool_calls[0].function.arguments` 中结构化填充字段，对兼容网关稳定性显著更高。注意当模型未按预期触发 `tool_calls` 时，`PydanticToolsParser` 会返回 `None`，生产代码应增加显式判空或重试。
2. **原生 OpenAI SDK Wire Protocol 状态机**：
   - 请求侧：`tools`（JSON Schema 列表）+ `tool_choice="auto"` + `temperature=0.0`。
   - 响应侧：`msg = resp.choices[0].message`，当 `msg.tool_calls` 非空时，`msg.content` 常为空；每个 `tc` 含 `tc.id`、`tc.function.name`、`tc.function.arguments`（JSON 字符串）。
   - 回传成对铁律：`messages` 中必须先追加规范化的 `assistant` 字典（含纯字典化的 `tool_calls` 列表，避免直接传 Pydantic 对象带出多余 `None` 字段），再为**每一个** `tc.id` 追加 `role: "tool"` + `tool_call_id` + 字符串 `content`。
3. **两段式分诊流水线（Extract-then-Act）**：
   - 先将非结构化故障描述收敛为强类型 `ISPBugTicket`（动态提取平台、模块、缺陷类型、严重度、可疑参数），再以通用分诊 SOP 驱动原生 Function Calling 核查参数基线与 SLA 排期。

---

## 二、 实战踩坑记录

1. **兼容端点 `json_schema` 静默失效**：`auto/hy4flash-free` 未强制执行 `response_format.json_schema`，导致字段缺失报错；显式切换 `method="function_calling"` 后稳定抽取 `ISPBugTicket`。
2. **`Completions.create` 关键字参数名**：原生 SDK 入参为 `tools=...` 而非 `tools_schema=...`。
3. **`msg.tool_calls` 序列化与异常边界**：将 `msg.tool_calls` 转为纯字典列表再写入 `messages`；将 `json.loads(arguments)` 纳入 `try ... except Exception as e:` 内，异常时返回错误信息字符串供模型自愈。
4. **Python 多行隐式拼接误加逗号变 `tuple`**：关卡 3 构造 `query` 时括号内每行末尾加了 `,`，导致 `query` 变为元组而非 `str`，序列化异常使模型未调用工具；去掉逗号并采用 `.model_dump_json(ensure_ascii=False)` + 通用 SOP 指令后，测试 3 在 `13332.26 ms` 内调用 `lookup_isp_param_baseline` 与 `estimate_fix_priority` 圆满通关。
