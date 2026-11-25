# Day 16 · 自定义工具开发与原生 Tool Calling 复盘笔记

> **日期**：2026-11-25（周三）｜**阶段**：第 3 周 · Agent 开发与 Tool Calling  
> **代码入口**：`learning/Daily/Day-16-自定义工具开发与ToolCalling/day16.py`

---

## 一、 核心知识点沉淀

1. **从 ReAct 文本解析到原生 Tool Calling 的代际演进**
   - **请求侧**：`llm.bind_tools(tools)` 将工具名称、Docstring 和 Pydantic `args_schema` 编译为标准 JSON Schema，挂载在请求体 `tools` 字段（而非混入 Prompt 文本）。
   - **响应侧**：模型返回 `AIMessage.tool_calls`，每个调用自带唯一 `id`、工具名 `name` 和已反序列化的字典 `args`，支持单轮多工具并发调用（Parallel Tool Calling）。
   - **协议配对铁律**：`messages` 必须先追加包含 `tool_calls` 的 `AIMessage`，再为每个调用追加 `ToolMessage(content=str(result), tool_call_id=tc["id"])`，否则服务端协议校验会直接报错。

2. **`@tool` vs `BaseTool` 双模式选型**
   - **`@tool(args_schema=...)`**：适合无状态、轻量级计算/函数工具；
   - **`BaseTool` 子类**：适合工业级生产工具，支持持有状态依赖（数据库连接、Milvus 客户端）及 `handle_tool_error` 异常自愈。

3. **两层异常防御边界**
   - `args_schema` 的 Pydantic `ValidationError` 发生在进入 `_run()` **之前**；
   - 因此除了工具内部的 `handle_tool_error`，Agent 调度层（`run_native_tool_agent`）外层也必须用 `try ... except Exception as e:` 捕获执行与校验异常，并转为语义化诊断文本写回 `ToolMessage`，让模型有机会在下一轮修正参数自愈。

---

## 二、 实战踩坑与修复记录

1. **`ToolMessage` 关键字参数名**：必须是 `tool_call_id=tool_id`（误写为 `tool_id` 会导致协议字段缺失）。
2. **返回契约与拼写对齐**：返回字典键名 `"tool_call_history"` 及子字典字段 `"id"` / `"name"` / `"args"` / `"output"` 需与下游消费方严格一致。
3. **异常区分与抛出规范**：调度器中先判断 `if tool_name in tool_map:` 再 `try ... except`，避免参数报错被误报为“未找到工具”；外层 `except` 不可 `raise "..."` 字符串，应直接 `raise` 或抛出 `Exception` 子类。

---

## 三、 实测验收结果

- **关卡 1（`@tool`）**：`calc_halo_overshoot` 正常计算 `halo_risk_index=3.0`，负数 `edge_contrast` 被 Pydantic 成功拦截。
- **关卡 2（`BaseTool`）**：`ISPExposureGainTool` 精准算出 `total_gain=96.0` 并命中 `Open MFNR Super Night`，`iso_value=10.0` 被 `ValidationError` 拦截。
- **关卡 3（端到端闭环）**：连接 `auto/hy4flash-free`，**2 轮模型交互（总耗时 12835.64 ms）** 完成：
  - 第 1 轮并发调用 `isp_exposure_gain_eval`（`total_gain=160.0`）与 `calc_halo_overshoot`（`halo_risk_index=4.5, is_severe=True`）；
  - 第 2 轮汇总输出完整《夜景高反差人像 · ISP 调试优化诊断书》，全关卡绿灯通关。
