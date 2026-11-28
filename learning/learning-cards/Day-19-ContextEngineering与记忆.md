# Day 19 · Context Engineering 与记忆（窗口裁剪、工具截断与轻量持久化）

> 第 19 / 60 个学习日｜第 3 周 · Agent 开发、Context Engineering 与 MCP  
> **范式演进**：跳过已过时的旧版 `LangChain Memory` 黑盒封装，掌握生产级 **Context Engineering（上下文工程）**——在有限 Token 预算内保住关键信息、防止孤立 `tool` 消息炸穿协议，并用 SQLite 实现可跨重启恢复的轻量会话记忆。

---

## 一、 承前启后与痛点剖析

在 Day 18 中，我们打通了原生的 Function Calling `messages` 状态机。但一旦把它放到真实多轮 ISP 排障场景中，马上会撞上三个工程瓶颈：

1. **工具输出爆炸**：比如调用一个 `dump_isp_registers` 或日志读取工具，单次返回 20,000 字符，直接塞进 `messages` 会瞬间击穿上下文窗口、拖慢首字延迟并烧光 Token 预算；
2. **粗暴切片炸协议**：多轮对话变长后，如果直接写 `messages[-6:]` 截断历史，不仅会丢掉开头的 `system` 人设，更致命的是可能把 `assistant(tool_calls)` 切走而只留下 `role: "tool"`，导致下一轮请求直接触发 **HTTP 400**；
3. **进程重启失忆**：内存里的 `messages` 列表一旦服务重启就清零，需要一个零重型依赖、结构透明的会话状态持久化方案。

---

## 二、 今日主题与核心目标

- **主题**：Context Engineering（上下文工程）与轻量会话持久化
- **核心目标**：
  1. 实现**首尾保留式工具输出截断器**（`truncate_tool_output`），在控制长度的同时保住头部报错与尾部汇总；
  2. 实现**协议安全的上下文窗口管理器**（`manage_context_window`），保护 `system` 消息与 `assistant(tool_calls) + role: "tool"` 原子组，并对被淘汰的早期历史生成摘要；
  3. 基于标准库 `sqlite3` 实现**轻量级会话存储**（`SessionStore`），支持幂等保存与恢复。

---

## 三、 精选参考资源（≤3 条）

1. **Anthropic Engineering: Effective context engineering for AI agents**：<https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
2. **LangChain Docs: Short-term memory & trimming messages**：<https://python.langchain.com/docs/how_to/trim_messages/>
3. **Python `sqlite3` Official Docs**：<https://docs.python.org/3/library/sqlite3.html>

---

## 四、 时间规划（提前预习 + 实操 ~2 小时，纯编码 ~90 分钟）

- **提前预习（15~25 分钟）**：
  - 思考为什么工具长文本截断往往采用 **“首尾保留（Head + Tail）”** 而不是单纯保留前 N 个字符；
  - 推演“按原子组倒序贪心保留 + 早期历史摘要”的状态流转，确保不破坏 `tool_calls` 与 `role: "tool"` 的成对契约。
- **集中实操（总 ~2 小时，纯编码+调试 ~90 分钟）**：
  - 环境就绪（10min）→ 3 个紧凑关卡手撕（70min）：
    - 【关卡 1】：实现 `truncate_tool_output`；
    - 【关卡 2】：实现 `manage_context_window`；
    - 【关卡 3】：实现 `SessionStore`；
  - `uv run` 验收与复盘（15min）。

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：实现 `truncate_tool_output`，支持超长工具输出按比例首尾保留并标注截断字符数；
- [ ] **必做**：实现 `manage_context_window`，永远保留主 `system`、不拆散 `tool_calls` 原子组，并在超预算淘汰旧消息时支持摘要注入；
- [ ] **必做**：实现 `SessionStore`（`save_session` / `load_session`），完成 SQLite 幂等读写与跨实例恢复验证；
- [ ] **进阶（读懂即可）**：理解 Context Engineering 中“写（Write）/ 选（Select）/ 压（Compress）/ 隔（Isolate）”四大核心动作。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-19-ContextEngineering与记忆/day19.py
```
**判定标准**：
1. 关卡 1 将 5000+ 字符模拟 ISP 寄存器日志准确截断至预算内，且同时包含原始头部标志与尾部汇总标志；
2. 关卡 2 在严格 Token 预算下完整保留 `system` 与最新 `tool_calls + tool` 成对原子组，未出现孤立 `tool` 消息，且早期对话被压缩为摘要；
3. 关卡 3 使用全新 `SessionStore` 实例从 SQLite 文件成功恢复完整对话列表，终端打印：
   `🎉 [PASSED]: Day 19 关卡全通！Context Engineering 与轻量持久化闭环通关！`

---

## 七、 阶段里程碑倒计时与前置物料预警

- **第 3 周里程碑**：搭建「ISP 研究助手」，并将 1 个本地查询工具封装为最小可运行 **MCP Server** 跑通调用（Day 21）；
- **距离本周里程碑日还差**：**2 个学习日**；
- **前置物料预警**：Day 19–21 均使用本地环境与标准库，无需额外外部物料；下一个需提前准备的云端物料为 **Day 27 的 Langfuse Cloud 免费版 API Key**，可待第 4 周前空闲时间再注册。
