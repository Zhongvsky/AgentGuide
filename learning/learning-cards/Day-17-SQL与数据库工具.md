# Day 17 · SQL & 数据库工具（Text-to-SQL Agent 实战）

> 2026-11-30（周一）｜第 17 / 60 个学习日｜第 3 周 · Agent 开发与 Tool Calling  
> **范式演进**：从“无状态计算工具”迈向“连接企业结构化数据库”，打造具备 Schema 感知、只读安全防线与 SQL 报错自愈能力的 **Text-to-SQL Agent**！

---

## 一、 承前启后与痛点剖析

在 Day 16 中，我们掌握了 `@tool`、`BaseTool` 与原生 Tool Calling 闭环，让大模型能够调用纯 Python 计算工具。但真实生产环境中，大量核心数据（实验记录、ISP 配置参数表、画质 Bad Case 工单、业务指标）都存储在**关系型数据库（SQLite / PostgreSQL / MySQL）**中，无法靠硬编码几个静态计算函数覆盖所有查询需求。

让大模型直接生成并执行 SQL，面临三大核心工程挑战：
1. **盲人摸象（Schema 未知）**：模型不知道表名、字段名和外键关联关系，直接写 SQL 极易编造不存在的列名；
2. **高危越权（SQL 注入与破坏性写入）**：一旦模型生成或被诱导生成 `DROP TABLE`、`DELETE`、`UPDATE` 或多语句分号拼接，会直接造成生产事故；
3. **语法容错（SQL 报错自愈）**：真实查询常因方言差异或列名拼写出错而报错，Agent 必须能把数据库报错信息回传给模型，让模型自动修正 SQL 重试。

---

## 二、 今日主题与核心目标

- **主题**：SQL 数据库工具封装与 Text-to-SQL 原生 Tool Calling 闭环
- **核心目标**：
  1. 理解 **Schema-First（先探查表结构再写 SQL）** 的数据库 Agent 交互范式；
  2. 实现 **只读安全沙箱 SQL 执行器**：拦截一切写操作、DDL 与多语句注入，限制返回行数以保护上下文窗口；
  3. 构建 **具备报错自愈能力的端到端 SQL Agent**：能自主完成 Schema 探查 -> 多表关联 SQL 生成 -> 失败自愈修正 -> 事实总结输出。

---

## 三、 精选参考资源（≤3 条）

1. **LangChain How-to: Build a Question/Answering system over SQL data**：<https://python.langchain.com/docs/tutorials/sql_qa/>
2. **SQLite Official SQL Language Reference**：<https://www.sqlite.org/lang.html>
3. **OWASP SQL Injection Prevention Cheat Sheet**：<https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html>

---

## 四、 时间规划建议（约 2~2.5 小时）

- **原理理解（30% · 40 分钟）**：
  - 梳理 Text-to-SQL Agent 的标准执行链路：`Schema Inspection -> SQL Generation -> Safe Read-Only Execution -> Self-Healing / Synthesis`；
  - 思考为什么生产级 SQL 工具必须做**词法/语句级只读拦截**与**数据库权限隔离（Read-Only User）**双保险。
- **代码动手（50% · 65 分钟）**：
  - 【关卡 1】：实现 `inspect_sqlite_schema`，自动提取数据库表结构与字段契约；
  - 【关卡 2】：实现 `execute_safe_readonly_sql`，完成只读安全拦截、行数截断与结构化 JSON 报错回传；
  - 【关卡 3】：实现 `run_sql_agent`，将上述能力封装为工具接入原生 Tool Calling 闭环，跑通多表关联分析问答。
- **实测对比与复盘（20% · 25 分钟）**：
  - 运行 `uv run python learning/Daily/Day-17-SQL与数据库工具/day17.py`；
  - 复盘模型如何自主探查 Schema、生成 `JOIN` 查询，并在遇到非法 SQL 时如何被安全拦截。

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：实现 `inspect_sqlite_schema`，能完整输出 `isp_configs` 与 `badcase_records` 的表与字段信息；
- [ ] **必做**：实现 `execute_safe_readonly_sql`，拦截 `DROP`/`DELETE`/`UPDATE`/`INSERT`/多语句注入，且语法错误时返回结构化 `error`；
- [ ] **必做**：实现 `run_sql_agent`，完成真实大模型端到端多表关联问答与结果校验；
- [ ] **进阶（读懂即可）**：对比“静态预置 SQL 模板工具”与“动态 Text-to-SQL 工具”在安全性与灵活性上的权衡。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-17-SQL与数据库工具/day17.py
```
**判定标准**：
1. 关卡 1 成功提取出 `isp_configs` 与 `badcase_records` 全部字段结构；
2. 关卡 2 成功执行合法 `SELECT` 查询，同时 100% 拦截 `DELETE`、`DROP`、多语句分号注入，且对错误 SQL 返回 `ok=False` 与可读错误信息；
3. 关卡 3 真实大模型自主完成 Schema 探查 + 多表 `JOIN` 查询，精准回答测试问题，终端打印：
   `🎉 [PASSED]: Day 17 关卡全通！SQL 数据库工具与 Text-to-SQL Agent 闭环通关！`

---

## 七、 周度里程碑倒计时

- **第 3 周里程碑**：构建集成 RAG + Web 搜索的「ISP 算法与代码研究助手」Agent（Day 21）；
- **距离本周里程碑日还差**：**4 个学习日**；
- **后续节奏**：Day 17 打通结构化数据库查询后，Day 18 将继续攻克 OpenAI Function Calling 结构化抽取实战。
