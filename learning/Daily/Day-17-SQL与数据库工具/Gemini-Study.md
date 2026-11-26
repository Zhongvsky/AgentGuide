# Day 17 · SQL & 数据库工具（Text-to-SQL Agent 实战）复盘笔记

> **完成时间**：2026-11-26（周四，提前通关）  
> **核心代码**：[day17.py](file:///d:/Study/AgentGuide/learning/Daily/Day-17-SQL与数据库工具/day17.py)  
> **任务卡**：[Day-17-SQL与数据库工具.md](file:///d:/Study/AgentGuide/learning/learning-cards/Day-17-SQL与数据库工具.md)

---

## 一、 核心架构与工程收获

1. **Schema-First（`inspect_sqlite_schema`）**：
   - 直接查询 `sqlite_master` 的原始 `CREATE TABLE` DDL 语句（过滤 `sqlite_%` 系统表），完整保留列名、数据类型、主键与 `FOREIGN KEY` 外键约束，是 LLM 最易理解的 Schema 表示。
2. **只读安全沙箱与结构化自愈（`execute_safe_readonly_sql`）**：
   - **双保险**：应用层语句校验（仅放行单条 `SELECT` / `WITH`，拦截多语句 `;` 注入与写/DDL 操作） + 驱动层 `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)` 物理只读隔离。
   - **结构化报错自愈**：捕获数据库异常后返回 `{"ok": False, "rows": [], "row_count": 0, "error": str(e)}`，避免进程崩溃并向模型提供高信噪比的修正线索。
3. **工具参数与基础设施参数解耦（`args_schema` vs `BaseTool` 私有字段）**：
   - **`args_schema`** 只放模型需要生成的业务参数（如 `sql`、`max_rows`）；
   - **`db_path`** 属于后端基础设施配置，应声明在 `BaseTool` 子类自身属性上（`db_path: str`），在初始化时注入并通过 `self.db_path` 读取（或使用 `@tool` 闭包捕获），绝不暴露给大模型猜测。

---

## 二、 实战踩坑实录

1. **末尾分号误杀**：合法 SQL 常以 `;` 结尾，直接 `" ;" in sql` 会误拦；需先 `.strip().rstrip(";")` 再检测中间是否含分号。
2. **布尔运算符优先级**：`not A or B` 等价于 `(not A) or B`，导致 `WITH` 语句被误杀；必须写 `not (A or B)`。
3. **异常对象不可 JSON 序列化**：`except Exception as e` 中直接返回 `"error": e` 会触发 `TypeError: Object of type OperationalError is not JSON serializable`，需转为 `str(e)`。
4. **`db_path` 暴露给 LLM 引发空库死循环**：最初 `args_schema` 中保留了 `db_path`，导致模型反复猜测路径打开空库并连续查询 `sqlite_master` 13 次耗尽步数；将 `db_path` 收口为 `self.db_path` 后，模型 4 轮（`12877.52 ms`）即精准完成值域探查与多表 `JOIN` 问答。

---

## 三、 实测验收结果

- **关卡 1**：成功提取 `isp_configs` 与 `badcase_records` 完整 DDL（含外键）。
- **关卡 2**：合法 `SELECT` 返回 2 条记录；100% 拦截写操作/DDL/分号注入；错误 SQL 返回 `ok=False` 与 `no such column: non_existent_col`。
- **关卡 3**：模型自主执行 3 条 SQL（含枚举值域探查与多表 `JOIN`），耗时 `12877.52 ms`（4 轮交互），准确输出 `SA8155` / `FGF` / `v2.1` 高风险记录表格，终端打印 `🎉 [PASSED]: Day 17 关卡全通！`。
