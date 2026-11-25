"""
Day 17: 第 3 周进阶 - SQL & 数据库工具（Text-to-SQL Agent 实战）
===================================================================
本脚本为 Day 17 的闯关代码骨架（零剧透纯契约版）。
请根据各关卡的函数签名、I/O 契约与验收标准，独立完成架构设计与代码实现：
1. 关卡 1：SQLite Schema 自动探查（inspect_sqlite_schema）
2. 关卡 2：只读安全 SQL 执行器（execute_safe_readonly_sql）
3. 关卡 3：端到端 Text-to-SQL Agent 闭环（run_sql_agent）

【闯关打靶规则】：
- 骨架仅提供业务目标与 I/O 契约，不剧透具体实现步骤；
- 实现完成后删除对应的 raise NotImplementedError 行，运行脚本逐关验收；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import os
import sys
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Windows 控制台 UTF-8 防御
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==========================================
# 0. 环境初始化与 ISP 实验数据库构建（开箱即用）
# ==========================================

DB_PATH = str(Path(__file__).parent / "isp_experiments.db")


def get_agent_llm() -> ChatOpenAI:
    """初始化并返回兼容 OpenAI 接口的大模型客户端"""
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:3425/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model_name = os.getenv("MODEL_NAME", "auto/hy4flash-free")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.0,
        max_tokens=4096,
        timeout=60.0,
    )
    print(f"[+] 成功连接 Agent 大模型端点 ({base_url} | 模型: {model_name})！")
    return llm


def init_isp_sqlite_db(db_path: str) -> None:
    """初始化本地 SQLite 测试数据库（幂等重建，包含 ISP 配置表与 BadCase 记录表）"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS badcase_records")
    cur.execute("DROP TABLE IF EXISTS isp_configs")

    cur.execute(
        """
        CREATE TABLE isp_configs (
            config_id INTEGER PRIMARY KEY,
            chip_platform TEXT NOT NULL,
            module_name TEXT NOT NULL,
            param_name TEXT NOT NULL,
            param_value REAL NOT NULL,
            version TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE badcase_records (
            case_id INTEGER PRIMARY KEY,
            config_id INTEGER NOT NULL,
            scene_type TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            halo_risk_index REAL NOT NULL,
            total_gain REAL NOT NULL,
            severity TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (config_id) REFERENCES isp_configs(config_id)
        )
        """
    )

    configs = [
        (1, "SA8155", "FGF", "guide_radius", 8.0, "v2.1"),
        (2, "SA8155", "FGF", "smooth_eps", 0.02, "v2.2"),
        (3, "Snapdragon8Gen3", "MFNR", "temporal_weight", 0.75, "v3.0"),
        (4, "Snapdragon8Gen3", "TMC", "highlight_compress", 0.45, "v3.1"),
    ]
    cur.executemany("INSERT INTO isp_configs VALUES (?, ?, ?, ?, ?, ?)", configs)

    badcases = [
        (101, 1, "夜景高反差人像", "Halo", 4.5, 160.0, "HIGH", "2026-11-20"),
        (102, 1, "逆光路灯", "Halo", 3.8, 120.0, "HIGH", "2026-11-21"),
        (103, 2, "夜景高反差人像", "Halo", 1.2, 150.0, "LOW", "2026-11-22"),
        (104, 3, "暗光运动抓拍", "Ghosting", 0.8, 180.0, "MEDIUM", "2026-11-23"),
        (105, 4, "强光高光场景", "ColorCast", 1.1, 20.0, "LOW", "2026-11-24"),
    ]
    cur.executemany("INSERT INTO badcase_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", badcases)

    conn.commit()
    conn.close()


# ==========================================
# 1. 关卡 1：SQLite Schema 自动探查
# ==========================================


def inspect_sqlite_schema(db_path: str) -> str:
    """
    【关卡 1 契约】：探查指定 SQLite 数据库的表结构。

    输入契约：
    - db_path: SQLite 数据库文件路径。

    输出契约：
    - 返回非空字符串（建议为清晰的文本或 JSON），必须完整包含所有用户表的表名、
      各表字段名与字段类型；若表存在外键关联，建议一并体现。
    - 不得暴露或依赖内部系统表（如 sqlite_sequence）。
    """
    # TODO 1: 实现 SQLite Schema 探查
    raise NotImplementedError(
        "【关卡 1 待实现】：请实现 inspect_sqlite_schema！完成后删除本行。"
    )


# ==========================================
# 2. 关卡 2：只读安全 SQL 执行器
# ==========================================


def execute_safe_readonly_sql(
    db_path: str, sql: str, max_rows: int = 50
) -> str:
    """
    【关卡 2 契约】：在只读安全约束下执行 SQL 查询并返回结构化 JSON 字符串。

    输入契约：
    - db_path: SQLite 数据库文件路径。
    - sql: 待执行的 SQL 语句。
    - max_rows: 最大返回行数上限（防止结果集过大撑爆上下文）。

    安全与边界契约：
    - 仅允许单条只读查询（如 SELECT / WITH ... SELECT）；
    - 必须拦截所有写操作与高危语句（INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/REPLACE/TRUNCATE/ATTACH/DETACH/PRAGMA），
      以及多语句分号注入（如 "SELECT 1; DROP TABLE ..."）；
    - 无论被安全策略拦截还是 SQL 语法/字段报错，均不得向外抛出未捕获异常导致程序崩溃，
      必须返回结构化 JSON 字符串供上层 Agent 自愈。

    输出契约（JSON 字符串，反序列化后必须包含以下 4 个键）：
    - "ok": bool（执行成功为 True，被拦截或报错为 False）
    - "rows": list[dict]（成功时为字典列表，每行以列名作 key；失败时为空列表 []）
    - "row_count": int（实际返回的 rows 长度）
    - "error": str | None（成功时为 None；失败或被拦截时为清晰的错误原因说明）
    """
    # TODO 2: 实现只读安全 SQL 执行器
    raise NotImplementedError(
        "【关卡 2 待实现】：请实现 execute_safe_readonly_sql！完成后删除本行。"
    )


# ==========================================
# 3. 关卡 3：端到端 Text-to-SQL Agent 闭环
# ==========================================


def run_sql_agent(
    llm: ChatOpenAI,
    db_path: str,
    question: str,
    max_steps: int = 6,
) -> Dict[str, Any]:
    """
    【关卡 3 契约】：构建基于原生 Tool Calling 的 Text-to-SQL Agent，回答针对数据库的自然语言分析问题。

    输入契约：
    - llm: 已初始化的 ChatOpenAI 实例。
    - db_path: 目标 SQLite 数据库路径。
    - question: 用户的自然语言查询问题。
    - max_steps: 最大模型交互轮数上限。

    行为契约：
    - Agent 必须具备探查数据库 Schema 与执行只读 SQL 查询的能力（可将关卡 1、2 封装为工具）；
    - 若生成的 SQL 报错，Agent 应能根据错误信息在后续轮次中自愈修正；
    - 最终回答必须基于数据库真实查询结果，严禁编造数据。

    输出契约（返回字典，必须包含以下 3 个键）：
    - "final_answer": str（最终回答）
    - "executed_sqls": List[str]（Agent 实际发起执行过的 SQL 语句列表）
    - "total_steps": int（实际经历的模型交互轮数）
    """
    # TODO 3: 实现端到端 Text-to-SQL Agent
    raise NotImplementedError(
        "【关卡 3 待实现】：请实现 run_sql_agent！完成后删除本行。"
    )


# ==========================================
# 4. 领域实战与 Milestone 验收用例
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("【Day 17 实战】：SQL & 数据库工具（Text-to-SQL Agent 实战）")
    print("=" * 80)

    init_isp_sqlite_db(DB_PATH)
    print(f"[+] 已初始化测试数据库: {DB_PATH}")

    # ----------------------------------------------------
    # 测试关卡 1：Schema 自动探查
    # ----------------------------------------------------
    print("\n🔍 【测试 1：SQLite Schema 自动探查】")
    schema_text = inspect_sqlite_schema(DB_PATH)
    assert isinstance(schema_text, str) and len(schema_text) > 20, "Schema 返回内容过短"
    for expected in [
        "isp_configs",
        "badcase_records",
        "chip_platform",
        "module_name",
        "halo_risk_index",
        "severity",
    ]:
        assert expected in schema_text, f"Schema 缺少关键标识: {expected}"
    print(f"✅ 关卡 1 通过：成功提取完整表结构：\n{schema_text}")

    # ----------------------------------------------------
    # 测试关卡 2：只读安全 SQL 执行器
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 2：只读安全 SQL 执行器单元测试】")

    # 2.1 正常只读查询
    ok_res = json.loads(
        execute_safe_readonly_sql(
            DB_PATH,
            "SELECT case_id, defect_type, halo_risk_index FROM badcase_records WHERE severity = 'HIGH' ORDER BY case_id;",
        )
    )
    assert ok_res["ok"] is True, f"合法查询失败: {ok_res}"
    assert ok_res["row_count"] == 2, f"期望返回 2 条 HIGH 记录，实际: {ok_res}"
    assert ok_res["error"] is None
    print(f"✅ 关卡 2.1 通过：合法 SELECT 查询返回 {ok_res['row_count']} 条记录。")

    # 2.2 高危写操作与多语句注入拦截
    dangerous_sqls = [
        "DELETE FROM badcase_records WHERE case_id = 101",
        "DROP TABLE isp_configs",
        "UPDATE isp_configs SET param_value = 99.0 WHERE config_id = 1",
        "SELECT * FROM isp_configs; DROP TABLE badcase_records;",
    ]
    for bad_sql in dangerous_sqls:
        bad_res = json.loads(execute_safe_readonly_sql(DB_PATH, bad_sql))
        assert bad_res["ok"] is False, f"未能拦截高危 SQL: {bad_sql}"
        assert bad_res["rows"] == [] and bad_res["row_count"] == 0
        assert bad_res["error"], "被拦截时必须返回明确的 error 说明"
    print("✅ 关卡 2.2 通过：100% 拦截写操作、DDL 与分号多语句注入！")

    # 2.3 SQL 语法/字段错误容错
    syntax_err_res = json.loads(
        execute_safe_readonly_sql(DB_PATH, "SELECT non_existent_col FROM isp_configs")
    )
    assert syntax_err_res["ok"] is False and syntax_err_res["error"]
    print(f"✅ 关卡 2.3 通过：SQL 语法/字段报错未崩溃，已转为可读诊断：{syntax_err_res['error']}")

    # ----------------------------------------------------
    # 测试关卡 3：端到端 Text-to-SQL Agent 实测
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 3：端到端 Text-to-SQL Agent 多表关联实测】")

    llm = get_agent_llm()
    test_question = (
        "请查询数据库，找出所有 severity 为 'HIGH' 的 Halo 缺陷记录，"
        "关联它们对应的芯片平台 (chip_platform)、ISP 模块 (module_name) 与配置版本 (version)，"
        "并列出这些记录的 case_id 与 halo_risk_index。"
    )

    t0 = time.time()
    agent_res = run_sql_agent(llm=llm, db_path=DB_PATH, question=test_question, max_steps=6)
    t1 = time.time()

    print(f"\n📜 【Agent 执行过的 SQL】:")
    for i, s in enumerate(agent_res["executed_sqls"], 1):
        print(f"  {i}. {s}")
    print(f"\n🏆 【Agent 最终回答】:\n{agent_res['final_answer']}")
    print(f"\n总耗时: {(t1 - t0) * 1000:.2f} ms，共经历 {agent_res['total_steps']} 轮交互。")

    assert len(agent_res["executed_sqls"]) >= 1, "Agent 未执行任何 SQL 查询"
    ans = agent_res["final_answer"]
    assert "SA8155" in ans and "FGF" in ans, "回答遗漏了关联表中的芯片平台或模块信息"
    assert "101" in ans and "102" in ans, "回答遗漏了 HIGH 级别的 case_id (101, 102)"

    print("\n" + "=" * 80)
    print("🎉 [PASSED]: Day 17 关卡全通！SQL 数据库工具与 Text-to-SQL Agent 闭环通关！")
    print("=" * 80)


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
import re

# 关卡 1 参考实现：
def inspect_sqlite_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [row[0] for row in cur.fetchall()]
    parts = []
    for table in tables:
        cur.execute(f'PRAGMA table_info("{table}")')
        cols = [f"{r[1]} ({r[2]}{' PK' if r[5] else ''})" for r in cur.fetchall()]
        cur.execute(f'PRAGMA foreign_key_list("{table}")')
        fks = [f"{r[3]} -> {r[2]}.{r[4]}" for r in cur.fetchall()]
        block = f"Table {table}: columns=[{', '.join(cols)}]"
        if fks:
            block += f", foreign_keys=[{', '.join(fks)}]"
        parts.append(block)
    conn.close()
    return "\\n".join(parts)


# 关卡 2 参考实现：
def execute_safe_readonly_sql(
    db_path: str, sql: str, max_rows: int = 50
) -> str:
    cleaned = sql.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if not cleaned:
        return json.dumps({"ok": False, "rows": [], "row_count": 0, "error": "SQL 不能为空"}, ensure_ascii=False)

    if ";" in cleaned:
        return json.dumps({"ok": False, "rows": [], "row_count": 0, "error": "安全拦截：禁止多语句分号拼接"}, ensure_ascii=False)

    forbidden = re.search(
        r"\\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA)\\b",
        cleaned,
        re.IGNORECASE,
    )
    if forbidden:
        return json.dumps(
            {"ok": False, "rows": [], "row_count": 0, "error": f"安全拦截：禁止只读以外的关键字 {forbidden.group(1).upper()}"},
            ensure_ascii=False,
        )

    if not re.match(r"^(SELECT|WITH)\\b", cleaned, re.IGNORECASE):
        return json.dumps(
            {"ok": False, "rows": [], "row_count": 0, "error": "安全拦截：仅允许 SELECT 或 WITH 只读查询"},
            ensure_ascii=False,
        )

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(cleaned)
        fetched = cur.fetchmany(max_rows)
        rows = [dict(r) for r in fetched]
        conn.close()
        return json.dumps({"ok": True, "rows": rows, "row_count": len(rows), "error": None}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "rows": [], "row_count": 0, "error": f"SQL 执行错误: {e}"}, ensure_ascii=False)


# 关卡 3 参考实现：
def run_sql_agent(
    llm: ChatOpenAI,
    db_path: str,
    question: str,
    max_steps: int = 6,
) -> Dict[str, Any]:
    executed_sqls: List[str] = []

    @tool
    def get_db_schema() -> str:
        \"\"\"获取当前 SQLite 数据库的所有表名、字段名、类型与外键关系。\"\"\"
        return inspect_sqlite_schema(db_path)

    class SqlQueryInput(BaseModel):
        sql: str = Field(..., description="单条只读 SELECT SQL 查询语句")

    @tool(args_schema=SqlQueryInput)
    def query_database(sql: str) -> str:
        \"\"\"执行只读 SQL 查询并返回 JSON 结果。若报错请根据 error 修正 SQL 后重试。\"\"\"
        executed_sqls.append(sql)
        return execute_safe_readonly_sql(db_path, sql)

    tools = [get_db_schema, query_database]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="你是严谨的数据库分析专家。请先调用 get_db_schema 确认表结构，再编写精确的只读 SQL 查询，最后基于真实返回结果作答。"
        ),
        HumanMessage(content=question),
    ]

    for step in range(1, max_steps + 1):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {
                "final_answer": response.content,
                "executed_sqls": executed_sqls,
                "total_steps": step,
            }
        for tc in response.tool_calls:
            name, args, cid = tc["name"], tc["args"], tc["id"]
            if name in tool_map:
                try:
                    out = tool_map[name].invoke(args)
                except Exception as e:
                    out = f"工具调用异常: {e}"
            else:
                out = f"未知工具: {name}"
            messages.append(ToolMessage(content=str(out), tool_call_id=cid))

    return {
        "final_answer": "达到最大步数限制，未能完成查询。",
        "executed_sqls": executed_sqls,
        "total_steps": max_steps,
    }
"""
# endregion
