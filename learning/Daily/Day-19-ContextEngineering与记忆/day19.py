"""
Day 19: 第 3 周进阶 - Context Engineering 与轻量会话持久化
===================================================================
本脚本为 Day 19 的闯关代码骨架（零剧透纯契约版）。
请根据各关卡的函数/类签名、I/O 契约与验收标准，独立完成架构设计与代码实现：
1. 关卡 1：大工具输出首尾保留截断（truncate_tool_output）
2. 关卡 2：协议安全的上下文窗口裁剪与摘要（manage_context_window）
3. 关卡 3：基于 SQLite 的轻量会话状态持久化（SessionStore）

【闯关打靶规则】：
- 骨架仅提供业务目标与 I/O 契约，不剧透具体实现步骤；
- 实现完成后删除对应的 raise NotImplementedError 行，运行脚本逐关验收；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import os
import sys
import json
import time
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

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
# 0. 辅助工具与测试 Fixture（开箱即用）
# ==========================================


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """
    轻量 Token 估算器（无需联网下载 tokenizer）：
    将消息序列化后按 ~2.5 字符 / token 估算，并附加固定消息头开销。
    """
    serialized = json.dumps(msg, ensure_ascii=False)
    return max(1, int(len(serialized) / 2.5)) + 6


def estimate_total_tokens(messages: List[Dict[str, Any]]) -> int:
    """计算消息列表的总估算 Token 数"""
    return sum(estimate_message_tokens(m) for m in messages)


def build_huge_isp_register_dump() -> str:
    """构造一份超长模拟 ISP 寄存器与统计日志（约 5000+ 字符）"""
    header = "[ISP_DUMP_START] chip=SA8155 module=FGF frame_id=1024\n"
    body_lines = [
        f"REG_0x{idx:04X}: val=0x{(idx * 17) & 0xFFFF:04X} status=OK detail=normal_gradient_pass"
        for idx in range(1, 75)
    ]
    footer = "\n[ISP_DUMP_SUMMARY] error_count=2 max_halo_score=0.87 recommended_action=check_guide_radius"
    return header + "\n".join(body_lines) + footer


def mock_history_summarizer(evicted_messages: List[Dict[str, Any]]) -> str:
    """确定性本地摘要器（用于离线稳定验收；生产中可替换为 LLM 调用）"""
    roles = [m.get("role", "unknown") for m in evicted_messages]
    return f"共压缩 {len(evicted_messages)} 条早期消息（角色序列: {','.join(roles)}），核心背景：用户反馈 SA8155 夜景 Halo 问题。"


def validate_tool_call_pairing(messages: List[Dict[str, Any]]) -> bool:
    """校验消息序列中不存在孤立的 role='tool'，且每个 tool_call_id 均有前序 assistant 认领"""
    pending_ids = set()
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                pending_ids.add(tc["id"])
        elif role == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id not in pending_ids:
                return False
            pending_ids.discard(tc_id)
        else:
            if pending_ids:
                return False
    return len(pending_ids) == 0


# ==========================================
# 1. 关卡 1：大工具输出首尾保留截断
# ==========================================


def truncate_tool_output(
    raw_output: str,
    max_chars: int = 800,
    head_ratio: float = 0.7,
) -> str:
    """
    【关卡 1 契约】：对超长工具输出执行“首尾保留（Head + Tail）”安全截断，防止撑爆上下文窗口。

    输入契约：
    - raw_output: 工具原始输出字符串。
    - max_chars: 允许的最大总字符预算（必须 > 0）。
    - head_ratio: 头部保留配额占比（0.0 < head_ratio < 1.0）。

    输出契约：
    - 若 len(raw_output) <= max_chars，原样返回 raw_output；
    - 若超出 max_chars，必须在保留头部与尾部关键内容的前提下，在中间插入包含省略字符数统计的提示标记，
      且最终返回字符串长度必须 <= max_chars（当 max_chars 极小时也应安全不越界）。
    """
    # TODO 1: 实现首尾保留截断策略
    raise NotImplementedError(
        "【关卡 1 待实现】：请实现 truncate_tool_output！完成后删除本行。"
    )


# ==========================================
# 2. 关卡 2：协议安全的上下文窗口裁剪与摘要
# ==========================================


def manage_context_window(
    messages: List[Dict[str, Any]],
    token_budget: int,
    summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> List[Dict[str, Any]]:
    """
    【关卡 2 契约】：在给定 token_budget 内裁剪并可选摘要早期对话历史，严格保障 OpenAI 协议合法性。

    输入契约：
    - messages: 完整对话历史列表（可能包含 system、user、assistant(tool_calls)、tool）。
    - token_budget: 允许的最大总 Token 预算（可结合 estimate_total_tokens / estimate_message_tokens 计算）。
    - summarizer: 可选摘要回调函数，接收被淘汰的早期消息列表，返回摘要文本。

    行为与输出契约：
    - 必须保留原始首条 system 消息（若存在）；
    - 从最新消息往前在预算内尽可能保留近期交互，严禁拆散“带 tool_calls 的 assistant 消息”与其紧随的“role='tool'”消息组；
    - 若有早期消息被淘汰且提供了 summarizer，需将淘汰消息压缩为一条摘要 system 消息（插在主 system 之后），
      且最终消息列表应尽量贴合 token_budget 约束、顺序保持时间先后、且不得产生孤立 tool 消息。
    """
    # TODO 2: 实现协议安全的上下文裁剪与摘要
    raise NotImplementedError(
        "【关卡 2 待实现】：请实现 manage_context_window！完成后删除本行。"
    )


# ==========================================
# 3. 关卡 3：基于 SQLite 的轻量会话持久化
# ==========================================


class SessionStore:
    """
    【关卡 3 契约】：基于标准库 sqlite3 的轻量级会话状态持久化仓库。

    要求：
    - 构造函数 __init__(self, db_path: str)：确保父目录存在，初始化会话表（不存在则创建）。
    - save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> None：
      幂等保存/覆盖指定 session_id 的完整 messages 列表（JSON 序列化存储）及更新时间。
    - load_session(self, session_id: str) -> List[Dict[str, Any]]：
      读取并反序列化指定 session_id 的 messages 列表；若该 session_id 不存在，返回空列表 []。
    """

    def __init__(self, db_path: str) -> None:
        # TODO 3a: 初始化 SQLite 连接与表结构
        raise NotImplementedError(
            "【关卡 3 待实现】：请实现 SessionStore.__init__！完成后删除本行。"
        )

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # TODO 3b: 幂等写入/更新会话记录
        raise NotImplementedError(
            "【关卡 3 待实现】：请实现 SessionStore.save_session！完成后删除本行。"
        )

    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        # TODO 3c: 读取会话记录（不存在返回 []）
        raise NotImplementedError(
            "【关卡 3 待实现】：请实现 SessionStore.load_session！完成后删除本行。"
        )


# ==========================================
# 4. 验收测试主循环（开箱即用）
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("【Day 19 实战】：Context Engineering 与轻量会话持久化")
    print("=" * 80)

    # ------------------------------------------
    # 测试 1：大工具输出首尾保留截断
    # ------------------------------------------
    print("\n🔍 【测试 1：大工具输出首尾保留截断】")
    huge_dump = build_huge_isp_register_dump()
    assert len(huge_dump) > 4000
    short_ok = truncate_tool_output("short_ok", max_chars=800)
    assert short_ok == "short_ok", "未超限文本应原样返回"

    truncated = truncate_tool_output(huge_dump, max_chars=600, head_ratio=0.7)
    assert len(truncated) <= 600, f"截断后长度超标: {len(truncated)} > 600"
    assert "[ISP_DUMP_START]" in truncated, "截断后丢失头部关键标志"
    assert "[ISP_DUMP_SUMMARY]" in truncated, "截断后丢失尾部关键汇总"
    assert "截断" in truncated or "omitted" in truncated.lower(), "缺少截断提示信息"
    print(f"✅ 关卡 1 通过：原始 {len(huge_dump)} 字符 -> 截断后 {len(truncated)} 字符（首尾完整保留）")

    # ------------------------------------------
    # 测试 2：协议安全的上下文窗口裁剪与摘要
    # ------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 2：协议安全的上下文窗口裁剪与摘要】")
    sample_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是严谨的 ISP 算法诊断助手。"},
        {"role": "user", "content": "SA8155 昨晚夜景人像出现严重 Halo 光晕，帮我初步分析可能原因。"},
        {"role": "assistant", "content": "Halo 通常与 FGF 引导滤波半径过大或边缘权重过高有关，需要查看寄存器配置。"},
        {"role": "user", "content": "请帮我调用工具检查 FGF 的 guide_radius 基线和最新报错日志。"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {"name": "lookup_isp_param_baseline", "arguments": '{"module_name": "FGF", "param_name": "guide_radius"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_001", "content": '{"safe_range": [2.0, 5.0], "recommended": 4.0}'},
        {"role": "assistant", "content": "基线核查完成，建议将 guide_radius 回调至 4.0 px。"},
    ]
    full_tokens = estimate_total_tokens(sample_messages)
    tight_budget = int(full_tokens * 0.72)
    managed = manage_context_window(
        messages=sample_messages,
        token_budget=tight_budget,
        summarizer=mock_history_summarizer,
    )
    assert managed[0]["role"] == "system" and "ISP" in managed[0]["content"], "主 system 消息必须保留在首位"
    assert validate_tool_call_pairing(managed), "裁剪后破坏了 tool_calls 与 tool 的成对协议！"
    assert any("历史" in m.get("content", "") or "压缩" in m.get("content", "") for m in managed), (
        "淘汰早期消息后未注入摘要消息"
    )
    assert len(managed) < len(sample_messages) + 1, "未实际淘汰超预算早期消息"
    print(
        f"✅ 关卡 2 通过：原始 {len(sample_messages)} 条（~{full_tokens} tokens）-> "
        f"裁剪摘要后 {len(managed)} 条（~{estimate_total_tokens(managed)} tokens，协议成对校验 100% 通过）"
    )

    # ------------------------------------------
    # 测试 3：SQLite 轻量会话状态持久化
    # ------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 3：SQLite 轻量会话状态持久化】")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = str(Path(tmpdir) / "nested" / "sessions.db")
        store1 = SessionStore(db_file)
        assert store1.load_session("non_existent") == [], "不存在的会话应返回空列表"

        store1.save_session("isp_case_42", managed)
        # 幂等覆盖更新测试
        updated_msgs = managed + [{"role": "user", "content": "收到，我先改成 4.0 复测。"}]
        store1.save_session("isp_case_42", updated_msgs)

        # 模拟进程重启：新建实例读取
        store2 = SessionStore(db_file)
        restored = store2.load_session("isp_case_42")
        assert restored == updated_msgs, "重启后恢复的会话内容与保存内容不一致"
        print(f"✅ 关卡 3 通过：SQLite 幂等保存与跨实例恢复 {len(restored)} 条消息成功")

    print("\n" + "=" * 80)
    print("🎉 [PASSED]: Day 19 关卡全通！Context Engineering 与轻量持久化闭环通关！")
    print("=" * 80)


# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# def truncate_tool_output(
#     raw_output: str,
#     max_chars: int = 800,
#     head_ratio: float = 0.7,
# ) -> str:
#     if len(raw_output) <= max_chars:
#         return raw_output
#
#     omitted = len(raw_output) - max_chars
#     marker = f"\n...[已截断约 {omitted} 字符，保留首尾关键上下文]...\n"
#     if max_chars <= len(marker):
#         return raw_output[:max_chars]
#
#     available = max_chars - len(marker)
#     head_len = int(available * head_ratio)
#     tail_len = available - head_len
#     head_part = raw_output[:head_len] if head_len > 0 else ""
#     tail_part = raw_output[-tail_len:] if tail_len > 0 else ""
#     return f"{head_part}{marker}{tail_part}"
#
#
# def manage_context_window(
#     messages: List[Dict[str, Any]],
#     token_budget: int,
#     summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
# ) -> List[Dict[str, Any]]:
#     if not messages or estimate_total_tokens(messages) <= token_budget:
#         return list(messages)
#
#     system_msgs: List[Dict[str, Any]] = []
#     rest: List[Dict[str, Any]] = list(messages)
#     if rest and rest[0].get("role") == "system":
#         system_msgs.append(rest.pop(0))
#
#     # 将剩余消息划分为不可拆分的原子组（assistant(tool_calls) + 紧随的 tool 消息）
#     groups: List[List[Dict[str, Any]]] = []
#     idx = 0
#     while idx < len(rest):
#         msg = rest[idx]
#         if msg.get("role") == "assistant" and msg.get("tool_calls"):
#             group = [msg]
#             idx += 1
#             while idx < len(rest) and rest[idx].get("role") == "tool":
#                 group.append(rest[idx])
#                 idx += 1
#             groups.append(group)
#         else:
#             groups.append([msg])
#             idx += 1
#
#     base_tokens = estimate_total_tokens(system_msgs)
#     kept_groups: List[List[Dict[str, Any]]] = []
#     used_tokens = base_tokens
#     evicted_groups: List[List[Dict[str, Any]]] = []
#
#     # 预留摘要空间（若提供 summarizer）
#     summary_reserve = 80 if summarizer else 0
#     for group in reversed(groups):
#         grp_tokens = sum(estimate_message_tokens(m) for m in group)
#         if used_tokens + grp_tokens + summary_reserve <= token_budget or not kept_groups:
#             kept_groups.append(group)
#             used_tokens += grp_tokens
#         else:
#             evicted_groups.append(group)
#
#     kept_groups.reverse()
#     evicted_groups.reverse()
#     evicted_flat = [m for grp in evicted_groups for m in grp]
#
#     result = list(system_msgs)
#     if evicted_flat and summarizer is not None:
#         summary_text = summarizer(evicted_flat)
#         result.append({"role": "system", "content": f"[历史对话摘要] {summary_text}"})
#     for grp in kept_groups:
#         result.extend(grp)
#     return result
#
#
# class SessionStore:
#     def __init__(self, db_path: str) -> None:
#         self.db_path = db_path
#         Path(db_path).parent.mkdir(parents=True, exist_ok=True)
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute(
#                 """
#                 CREATE TABLE IF NOT EXISTS chat_sessions (
#                     session_id TEXT PRIMARY KEY,
#                     messages_json TEXT NOT NULL,
#                     updated_at REAL NOT NULL
#                 )
#                 """
#             )
#             conn.commit()
#
#     def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
#         payload = json.dumps(messages, ensure_ascii=False)
#         now = time.time()
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute(
#                 """
#                 INSERT INTO chat_sessions (session_id, messages_json, updated_at)
#                 VALUES (?, ?, ?)
#                 ON CONFLICT(session_id) DO UPDATE SET
#                     messages_json = excluded.messages_json,
#                     updated_at = excluded.updated_at
#                 """,
#                 (session_id, payload, now),
#             )
#             conn.commit()
#
#     def load_session(self, session_id: str) -> List[Dict[str, Any]]:
#         with sqlite3.connect(self.db_path) as conn:
#             row = conn.execute(
#                 "SELECT messages_json FROM chat_sessions WHERE session_id = ?",
#                 (session_id,),
#             ).fetchone()
#         if row is None:
#             return []
#         return json.loads(row[0])
# endregion
