"""
Day 18: 第 3 周进阶 - OpenAI Function Calling 与结构化抽取实战
===================================================================
本脚本为 Day 18 的闯关代码骨架（零剧透纯契约版）。
请根据各关卡的函数签名、I/O 契约与验收标准，独立完成架构设计与代码实现：
1. 关卡 1：Pydantic 结构化数据抽取（extract_isp_bug_ticket）
2. 关卡 2：原生 OpenAI SDK Function Calling 闭环（run_native_openai_fc）
3. 关卡 3：结构化抽取 + 工具分诊端到端流水线（run_isp_triage_pipeline）

【闯关打靶规则】：
- 骨架仅提供业务目标与 I/O 契约，不剧透具体实现步骤；
- 实现完成后删除对应的 raise NotImplementedError 行，运行脚本逐关验收；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Literal, Callable

from langchain.messages import HumanMessage
from openai import OpenAI
from pydantic import BaseModel, Field
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
# 0. 环境初始化、数据契约与内置诊断工具（开箱即用）
# ==========================================


def get_llm_config() -> Dict[str, str]:
    """返回统一的本地模型端点配置"""
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:3425/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
        "model": os.getenv("MODEL_NAME", "auto/hy4flash-free"),
    }


def get_langchain_llm() -> ChatOpenAI:
    cfg = get_llm_config()
    llm = ChatOpenAI(
        model=cfg["model"],
        openai_api_key=cfg["api_key"],
        openai_api_base=cfg["base_url"],
        temperature=0.0,
        max_tokens=2048,
        timeout=60.0,
    )
    return llm


def get_native_openai_client() -> OpenAI:
    cfg = get_llm_config()
    return OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=60.0,
    )


class ISPBugTicket(BaseModel):
    """结构化 ISP 画质缺陷工单契约"""

    chip_platform: str = Field(description="芯片平台名称，如 SA8155 / Snapdragon8Gen3")
    module_name: str = Field(description="涉事 ISP 模块名，如 FGF / MFNR / TMC")
    defect_type: str = Field(
        description="画质缺陷类型，如 Halo / Ghosting / ColorCast / Noise"
    )
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="严重程度等级")
    suspected_params: List[str] = Field(
        default_factory=list, description="文本中提及的可疑参数名列表"
    )
    summary: str = Field(description="一句话概括核心故障现象")


def lookup_isp_param_baseline(module_name: str, param_name: str) -> str:
    """内置 ISP 参数基线查询函数（供 Function Calling 调度使用）"""
    baselines = {
        ("FGF", "guide_radius"): {
            "safe_range": [2.0, 5.0],
            "recommended": 4.0,
            "unit": "px",
        },
        ("FGF", "smooth_eps"): {
            "safe_range": [0.005, 0.03],
            "recommended": 0.015,
            "unit": "norm",
        },
        ("MFNR", "temporal_weight"): {
            "safe_range": [0.3, 0.65],
            "recommended": 0.5,
            "unit": "ratio",
        },
        ("TMC", "highlight_compress"): {
            "safe_range": [0.2, 0.5],
            "recommended": 0.35,
            "unit": "ratio",
        },
    }
    key = (module_name.upper(), param_name)
    if key in baselines:
        return json.dumps(
            {
                "found": True,
                "module": module_name,
                "param": param_name,
                **baselines[key],
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {"found": False, "message": f"未收录 {module_name}.{param_name} 的基线配置"},
        ensure_ascii=False,
    )


def estimate_fix_priority(severity: str, defect_type: str) -> str:
    """内置工单排期优先级评估函数（供 Function Calling 调度使用）"""
    sev = severity.upper()
    if sev == "HIGH":
        sla = "P0-24小时内热修复"
    elif sev == "MEDIUM":
        sla = "P1-本周迭代修复"
    else:
        sla = "P2-下个版本排期"
    return json.dumps(
        {"severity": sev, "defect_type": defect_type, "sla": sla}, ensure_ascii=False
    )


# ==========================================
# 1. 关卡 1：Pydantic 结构化数据抽取
# ==========================================


def extract_isp_bug_ticket(llm: ChatOpenAI, raw_report: str) -> ISPBugTicket:
    """
    【关卡 1 契约】：从非结构化自然语言故障报告中抽取强类型 ISPBugTicket 对象。

    输入契约：
    - llm: 已初始化的 ChatOpenAI 实例。
    - raw_report: 工程师提交的非结构化故障描述文本。

    输出契约：
    - 返回校验通过的 ISPBugTicket 实例；
    - 必须准确提取 chip_platform、module_name、defect_type、severity（限定 HIGH/MEDIUM/LOW）、
      suspected_params（列表）与 summary。
    - 严禁依赖手工硬编码正则匹配特定测试文本。
    """
    # TODO 1: 实现结构化抽取
    structured_llm = llm.with_structured_output(ISPBugTicket, method="function_calling")
    return structured_llm.invoke(
        [
            HumanMessage(
                f"请从以下 ISP 画质缺陷报告中提取结构化字段，severity 只能取 HIGH / MEDIUM / LOW。\n\n报告内容：{raw_report}"
            )
        ]
    )


# ==========================================
# 2. 关卡 2：原生 OpenAI SDK Function Calling 闭环
# ==========================================


def run_native_openai_fc(
    client: OpenAI,
    model: str,
    user_query: str,
    tools_schema: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., str]],
    max_steps: int = 5,
) -> Dict[str, Any]:
    """
    【关卡 2 契约】：脱离 LangChain 封装，使用原生 OpenAI SDK 完成 Function Calling 工具调度闭环。

    输入契约：
    - client: 原生 openai.OpenAI 客户端实例。
    - model: 模型名称。
    - user_query: 用户自然语言提问。
    - tools_schema: 符合 OpenAI Chat Completions 规范的 tools JSON Schema 列表。
    - tool_registry: 工具名 -> 本地 Python 可调用函数的映射字典。
    - max_steps: 最大模型交互轮数上限。

    行为契约：
    - 使用原生 client.chat.completions.create(...) 携带 tools 发起请求；
    - 若响应包含 tool_calls，需解析工具名与 JSON 参数、执行对应本地函数，
      并将工具结果以原生协议消息格式回传给下一轮对话；
    - 直到模型不再调用工具并给出最终回复，或达到 max_steps 上限。

    输出契约（返回字典，必须包含以下 3 个键）：
    - "final_answer": str（模型最终回复文本）
    - "called_tools": List[str]（按实际调用顺序记录的工具名列表）
    - "total_steps": int（实际经历的 API 请求轮数）
    """
    # TODO 2: 实现原生 OpenAI Function Calling 闭环

    final_answer = f"Not get the final answer within {max_steps} steps"
    called_tools = []
    messages = [
        {
            "role": "system",
            "content": "你是严谨的 ISP 算法诊断助手，请按需调用工具获取真实参数基线与排期后再回答，不要凭空猜测",
        },
        {"role": "user", "content": user_query},
    ]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

        msg = resp.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return {
                "final_answer": msg.content,
                "called_tools": called_tools,
                "total_steps": step + 1,
            }
        else:
            _tool_calls = []
            for tool_call in msg.tool_calls:
                _tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": _tool_calls,
                }
            )
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                if tool_name in tool_registry:
                    tool = tool_registry[tool_name]
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        tool_res = tool(**arguments)
                        called_tools.append(tool_name)
                    except Exception as e:
                        tool_res = f"Tool Call Failed. Error: {str(e)}"
                else:
                    tool_res = f"Tool: {tool_name} Not Found in tool_registry: {tool_registry.keys()}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_res,
                    }
                )
    return {
        "final_answer": final_answer,
        "called_tools": called_tools,
        "total_steps": max_steps,
    }


# ==========================================
# 3. 关卡 3：结构化抽取 + 工具分诊端到端流水线
# ==========================================


def run_isp_triage_pipeline(
    llm: ChatOpenAI,
    client: OpenAI,
    model: str,
    raw_report: str,
) -> Dict[str, Any]:
    """
    【关卡 3 契约】：串联“结构化抽取”与“原生 Function Calling 工具诊断”，完成端到端 ISP 故障自动分诊。

    输入契约：
    - llm: ChatOpenAI 实例（用于结构化抽取阶段）。
    - client: 原生 OpenAI 实例（用于工具诊断阶段）。
    - model: 模型名称。
    - raw_report: 原始非结构化故障报告。

    行为契约：
    - 先调用关卡 1 将 raw_report 抽取为 ISPBugTicket；
    - 再基于抽取出的结构化字段，构造 OpenAI tools（接入 lookup_isp_param_baseline 与 estimate_fix_priority），
      调用关卡 2 完成参数基线核查与 SLA 优先级评估，输出最终分诊结论。

    输出契约（返回字典，必须包含以下 3 个键）：
    - "ticket": ISPBugTicket（抽取出的结构化故障工单）
    - "triage_report": str（结合工具核查结果的最终分诊建议）
    - "called_tools": List[str]（分诊阶段实际调用过的工具名列表）
    """
    # TODO 3: 实现端到端分诊流水线

    isp_bug_ticket: ISPBugTicket = extract_isp_bug_ticket(llm, raw_report)

    fc_tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "lookup_isp_param_baseline",
                "description": "查询指定 ISP 模块参数的安全基线范围与推荐默认值",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_name": {
                            "type": "string",
                            "description": "ISP 模块名，如 FGF / MFNR / TMC",
                        },
                        "param_name": {
                            "type": "string",
                            "description": "参数名称，如 guide_radius",
                        },
                    },
                    "required": ["module_name", "param_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_fix_priority",
                "description": "根据严重度与缺陷类型评估 SLA 修复排期优先级",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                        },
                        "defect_type": {"type": "string", "description": "缺陷类型"},
                    },
                    "required": ["severity", "defect_type"],
                },
            },
        },
    ]
    fc_registry = {
        "lookup_isp_param_baseline": lookup_isp_param_baseline,
        "estimate_fix_priority": estimate_fix_priority,
    }

    query = (
        f"请对以下已结构化的 ISP 缺陷工单进行分诊核查：\n"
        f"{isp_bug_ticket.model_dump_json(ensure_ascii=False)}\n"
        "分诊要求：1) 若工单包含可疑参数，核查其基线安全范围；"
        "2) 评估修复 SLA 优先级；3) 给出综合分诊建议。"
    )

    res = run_native_openai_fc(client, model, query, fc_tools_schema, fc_registry)

    return {
        "ticket": isp_bug_ticket,
        "triage_report": res["final_answer"],
        "called_tools": res["called_tools"],
    }


# ==========================================
# 4. 验收测试主循环（开箱即用）
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("【Day 18 实战】：OpenAI Function Calling 与结构化抽取实战")
    print("=" * 80)

    cfg = get_llm_config()
    llm = get_langchain_llm()
    client = get_native_openai_client()
    print(f"[+] 模型端点就绪: {cfg['base_url']} | 模型: {cfg['model']}\n")

    # ------------------------------------------
    # 测试 1：Pydantic 结构化抽取
    # ------------------------------------------
    print("🔍 【测试 1：Pydantic 结构化数据抽取】")
    sample_report_1 = (
        "昨晚路测反馈，SA8155 平台的 FGF 模块在夜景高反差人像场景下出现明显边缘 Halo 光晕，"
        "严重影响主观观感（紧急程度 HIGH）。排查发现当前配置把 guide_radius 拉到了 8.0，请尽快定位！"
    )
    ticket1 = extract_isp_bug_ticket(llm, sample_report_1)
    assert isinstance(ticket1, ISPBugTicket), "返回值必须为 ISPBugTicket 实例"
    assert (
        "SA8155" in ticket1.chip_platform
    ), f"芯片平台抽取错误: {ticket1.chip_platform}"
    assert (
        "FGF" in ticket1.module_name.upper()
    ), f"模块名抽取错误: {ticket1.module_name}"
    assert ticket1.severity == "HIGH", f"严重度抽取错误: {ticket1.severity}"
    assert (
        "guide_radius" in ticket1.suspected_params
    ), f"可疑参数漏抽: {ticket1.suspected_params}"
    print(f"✅ 关卡 1 通过：抽取结果 -> {ticket1.model_dump_json(indent=2)}\n")

    # ------------------------------------------
    # 测试 2：原生 OpenAI SDK Function Calling
    # ------------------------------------------
    print("=" * 80)
    print("🔍 【测试 2：原生 OpenAI SDK Function Calling 闭环】")
    fc_tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "lookup_isp_param_baseline",
                "description": "查询指定 ISP 模块参数的安全基线范围与推荐默认值",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_name": {
                            "type": "string",
                            "description": "ISP 模块名，如 FGF / MFNR / TMC",
                        },
                        "param_name": {
                            "type": "string",
                            "description": "参数名称，如 guide_radius",
                        },
                    },
                    "required": ["module_name", "param_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_fix_priority",
                "description": "根据严重度与缺陷类型评估 SLA 修复排期优先级",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                        },
                        "defect_type": {"type": "string", "description": "缺陷类型"},
                    },
                    "required": ["severity", "defect_type"],
                },
            },
        },
    ]
    fc_registry = {
        "lookup_isp_param_baseline": lookup_isp_param_baseline,
        "estimate_fix_priority": estimate_fix_priority,
    }
    fc_res = run_native_openai_fc(
        client=client,
        model=cfg["model"],
        user_query="请帮我查一下 FGF 模块 guide_radius 参数的安全基线范围和推荐值是多少？",
        tools_schema=fc_tools_schema,
        tool_registry=fc_registry,
    )
    assert (
        "lookup_isp_param_baseline" in fc_res["called_tools"]
    ), f"未调用基线查询工具: {fc_res}"
    assert (
        "4.0" in fc_res["final_answer"] or "4" in fc_res["final_answer"]
    ), f"回答未包含查询到的推荐值: {fc_res['final_answer']}"
    print(
        f"✅ 关卡 2 通过：原生 FC 共 {fc_res['total_steps']} 轮，调用工具 {fc_res['called_tools']}"
    )
    print(f"   回复摘要: {fc_res['final_answer'][:150]}...\n")

    # ------------------------------------------
    # 测试 3：端到端分诊流水线
    # ------------------------------------------
    print("=" * 80)
    print("🔍 【测试 3：结构化抽取 + 工具分诊端到端流水线】")
    t0 = time.perf_counter()
    sample_report_2 = (
        "Snapdragon8Gen3 平台 MFNR 模块在暗光运动抓拍下出现中等程度 Ghosting 鬼影（MEDIUM），"
        "怀疑是 temporal_weight 设得过高导致拖影，请核查参数基线并给出修复 SLA 排期。"
    )
    triage = run_isp_triage_pipeline(llm, client, cfg["model"], sample_report_2)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert isinstance(triage["ticket"], ISPBugTicket)
    assert "lookup_isp_param_baseline" in triage["called_tools"], "分诊未核查参数基线"
    assert "estimate_fix_priority" in triage["called_tools"], "分诊未评估修复排期"
    assert (
        "P1" in triage["triage_report"] or "0.5" in triage["triage_report"]
    ), f"分诊报告遗漏关键结论: {triage['triage_report']}"
    print(
        f"✅ 关卡 3 通过：耗时 {elapsed_ms:.2f} ms | 调用工具: {triage['called_tools']}"
    )
    print(f"   分诊结论:\n{triage['triage_report']}\n")
    print("=" * 80)
    print("🎉 [PASSED]: Day 18 关卡全通！Function Calling 与结构化抽取闭环通关！")
    print("=" * 80)


# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# def extract_isp_bug_ticket(llm: ChatOpenAI, raw_report: str) -> ISPBugTicket:
#     structured_llm = llm.with_structured_output(ISPBugTicket)
#     prompt = (
#         "请从以下 ISP 画质缺陷报告中提取结构化字段，"
#         "severity 只能取 HIGH / MEDIUM / LOW。\n\n"
#         f"报告内容：{raw_report}"
#     )
#     return structured_llm.invoke(prompt)
#
#
# def run_native_openai_fc(
#     client: OpenAI,
#     model: str,
#     user_query: str,
#     tools_schema: List[Dict[str, Any]],
#     tool_registry: Dict[str, Callable[..., str]],
#     max_steps: int = 5,
# ) -> Dict[str, Any]:
#     messages: List[Dict[str, Any]] = [
#         {"role": "system", "content": "你是严谨的 ISP 算法诊断助手，请按需调用工具获取真实参数后再回答。"},
#         {"role": "user", "content": user_query},
#     ]
#     called_tools: List[str] = []
#     final_answer = f"未能在 {max_steps} 轮内完成响应"
#     step = 0
#
#     for step in range(1, max_steps + 1):
#         resp = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             tools=tools_schema,
#             tool_choice="auto",
#             temperature=0.0,
#         )
#         msg = resp.choices[0].message
#         assistant_dict: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
#         if msg.tool_calls:
#             assistant_dict["tool_calls"] = [
#                 {
#                     "id": tc.id,
#                     "type": "function",
#                     "function": {"name": tc.function.name, "arguments": tc.function.arguments},
#                 }
#                 for tc in msg.tool_calls
#             ]
#         messages.append(assistant_dict)
#
#         if not msg.tool_calls:
#             final_answer = msg.content or ""
#             break
#
#         for tc in msg.tool_calls:
#             fn_name = tc.function.name
#             called_tools.append(fn_name)
#             try:
#                 fn_args = json.loads(tc.function.arguments or "{}")
#                 result = tool_registry[fn_name](**fn_args) if fn_name in tool_registry else f"未知工具: {fn_name}"
#             except Exception as e:
#                 result = json.dumps({"error": str(e)}, ensure_ascii=False)
#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tc.id,
#                 "content": result,
#             })
#
#     return {"final_answer": final_answer, "called_tools": called_tools, "total_steps": step}
#
#
# def run_isp_triage_pipeline(
#     llm: ChatOpenAI,
#     client: OpenAI,
#     model: str,
#     raw_report: str,
# ) -> Dict[str, Any]:
#     ticket = extract_isp_bug_ticket(llm, raw_report)
#     tools_schema = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "lookup_isp_param_baseline",
#                 "description": "查询指定 ISP 模块参数的安全基线范围与推荐默认值",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "module_name": {"type": "string"},
#                         "param_name": {"type": "string"},
#                     },
#                     "required": ["module_name", "param_name"],
#                 },
#             },
#         },
#         {
#             "type": "function",
#             "function": {
#                 "name": "estimate_fix_priority",
#                 "description": "根据严重度与缺陷类型评估 SLA 修复排期优先级",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
#                         "defect_type": {"type": "string"},
#                     },
#                     "required": ["severity", "defect_type"],
#                 },
#             },
#         },
#     ]
#     registry = {
#         "lookup_isp_param_baseline": lookup_isp_param_baseline,
#         "estimate_fix_priority": estimate_fix_priority,
#     }
#     query = (
#         f"请对以下已结构化的 ISP 缺陷工单进行分诊核查：\n"
#         f"{ticket.model_dump_json(ensure_ascii=False)}\n"
#         f"要求：1) 核查其中所有可疑参数的基线安全范围；2) 评估修复 SLA 优先级；3) 给出综合分诊建议。"
#     )
#     fc_res = run_native_openai_fc(client, model, query, tools_schema, registry)
#     return {
#         "ticket": ticket,
#         "triage_report": fc_res["final_answer"],
#         "called_tools": fc_res["called_tools"],
#     }
# endregion
