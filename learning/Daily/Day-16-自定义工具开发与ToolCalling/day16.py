"""
Day 16: 第 3 周进阶 - 自定义工具开发与 Tool Calling 原生协议实战
=====================================================================
本脚本为 Day 16 的工具工程化与原生 Tool Calling 闭环闯关代码骨架。
请根据契约定义与关卡提示，手撕三大核心模块：
1. 关卡 1：基于 `@tool` 装饰器与 Pydantic v2 构建轻量级强类型计算摄影工具；
2. 关卡 2：继承 `BaseTool` 深度定制工业级结构化工具（支持异常自愈与自定义错误提示）；
3. 关卡 3：利用 `llm.bind_tools` 绑定工具，手撕原生 Tool Calling 通信闭环（支持 tool_calls 分发与 ToolMessage 状态流转）。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import sys
import os
import json
import math
import time
from typing import List, Dict, Any, Optional, Type, Union
from pydantic import BaseModel, Field, ValidationError

from langchain_core.tools import tool, BaseTool, ToolException
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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
# 0. 环境与大模型客户端初始化
# ==========================================


def get_agent_llm() -> ChatOpenAI:
    """初始化并返回兼容 OpenAI 接口的大模型客户端"""
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:3425/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model_name = os.getenv("MODEL_NAME", "auto/hy4flash-free")

    try:
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
    except Exception as e:
        print(f"[-] 初始化 LLM 失败: {e}")
        raise e


# ==========================================
# 1. 关卡 1：基于 @tool 装饰器的轻量级工具
# ==========================================


class HaloOvershootInput(BaseModel):
    """高反差导向滤波 (FGF) 边缘光晕过冲计算输入契约"""

    edge_contrast: float = Field(
        ..., gt=0.0, description="边缘内外法向对比度阶跃比 (C > 0)"
    )
    sdf_overflow: float = Field(
        ..., ge=0.0, description="法向有向距离场 (SDF) 溢出距离百分比 (%)"
    )


# TODO 1: 请使用 @tool 装饰器封装 calc_halo_overshoot 工具函数
# 要求：
# 1. 使用 args_schema=HaloOvershootInput 绑定强类型入参校验；
# 2. 具备详尽的 Docstring 供 LLM 理解工具使用场景与功能；
# 3. 计算公式：halo_risk_index = edge_contrast * (sdf_overflow / 10.0)；
# 4. 返回标准 JSON 字符串（包含 halo_risk_index, is_severe, advice 等字段）。


@tool(args_schema=HaloOvershootInput)
def calc_halo_overshoot(edge_contrast: float, sdf_overflow: float) -> str:
    """
    计算导向滤波 (Fast Guided Filter, FGF) 在高反差边缘引起的光晕 (Halo) 过冲风险指标。
    """
    halo_risk_index = edge_contrast * (sdf_overflow / 10.0)
    if halo_risk_index >= 2.5:
        is_severe = True
        advice = "存在严重光晕过冲风险，建议减小导向滤波半径 r、增大平滑因子 eps 或启用梯度自适应保边引导"
    else:
        is_severe = False
        advice = "光晕在可接受容差范围内，保持当前滤波参数即可"

    res = {
        "halo_risk_index": halo_risk_index,
        "is_severe": is_severe,
        "advice": advice,
    }

    return json.dumps(res, ensure_ascii=False)


# ==========================================
# 2. 关卡 2：继承 BaseTool 的工业级结构化工具
# ==========================================


class ExposureGainInput(BaseModel):
    """ISP 多帧降噪曝光增益评估输入参数"""

    lux_index: float = Field(..., ge=0.0, description="环境照度指数 Lux Index")
    iso_value: float = Field(..., ge=50.0, description="传感器感光度 ISO (>= 50)")
    exposure_time_ms: float = Field(
        ..., gt=0.0, description="单帧曝光时间 (毫秒，必须 > 0)"
    )


def custom_exposure_error_handler(error: Exception) -> str:
    """工具异常拦截与自愈提示函数"""
    return f"【工具入参异常拦截】：检测到非法曝光参数输入 ({str(error)})！建议检查 ISO >= 50 以及曝光时间 > 0 后重新调用。"


# TODO 2: 请继承 BaseTool 类，定制 ISPExposureGainTool
# 要求：
# 1. 设置 name, description, args_schema = ExposureGainInput；
# 2. 设置 handle_tool_error = custom_exposure_error_handler 或开启容错机制；
# 3. 实现同步执行核心方法：def _run(self, lux_index: float, iso_value: float, exposure_time_ms: float) -> str；
# 4. 业务逻辑：
#    - total_gain = (iso_value / 100.0) * (exposure_time_ms / 10.0)
#    - 若 total_gain > 80.0，判定建议启用多帧时空融合超级夜景 (MFNR Super Night)；
#    - 否则建议常规单帧降噪与适度锐化；
#    - 返回 JSON 字符串。
class ISPExposureGainTool(BaseTool):
    """
    【关卡 2】：ISP 曝光增益与多帧时空融合降噪决策工具
    """

    name: str = "isp_exposure_gain_eval"
    description: str = (
        "评估相机 ISP 当前画面的总曝光增益 Total Gain，并基于环境照度推荐降噪策略。"
    )
    args_schema: Type[BaseModel] = ExposureGainInput

    def _run(self, lux_index: float, iso_value: float, exposure_time_ms: float) -> str:
        try:
            total_gain = (iso_value / 100.0) * (exposure_time_ms / 10.0)
            if total_gain > 80.0:
                strategy = "Open MFNR Super Night"
            else:
                strategy = "Retain Current Single-Frame Denoise and Sharpness"
            return json.dumps(
                {
                    "total_gain": total_gain,
                    "strategy": strategy,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return custom_exposure_error_handler(e)


# ==========================================
# 3. 关卡 3：原生 Tool Calling 驱动循环引擎
# ==========================================


def run_native_tool_agent(
    llm: ChatOpenAI,
    tools: List[Any],
    query: str,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """
    【关卡 3】：基于原生 Tool Calling 的智能体执行引擎
    1. 使用 llm.bind_tools(tools) 获得具备工具调用能力的大模型实例；
    2. 维护对话消息列表 messages = [SystemMessage(...), HumanMessage(query)]；
    3. 维护工具映射字典 tool_map = {t.name: t for t in tools}；
    4. 进入 while/for 循环驱动多步推理：
       a. 调用 llm_with_tools.invoke(messages)；
       b. 将模型的回复追加至 messages；
       c. 检查回复中的 tool_calls（response.tool_calls）：
          - 若 tool_calls 为空：说明模型已完成推导，直接以 response.content 作为 final_answer 退出；
          - 若包含 tool_calls：
            遍历每一个 tool_call：
              i. 提取 call_id = tool_call["id"], tool_name = tool_call["name"], tool_args = tool_call["args"]；
              ii. 从 tool_map 中调度对应工具执行：tool.invoke(tool_args)；
              iii. 包装为 ToolMessage(content=str(result), tool_call_id=call_id)，追加回 messages；
       d. 进入下一轮调用，直到得出最终答案或超出最大步数限制。
    5. 返回包含 final_answer, tool_call_history, total_steps 的字典。
    """
    # TODO 3: 请实现原生 Tool Calling 分发与回传循环
    try:
        llm_with_tools = llm.bind_tools(tools)
        messages = [
            SystemMessage("你是一名专业的相机画质和ISP专家，为用户解答问题。"),
            HumanMessage(query),
        ]
        tool_map = {t.name: t for t in tools}

        tool_call_history = []
        final_answer = f"Not get the final answer yet within {max_steps} steps"
        for step in range(max_steps):
            response = llm_with_tools.invoke(messages)  # Return AIMessage
            messages.append(response)

            if not response.tool_calls:
                final_answer = response.content
                break
            else:
                for tool_call in response.tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name in tool_map:
                        try:
                            tool_output = tool_map[tool_name].invoke(tool_args)
                        except Exception as e:
                            tool_output = f"Tool execution error: {e}"
                    else:
                        tool_output = f"Not found tool: {tool_name} in tools list"
                    messages.append(
                        ToolMessage(
                            content=tool_output,
                            tool_call_id=tool_id,
                        )
                    )
                    tool_call_history.append(
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "args": tool_args,
                            "output": tool_output,
                        }
                    )
        return {
            "final_answer": final_answer,
            "tool_call_history": tool_call_history,
            "total_steps": step + 1,
        }

    except Exception as e:
        raise


# ==========================================
# 4. 领域实战与 Milestone 验收用例
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("【Day 16 实战】：LangChain 自定义工具开发与 Tool Calling 原生协议实战")
    print("=" * 80)

    # ----------------------------------------------------
    # 测试关卡 1：@tool 装饰器工具测试
    # ----------------------------------------------------
    print("\n🔍 【测试 1：@tool 装饰器轻量级工具单元测试】")

    assert hasattr(calc_halo_overshoot, "args_schema"), "未绑定 args_schema"
    assert (
        calc_halo_overshoot.name == "calc_halo_overshoot"
    ), f"工具名称不匹配: {calc_halo_overshoot.name}"

    # 1.1 正常参数调用
    res1_str = calc_halo_overshoot.invoke({"edge_contrast": 2.5, "sdf_overflow": 12.0})
    res1 = json.loads(res1_str)
    assert res1["halo_risk_index"] == 3.0, f"计算结果偏差: {res1}"
    assert res1["is_severe"] is True, "未正确判定高危光晕"
    print(f"✅ 关卡 1.1 通过：正常参数执行成功，光晕风险指数={res1['halo_risk_index']}")

    # 1.2 非法参数拦截（负对比度）
    try:
        calc_halo_overshoot.invoke({"edge_contrast": -1.0, "sdf_overflow": 12.0})
        print("❌ 关卡 1.2 失败：未能拦截非法入参！")
        assert False
    except ValidationError:
        print("✅ 关卡 1.2 通过：Pydantic 成功拦截非法负数对比度入参！")

    # ----------------------------------------------------
    # 测试关卡 2：继承 BaseTool 的结构化工具测试
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 2：继承 BaseTool 的工业级结构化工具测试】")

    exp_tool = ISPExposureGainTool()
    assert isinstance(exp_tool, BaseTool), "未成功继承 BaseTool"

    # 2.1 正常高增益夜景场景测试
    res2_str = exp_tool.invoke(
        {"lux_index": 380.0, "iso_value": 1600.0, "exposure_time_ms": 60.0}
    )
    res2 = json.loads(res2_str)
    # total_gain = (1600/100) * (60/10) = 16 * 6 = 96.0 (> 80.0)
    assert res2["total_gain"] == 96.0, f"Total Gain 计算有误: {res2['total_gain']}"
    assert "Super Night" in res2["strategy"], "未命中超级夜景降噪策略"
    print(
        f"✅ 关卡 2.1 通过：高感光场景计算精确，Total Gain={res2['total_gain']}，推荐策略={res2['strategy']}"
    )

    # 2.2 异常拦截与自愈测试（ISO < 50）
    try:
        err_res = exp_tool.invoke(
            {"lux_index": 100.0, "iso_value": 10.0, "exposure_time_ms": 10.0}
        )
        # 如果配置了 handle_tool_error，invoke 会返回友好错误字符串
        assert "异常拦截" in err_res or "建议检查" in err_res
        print("✅ 关卡 2.2 通过：工具成功触发自定义 handle_tool_error 拦截防御！")
    except ValidationError:
        print("✅ 关卡 2.2 通过：Pydantic 严格模式抛出 ValidationError 拦截！")

    # ----------------------------------------------------
    # 测试关卡 3：端到端原生 Tool Calling 智能体推理实测
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 3：端到端原生 Tool Calling 智能体多步推理实测】")

    llm = get_agent_llm()
    tools = [calc_halo_overshoot, exp_tool]

    test_query = (
        "夜景高反差人像分析：实测环境 lux_index=420.0，传感器感光度 ISO=3200.0，单帧曝光时间 exposure_time_ms=50.0；"
        "同时路灯边缘外侧实测对比度阶跃比 edge_contrast=3.0，法向 SDF 溢出距离 sdf_overflow=15.0%。"
        "请调用相关专业工具：1. 评估当前画面的曝光增益与推荐降噪策略；"
        "2. 计算该路灯边缘的光晕过冲风险指数并判断是否属于严重光晕；"
        "3. 综合两者输出一份结构化的 ISP 调试优化诊断书。"
    )

    t0 = time.time()
    agent_result = run_native_tool_agent(
        llm=llm, tools=tools, query=test_query, max_steps=5
    )
    t1 = time.time()

    print("\n" + "-" * 60)
    print("📊 【原生 Tool Calling 调用记录全貌】")
    print("-" * 60)
    for idx, item in enumerate(agent_result["tool_call_history"], 1):
        print(f"调用 {idx}:")
        print(f"  ⚡ Tool Name: {item['name']}")
        print(f"  🔑 Call ID:   {item['id']}")
        print(f"  📥 Tool Args: {item['args']}")
        print(f"  📤 Output:    {item['output']}")
        print()

    print(f"🏆 【Agent 最终诊断报告】:\n{agent_result['final_answer']}")
    print(
        f"\n总耗时: {(t1 - t0) * 1000:.2f} ms，共经历 {agent_result['total_steps']} 轮模型交互。"
    )

    # 验证业务闭环：
    assert (
        len(agent_result["tool_call_history"]) >= 2
    ), "Agent 未经历至少两个原生工具调用"
    tool_names = [call["name"] for call in agent_result["tool_call_history"]]
    assert "calc_halo_overshoot" in tool_names, "Agent 遗漏光晕过冲计算工具调用"
    assert "isp_exposure_gain_eval" in tool_names, "Agent 遗漏曝光增益评估工具调用"
    assert (
        "final_answer" in agent_result and len(agent_result["final_answer"]) > 50
    ), "最终诊断报告内容不充实"

    print("\n" + "=" * 80)
    print(
        "🎉 [PASSED]: Day 16 关卡全通！LangChain 自定义工具与原生 Tool Calling 闭环手撕成功！"
    )
    print("=" * 80)


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
# 关卡 1 参考实现：
@tool(args_schema=HaloOvershootInput)
def calc_halo_overshoot(edge_contrast: float, sdf_overflow: float) -> str:
    \"\"\"计算导向滤波 (FGF) 在高反差边缘引起的光晕 (Halo) 过冲风险指标。\"\"\"
    risk_index = round(edge_contrast * (sdf_overflow / 10.0), 2)
    is_severe = risk_index >= 2.5
    advice = (
        "【严重光晕过冲风险】：建议减小导向滤波半径 r，增大平滑因子 eps，启用梯度自适应保边引导。"
        if is_severe
        else "【轻度光晕容许】：在可接受容差范围内，保持当前保边滤波参数即可。"
    )
    return json.dumps({
        "edge_contrast": edge_contrast,
        "sdf_overflow": sdf_overflow,
        "halo_risk_index": risk_index,
        "is_severe": is_severe,
        "advice": advice,
    }, ensure_ascii=False)

# 关卡 2 参考实现：
class ISPExposureGainTool(BaseTool):
    name: str = "isp_exposure_gain_eval"
    description: str = "评估相机 ISP 当前画面的总曝光增益 Total Gain，并基于环境照度推荐降噪策略。"
    args_schema: Type[BaseModel] = ExposureGainInput
    handle_tool_error: Any = custom_exposure_error_handler

    def _run(self, lux_index: float, iso_value: float, exposure_time_ms: float) -> str:
        total_gain = round((iso_value / 100.0) * (exposure_time_ms / 10.0), 2)
        if total_gain > 80.0:
            strategy = "【MFNR Super Night 超级夜景】：建议开启 8 帧 RAW 域对齐融合与时空时序滤波 (TNR)。"
        elif total_gain > 30.0:
            strategy = "【MFNR Standard 中感光融合】：建议开启 4 帧中度降噪与小波保边抑制。"
        else:
            strategy = "【Single Frame 常规单帧降噪】：建议采用标准双边滤波与自适应锐化。"
        return json.dumps({
            "lux_index": lux_index,
            "iso_value": iso_value,
            "exposure_time_ms": exposure_time_ms,
            "total_gain": total_gain,
            "strategy": strategy,
        }, ensure_ascii=False)

# 关卡 3 参考实现：
def run_native_tool_agent(
    llm: ChatOpenAI,
    tools: List[Any],
    query: str,
    max_steps: int = 5,
) -> Dict[str, Any]:
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    system_msg = SystemMessage(content="你是一个专业的相机图像画质与 ISP 算法工程专家。请善于使用工具辅助计算与策略核查，给出条理清晰、严谨专业的诊断报告。")
    messages = [system_msg, HumanMessage(content=query)]
    tool_call_history = []

    for step in range(1, max_steps + 1):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {
                "final_answer": response.content,
                "tool_call_history": tool_call_history,
                "total_steps": step,
            }

        for tool_call in response.tool_calls:
            call_id = tool_call["id"]
            name = tool_call["name"]
            args = tool_call["args"]

            if name in tool_map:
                try:
                    tool_output = tool_map[name].invoke(args)
                except Exception as e:
                    tool_output = f"工具执行异常: {str(e)}"
            else:
                tool_output = f"未找到名为 '{name}' 的工具。"

            tool_call_history.append({
                "id": call_id,
                "name": name,
                "args": args,
                "output": tool_output,
            })
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=call_id))

    return {
        "final_answer": "达到最大迭代步数，未能产生最终报告。",
        "tool_call_history": tool_call_history,
        "total_steps": max_steps,
    }
"""
# endregion
