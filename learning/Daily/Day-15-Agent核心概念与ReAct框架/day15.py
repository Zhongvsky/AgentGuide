"""
Day 15: 第 3 周启航 - Agent 核心概念与 ReAct 框架手撕实战
=====================================================================
本脚本为 Day 15 的智能体认知闭环闯关代码骨架。
请根据契约定义（Pydantic & Type Hints）与关卡提示（Docstring），
从零手撕经典 ReAct (Reasoning + Acting) 状态机循环引擎：
1. ToolRegistry: 强类型工具注册表与参数校验；
2. ReActParser: Thought / Action / Action Input / Final Answer 正则状态机解析器；
3. ReActAgentLoop: 多步动态上下文（Scratchpad）组装、工具分发与闭环执行。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import sys
import os
import re
import json
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Callable, Type, Union
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
            temperature=0.0,  # Agent 决策必须保证严格确定性
            max_tokens=1024,
            timeout=30.0,
        )
        print(f"[+] 成功连接 Agent 大模型端点 ({base_url} | 模型: {model_name})！")
        return llm
    except Exception as e:
        print(f"[-] 初始化 LLM 失败: {e}")
        raise e


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class ToolSpec(BaseModel):
    """工具规范契约"""

    name: str = Field(..., description="工具唯一标识名称")
    description: str = Field(..., description="工具功能与使用场景描述")
    args_schema: Type[BaseModel] = Field(..., description="Pydantic 参数校验模型")
    func: Any = Field(..., description="实际执行的可调用函数")


class AgentAction(BaseModel):
    """Agent 发起的动作指令"""

    thought: str = Field(..., description="该步动作前的思考推理过程")
    tool_name: str = Field(..., description="调用的目标工具名称")
    tool_input: Dict[str, Any] = Field(..., description="传递给工具的结构化参数字典")


class AgentFinish(BaseModel):
    """Agent 完成任务的终极输出"""

    thought: str = Field(..., description="得出结论的最终思考过程")
    final_answer: str = Field(..., description="给用户的最终答案")


class AgentStep(BaseModel):
    """单步执行轨迹记录"""

    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str


# ==========================================
# 2. 核心架构手撕区：Day 15 闯关打靶区
# ==========================================


class ToolRegistry:
    """
    【关卡 1】：强类型工具注册表系统
    管理 Agent 可调用的工具集，负责元数据渲染、Pydantic 参数校验与安全执行。
    """

    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        func: Callable,
    ) -> None:
        """
        注册一个新工具到注册表中。
        """
        # TODO 1.1: 请实现工具注册逻辑
        raise NotImplementedError("【关卡 1.1 待实现】：请实现 ToolRegistry.register！")

    def get_tools_prompt(self) -> str:
        """
        将所有已注册工具的名字、描述和 JSON 参数 Schema 格式化为 Prompt 描述文本。
        格式示例：
        - tool_name: description. Args: {"param1": {"type": "float", "description": "..."}}
        """
        # TODO 1.2: 请实现工具元数据渲染为 Prompt 文本
        raise NotImplementedError(
            "【关卡 1.2 待实现】：请实现 ToolRegistry.get_tools_prompt！"
        )

    def execute(self, tool_name: str, tool_input: Union[Dict[str, Any], str]) -> str:
        """
        分发执行工具：
        1. 检查工具是否存在，若不存在返回错误提示；
        2. 若 tool_input 为 JSON 字符串，尝试反序列化为 dict；
        3. 利用对应工具的 args_schema 进行强类型入参校验；
        4. 执行目标函数并捕获所有异常，将结果统一转换为 str 返回。
        """
        # TODO 1.3: 请实现基于 Pydantic 校验的安全工具分发执行
        raise NotImplementedError("【关卡 1.3 待实现】：请实现 ToolRegistry.execute！")


class ReActParser:
    """
    【关卡 2】：ReAct 状态机文本解析器
    从 LLM 输出的原始文本中提取 Thought, Action, Action Input 或 Final Answer。

    期望的 LLM 输出协议：
    --------------------
    情境 A (中间决策步):
    Thought: 思考过程描述...
    Action: 工具名称
    Action Input: {"param1": 1.0, "param2": "value"}

    情境 B (终止完成步):
    Thought: 最终综合思考...
    Final Answer: 最终输出的完整报告或回答...
    --------------------
    """

    @staticmethod
    def parse(llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 LLM 的输出文本：
        1. 优先检测是否包含 `Final Answer:`：
           - 提取 Thought 与 Final Answer，封装并返回 `AgentFinish`；
        2. 若不包含 Final Answer，则检测 `Action:` 与 `Action Input:`：
           - 提取 Thought, Action (去除多余字符)；
           - 提取 Action Input 并解析为合法 JSON 字典（容错防御：支持 Markdown ```json 块包裹）；
           - 封装并返回 `AgentAction`；
        3. 若都不匹配，抛出 ValueError 或尝试兜底。
        """
        # TODO 2: 请实现 ReActParser 解析逻辑
        raise NotImplementedError("【关卡 2 待实现】：请实现 ReActParser.parse！")


class ReActAgentLoop:
    """
    【关卡 3】：ReAct Agent 核心执行循环引擎
    将用户 Query、可用工具清单与历史行动轨迹 (Scratchpad) 组装并驱动 LLM 进行交替推理与行动。
    """

    SYSTEM_PROMPT = """你是一个具备严谨逻辑推理与计算能力的 AI Agent。
你可以通过交替进行思考（Thought）与行动（Action）来解答用户的复杂多步问题。

## 可用工具清单：
{tools_description}

## 输出协议格式规范：
每次回复必须严格遵循以下两种格式之一：

【格式 1 - 需要调用工具时】：
Thought: 分析当前已知信息，阐明为什么需要调用该工具，以及下一步计划
Action: 工具名称（必须严格为上述可用工具清单中的名字之一）
Action Input: 严格符合工具要求的 JSON 字典，例如 {{"param": 1.0}}

【格式 2 - 收集完所有必要事实，准备给出最终答案时】：
Thought: 综合前面所有 Observation 事实，推导最终结论
Final Answer: 详细、严谨、条理清晰的最终答案

重要约束：
1. 每次回复只能输出一组 Thought 和 Action/Action Input，或者一组 Thought 和 Final Answer！
2. 严禁自行编造工具返回的 Observation 内容！
3. Action Input 必须是合法且能够被 json.loads 解析的 JSON 格式！
"""

    def __init__(self, llm: ChatOpenAI, tool_registry: ToolRegistry):
        self.llm = llm
        self.registry = tool_registry

    def run(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        驱动 ReAct 循环执行：
        1. 初始化 scratchpad 为空字符串，记录执行步骤列表 steps = []；
        2. 进入 for 循环 (最多 max_iterations 次)：
           a. 组装 Prompt：系统提示词 + 用户 Query + 当前累积的 scratchpad；
           b. 调用 self.llm.invoke() 获取输出；
           c. 调用 ReActParser.parse() 进行解析：
              - 若为 AgentFinish：记录最终思考，返回包含 final_answer、steps 与总耗时的字典；
              - 若为 AgentAction：
                i. 调用 self.registry.execute 执行工具，获得 observation 字符串；
                ii. 将单步轨迹记录加入 steps；
                iii. 更新 scratchpad：追加模型输出文本 + "\nObservation: " + observation + "\n"；
                iv. 打印该步详细日志，进入下一轮迭代；
        3. 若超出最大循环次数仍未达成 Final Answer，抛出异常或返回兜底响应。
        """
        # TODO 3: 请实现 ReActAgentLoop 循环引擎
        raise NotImplementedError("【关卡 3 待实现】：请实现 ReActAgentLoop.run！")


# ==========================================
# 3. 领域实战工具集与 Milestone 验收用例
# ==========================================

# 实战工具 1：计算摄影暗部信噪比计算器
class NoiseSNRInput(BaseModel):
    local_variance: float = Field(..., gt=0, description="局部图像暗部方差 sigma^2，必须大于0")
    mean_intensity: float = Field(..., ge=0, description="局部图像暗部像素均值 mu")


def calc_noise_snr_func(local_variance: float, mean_intensity: float) -> str:
    """计算图像信噪比 SNR (dB) = 10 * log10(mu^2 / sigma^2)"""
    snr_linear = (mean_intensity ** 2) / local_variance
    snr_db = 10.0 * math.log10(snr_linear) if snr_linear > 0 else 0.0
    return json.dumps({
        "snr_db": round(snr_db, 2),
        "mean_intensity": mean_intensity,
        "local_variance": local_variance,
        "status": "success",
    }, ensure_ascii=False)


# 实战工具 2：ISP 缺陷机理与阈值规章查询器
class ISPDefectInput(BaseModel):
    defect_type: str = Field(..., description="缺陷类型，可选: 'Noise', 'Halo', 'ColorCast'")


def query_isp_defect_func(defect_type: str) -> str:
    """查询 ISP 缺陷评级与判定阈值规范"""
    knowledge_base = {
        "Noise": (
            "【ISP 噪声缺陷评级规范】：低照度夜景场景下，若实测暗部信噪比 SNR < 15.0 dB，"
            "判定为【严重噪声塌陷（Defect: Noise Level-3）】！"
            "工程修复方案：需在 ISP Pipeline 中将双边滤波时空降噪（TNR）增益权重提升 25%，并开启小波去噪高频抑制。"
        ),
        "Halo": (
            "【ISP 光晕缺陷评级规范】：高反差边缘若法向 SDF 溢出度 > 12.0%，判定为【FGF 导向滤波光晕过冲】。"
            "工程修复方案：调小导向滤波半径，启用梯度自适应保边系数。"
        ),
    }
    return knowledge_base.get(defect_type, f"未收录缺陷类型: {defect_type}")


if __name__ == "__main__":
    print("=" * 80)
    print("【Day 15 实战】：Agent 核心概念与 ReAct 框架手撕实战")
    print("=" * 80)

    # ----------------------------------------------------
    # 测试关卡 1：ToolRegistry 工具注册与 Pydantic 校验测试
    # ----------------------------------------------------
    print("\n🔍 【测试 1：强类型工具注册表 ToolRegistry 单元测试】")
    registry = ToolRegistry()

    registry.register(
        name="calc_noise_snr",
        description="计算图像局部暗部信噪比 SNR (dB)，输入为方差与均值。",
        args_schema=NoiseSNRInput,
        func=calc_noise_snr_func,
    )
    registry.register(
        name="query_isp_defect",
        description="查询 ISP 缺陷评级判定标准与工程调优建议，输入为缺陷类型。",
        args_schema=ISPDefectInput,
        func=query_isp_defect_func,
    )

    # 1.1 校验 Prompt 描述生成
    tools_prompt = registry.get_tools_prompt()
    assert "calc_noise_snr" in tools_prompt, "工具描述中遗漏 calc_noise_snr"
    assert "local_variance" in tools_prompt, "未提取出参数字段 local_variance"
    print("✅ 关卡 1.1 通过：工具清单成功序列化为符合 LLM 阅读的 Schema 描述！")

    # 1.2 正常调用执行
    valid_res = registry.execute(
        "calc_noise_snr", {"local_variance": 16.0, "mean_intensity": 80.0}
    )
    valid_data = json.loads(valid_res)
    assert valid_data["snr_db"] == 26.02, f"SNR 计算偏差: 期望 26.02, 实际 {valid_data['snr_db']}"
    print(f"✅ 关卡 1.2 通过：正常参数执行成功，SNR={valid_data['snr_db']} dB")

    # 1.3 异常拦截：方差必须 > 0
    invalid_res = registry.execute(
        "calc_noise_snr", {"local_variance": -5.0, "mean_intensity": 80.0}
    )
    assert "参数校验失败" in invalid_res or "ValidationError" in invalid_res, "未正确拦截非法参数"
    print("✅ 关卡 1.3 通过：Pydantic 成功拦截非法参数输入！")

    # ----------------------------------------------------
    # 测试关卡 2：ReActParser 状态机解析器测试
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 2：ReActParser 状态机文本解析器测试】")

    # 2.1 测试解析 Action 输出
    sample_action_text = (
        "Thought: 用户给出了局部暗部方差和均值，我需要先计算信噪比。\n"
        "Action: calc_noise_snr\n"
        "Action Input: {\"local_variance\": 16.0, \"mean_intensity\": 80.0}\n"
    )
    parsed_action = ReActParser.parse(sample_action_text)
    assert isinstance(parsed_action, AgentAction), "未解析为 AgentAction 实例"
    assert parsed_action.tool_name == "calc_noise_snr", f"工具名解析有误: {parsed_action.tool_name}"
    assert parsed_action.tool_input["local_variance"] == 16.0, "参数字典解析有误"
    print("✅ 关卡 2.1 通过：成功精准解析 Action 指令与结构化参数字典！")

    # 2.2 测试解析 Final Answer 输出
    sample_finish_text = (
        "Thought: 根据知识库，当前 SNR=13.98 dB 低于 15 dB 门限，确实属于严重噪声。\n"
        "Final Answer: 经计算该区域 SNR 为 13.98 dB，判定为严重噪声塌陷（Noise Level-3）。建议调大 TNR 权重。"
    )
    parsed_finish = ReActParser.parse(sample_finish_text)
    assert isinstance(parsed_finish, AgentFinish), "未解析为 AgentFinish 实例"
    assert "严重噪声塌陷" in parsed_finish.final_answer, "最终答案提取不全"
    print("✅ 关卡 2.2 通过：成功精准截断并提取 Final Answer 最终答案！")

    # ----------------------------------------------------
    # 测试关卡 3：端到端 ReActAgentLoop 多步智能体推理实测
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 3：端到端 ReActAgentLoop 多步自主决策实操】")

    agent_llm = get_agent_llm()
    agent_loop = ReActAgentLoop(llm=agent_llm, tool_registry=registry)

    test_query = (
        "请分析暗部测试图块：实测局部方差 local_variance=16.0，像素均值 mean_intensity=30.0。"
        "请先计算该区域的实际信噪比 SNR（分贝）；然后查询知识库判断此 SNR 水平在夜景下是否达到'Noise'严重缺陷标准？"
        "最后给出详细的诊断评级与修复调优建议。"
    )

    result = agent_loop.run(query=test_query, max_iterations=5)

    print("\n" + "-" * 60)
    print("📊 【Agent 端到端推理轨迹全貌】")
    print("-" * 60)
    for s in result["steps"]:
        print(f"第 {s.step} 步:")
        print(f"  💭 Thought: {s.thought}")
        print(f"  ⚡ Action: {s.action}({s.action_input})")
        print(f"  👁️ Observation: {s.observation}")
        print()

    print(f"🏆 【Final Answer 最终答案】:\n{result['final_answer']}")
    print(f"总耗时: {result['latency_ms']:.2f} ms，共经历 {len(result['steps'])} 轮环境交互。")

    # 验证业务闭环：
    # 30^2 / 16 = 900 / 16 = 56.25 -> 10 * log10(56.25) = 17.50 dB (大于 15dB)
    # 或者如果数值是均值 20，方差 16 -> 400 / 16 = 25 -> 10 * log10(25) = 13.98 dB
    assert len(result["steps"]) >= 2, "Agent 未经历至少两步工具调用闭环"
    assert "calc_noise_snr" in [s.action for s in result["steps"]], "Agent 未调用计算信噪比工具"
    assert "query_isp_defect" in [s.action for s in result["steps"]], "Agent 未调用知识库检索工具"
    assert "final_answer" in result and len(result["final_answer"]) > 20, "未产生有效最终回答"

    print("\n" + "=" * 80)
    print("🎉 [PASSED]: Day 15 关卡全通！ReAct 认知闭环引擎手撕成功！")
    print("=" * 80)


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
# 关卡 1 参考实现：
def register(
    self,
    name: str,
    description: str,
    args_schema: Type[BaseModel],
    func: Callable,
) -> None:
    self.tools[name] = ToolSpec(
        name=name,
        description=description,
        args_schema=args_schema,
        func=func,
    )

def get_tools_prompt(self) -> str:
    lines = []
    for t in self.tools.values():
        schema_dict = t.args_schema.model_json_schema().get("properties", {})
        param_desc = {k: {"type": v.get("type"), "description": v.get("description")} for k, v in schema_dict.items()}
        lines.append(f"- **{t.name}**: {t.description}\n  参数约束: {json.dumps(param_desc, ensure_ascii=False)}")
    return "\\n".join(lines)

def execute(self, tool_name: str, tool_input: Union[Dict[str, Any], str]) -> str:
    if tool_name not in self.tools:
        return f"错误：未找到名为 '{tool_name}' 的工具，请检查工具拼写。"
    tool = self.tools[tool_name]
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except Exception as e:
            return f"参数 JSON 反序列化失败: {e}"
    try:
        validated_args = tool.args_schema(**tool_input)
    except ValidationError as ve:
        return f"参数校验失败 (ValidationError): {ve}"
    try:
        result = tool.func(**validated_args.model_dump())
        return str(result)
    except Exception as e:
        return f"工具执行发生异常: {str(e)}"


# 关卡 2 参考实现：
@staticmethod
def parse(llm_output: str) -> Union[AgentAction, AgentFinish]:
    clean_text = llm_output.strip()
    if "Final Answer:" in clean_text:
        parts = clean_text.split("Final Answer:", 1)
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nFinal Answer:|$)", clean_text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else parts[0].replace("Thought:", "").strip()
        final_answer = parts[1].strip()
        return AgentFinish(thought=thought, final_answer=final_answer)

    action_match = re.search(r"Action:\s*([^\n]+)", clean_text)
    action_input_match = re.search(r"Action Input:\s*(\{.*?\}|```json\s*(\{.*?\})\s*```)", clean_text, re.DOTALL)

    if not action_match:
        raise ValueError(f"无法从模型输出中识别 Action 指令: {clean_text}")

    tool_name = action_match.group(1).strip().strip("`").strip("'").strip('"')
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", clean_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    raw_input = "{}"
    if action_input_match:
        raw_input = action_input_match.group(2) if action_input_match.group(2) else action_input_match.group(1)
        raw_input = raw_input.strip()

    try:
        tool_input = json.loads(raw_input)
    except Exception:
        tool_input = {"input": raw_input}

    return AgentAction(thought=thought, tool_name=tool_name, tool_input=tool_input)


# 关卡 3 参考实现：
def run(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
    t0 = time.time()
    tools_desc = self.registry.get_tools_prompt()
    system_prompt = self.SYSTEM_PROMPT.format(tools_description=tools_desc)

    scratchpad = ""
    steps: List[AgentStep] = []

    for step_idx in range(1, max_iterations + 1):
        full_user_prompt = f"用户问题: {query}\n\n{scratchpad}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content.strip()

        parsed = ReActParser.parse(content)

        if isinstance(parsed, AgentFinish):
            t1 = time.time()
            return {
                "final_answer": parsed.final_answer,
                "thought": parsed.thought,
                "steps": steps,
                "latency_ms": (t1 - t0) * 1000,
            }

        elif isinstance(parsed, AgentAction):
            observation = self.registry.execute(parsed.tool_name, parsed.tool_input)
            steps.append(
                AgentStep(
                    step=step_idx,
                    thought=parsed.thought,
                    action=parsed.tool_name,
                    action_input=parsed.tool_input,
                    observation=observation,
                )
            )
            scratchpad += f"Thought: {parsed.thought}\nAction: {parsed.tool_name}\nAction Input: {json.dumps(parsed.tool_input, ensure_ascii=False)}\nObservation: {observation}\n"

    t1 = time.time()
    return {
        "final_answer": "已达到最大步数限制，未能得出最终结论。",
        "thought": "Exceeded max iterations",
        "steps": steps,
        "latency_ms": (t1 - t0) * 1000,
    }
"""
# endregion
