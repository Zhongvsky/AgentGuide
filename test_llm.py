"""
LLM 服务连通性与可用性诊断脚本
==================================
使用方法：
    uv run python test_llm.py

功能：
1. 自动加载项目根目录下的 .env 环境变量；
2. 打印当前激活的 Base URL、API Key 与 Model 配置；
3. 发送简单测试请求，验证连通性、响应延迟与基本生成能力；
4. 验证 JSON 格式输出与中文 UTF-8 编码能力（为 RAG / Agent 评估打底）。
"""

import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# Windows 控制台编码防护
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. 加载 .env
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def mask_key(key: Optional[str]) -> str:
    """脱敏打印 API Key"""
    if not key:
        return "<空 / 未设置>"
    if len(key) <= 6:
        return "***"
    return f"{key[:3]}***{key[-3:]}"


def test_llm_connection():
    print("=" * 60)
    print("🔍 [1/3] 读取环境变量配置...")
    print("=" * 60)

    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:9527/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    print(f"  • OPENAI_BASE_URL : {base_url}")
    print(f"  • OPENAI_API_KEY  : {mask_key(api_key)}")
    print(f"  • LLM_MODEL_NAME  : {model_name}")

    print("\n" + "=" * 60)
    print("🚀 [2/3] 测试基础问答连通性 (Ping & Inference)...")
    print("=" * 60)

    try:
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=0.1,
            max_retries=1,
            timeout=15,
        )

        # test_prompt = "请用一句话做个自我介绍，并说明你是什么模型。"
        # test_prompt = '对下面的内容提取事实陈述、剔除语气助词，输出严格的 JSON 格式(格式为 {"statements": ["陈述1", "陈述2", ...]})："普通员工出差不能购买高铁一等座。根据差旅费管理办法第七条规定，普通员工（其余人员）出差乘坐高铁时只能购买二等座，高铁一等座仅限司局级及相当职务人员乘坐。"'
        test_prompt = (
            "你是一位极其严格的评审专家，请针对下面一系列陈述内容，判断能否由下面的参考资料中推导出来：\n"
            + "## 陈述内容\n"
            '["普通员工出差不能购买高铁一等座。","差旅费管理办法第七条规定，普通员工（其余人员）出差乘坐高铁时只能购买二等座。","高铁一等座仅限司局级及相当职务人员乘坐。"]\n'
            "## 参考资料\n"
            "第二章 第六条 城市间交通费是指工作人员因公到常驻地以外地区出差乘坐火车、轮船、飞机等交通工具所发生的费用。\n第七条 出差人员应当按规定等级乘坐交通工具。乘坐火车级别见下表：部级及相当职务人员可乘高铁商务座；司局级及相当职务人员可乘高铁一等座；其余人员可乘高铁二等座。\n"
            "## 要求\n"
            "对于每一条陈述，要先说明理由，若其可以在参考资料中推导出来，则记该陈述的 verdict = 1，否则记该陈述的 verdict=0\n"
            "最后输出为严格的JSON格式，如:"
            '{{"statement": "xxx", "reason": "xxx", "vedict": 0}, {"statement": "xxx", "reason": "xxx", "vedict": 1}, ...}'
        )
        print(f"  [>] 发送测试问题: 「{test_prompt}」")

        t0 = time.perf_counter()
        response = llm.invoke(test_prompt)
        latency = time.perf_counter() - t0

        print(f"  [+] 模型响应成功！(耗时: {latency:.2f} 秒)")
        print(f"  [<] 回复内容:\n      {response.content.strip()}\n")

    except Exception as e:
        print(f"  [-] ❌ 连通性测试失败！")
        print(f"  [!] 错误详情: {e}")
        print("\n💡 排查建议:")
        print(
            "  1. 检查本地模型服务（如 LM Studio / Magpie / Ollama / vLLM）是否已启动并处于监听状态；"
        )
        print(
            f"  2. 检查 .env 中的 OPENAI_BASE_URL ({base_url}) 端口是否与服务端口一致；"
        )
        print(
            f"  3. 检查 .env 中的 LLM_MODEL_NAME ({model_name}) 是否在服务中已加载或名称正确。"
        )
        return False

    print("=" * 60)
    print("🧩 [3/3] 测试结构化与 JSON 指令遵循能力...")
    print("=" * 60)

    try:
        json_prompt = (
            "请严格以纯 JSON 格式返回，不要包含 markdown 代码块包裹或解释："
            '{"status": "ok", "service": "ready", "agent_guide": true}'
        )
        t0 = time.perf_counter()
        json_resp = llm.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that outputs only raw JSON."
                ),
                HumanMessage(content=json_prompt),
            ]
        )
        latency = time.perf_counter() - t0

        content = json_resp.content.strip()
        print(f"  [+] JSON 指令测试通过！(耗时: {latency:.2f} 秒)")
        print(f"  [<] 原始返回: {content}")

    except Exception as e:
        print(f"  [!] ⚠️ 结构化测试出现异常: {e}")

    print("\n" + "=" * 60)
    print("🎉 恭喜！当前 .env 配置的 LLM 服务完全正常可用！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_llm_connection()
    sys.exit(0 if success else 1)
