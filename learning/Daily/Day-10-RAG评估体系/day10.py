"""
Day 10: Advanced RAG - RAG 科学评估体系与 Ragas 实战
===================================================
本脚本为 Day 10 的实战闯关代码骨架。
请根据其中的契约定义（Pydantic & Type Hints）与关卡提示（Docstring），
逐步实现基于 LLM-as-a-Judge 的 Faithfulness（真实度/抗幻觉）与
Answer Relevancy（回答相关度）评估链。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np

# Windows 控制台 UTF-8 防御（避免 GBK 编码崩溃）
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

# 镜像加速与依赖库
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class EvaluationSample(BaseModel):
    """单个评估样本输入契约"""

    query: str = Field(..., description="用户的原始提问")
    retrieved_contexts: List[str] = Field(..., description="检索回来的参考上下文片段列表")
    generated_answer: str = Field(..., description="RAG 系统生成的最终回答")
    ground_truth: Optional[str] = Field(None, description="标准人工标注真值答案（可选）")


class StatementVerification(BaseModel):
    """单个原子命题的判定契约"""

    statement: str = Field(..., description="拆解出的原子陈述句")
    verdict: int = Field(..., description="是否得到上下文支撑：1 为得到支撑，0 为无支撑/幻觉")
    reason: str = Field(..., description="裁判模型的判决依据与理由")


class EvaluationReport(BaseModel):
    """单条样本的综合量化评测报告"""

    query: str
    faithfulness_score: float = Field(..., description="真实度得分 (0.0 ~ 1.0)")
    answer_relevance_score: float = Field(..., description="回答相关度得分 (0.0 ~ 1.0)")
    statements: List[StatementVerification] = Field(default_factory=list, description="原子命题验证明细")
    reverse_queries: List[str] = Field(default_factory=list, description="逆向反推生成的问题")


# ==========================================
# 2. 裁判模型与打分脚手架 (开箱即用)
# ==========================================


def get_judge_llm(base_url: str = "http://127.0.0.1:9527/v1", api_key: str = "dummy") -> Any:
    """初始化用于打分的裁判大模型（兼容 OpenAI 协议本地端点）"""
    try:
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.01,  # 裁判任务必须低温确定性
            max_retries=1,
            timeout=15,
        )
        # 简单心跳探测
        llm.invoke("Hi")
        print("[+] 成功连接本地裁判大模型 (127.0.0.1:9527)！")
        return llm
    except Exception as e:
        print(f"[-] 未能连接本地大模型端点 ({e})")
        print("[*] 启用内置高拟真裁判模拟器 (MockJudgeLLM) 作为自适应离线环境...")

        class MockJudgeLLM:
            """自适应裁判模拟器：基于规则与关键词推导原子命题与验证结果"""

            def invoke(self, prompt_text: Any) -> Any:
                text = str(prompt_text)
                class MockMsg:
                    content = ""
                msg = MockMsg()

                # 分支 1：如果是拆解命题
                if "原子陈述" in text or "statements" in text.lower():
                    # 模拟将回答按句号分句
                    ans_part = text.split("回答：")[-1].split("\n")[0]
                    sentences = [s.strip() for s in ans_part.replace("。", "。\n").split("\n") if len(s.strip()) > 5]
                    msg.content = json.dumps({"statements": sentences or ["出差人员按规定乘坐交通工具。"]}, ensure_ascii=False)
                    return msg

                # 分支 2：如果是验证支撑性
                elif "忠实度" in text or "verdict" in text.lower():
                    # 简单的包含度逻辑模拟
                    if "特等座" in text or "800元" in text or "豪华" in text:
                        msg.content = json.dumps([{"verdict": 0, "reason": "上下文无此金额或席位规定"}], ensure_ascii=False)
                    else:
                        msg.content = json.dumps([{"verdict": 1, "reason": "上下文中有明确依据"}], ensure_ascii=False)
                    return msg

                # 分支 3：如果是逆向反推问题
                elif "反推问题" in text or "reverse" in text.lower():
                    msg.content = json.dumps({
                        "questions": [
                            "出差乘坐交通工具有什么等级规定？",
                            "差旅费管理办法对于出行工具有哪些约束？",
                            "哪些级别的干部可以坐一等座？"
                        ]
                    }, ensure_ascii=False)
                    return msg

                msg.content = "1"
                return msg

        return MockJudgeLLM()


# ==========================================
# 3. 核心任务手撕区：RAG 评估链 (闯关打靶区)
# ==========================================


def extract_atomic_statements(answer: str, judge_llm: Any) -> List[str]:
    """
    【关卡 1】：原子命题拆解 (Atomic Statement Extraction)

    目标：
    将 RAG 系统生成的复杂长回答（answer）拆解为若干个语义独立、不可再分的「原子事实陈述句（Atomic Statements）」。
    例如："出差人员可乘坐高铁，且司局级干部可坐一等座。" 
    应拆解为：
    - "出差人员可乘坐高铁。"
    - "司局级干部可坐一等座。"

    实现提示：
    1. 设计 Prompt 要求大模型仅提取事实陈述，剔除语气助词；
    2. 要求输出严格的 JSON 格式：{"statements": ["陈述1", "陈述2", ...]}；
    3. 调用 judge_llm，解析返回的 JSON，返回 List[str]。
    """
    # TODO 1: 请实现原子命题拆解逻辑
    raise NotImplementedError(
        "【关卡 1 待实现】：请实现 extract_atomic_statements 原子命题拆解！完成后删除本行。"
    )


def evaluate_faithfulness(
    statements: List[str],
    contexts: List[str],
    judge_llm: Any,
) -> Tuple[float, List[StatementVerification]]:
    """
    【关卡 2】：Faithfulness（真实度 / 忠实度 / 抗幻觉）评估

    数学公式：
        Faithfulness = (得到 Context 有效支撑的原子命题数量) / (总原子命题数量)

    目标：
    让裁判大模型核验每一个 statement 是否可以从 contexts 中直接推导出来。
    若存在上下文未提及的事实、捏造的金额或过度发散的结论，判为 0（无支撑）；
    完全由上下文支撑的判为 1。

    实现提示：
    1. 将 contexts 拼接为完整的参考资料 text_context；
    2. 对每个 statement，构造 Prompt 让 judge_llm 判定其 verdict (0 或 1) 以及理由 reason；
       （也可将所有 statements 打包一次性送给 LLM 批量打分以节省 Token）；
    3. 统计 verdict == 1 的数量 supported_count；
    4. 计算 faithfulness_score = supported_count / len(statements)；
    5. 返回 (score, verification_list)。
    """
    # TODO 2: 请实现 Faithfulness 忠实度验证算法
    raise NotImplementedError(
        "【关卡 2 待实现】：请实现 evaluate_faithfulness 真实度核验！完成后删除本行。"
    )


def evaluate_answer_relevance(
    query: str,
    answer: str,
    embed_model: SentenceTransformer,
    judge_llm: Any,
    num_reverse_questions: int = 3,
) -> Tuple[float, List[str]]:
    """
    【关卡 3】：Answer Relevancy（回答相关度）评估

    数学公式：
        Answer_Relevance = (1 / N) * sum_{i=1}^N Cosine_Similarity(E(Q_original), E(Q_reverse_i))

    核心思想（逆向提问法）：
    - 如果一个回答非常切中问题，那么光看这个回答，大模型就能精准“猜”出当初用户问了什么；
    - 让裁判模型基于 answer 逆向反推 N 个可能的问题；
    - 计算这 N 个反推问题与原始 query 的向量余弦相似度均值；
    - 得分越接近 1.0，说明回答与原始问题的主题契合度极高，没有顾左右而言他。

    实现提示：
    1. 构造 Prompt，给入 answer，让 judge_llm 逆向生成 num_reverse_questions 个问题；
    2. 要求输出 JSON：{"questions": ["反推问题1", "反推问题2", ...]}；
    3. 解析出反推问题列表 reverse_queries；
    4. 使用 embed_model.encode 对 query 和 reverse_queries 进行向量化（normalize_embeddings=True）；
    5. 计算余弦相似度（点积即为余弦值），取平均值作为最终得分；
    6. 返回 (relevance_score, reverse_queries)。
    """
    # TODO 3: 请实现基于逆向提问与余弦相似度的 Answer Relevancy 算法
    raise NotImplementedError(
        "【关卡 3 待实现】：请实现 evaluate_answer_relevance 回答相关度评估！完成后删除本行。"
    )


# ==========================================
# 4. 自动化评测流水线驱动 (开箱即用)
# ==========================================


def run_evaluation_pipeline(
    samples: List[EvaluationSample],
    judge_llm: Any,
    embed_model: SentenceTransformer,
):
    """运行批量评估流水线并输出量化台账"""
    print("\n" + "=" * 80)
    print("[*] 开始执行 RAG 科学量化评测体系 (RAG Triad & Ragas) 自动化评估")
    print("=" * 80)

    for idx, sample in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] 正在评估样本:")
        print(f"  [Q] 原始提问: 「{sample.query}」")
        print(f"  [A] 生成回答: 「{sample.generated_answer[:80]}...」")
        print("-" * 80)

        t0 = time.perf_counter()

        # 1. 拆解命题
        statements = extract_atomic_statements(sample.generated_answer, judge_llm)
        print(f"  [*] 拆解出 {len(statements)} 个原子命题:")
        for s in statements:
            print(f"      - {s}")

        # 2. 评估 Faithfulness
        faith_score, verifications = evaluate_faithfulness(
            statements=statements,
            contexts=sample.retrieved_contexts,
            judge_llm=judge_llm,
        )

        # 3. 评估 Answer Relevance
        relevance_score, reverse_queries = evaluate_answer_relevance(
            query=sample.query,
            answer=sample.generated_answer,
            embed_model=embed_model,
            judge_llm=judge_llm,
        )

        elapsed = time.perf_counter() - t0

        # 打印单样本得分卡
        print(f"\n  📊 【样本评估指标卡】 (耗时: {elapsed:.2f}s):")
        print(f"  ------------------------------------------------------------")
        print(f"  1. 真实度 / 抗幻觉 (Faithfulness)  : {faith_score:.4f}  ({'🟢 极高' if faith_score > 0.8 else '🔴 存严重幻觉'})")
        print(f"  2. 回答相关性 (Answer Relevance) : {relevance_score:.4f}  ({'🟢 高度契合' if relevance_score > 0.75 else '🟡 存在偏题'})")
        print(f"  ------------------------------------------------------------")
        print(f"  [命题支撑判定明细]:")
        for v in verifications:
            tag = "✅ 支撑" if v.verdict == 1 else "❌ 幻觉"
            print(f"    {tag} | 「{v.statement}」 -> 理由: {v.reason}")
        print(f"  [逆向推导问题]:")
        for rq in reverse_queries:
            print(f"    ↳ 「{rq}」")

    print("\n" + "=" * 80)
    print("[+] 评估流水线执行完毕！量化指标成功度量生成端质量与抗幻觉能力。")


# ==========================================
# 5. 主执行入口与 Benchmark 测试用例
# ==========================================

if __name__ == "__main__":
    print("[*] 正在加载 Embedding 模型用于相关度打分...")
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    print("[*] 正在初始化裁判 LLM...")
    judge_llm = get_judge_llm()

    # 准备两组具有鲜明对比的 Benchmark 测试样本
    benchmark_samples = [
        # 样本 1：标准高质量忠实回答（无幻觉）
        EvaluationSample(
            query="普通员工出差去北京，坐高铁能买一等座吗？",
            retrieved_contexts=[
                "第二章 第六条 城市间交通费是指工作人员因公到常驻地以外地区出差乘坐火车、轮船、飞机等交通工具所发生的费用。\n第七条 出差人员应当按规定等级乘坐交通工具。乘坐火车级别见下表：部级及相当职务人员可乘高铁商务座；司局级及相当职务人员可乘高铁一等座；其余人员可乘高铁二等座。"
            ],
            generated_answer="普通员工出差不能购买高铁一等座。根据差旅费管理办法第七条规定，普通员工（其余人员）出差乘坐高铁时只能购买二等座，高铁一等座仅限司局级及相当职务人员乘坐。",
        ),
        # 样本 2：典型幻觉样本（伪造特等座、编造 800 元补助、答非所问）
        EvaluationSample(
            query="出差人员乘坐高铁有什么等级限制？",
            retrieved_contexts=[
                "第七条 出差人员应当按规定等级乘坐交通工具。部级人员可乘高铁商务座；司局级人员可乘高铁一等座；其余人员可乘高铁二等座。"
            ],
            generated_answer="出差人员乘坐高铁可以根据需要购买特等座或商务座。另外，出差每人每天可享受 800 元豪华伙食补助费，住宿不设上限。",
        ),
    ]

    # 执行评测
    run_evaluation_pipeline(
        samples=benchmark_samples,
        judge_llm=judge_llm,
        embed_model=embed_model,
    )


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
def extract_atomic_statements_reference(answer: str, judge_llm: Any) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名严谨的自然语言逻辑分析师。请将用户的回答拆解为若干个语义独立的原子事实陈述句（Atomic Statements）。每个陈述句必须是不可再分的事实判断，排除主观修辞。以严格的 JSON 格式输出：{{\"statements\": [\"陈述1\", \"陈述2\"]}}，不要输出任何额外文字。"),
        ("human", "待拆解回答：\n{answer}")
    ])
    chain = prompt | judge_llm | StrOutputParser()
    res = chain.invoke({"answer": answer})
    try:
        data = json.loads(res.strip().replace("```json", "").replace("```", ""))
        return data.get("statements", [answer])
    except Exception:
        return [s.strip() for s in answer.split("。") if s.strip()]

def evaluate_faithfulness_reference(
    statements: List[str],
    contexts: List[str],
    judge_llm: Any,
) -> Tuple[float, List[StatementVerification]]:
    if not statements:
        return 1.0, []
    context_text = "\\n\\n".join(contexts)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名客观严谨的裁判员。请判断给定的陈述句是否能够根据参考资料直接推导出来。若完全被参考资料支撑，verdict 为 1；若包含资料中未提及的事实、捏造的数据或违背资料，verdict 为 0。请以严格的 JSON 列表格式输出：[{{\"verdict\": 1或0, \"reason\": \"判定理由\"}}]。"),
        ("human", "参考资料：\\n{context}\\n\\n待核验陈述句列表：\\n{statements}")
    ])
    chain = prompt | judge_llm | StrOutputParser()
    raw = chain.invoke({"context": context_text, "statements": json.dumps(statements, ensure_ascii=False)})
    try:
        verdicts = json.loads(raw.strip().replace("```json", "").replace("```", ""))
    except Exception:
        verdicts = [{"verdict": 1, "reason": "解析异常默认支撑"} for _ in statements]

    verifications = []
    supported = 0
    for s, v in zip(statements, verdicts):
        verdict_val = v.get("verdict", 0)
        supported += verdict_val
        verifications.append(StatementVerification(statement=s, verdict=verdict_val, reason=v.get("reason", "")))
    
    score = supported / len(statements)
    return score, verifications

def evaluate_answer_relevance_reference(
    query: str,
    answer: str,
    embed_model: SentenceTransformer,
    judge_llm: Any,
    num_reverse_questions: int = 3,
) -> Tuple[float, List[str]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "基于给定的文本内容，逆向推导并生成 {num} 个最有可能引出该回答的用户提问。以严格的 JSON 格式输出：{{\"questions\": [\"问题1\", \"问题2\", \"问题3\"]}}。"),
        ("human", "回答内容：\\n{answer}")
    ])
    chain = prompt | judge_llm | StrOutputParser()
    raw = chain.invoke({"answer": answer, "num": num_reverse_questions})
    try:
        questions = json.loads(raw.strip().replace("```json", "").replace("```", "")).get("questions", [])
    except Exception:
        questions = [query]

    if not questions:
        return 1.0, []

    q_emb = embed_model.encode([query], normalize_embeddings=True)
    rev_embs = embed_model.encode(questions, normalize_embeddings=True)
    sims = np.dot(rev_embs, q_emb.T).flatten()
    relevance_score = float(np.mean(sims))
    return relevance_score, questions
"""
# endregion
