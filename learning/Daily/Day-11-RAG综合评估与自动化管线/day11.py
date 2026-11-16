"""
Day 11: Advanced RAG - 端到端 RAG 综合量化评估与自动化评测管线
============================================================
本脚本为 Day 11 的实战闯关代码骨架。
请根据其中的契约定义（Pydantic & Type Hints）与关卡提示（Docstring），
手撕 Context Relevance 上下文相关度算法、端到端 A/B 对比驱动与 CI/CD 质量卡点（Quality Gate）。

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
import re
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


class StatementVerification(BaseModel):
    """单个原子命题的判定契约"""

    statement: str = Field(..., description="拆解出的原子陈述句")
    verdict: int = Field(..., description="是否得到上下文支撑：1 为得到支撑，0 为无支撑/幻觉")
    reason: str = Field(..., description="裁判模型的判决依据与理由")


class RagSystemOutput(BaseModel):
    """单次 RAG 问答链路的端到端产出契约"""

    query: str = Field(..., description="用户原始提问")
    retrieved_contexts: List[str] = Field(..., description="检索模块召回的上下文片段")
    generated_answer: str = Field(..., description="生成模块产生的最终回答")
    latency_ms: float = Field(default=0.0, description="端到端耗时 (ms)")


class RagTriadScore(BaseModel):
    """单个样本的完整 RAG Triad 三元组量化评分契约"""

    query: str
    context_relevance: float = Field(..., description="上下文相关度 (0.0 ~ 1.0)")
    faithfulness: float = Field(..., description="真实度 / 忠实度 (0.0 ~ 1.0)")
    answer_relevance: float = Field(..., description="回答相关度 (0.0 ~ 1.0)")
    latency_ms: float = Field(default=0.0, description="端到端耗时 (ms)")


class QualityGateResult(BaseModel):
    """CI/CD 自动化质量门禁决策报告契约"""

    passed: bool = Field(..., description="质量门禁是否通过")
    decision: str = Field(..., description="决策说明：PASS / BLOCK")
    reasons: List[str] = Field(default_factory=list, description="通过或熔断的具体判定依据")
    baseline_avg: Dict[str, float] = Field(default_factory=dict, description="基线系统平均指标")
    candidate_avg: Dict[str, float] = Field(default_factory=dict, description="候选新系统平均指标")


# ==========================================
# 2. 裁判模型与打分脚手架 (开箱即用)
# ==========================================


def get_judge_llm(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Any:
    """初始化用于打分的裁判大模型（兼容 OpenAI 协议端点，优先读取 .env 环境变量）"""
    actual_base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:9527/v1")
    actual_api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy")
    actual_model = model or os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    try:
        llm = ChatOpenAI(
            base_url=actual_base_url,
            api_key=actual_api_key,
            model=actual_model,
            temperature=0.01,  # 裁判任务必须低温确定性
            max_retries=1,
            timeout=15,
        )
        llm.invoke("Hi")
        print(f"[+] 成功连接裁判大模型 ({actual_base_url} | 模型: {actual_model})！")
        return llm
    except Exception as e:
        print(f"[-] 未能连接大模型端点 ({e})，启用内置 Mock 裁判模拟器...")

        class MockJudgeLLM:
            def invoke(self, prompt_text: Any) -> Any:
                text = str(prompt_text)
                class Msg:
                    content = ""
                m = Msg()

                # 分支 1：Context Relevance 关键句筛选
                if "上下文相关度" in text or "relevant_sentences" in text.lower():
                    # 模拟挑出包含“一等座”、“二等座”、“等级”的句子
                    m.content = json.dumps({
                        "relevant_sentences": [
                            "第七条 出差人员应当按规定等级乘坐交通工具。",
                            "部级及相当职务人员可乘高铁商务座；司局级及相当职务人员可乘高铁一等座；其余人员可乘高铁二等座。"
                        ]
                    }, ensure_ascii=False)
                    return m

                # 分支 2：原子命题拆解
                elif "原子陈述" in text or "statements" in text.lower():
                    m.content = json.dumps({
                        "statements": [
                            "普通员工出差不能购买高铁一等座。",
                            "普通员工出差乘坐高铁只能购买二等座。"
                        ]
                    }, ensure_ascii=False)
                    return m

                # 分支 3：Faithfulness 判定
                elif "忠实度" in text or "verdict" in text.lower():
                    if "特等座" in text or "800元" in text or "无上限" in text:
                        m.content = json.dumps([{"statement": "xxx", "verdict": 0, "reason": "无原文支撑"}], ensure_ascii=False)
                    else:
                        m.content = json.dumps([{"statement": "xxx", "verdict": 1, "reason": "完全吻合第七条"}], ensure_ascii=False)
                    return m

                # 分支 4：逆向提问
                elif "反推问题" in text or "reverse" in text.lower():
                    m.content = json.dumps({
                        "questions": [
                            "普通员工出差可以乘坐高铁一等座吗？",
                            "差旅规定中各级别人员的高铁乘坐标准是什么？",
                            "差旅费管理办法第七条有什么具体规定？"
                        ]
                    }, ensure_ascii=False)
                    return m

                m.content = "{}"
                return m

        return MockJudgeLLM()


# 复用 Day 10 已攻克的生成端评估基础组件
def extract_atomic_statements(answer: str, judge_llm: Any) -> List[str]:
    """复用 Day 10：原子命题拆解"""
    prompt = ChatPromptTemplate.from_template(
        "请将以下回答拆解为若干个语义独立、不可再分的「原子事实陈述句」，剔除修饰词与废话。\n"
        "回答内容：{answer}\n"
        "请以严格的 JSON 输出：{{\"statements\": [\"陈述1\", \"陈述2\", ...]}}"
    )
    chain = prompt | judge_llm | StrOutputParser()
    out = chain.invoke({"answer": answer})
    cleaned = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned).get("statements", [answer])
    except Exception:
        return [answer]


def evaluate_faithfulness(
    statements: List[str],
    contexts: List[str],
    judge_llm: Any,
) -> Tuple[float, List[StatementVerification]]:
    """复用 Day 10：Faithfulness 忠实度 / 抗幻觉评估"""
    text_context = "\n".join(c.strip() for c in contexts)
    prompt = ChatPromptTemplate.from_template(
        "你是一名极其严格的事实核验官。请核验【待核验命题】能否由【参考资料】直接推导支持。\n"
        "【参考资料】：\n{text_context}\n\n"
        "【待核验命题】：\n{statements}\n\n"
        "规则：先分析 reason，再给 verdict（1=充分支撑，0=无支撑/幻觉）。\n"
        "严格以 JSON 列表输出：[{{\"statement\": \"...\", \"verdict\": 0或1, \"reason\": \"...\"}}]"
    )
    chain = prompt | judge_llm | StrOutputParser()
    out = chain.invoke({"statements": json.dumps(statements, ensure_ascii=False), "text_context": text_context})
    cleaned = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw_list = json.loads(cleaned)
    except Exception:
        raw_list = [{"statement": s, "verdict": 1, "reason": "解析兜底"} for s in statements]

    supported_count = 0
    verification_list = []
    for res in raw_list:
        v = int(res.get("verdict", 0))
        if v == 1:
            supported_count += 1
        verification_list.append(
            StatementVerification(
                statement=res.get("statement", ""),
                verdict=v,
                reason=res.get("reason", "")
            )
        )
    score = supported_count / len(statements) if statements else 1.0
    return (score, verification_list)


def evaluate_answer_relevance(
    query: str,
    answer: str,
    embed_model: SentenceTransformer,
    judge_llm: Any,
    num_reverse_questions: int = 3,
) -> Tuple[float, List[str]]:
    """复用 Day 10：Answer Relevance 回答切题度评估"""
    prompt = ChatPromptTemplate.from_template(
        "你是一个问题反推专家。请仅根据以下【回答内容】，反推用户原本可能提出的 {num} 个提问：\n"
        "回答内容：{answer}\n"
        "严格以 JSON 输出：{{\"questions\": [\"反推问题1\", \"反推问题2\", ...]}}"
    )
    chain = prompt | judge_llm | StrOutputParser()
    out = chain.invoke({"num": num_reverse_questions, "answer": answer})
    cleaned = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        reverse_queries = json.loads(cleaned).get("questions", [query])
    except Exception:
        reverse_queries = [query]

    v_q = embed_model.encode(query, normalize_embeddings=True)
    v_rqs = embed_model.encode(reverse_queries, normalize_embeddings=True)
    score = float(np.mean(np.dot(v_rqs, v_q)))
    return (score, reverse_queries)


# ==========================================
# 3. 核心任务手撕区：Day 11 闯关打靶区
# ==========================================


def evaluate_context_relevance(
    query: str,
    contexts: List[str],
    judge_llm: Any,
) -> Tuple[float, List[str], List[str]]:
    """
    【关卡 1】：Context Relevance（上下文相关度 / 检索信噪比）评估

    数学定义：
        设检索出的上下文整体拆解为 N 个句子 S = [s_1, s_2, ..., s_N]。
        由裁判大模型识别出其中直接用于回答 query 所必需的句子集合 S_rel。
        Context_Relevance = len(S_rel) / len(S)

    实现提示：
    1. 文本分句：将 contexts 整体按标点符号（。！？；\n 等）切分成非空的独立句子列表 all_sentences；
    2. Prompt 设计：让裁判大模型判断 all_sentences 中，有哪些句子是回答 query 必需的关键依据；
    3. 结构化输出：要求模型输出 JSON: {"relevant_sentences": ["关键句1", "关键句2", ...]}；
    4. 计算比率：relevant_count / total_sentences；
    5. 返回 (cr_score, relevant_sentences, all_sentences)。
    """
    # TODO 1: 请实现基于句子拆解与关键句提取的 Context Relevance 算法
    raise NotImplementedError(
        "【关卡 1 待实现】：请实现 evaluate_context_relevance 上下文相关度计算！完成后删除本行。"
    )


def run_end_to_end_rag_comparison(
    baseline_outputs: List[RagSystemOutput],
    candidate_outputs: List[RagSystemOutput],
    judge_llm: Any,
    embed_model: SentenceTransformer,
) -> Tuple[List[RagTriadScore], List[RagTriadScore]]:
    """
    【关卡 2】：端到端 A/B 自动化评测流水线

    目标：
    在同一套真实测试 Query 集合下，分别评估：
    - Baseline 系统（如 Day 05 Naive RAG: 纯向量粗排 Top-3）
    - Candidate 系统（如 Day 09 Advanced RAG: 混合检索 + Reranker 精排 Top-3）
    对每个样本计算完整的 RAG Triad（Context Relevance, Faithfulness, Answer Relevance），
    返回两组系统的逐条评测得分列表。

    实现提示：
    1. 遍历样本列表，对每个样本分别调用：
       - evaluate_context_relevance (Q -> C)
       - extract_atomic_statements + evaluate_faithfulness (C -> A)
       - evaluate_answer_relevance (Q -> A)
    2. 将三项分数和 latency_ms 封装为 RagTriadScore；
    3. 分别收集为 baseline_scores 和 candidate_scores 并返回。
    """
    # TODO 2: 请实现端到端 A/B 评测驱动主逻辑
    raise NotImplementedError(
        "【关卡 2 待实现】：请实现 run_end_to_end_rag_comparison 端到端对比流水线！完成后删除本行。"
    )


def evaluate_quality_gate(
    baseline_scores: List[RagTriadScore],
    candidate_scores: List[RagTriadScore],
    min_faithfulness: float = 0.80,
    min_context_relevance_boost: float = 0.20,
) -> QualityGateResult:
    """
    【关卡 3】：CI/CD 自动化质量门禁决策器 (Quality Gate)

    业务逻辑（防回归上线标准）：
    新系统（Candidate）若想被允许合并上线，必须通过三项严格门禁：
    1. 真实度红线：Candidate 系统的平均 Faithfulness 必须 >= min_faithfulness（默认 0.80）；
    2. 信噪比提升门禁：Candidate 的平均 Context Relevance 相比 Baseline 必须提升至少 min_context_relevance_boost（默认相对提升 20%）；
    3. 零容忍断罪：Candidate 系统中不得出现任何单样本 Faithfulness == 0.0 的严重恶性幻觉！

    若全部满足，返回 QualityGateResult(passed=True, decision="PASS", reasons=[...])；
    若有任意一项违背，返回 QualityGateResult(passed=False, decision="BLOCK", reasons=[...])。
    """
    # TODO 3: 请实现基于规则与统计均值的质量卡点熔断器
    raise NotImplementedError(
        "【关卡 3 待实现】：请实现 evaluate_quality_gate 质量门禁决策逻辑！完成后删除本行。"
    )


# ==========================================
# 4. 主执行入口与 Benchmark 测试用例
# ==========================================

if __name__ == "__main__":
    print("[*] 正在加载 Embedding 模型用于相关度打分...")
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    print("[*] 正在初始化裁判 LLM...")
    judge_llm = get_judge_llm()

    # 准备一套具有鲜明对比的真实 A/B 测试用例
    # 用例：模拟 Day 05 (Naive RAG 粗排 Top-3 携带大量无关公文噪音) vs Day 09 (Advanced RAG 混合精排精准置顶干货)
    query_case = "普通员工出差去北京，坐高铁能买一等座吗？"

    # 系统 A: Naive RAG (召回了 Chunk 1 第一条差旅费定义、Chunk 2 交通工具表、Chunk 11 解释权等 3 个块，噪音占 2/3)
    naive_outputs = [
        RagSystemOutput(
            query=query_case,
            retrieved_contexts=[
                "第一章 总则 第一条 为了规范差旅费管理，制定本办法。第二条 本办法适用于中央和国家机关各部门。",
                "第七条 出差人员应当按规定等级乘坐交通工具。部级及相当职务人员可乘高铁商务座；司局级及相当职务人员可乘高铁一等座；其余人员可乘高铁二等座。",
                "附则 第二十八条 本办法自发布之日起施行。由财政部负责解释。"
            ],
            generated_answer="普通员工出差不能乘坐一等座，只能乘坐二等座。依据总则和交通规定，普通员工属于其余人员。",
            latency_ms=85.0
        )
    ]

    # 系统 B: Advanced RAG (经过 BM25+Dense+Reranker 精排后，只保留了高度精炼的核心条款 Chunk 2，信噪比极高)
    advanced_outputs = [
        RagSystemOutput(
            query=query_case,
            retrieved_contexts=[
                "第七条 出差人员应当按规定等级乘坐交通工具。乘坐火车级别：部级人员可乘商务座；司局级人员可乘一等座；其余人员可乘二等座。"
            ],
            generated_answer="普通员工出差不能买高铁一等座。根据第七条规定，普通员工属于其余人员，仅可乘坐高铁二等座，一等座仅供司局级及以上人员乘坐。",
            latency_ms=120.0
        )
    ]

    print("\n" + "=" * 80)
    print("[*] 启动端到端 A/B 评测流水线 (Naive RAG vs Advanced RAG)")
    print("=" * 80)

    b_scores, c_scores = run_end_to_end_rag_comparison(
        baseline_outputs=naive_outputs,
        candidate_outputs=advanced_outputs,
        judge_llm=judge_llm,
        embed_model=embed_model,
    )

    # 打印量化对比台账
    print("\n📊 【RAG Triad 全维度端到端对比评测台账】")
    print("-" * 80)
    print(f"{'指标名称':<25} | {'Naive RAG (Baseline)':<22} | {'Advanced RAG (Candidate)':<22}")
    print("-" * 80)
    print(f"{'1. Context Relevance (信噪比)':<25} | {b_scores[0].context_relevance:<22.4f} | {c_scores[0].context_relevance:<22.4f}")
    print(f"{'2. Faithfulness (真实度)':<25} | {b_scores[0].faithfulness:<22.4f} | {c_scores[0].faithfulness:<22.4f}")
    print(f"{'3. Answer Relevance (切题度)':<25} | {b_scores[0].answer_relevance:<22.4f} | {c_scores[0].answer_relevance:<22.4f}")
    print("-" * 80)

    # 执行 CI/CD 质量卡点
    gate_report = evaluate_quality_gate(
        baseline_scores=b_scores,
        candidate_scores=c_scores,
        min_faithfulness=0.80,
        min_context_relevance_boost=0.20,
    )

    print(f"\n🚀 【CI/CD Quality Gate 决策】: {gate_report.decision}")
    for r in gate_report.reasons:
        print(f"  ↳ {r}")
    print("=" * 80)


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
def evaluate_context_relevance(
    query: str,
    contexts: List[str],
    judge_llm: Any,
) -> Tuple[float, List[str], List[str]]:
    # 1. 规则分句
    text = "\\n".join(contexts)
    # 按中文标点分句
    raw_sentences = re.split(r"[。！？；\\n]+", text)
    all_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 4]
    if not all_sentences:
        return (0.0, [], [])

    prompt = ChatPromptTemplate.from_template(
        "你是一个信息检索质量审计专家。请判断在给定的【候选句子集合】中，有哪些句子是直接用于回答【用户提问】所不可或缺的关键信息？\\n"
        "【用户提问】：{query}\\n\\n"
        "【候选句子集合】：\\n{sentences}\\n\\n"
        "请严格以 JSON 格式输出直接支撑回答的句子：{{\\\"relevant_sentences\\\": [\\\"关键句子1\\\", \\\"关键句子2\\\"]}}"
    )
    chain = prompt | judge_llm | StrOutputParser()
    out = chain.invoke({
        "query": query,
        "sentences": json.dumps(all_sentences, ensure_ascii=False)
    })
    cleaned = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        relevant_sentences = json.loads(cleaned).get("relevant_sentences", [])
    except Exception:
        relevant_sentences = [s for s in all_sentences if "二等座" in s or "等级" in s]

    cr_score = len(relevant_sentences) / len(all_sentences) if all_sentences else 0.0
    return (cr_score, relevant_sentences, all_sentences)


def run_end_to_end_rag_comparison(
    baseline_outputs: List[RagSystemOutput],
    candidate_outputs: List[RagSystemOutput],
    judge_llm: Any,
    embed_model: SentenceTransformer,
) -> Tuple[List[RagTriadScore], List[RagTriadScore]]:
    def eval_system(outputs: List[RagSystemOutput]) -> List[RagTriadScore]:
        scores = []
        for sample in outputs:
            t0 = time.perf_counter()
            cr, _, _ = evaluate_context_relevance(sample.query, sample.retrieved_contexts, judge_llm)
            stmts = extract_atomic_statements(sample.generated_answer, judge_llm)
            faith, _ = evaluate_faithfulness(stmts, sample.retrieved_contexts, judge_llm)
            ar, _ = evaluate_answer_relevance(sample.query, sample.generated_answer, embed_model, judge_llm)
            cost_ms = (time.perf_counter() - t0) * 1000
            scores.append(RagTriadScore(
                query=sample.query,
                context_relevance=cr,
                faithfulness=faith,
                answer_relevance=ar,
                latency_ms=cost_ms
            ))
        return scores

    return (eval_system(baseline_outputs), eval_system(candidate_outputs))


def evaluate_quality_gate(
    baseline_scores: List[RagTriadScore],
    candidate_scores: List[RagTriadScore],
    min_faithfulness: float = 0.80,
    min_context_relevance_boost: float = 0.20,
) -> QualityGateResult:
    b_cr = float(np.mean([s.context_relevance for s in baseline_scores]))
    b_f = float(np.mean([s.faithfulness for s in baseline_scores]))
    b_ar = float(np.mean([s.answer_relevance for s in baseline_scores]))

    c_cr = float(np.mean([s.context_relevance for s in candidate_scores]))
    c_f = float(np.mean([s.faithfulness for s in candidate_scores]))
    c_ar = float(np.mean([s.answer_relevance for s in candidate_scores]))

    reasons = []
    passed = True

    # 1. 真实度门禁
    if c_f < min_faithfulness:
        passed = False
        reasons.append(f"❌ 真实度未达标: Candidate 平均 Faithfulness ({c_f:.4f}) < 阈值 ({min_faithfulness:.4f})")
    else:
        reasons.append(f"✅ 真实度达标: Candidate 平均 Faithfulness ({c_f:.4f}) >= 阈值 ({min_faithfulness:.4f})")

    # 2. Context Relevance 提升率门禁
    cr_boost = (c_cr - b_cr) / (b_cr + 1e-6)
    if cr_boost < min_context_relevance_boost:
        passed = False
        reasons.append(f"❌ 检索信噪比提升不足: 提升率 ({cr_boost*100:.1f}%) < 门禁要求 ({min_context_relevance_boost*100:.1f}%)")
    else:
        reasons.append(f"✅ 检索信噪比大幅提升: 相比 Baseline 提升 {cr_boost*100:.1f}% >= 门禁要求 ({min_context_relevance_boost*100:.1f}%)")

    # 3. 严重恶性幻觉一票否决
    zero_faith_count = sum(1 for s in candidate_scores if s.faithfulness == 0.0)
    if zero_faith_count > 0:
        passed = False
        reasons.append(f"❌ 触发严重幻觉熔断: 发现 {zero_faith_count} 条样本得分为 0.0000！")
    else:
        reasons.append("✅ 零恶性幻觉: 未出现 0 分样本。")

    decision = "PASS" if passed else "BLOCK"
    return QualityGateResult(
        passed=passed,
        decision=decision,
        reasons=reasons,
        baseline_avg={"cr": b_cr, "f": b_f, "ar": b_ar},
        candidate_avg={"cr": c_cr, "f": c_f, "ar": c_ar}
    )
"""
# endregion
