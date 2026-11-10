# AI Agent 算法工程师 · 周末版学习计划（20 个周末）

> **来源文档**：https://github.com/adongwanai/AgentGuide/blob/main/docs/05-roadmaps/learning-roadmap-algorithm.md
> **原计划**：9 周 / 65 天（全职每日制）
> **本计划**：压缩为每周末一次、每次约 3 天内容（2-4 天弹性），共 20 个周末，约 5 个月
> **节奏**：每周六 09:00 推送学习单，周六 + 周日两个学习日完成

## 编排原则

1. **3 天为基准，内容边界优先**：不把同一篇论文的精读拆到两个周末，也不把「理论 + 手撕」切开；因此实际区间在 2-4 天之间浮动。
2. **每个周末必须有硬产出**：可运行的代码、流程图、伪代码推导或笔记中的一种，不接受「只看视频」。
3. **复盘周末不单独占位**：原文档的预留/复盘日并入同主题周末的第二学习日。
4. **项目期（D43-D65）按里程碑切**：编码类周末只要求跑通一个明确里程碑，不追求进度数字。

## 周末总览

| 周末 | 原文档天数 | 主题 | 阶段 |
| --- | --- | --- | --- |
| W01 | D1-3 | Python/PyTorch 基础 + 手撕神经网络 | 一、基础 |
| W02 | D4-7 | Transformer 架构全解剖 + 手撕 Multi-Head Attention | 一、基础 |
| W03 | D8-10 | Agent 核心概念 + ReAct 精读与复现 | 二、Agent 理论 |
| W04 | D11-14 | ReAct 算法反思 + 周度复盘 | 二、Agent 理论 |
| W05 | D15-17 | Reflexion 精读 + ToT 精读 | 三、高级架构 |
| W06 | D18-21 | ToT 实战（24 点）+ Multi-Agent 协作 | 三、高级架构 |
| W07 | D22-24 | BM25 + DPR + Reranker/混合检索 | 四、RAG |
| W08 | D25-28 | GraphRAG + Agentic RAG + RAG 评估 | 四、RAG |
| W09 | D29-31 | Agent Memory 概述 + Generative Agents | 五、记忆 |
| W10 | D32-35 | 记忆压缩/上下文工程 + 分层记忆设计 | 五、记忆 |
| W11 | D36-38 | RL 基础 + PPO + DPO 推导 | 六、RL 对齐 |
| W12 | D39-42 | DPO 实战 + GRPO + RL for Tool + 奖励模型 | 六、RL 对齐 |
| W13 | D43-46 | 项目选题 + 文献调研 + 实验方案设计 | 七、项目 |
| W14 | D47-49 | 实验框架 + Baseline 实现 | 七、项目 |
| W15 | D50-52 | Baseline 跑通 + 核心算法模块起步 | 七、项目 |
| W16 | D53-55 | 核心创新算法实现完成 | 七、项目 |
| W17 | D56-58 | 全量实验 + 消融 + 结果可视化 | 七、项目 |
| W18 | D59-61 | 论文 Method/Exp + Intro/Conclusion | 七、项目 |
| W19 | D62-64 | 开源与博客 + 简历打磨 + 项目逐字稿 | 八、冲刺 |
| W20 | D65 + 全链路复盘 | 模拟面试 + 65 天总复盘 | 八、冲刺 |

---

## 分周末详情

### W01 · D1-3 · 环境与手撕第一课
- **内容**：Python 核心语法、NumPy/Pandas；神经网络前向/反向传播、损失函数；PyTorch Tensor、自动求导、训练循环；从零实现前馈网络
- **必读/必看**：《动手学深度学习》前 3 章、3Blue1Brown 线性代数精髓、Karpathy *Neural Networks: Zero to Hero* 第 1-2 集
- **硬产出**：一份纯 NumPy 实现的两层 MLP（手写 forward + backward，不用 autograd），在 MNIST 子集上训练到 >90% 准确率
- **验收**：能口述梯度推导中每个矩阵的维度变化

### W02 · D4-7 · Transformer 全解剖
- **内容**：Encoder/Decoder 结构、Self-Attention、Multi-Head Attention；Positional Encoding、LayerNorm、残差、FFN；MoE（专家网络、门控、Top-K）初探；用 Excel 复算一次注意力矩阵
- **必读**：*Attention Is All You Need*
- **硬产出**：手撕 Multi-Head Attention（含 mask 分支），并与 `nn.MultiheadAttention` 数值对齐（误差 < 1e-5）；Excel 里复现 4×4 注意力权重
- **验收**：能解释 Bert/T5/GPT 三者架构差异，以及为什么 Decoder 需要 causal mask

### W03 · D8-10 · Agent 与 ReAct
- **内容**：什么是 Agent；Planning/Memory/Tool Use 三组件；Agent vs LLM vs RAG 的本质区别；ReAct 的 Thought-Action-Observation 循环
- **必读**：ReAct（arXiv 2210.03629）、LLM Powered Autonomous Agents
- **硬产出**：手绘 ReAct 算法流程图 + 基于 LangChain（或 Qwen-Agent）跑通一个带 2 个工具的 ReAct Agent
- **验收**：能回答「ReAct 与 CoT 的本质区别」

### W04 · D11-14 · ReAct 反思与复盘
- **内容**：ReAct 的失败模式（循环、工具误用、幻觉动作）；对比 Lagent/Qwen-Agent 实现差异；第 1-4 周知识复盘
- **硬产出**：一篇 800 字以内的 ReAct 局限性分析（含至少 2 个你自己复现出的失败案例）；第 1 周 MLP 与第 2 周 MHA 代码合并成一个可复用模块
- **验收**：能说出 3 种改进 ReAct 的思路及各自的代价

### W05 · D15-17 · Reflexion 与 ToT
- **内容**：Reflexion 的 Actor→Evaluator→Self-Reflection 流程；ToT 把 LLM 当启发式函数的搜索框架；Self-Consistency
- **必读**：Reflexion（2303.11366）、Tree of Thoughts（2305.10601）、Self-Refine
- **硬产出**：Reflexion 伪代码实现（含记忆缓冲区）+ 手绘 ToT 搜索树示意
- **验收**：能讲清 Reflexion 与 RL 中 credit assignment 的对应关系

### W06 · D18-21 · ToT 实战与 Multi-Agent
- **内容**：用 ToT 解 24 点游戏；MetaGPT 的 SOP 角色定义、Communicative Agents、AutoGen 对话式多智能体
- **硬产出**：可运行的 ToT 24 点求解器（带剪枝和自评函数）；AutoGen 版 coder+critic 双 Agent 协作 demo
- **验收**：对比 ReAct / Reflexion / ToT 三者的搜索空间与成本

### W07 · D22-24 · 检索基础三件套
- **内容**：TF-IDF 与 BM25 原理；DPR 双编码器；ColBERT 延迟交互；Reranker 与 Hybrid Search
- **必读**：DPR（2004.04906）、TF-IDF、ColBERT
- **硬产出**：纯 Python 手写 BM25（不用现成库）+ FAISS 向量索引 + BM25&Embedding 混合检索管线
- **验收**：能在小语料上给出三种检索方式的 Recall@K 对比表

### W08 · D25-28 · 前沿 RAG 与评估
- **内容**：GraphRAG 的子图采样与路径排序；Agentic RAG 与多跳推理；Self-RAG 的自主检索决策；RAGAs 评估
- **必读**：GraphRAG 技术报告、Self-RAG（2310.11511）、Modular RAG
- **硬产出**：用 RAGAs 对第 7 周的 RAG 打分（faithfulness/answer relevancy/context recall 三项）+ Agentic RAG 查询规划模块伪代码
- **验收**：能说明 GraphRAG 相对于向量 RAG 的收益边界

### W09 · D29-31 · 记忆机制入门
- **内容**：短期 vs 长期记忆；记忆分类与挑战；Generative Agents 的 Recency/Importance/Relevance 评分与检索
- **必读**：Generative Agents（2304.03442）、Agent Memory 综述
- **硬产出**：实现一个三维加权的记忆检索函数（含时间衰减），并用构造数据验证排序合理性
- **验收**：能解释为什么单纯的向量相似度检索在长期对话中会失效

### W10 · D32-35 · 上下文工程
- **内容**：记忆压缩与总结（聚类+摘要+去重）；上下文选择策略、层级笔记、QA 对转换；MemGPT 的虚拟上下文管理；分层记忆架构设计
- **必读**：MemGPT（2310.08560）
- **硬产出**：分层记忆（Event Memory + Working Memory）完整设计文档 + 可运行的检索 demo；自定义一个 LlamaIndex NodePostprocessor
- **验收**：给出压缩率与信息保留率的实测数字

### W11 · D36-38 · RL 与偏好对齐基础
- **内容**：MDP、Policy、Value Function、Policy Gradient；PPO 目标函数与裁剪；DPO 如何从偏好数据隐式学奖励
- **必读**：PPO、DPO（2305.18290）
- **硬产出**：**手推 DPO 损失函数**（从 Bradley-Terry 模型出发，拍照上传推导过程）；PPO clip 机制的数值小实验
- **验收**：能解释 DPO 为什么不需要 reward model，以及它的隐含假设

### W12 · D39-42 · 对齐实战与 GRPO
- **内容**：TRL DPOTrainer 跑通一次 DPO；GRPO 的组内相对偏好；Toolformer 与 RL for Tool Learning；奖励模型训练
- **必读**：GRPO（2402.03300）、DeepSeek-R1、Toolformer
- **硬产出**：一次完整 DPO 训练（小模型 + 小数据集，记录 loss 曲线与前后对比样本）+ 一个 Agent 工具调用任务的奖励函数设计
- **验收**：能比较 PPO/DPO/GRPO 在 Agent 场景的取舍

### W13 · D43-46 · 项目启动
- **内容**：在「Agentic RAG with Self-Correction」与「Hierarchical Memory Agent」中选定一个；精读 5-7 篇核心论文；画算法流程图；确定数据集、Baseline、指标、消融方案
- **硬产出**：项目 README（问题定义/创新点/实验设计）+ Related Work 初稿 + 算法流程图
- **验收**：能用一段话讲清「我的方法比 baseline 好在哪，用什么指标证明」

### W14 · D47-49 · 实验框架与 Baseline
- **内容**：数据加载与预处理、评估脚本、日志与实验管理；实现 Naive RAG / ReAct / Sliding Window 等 Baseline
- **硬产出**：可一键跑通的 experiment 入口脚本 + Baseline 指标基线数字
- **验收**：换机器/换随机种子能复现同一结果

### W15 · D50-52 · 核心模块起步
- **内容**：实现创新算法的第一个核心模块（Self-Correction 判定器 或 Hierarchical Memory 写入/检索）
- **硬产出**：核心模块单测 + 小规模端到端跑通
- **验收**：模块可独立开关（为后续消融实验留好接口）

### W16 · D53-55 · 算法实现完成
- **内容**：补齐第二个创新模块（Iterative Retrieval 或 Autonomous Summarization），打通完整 pipeline
- **硬产出**：完整方法可跑，主表指标首次出数
- **验收**：相对 Baseline 已有非平凡差异（不论正负），能解释原因

### W17 · D56-58 · 实验与消融
- **内容**：全量实验、消融实验、多组随机种子、效率指标（查询次数、检索延迟、压缩率）；图表可视化
- **硬产出**：主实验结果表 + 消融表 + 3 张核心图（含误差棒）
- **验收**：每个创新模块的贡献有单独可量化的证据

### W18 · D59-61 · 论文写作
- **内容**：Method 与 Experiments 章节撰写；Introduction、Abstract、Conclusion；Overleaf ACL 模板排版
- **硬产出**：论文初稿（Method + Exp + Intro + Conclusion 齐全，图表就位）
- **验收**：Method 部分能让同行不看代码复现出算法

### W19 · D62-64 · 开源与求职材料
- **内容**：代码整理与注释、README 与技术博客；按 STAR-算法版重写简历项目经历；3-5 分钟项目介绍逐字稿
- **硬产出**：GitHub 仓库（README + 示例）+ 技术博客一篇 + 简历项目段 + 逐字稿
- **验收**：README 能让陌生人在 10 分钟内跑起来

### W20 · D65 + 总复盘 · 面试冲刺
- **内容**：模拟项目深挖（「为什么不用 XX 方法」「算法局限在哪」）；65 天全链路知识地图复盘；补齐短板
- **硬产出**：一份自问自答题库（≥30 题）+ 一张覆盖 9 周的知识地图
- **验收**：能连续 20 分钟被追问而不掉线

---

## 使用说明

- 进度记录在 `learning/progress.md`，每次推送后自动 +1。
- 学习单文件命名：`learning/weekend-W##-学习单.md`。
- 若某个周末没完成，直接在下次推送时告诉助手「W## 未完成」，助手会把内容并入当周学习单（不跳过）。

### 漏推送怎么补

定时任务是**本地执行**：周六 09:00 电脑关机、客户端未运行、系统休眠或断网，这次触发就直接错过，不会自动补跑。
因为学哪个周末由 `progress.md` 的指针决定、不按日期推算，所以手动补跑不会错位：

> 周日（或其他任意时间）打开电脑后，在对话里说一句 **「补跑本周末」** 即可。

助手会读 `progress.md` 取当前指针，生成同一份 W## 学习单，并把指针 +1。连续漏两周就说「补跑本周末，上周也没做」，会把两周内容合并（总量可到 4-5 天）。

### 手动触发会怎样

- **手动触发不会取消定时**：立即运行一次后，定时规则照旧，本周六 09:00 仍会再跑一次。
- **手动触发会消耗进度**：每次正常运行都会让指针 +1，所以提前手动跑一次 = 该周末被提前发掉，到周六就变成下一个 W##，整体进度前移一周。
- **只想看格式**：说「试跑」或「不要推进进度」，只在对话里输出学习单，不写文件、不动指针。
- **同周重复触发**：若该编号学习单已存在且本周已推送过，只提示「已推送」，不重复生成、不推进指针；要重做请说「重做 W##」。

