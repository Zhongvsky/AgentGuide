# 主流 AI Agent 与 Harness 记忆模块架构设计与前沿演进深度调研报告 (2026)

**从声明式文件层级、渐进式上下文压缩、双时态知识图谱到睡眠期巩固（Sleep-Time Compute）与强化学习原生记忆策略的全景解析**

**调研基线**：2026 年 10 月最新工程与学术基线  
**覆盖系统**：Claude Code (v2.1+)、OpenAI Codex CLI (0.159+)、OpenCode、Cursor 3、Devin Desktop / Cascade、OpenHands SDK V1、Letta (MemGPT 2.0)、Mem0 / Mem0g、Zep / Graphiti、ByteRover 2.1.5、Cognee、LangMem、Supermemory  
**核心文献**：arXiv:2512.13564、arXiv:2604.14228、AgeMem (arXiv:2601.01885)、A-Mem (arXiv:2502.12110)、Memory-R1 (arXiv:2508.19828)、MemRL (arXiv:2601.03192)、MemEvolve (arXiv:2512.18746)、Mem^p (arXiv:2508.06433)、Titans (arXiv:2501.00663)、MINJA / A-MemGuard

## 核心调研摘要 (Executive Abstract)

随着大语言模型（LLM）从单轮对话助手全面演进为能够执行数小时长程任务的自主智能体（Autonomous Agents），决定智能体生产级可靠性的核心瓶颈已从“模型单步推理能力”转向“运行时容器（Agent Harness）的记忆与状态管理架构”。尽管前沿基础模型纷纷提供百万级 Token 的上下文窗口，但实证研究表明，纯上下文窗口堆砌（Full-Context Stuffing）在超过 50 轮交互后会出现严重的“中间迷失（Lost in the Middle）”注意力退化、平方级算力膨胀以及 Prompt Cache 频繁失效，在 [LoCoMo 跨会话长程记忆基准](https://niteagent.com/blog/agent-memory-research-2026/)上的召回准确率骤降至 35% 左右。

本报告基于 2025 至 2026 年间发表的核心学术文献（包括 [Memory in the Age of AI Agents (arXiv:2512.13564)](https://arxiv.org/abs/2512.13564)、[Dive into Claude Code (arXiv:2604.14228)](https://arxiv.org/abs/2604.14228)、[AgeMem (arXiv:2601.01885)](https://arxiv.org/abs/2601.01885)、[A-Mem (arXiv:2502.12110)](https://arxiv.org/abs/2502.12110)、[Mem^p (arXiv:2508.06433)](https://arxiv.org/abs/2508.06433) 及 [Titans (arXiv:2501.00663)](https://arxiv.org/abs/2501.00663)）与主流生产级工程实践，系统拆解了三类核心架构：第一，以 [Claude Code](https://code.claude.com/docs/en/memory)、[OpenAI Codex CLI](https://kangwooklee.com/blogs/codex_context_compaction.html)、[OpenCode](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)、[Cursor 3](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)、[Devin Desktop (原 Windsurf)](https://omidsaffari.com/blog/windsurf-vs-cursor-after-devin-desktop) 和 [OpenHands SDK](https://arxiv.org/abs/2511.03690) 为代表的内置 Harness 上下文与压缩生命周期设计；第二，以 [Letta (MemGPT 2.0)](https://www.letta.com/blog/sleep-time-compute/)、[Mem0](https://arxiv.org/pdf/2504.19413)、[Zep / Graphiti](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)、[ByteRover Context Tree](https://niteagent.com/blog/agent-memory-research-2026/) 与 [Cognee](https://thegenios.com/blog/open-source-memory-layers-2026/) 为代表的独立记忆中间件；第三，涵盖睡眠期计算与“做梦（Dreaming）”记忆巩固、强化学习端到端记忆策略（Memory-R1 / MemRL）、程序性技能蒸馏、测试时参数化记忆及防投毒安全机制的六大前沿演进方向。

## 一、 范式重构：突破“无限上下文幻觉”与三维记忆系统分类学

在 2025 年至 2026 年的智能体工程演进中，业界形成了一项至关重要的工程共识：**“AI 智能体不是提示词，而是运行时底座（AI agents are not prompts; they are harnesses）”**（[Agentic Harness Engineering, 2026](https://medium.com/@adnanmasood/agentic-harness-engineering-the-operating-system-for-machine-intelligence-ce4fa48e9520)）。正如 [freeCodeCamp 对 Agent Harness 架构的剖析](https://www.freecodecamp.org/news/what-is-an-agent-harness/)所指出，一个完整的 Agent Harness 是围绕无状态基础模型构建的运行时外壳（Runtime Shell），由五个不可或缺的核心子系统组成：控制循环（Loop）、工具路由器（Tool Router）、记忆系统（Memory）、规划器（Planning）与沙箱边界（Sandbox Boundary）。在 [2026 Agent System and Harness Design 综述](https://github.com/FlorianBruniaux/claude-code-ultimate-guide/blob/main/guide/ecosystem/agent-harness-landscape.md)与 [Harness-Bench](https://medium.com/@adnanmasood/agentic-harness-engineering-the-operating-system-for-machine-intelligence-ce4fa48e9520) 定义的四平面模型中，状态平面（State Plane）专门负责管理草稿纸（Scratchpads）、待办清单（Todo Files）、情景与语义记忆以及跨会话持久化；实证对比实验（The Scaffold Effect，跨 2 个模型与 3 个 Harness 的 300 次受控运行）表明，不同 Harness 的上下文与记忆脚手架设计在解决相同任务时可产生高达 **40 倍的单任务 Token 消耗差异**。

### 1.1 为什么百万级上下文窗口无法替代专用记忆架构？

随着 Gemini 1.5/2.5 Pro、Claude 3.7/4 系列及 GPT-4.1/5 等前沿模型将标称上下文窗口扩展至 20 万乃至 200 万 Token，部分工程团队曾试图放弃外部记忆模块，转而采用将全部历史对话与工具输出直接追加进上下文窗口的“全量上下文堆砌（Full-Context Stuffing）”策略。然而，生产环境的真实评测与学术研究揭示了这一做法背后的“无限上下文幻觉（The Infinite Context Illusion）”（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)）：

首先是**注意力退化与“上下文焦虑（Context Anxiety）”**。根据 [Liu et al. (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) 揭示的“Lost in the Middle”效应，Transformer 架构对位于长序列首尾两端的信息召回率较高，但对处于长上下文中段的历史事实与约束条件会出现显著的注意力稀释。在编程智能体的实际运行中，当上下文窗口接近饱和时，模型还会表现出明显的“上下文焦虑”——即倾向于忽略系统安全检查、跳过测试验证并草率结束任务（[Justin3go, 2026](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)）。在 [LoCoMo 长会话评测](https://niteagent.com/blog/agent-memory-research-2026/)（涵盖多达 35 个会话、300 至 10,000+ 轮交互）中，即便底层模型具备 200K 上下文窗口，全量上下文基线（Full-Context Baseline）的事实召回准确率也仅能维持在 **35% 左右**，且一旦交互超过 50 轮，其准确率便断崖式跌破 50% 分水岭；而在 100 万 Token 规模的 [BEAM 1M 基准](https://niteagent.com/blog/agent-memory-research-2026/)下，全量上下文基线更是崩溃至 **18%**。

其次是**算力成本与键值缓存（KV Cache / Prompt Cache）失效问题**。自注意力机制的计算复杂度随序列长度呈平方级增长（$O(N^2)$）。在每次工具调用动辄产生数千 Token 输出的编码或深度调研场景中，若不加筛选地保留全部历史，单次长会话的输入 Token 成本可达数美元。更严重的是，现代推理服务高度依赖前缀键值缓存（Prefix Prompt Caching，缓存命中部分的价格通常仅为未缓存输入的 1/10，例如 [Claude Sonnet 5 在 Cursor 中的定价](https://omidsaffari.com/blog/windsurf-vs-cursor-after-devin-desktop)为未缓存输入 $3/M tokens，而缓存命中仅需 $0.30/M tokens）。若采用简单的滑动窗口（Sliding Window）丢弃最早的对话轮次，会导致整个请求前缀发生偏移，瞬间击穿全部 Prompt Cache，迫使服务端在每一轮交互中从头执行昂贵的 Prefill 预填充计算。

### 1.2 超越“长短期二分法”：2026 智能体记忆三维统一分类学

长期以来，AI 社区习惯沿用粗糙的“短期记忆（上下文窗口）vs. 长期记忆（外部向量库）”二分法。然而，由 47 位研究者联合撰写的里程碑式综述论文 [《Memory in the Age of AI Agents: A Survey》(Hu et al., arXiv:2512.13564, 2026)](https://arxiv.org/abs/2512.13564) 指出，传统二分法已无法指导现代智能体系统的工程设计。该综述首先严格划定了**“智能体记忆（Agent Memory）”与“传统 RAG（检索增强生成）”的本质边界**：RAG 本质上是面向静态外部知识库（如企业文档、代码仓、网页快照）的只读检索系统；而智能体记忆则是面向智能体自身交互演化的动态状态机，必须持续追踪“智能体做过什么、用户偏好发生了何种改变、哪条工具调用路径曾导致失败”。将静态向量库直接当作智能体记忆使用，虽然能完成字面相似片段召回，但在面对时序推理（如“在执行那次报错的部署脚本之前，用户对数据库配置提出了什么修改意见？”）和状态覆写（如用户从偏好 Python 转向偏好 Rust）时会彻底失效。实际上，[Hu et al. (2026)](https://arxiv.org/abs/2512.13564) 对 47 个现有智能体记忆系统的排查发现，仅有 **4 个系统**原生支持事实的原地更新（In-place Updates），其余 43 个系统均采用只追加（Append-only）模式，导致新旧事实并存引发严重的推理矛盾。

为了建立清晰的架构坐标系，[arXiv:2512.13564](https://arxiv.org/abs/2512.13564) 与 [NiteAgent 2026 研究综述](https://niteagent.com/blog/agent-memory-research-2026/) 共同确立了从**存在形式（Forms）、认知功能（Functions）与演化动力学（Dynamics）**切入的三维分类体系，并在工程实现上细化为七大功能层级：

| 分类维度与记忆层级 | 认知与系统定义 | 在主流 Agent / Harness 中的典型实现载体 | 生产环境核心失效模式 (Failure Modes) |
| :--- | :--- | :--- | :--- |
| **形式维度 1：Token 级记忆 (Token-Level)** | 以显式自然语言文本、JSON、Markdown 或图谱节点形式存储在模型外部或上下文中 | `CLAUDE.md`、`MEMORY.md`、向量库 (Chroma/Qdrant)、时序图谱 (Neo4j/Graphiti) | 检索延迟、语义漂移、Token 预算挤占、长上下文注意力稀释 |
| **形式维度 2：参数化记忆 (Parametric)** | 将历史经验或长期知识编码进神经网络权重、LoRA 适配器或测试时记忆模块中 | Google [Titans](https://arxiv.org/abs/2501.00663) 深度神经记忆模块、测试时训练 (TTT)、在线 RL 微调权重 | 灾难性遗忘 (Catastrophic Forgetting)、可解释性差、难以精确擦除单条隐私数据 |
| **形式维度 3：潜空间记忆 (Latent Memory)** | 以模型内部激活状态、预计算 KV Cache 块、隐藏状态向量或加密状态流形式驻留 | Anthropic Prompt Cache、Letta [Sleep-Time KV 预填充](https://www.spheron.network/blog/sleep-time-compute-gpu-cloud/)、OpenAI [Fernet 加密压缩块](https://kangwooklee.com/blogs/codex_context_compaction.html) | 跨模型不可迁移、前缀微小变动即导致缓存雪崩、黑盒状态难以人工审计 |
| **功能层级 1：感官缓冲 (Sensor Buffer)** | 当前单步交互的原始输入（最新用户指令、刚返回的终端 stdout/stderr 输出） | 运行时事件流 (Event Stream) 最新节点、工具执行拦截器缓冲区（存活毫秒至数秒） | 单次工具输出过大（如数万行日志）直接撑爆上下文窗口 |
| **功能层级 2：工作记忆 (Working Memory)** | 当前任务推理周期内的活跃上下文、待办进度与临时中间变量 | 活跃消息数组、Letta `Core Memory` (2–4KB RAM 块)、`todo.md` / 状态快照 | 多轮工具调用后超过 50 轮阈值引发上下文焦虑与目标漂移 |
| **功能层级 3：情景记忆 (Episodic Memory)** | 带有明确时间戳与上下文锚点的历史具体事件记录（“何时、何地、发生了什么”） | Letta `Recall Memory`、Zep [Graphiti 情景子图](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)、OpenHands 事件溯源日志 | 缺乏双时态索引导致无法回答“某事件发生前后的状态变化”等多跳时序问题 |
| **功能层级 4：语义记忆 (Semantic Memory)** | 从交互历史中提炼出的去时间化陈述性事实、领域知识、实体属性与用户偏好 | [Mem0](https://arxiv.org/pdf/2504.19413) 事实与图谱存储、LangMem 命名空间 JSON、Cognee 语义知识图谱 | 事实陈旧 (Stale Facts) 与冲突累积；新偏好追加而非覆盖旧偏好导致行为摇摆 |
| **功能层级 5：程序性记忆 (Procedural Memory)** | 关于“如何执行任务”的技能、工具调用序列、调试启发式规则与工作流规范 | `SKILL.md`、`.claude/rules/`、LangMem 动态系统提示词重写、[Mem^p 轨迹蒸馏](https://arxiv.org/abs/2508.06433) | 硬编码工作流难以适应环境变化；将操作规范误存为向量数据而非指令注入 |
| **功能层级 6 & 7：空间与社交记忆 (Spatial & Social)** | 代码库目录拓扑、API 端点映射（空间记忆）；多用户身份、权限边界与协作关系（社交记忆） | [ByteRover Context Tree](https://niteagent.com/blog/agent-memory-research-2026/)、仓库 AST 映射、Mem0 `user/session/agent` 作用域 | 多智能体并发修改冲突、跨用户权限越界、大型 Monorepo 路径迷失 |

## 二、 主流编码与通用 Agent Harness 的内置记忆架构深度拆解

在软件工程与通用任务自动化领域，以 [Claude Code](https://code.claude.com/docs/en/memory)、[OpenAI Codex CLI](https://kangwooklee.com/blogs/codex_context_compaction.html)、[OpenCode](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)、[Cursor 3](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)、[Devin Desktop (原 Windsurf)](https://omidsaffari.com/blog/windsurf-vs-cursor-after-devin-desktop) 以及 [OpenHands Software Agent SDK](https://arxiv.org/abs/2511.03690) 为代表的工业级 Agent Harness，并未盲目引入笨重的外部图数据库，而是围绕**“声明式文件系统层级 + 渐进式上下文压缩（Compaction） + 缓存感知优化 + 子智能体上下文隔离”**构建了一套极度务实且高吞吐的内置记忆体系。

### 2.1 Claude Code：双轨制声明式记忆与多级“精准遗忘”引擎

根据 Anthropic 官方文档（[How Claude remembers your project, 2026](https://code.claude.com/docs/en/memory)）以及技术剖析论文 [《Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems》(Liu et al., arXiv:2604.14228v3, 2026)](https://arxiv.org/abs/2604.14228)，Claude Code 的记忆架构建立在“人类编写的确定性指令轨（`CLAUDE.md` / `AGENTS.md`）”与“智能体自主积累的学习轨（Auto Memory / `MEMORY.md`）”双轨并行机制之上，并辅以精密的上下文压缩管线。

**第一，四级作用域声明式记忆与按需动态加载（On-Demand Subtree & Path-Scoped Loading）。** Claude Code 将持久化语义与程序性记忆显式存储在文件系统中，按从宽泛到具体的优先级顺序拼接加载进上下文（后加载的更贴近工作目录，在注意力分布中权重更高）：
1. **组织策略级（Managed Policy）**：位于系统受控路径（如 Linux `/etc/claude-code/CLAUDE.md` 或 macOS `/Library/Application Support/ClaudeCode/CLAUDE.md`），由企业 IT/DevOps 统一分发且无法被个人通过 `claudeMdExcludes` 屏蔽；
2. **用户全局级（User Scope）**：位于 `~/.claude/CLAUDE.md` 与 `~/.claude/rules/`，记录开发者跨项目的个人编码风格与快捷键偏好；
3. **项目共享级（Project Scope）**：位于仓库根目录的 `./CLAUDE.md`、`./.claude/CLAUDE.md` 或跨工具兼容的 `./AGENTS.md`（自 v2.1.277 起原生支持直接读取 `AGENTS.md`），支持通过 `@path/to/file` 语法递归导入最多 4 层外部文档（启动时会剥离 HTML 注释 `<!-- -->` 以节省 Token）；
4. **本地私有级（Local Scope）**：位于被 Git 忽略的 `./CLAUDE.local.md`。

尤为关键的是其**防上下文膨胀的懒加载机制**：启动会话时，Claude Code 仅向上遍历加载当前工作目录及其祖先路径上的 `CLAUDE.md`，而**子目录中的嵌套 `CLAUDE.md` 在启动时绝不加载**，仅当智能体通过 `Read`、`Write`、`Edit` 或等效 Bash 命令实际触达该子目录内的文件时才动态注入上下文。此外，针对大型工程，`.claude/rules/*.md` 支持通过 YAML Frontmatter 中的 `paths` 字段定义 Glob 路径匹配规则（例如 `paths: ["src/api/**/*.ts"]`，花括号展开预算上限为 1,000 个模式与 4 MiB），实现“仅在编辑特定模块时唤醒对应领域规范”的精准按需记忆挂载。同时，Claude Code v2.1.283+ 提供了 `/doctor prompt-audit` 审计工具，自动扫描所有记忆文件中的过期模型指令、失效文件路径与跨文件逻辑冲突。

**第二，Auto Memory（自动记忆）与子智能体独立记忆空间。** 当启用 Auto Memory 时，Claude 在交互过程中一旦接收到用户的纠错、偏好澄清或发现代码库中难以从字面推导的隐性规律，便会自动向仓库专属的自动记忆区写入笔记。为防止自动记忆无限膨胀吞噬上下文，系统在每次会话启动及压缩恢复时，**仅加载 `MEMORY.md` 索引文件的前 200 行或前 25 KB（以先到者为准）**（[GuildBuild KN-09, 2026](https://www.guildbuild.ca/insights/agent-memory-after-compaction)）。[GuildBuild 团队的实测研究](https://www.guildbuild.ca/insights/agent-memory-after-compaction)证实，该索引本身不会在空启动时触发全量文件读取，而是作为路由表（Routing Table），当用户的具体提问命中索引条目触发词时，智能体才主动调用文件读取工具按需打开对应的详细笔记。此外，在 [Claude Code Subagents 架构](https://code.claude.com/docs/en/sub-agents)（如 `.claude/agents/code-reviewer.md`）中，主智能体通过派生拥有独立上下文窗口的子智能体来执行高噪探索任务，避免海量中间搜索结果污染主会话记忆；每个子智能体还可通过 Frontmatter 配置专属的持久化跨会话记忆目录，实现角色级的专业经验沉淀。

**第三，三阶段核心压缩策略（Three-Tier Precision Forgetting）与状态重建。** 当长会话持续运行导致 Token 逼近上限时，Claude Code 并不会粗暴地截断历史，而是执行一套兼顾缓存命中率与信息保真度的渐进式清理流程（[Justin3go, 2026](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)；[arXiv:2604.14228](https://arxiv.org/abs/2604.14228)）：
* **第一层：零 LLM 开销的工具输出修剪（Tool Result Trimming）**。在每次请求发送前由本地规则引擎自动执行。系统保护最近若干次活跃工具调用的完整返回结果，而将保护窗口之外的早期庞大工具输出（如数千行的文件读取内容或测试报错堆栈）替换为轻量级占位符 `[Old tool result content cleared]`，但**完整保留对应的 `tool_call` 调用记录本身**。这赋予了智能体“选择性遗忘”能力：智能体依然记得自己查阅过 `src/auth.ts`，若后续仍需细节，只需重新发起一次读取即可。
* **第二层：Prompt Cache 友好的尾部修剪策略（Cache-Friendly Strategy）**。为了最大化复用 Anthropic API 的前缀缓存（Prefix Prompt Cache），Claude Code 在清理消息时严禁修改消息序列的前半部分（Prefix），而是采取外科手术式的尾部裁剪，确保请求前缀与上一轮完全一致，从而将长程重构任务的 API 成本降低一个数量级。
* **第三层：会话内存压缩与 9 节结构化 LLM 摘要（Structured Compaction & State Reconstruction）**。当上下文占用触及自动压缩阈值（计算公式为：`effective_context_window - 13,000 tokens`）时，系统首先尝试零模型调用的 `Session Memory Compact`；若仍不足，则调用 LLM 生成严格包含 9 个固定章节的结构化交接摘要：（1）用户原始意图、（2）核心技术概念、（3）关注的文件与代码段、（4）遇到的报错及修复方式、（5）问题求解逻辑链、（6）全部用户消息汇总、（7）待办事项（TODOs）、（8）当前正在推进的工作、（9）建议的下一步操作，且提示词强制要求直接引用原文关键语句以防止“语义漂移（Context Drift）”。压缩完成后，Harness 立即从磁盘执行**确定性状态重建**：重新加载根目录 `CLAUDE.md`、`MEMORY.md` 索引、已激活的技能（Skills）正文，并**自动重读最近编辑或访问过的最多 5 个核心代码文件**（单文件上限 5,000 tokens，总预算 50,000 tokens），确保智能体在压缩后立即拥有最新鲜的代码上下文（[GuildBuild, 2026](https://www.guildbuild.ca/insights/agent-memory-after-compaction)；[Justin3go, 2026](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)）。

### 2.2 OpenAI Codex CLI 与 OpenCode：交接备忘录、加密状态块与阶梯式治理

与 Claude Code 的渐进式修剪不同，OpenAI 的开源终端智能体 **Codex CLI**（Rust 实现）以及新兴开源框架 **OpenCode**（TypeScript + Effect-TS 实现）展现了另外两种极具代表性的工程哲学。

**OpenAI Codex CLI 的双路径压缩与 Fernet 加密状态交接。** 根据源码与逆向工程研究（[Kangwook Lee, 2026](https://kangwooklee.com/blogs/codex_context_compaction.html)；[Justin3go, 2026](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)），Codex CLI 采用“总结并替换（Summarize and Replace）”的**交接备忘录（Handoff Memo）**模式，并提供两条执行路径：在面向通用第三方模型的**本地路径（`compact.rs`）**中，客户端提取最近的用户消息（硬上限约 20,000 tokens），使用内置模板 `templates/compact/prompt.md` 要求模型生成包含“当前进展与关键决策、重要约束与用户偏好、剩余待办事项、继续工作所需的关键数据”四要素的交接摘要；而在面向 OpenAI 原生 Codex 模型的**远端路径（`compact_remote.rs`）**中，客户端调用专用服务端接口 `responses.compact()`。

[Kangwook Lee (2026) 的两步提示注入实验](https://kangwooklee.com/blogs/codex_context_compaction.html)首次揭示了 `responses.compact()` 的内部机制：服务端使用隐藏的压缩提示词（`"You are performing a CONTEXT CHECKPOINT COMPACTION..."`）生成摘要后，使用 **Fernet 对称加密（AES-128-CBC + HMAC-SHA256）**将状态打包为客户端不可读的加密二进制块（`gAAAAAB...`）返回；在下一轮调用 `responses.create()` 时，服务端解密该块并自动前置一段交接引导词（`"Another language model started to solve this problem and produced a summary of its thinking process..."`）。在可见的历史替换策略上，Codex CLI 极度尊重人类原始意图——它会**物理删除所有历史助手回复与工具调用输出，但逐字原样保留所有人类用户消息（Verbatim User Messages）**（[GuildBuild, 2026](https://www.guildbuild.ca/insights/agent-memory-after-compaction)）。同时，加密压缩块中还隐式携带了工具状态与高维上下文（实测表明，未出现在可见历史中的长哈希值与状态数字在经历两次连续压缩后仍被模型精确复现）。但需注意，在多智能体协作中，若平级智能体（Peer Agent）通过工具通道发送交接消息，压缩后其消息正文可能被剥离为仅剩发送者名称的存根（Stub），因此工程最佳实践要求将关键交接负载写入磁盘文件并传递文件路径。

**OpenCode 的非破坏性时间戳隐藏（Non-Destructive Pruning）与末轮重放。** OpenCode 在 `session/compaction.ts` 中提出了“先修剪、后摘要”的**阶梯式治理（Stepped Governance）**方案（[Justin3go, 2026](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)）：第一步是**非物理删除的标记隐藏（Prune）**——仅当预计可释放超过 20,000 tokens 时才触发，始终保留最近 40,000 tokens 作为安全垫，且绝不修剪 `skill` 类型的操作指令输出与最近两轮用户对话；对于超龄旧消息，系统并不将其从数据库中抹除，而是打上 `compacted = Date.now()` 时间戳使其在后续请求组装中“隐形”，从而为审计回溯与状态回滚保留了完整的底层数据。第二步是当标记隐藏仍无法缓解容量压力时，唤醒后台隐藏摘要智能体生成 5 标题结构化摘要（自动匹配用户语言，如中文会话生成中文摘要），并在摘要生成后**自动重放用户的最后一条原始指令（Auto-Replay of Last Instruction）**，彻底消除了压缩后模型因面对冰冷摘要文本而丧失即时交互焦点的断层感。

### 2.3 IDE 与自主软件工程 Harness：Cursor 3、Devin Desktop (Windsurf) 与 OpenHands SDK

在图形化与云端混合工程环境中，记忆模块的设计重点转向了**代码库实时语义索引、隐式规则提炼以及多智能体空间编排**：
* **Cursor 3（静态 `.cursor/rules` 与动态 `Cursor Memories` 分离）**：根据 [Zenn 2026 架构评测](https://zenn.dev/taka4/articles/7221be19b93603?locale=en) 与 [Omid Saffari (2026) 的对比分析](https://omidsaffari.com/blog/windsurf-vs-cursor-after-devin-desktop)，Cursor 将项目级记忆拆分为两层：一是开发者手动维护的静态规则文件 `.cursor/rules/*.md`；二是自 v1.0 引入并在 Cursor 3 Agents Window 中成熟的 **Memories（自动记忆）**功能，它能从日常结对聊天中自动提炼出项目规范与架构约束并绑定至当前项目。然而，Cursor 的内置记忆严格局限于单项目与单一产品内部（Project-Scoped & Product-Bound），一旦开发者切换至终端 CLI 工具便无法直接共享。
* **Devin Desktop / Windsurf Cascade（工作区记忆与跨智能体控制平面）**：2026 年 6 月 Cognition 将 Windsurf 整合升级为 **Devin Desktop**，打造了支持 ACP（Agent Communication Protocol）协议的多智能体指挥中心（[Omid Saffari, 2026](https://omidsaffari.com/blog/windsurf-vs-cursor-after-devin-desktop)）。其核心引擎 **Cascade** 保留了本地持久化的工作区记忆系统（存储于 `~/.codeium/windsurf/memories/`），结合了 AI 自动判定重要性后写入的记忆与用户手动保存的条目；在云端，Devin 通过 **Spaces** 将关联会话、文件上下文、PR 与架构知识卡片（Devin Knowledge）聚合在同一看板视图下，使本地循环（Devin Local）与云端长程自主任务（Devin Cloud）共享统一的背景状态。
* **OpenHands Software Agent SDK V1（事件溯源状态与可插拔冷凝器）**：在发表于 MLSys 2026 的论文 [《The OpenHands Software Agent SDK》(Wang et al., arXiv:2511.03690v2, 2026)](https://arxiv.org/abs/2511.03690) 中，OpenHands 团队重构了其底层架构，采用严格的**不可变事件溯源日志（Immutable Event-Sourced State）**作为单一事实来源（Single Source of Truth）。智能体的每一次动作（Action）与环境观察（Observation）均被记录为持久化事件流，而传给 LLM 的工作记忆视图则由独立的**上下文冷凝器（Context Condenser）**动态派生——支持滑动窗口冷凝、摊销摘要冷凝（Amortized Summarization Condenser）以及结合仓库微智能体（Repo Microagents）的按需知识注入，在生产部署中以极低的事件溯源开销大幅降低了系统级崩溃率。

| 对比维度 | Claude Code (Anthropic) | OpenAI Codex CLI | OpenCode (OSS) | Cursor 3 / Devin Desktop | OpenHands SDK V1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **声明式指令载体** | `CLAUDE.md` (4级作用域)、`.claude/rules/*.md` (Glob路径触发)、`AGENTS.md` | `AGENTS.md` (目录层级继承) | `AGENTS.md` / 项目配置文件 | `.cursor/rules/*.md` / `.windsurf` 规则与 Devin Spaces | `.openhands/microagents/` (仓库级与触发词微智能体) |
| **自动记忆机制** | Auto Memory (`MEMORY.md` 200行/25KB索引 + 按需路由读取) + Dreaming 巩固 | `/memories` 跨会话通道 + 会话级加密压缩状态块 | 插件化外部记忆挂载 (支持 Supermemory / MCP) | Cursor Memories (对话自动提炼) / Cascade 本地记忆库 | 基于事件流的状态持久化与可编程 Memory 组件 |
| **上下文压缩策略** | 三层渐进式：零开销工具输出清理 → 保前缀缓裁剪 → 9节摘要 | 单层替换：本地 LLM 4点备忘录 或 远端 `responses/compact` 加密块 | 两步阶梯式：`compacted` 时间戳隐藏 → 5标题摘要 | 滑动窗口 + 动态 Token 预算分配 + 代码库语义重检索 | 模块化 Condenser (滑动窗口 / 摊销 LLM 摘要 / 观测掩码) |
| **用户原始消息处理** | 在第 3 层被合并进第 6 节“全部用户消息汇总”并强制引用原文 | **逐字完整保留**全部人类用户消息，仅物理删除助手与工具消息 | 摘要化处理，但在摘要后**自动重放最后一条用户指令** | 当前会话滑动保留，历史会话提炼为规则条目 | 保留任务初始目标事件与最近 K 轮交互事件 |
| **压缩后状态恢复** | 重载根 `CLAUDE.md`、`MEMORY.md`、已调技能及**最近5个修改文件** | 注入 Handoff 引导提示词 + 解密服务端隐式工具状态 | 隐藏旧节点保留于数据库供回溯 + 末轮指令无缝续接 | 依托后台持久化向量索引每轮动态检索注入 | 从不可变 Event Stream 重新投影生成精简上下文视图 |

## 三、 独立记忆中间件与开源框架的架构路线之争

当企业需求跨越单一编码工具、走向多智能体协同、长周期客户陪伴或跨平台状态共享时，产品内置的文件级记忆便暴露出结构单一、缺乏时序推理与跨产品锁定的短板。2025 至 2026 年间，独立记忆层（Standalone Memory Layers）赛道迎来了爆发式演进，形成了四大技术流派（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)；[GeniOS, 2026](https://thegenios.com/blog/open-source-memory-layers-2026/)；[Zenn, 2026](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)）：

### 3.1 操作系统虚拟内存范式：Letta (原 MemGPT) 与主从睡眠代理

源自 UC Berkeley 的 **Letta（原 MemGPT）** 开创了“将大语言模型视为操作系统内核（LLM as an Operating System）”的设计哲学。Letta 将智能体的存储空间严格划分为三层：（1）**核心内存（Core Memory）**：驻留在上下文窗口内的可读可写结构化内存块（Memory Blocks，默认大小 2–4 KB，如 `persona` 块与 `human` 块），相当于计算机的物理内存（RAM）；（2）**召回内存（Recall Memory）**：基于 SQLite 或 PostgreSQL 的全量历史会话日志，支持按时间或关键词分页检索；（3）**归档内存（Archival Memory）**：基于向量数据库的无上限外部长期存储，相当于磁盘（Disk）。

在初代 MemGPT 架构中，智能体通过自主调用 `core_memory_replace`、`core_memory_append`、`archival_memory_insert` 与 `archival_memory_search` 等工具函数完成内存分页（Paging）。这种设计的最大优势在于**极高的可审计性与自主性**，但也带来了显著的在线延迟——每当智能体决定编辑记忆时，都必须消耗额外的工具调用轮次与推理耗时。为彻底解决这一痛点，Letta 在 0.7.0+ 版本中推出了基于 [Sleep-Time Compute (Lin et al., arXiv:2504.13171)](https://www.letta.com/blog/sleep-time-compute/) 的 **MemGPT 2.0 睡眠期智能体架构**：系统在底层同时实例化一个负责低延迟在线交互的**主智能体（Primary Agent，如使用轻量高速的 GPT-4o-mini）**与一个运行在后台的**睡眠智能体（Sleep-Time Agent，如使用推理能力更强的 Claude 3.7/4 Sonnet 或 GPT-4.1）**。主智能体的核心内存编辑工具被剥离，转交由睡眠智能体在异步后台以“随时可用（Anytime）”方式持续重写与整理主智能体的 Core Memory 块，从而将在线对话延迟与后台深度记忆重构彻底解耦。

### 3.2 混合提取与双时态知识图谱：Mem0 vs. Zep (Graphiti)

在生产级 SaaS 与开源下载量排行榜上，**Mem0**（GitHub 逾 4.8万–6.1万 Stars，获 $24M Series A 融资，被选为 AWS Agent SDK 独家记忆提供商）与 **Zep / Graphiti**（Graphiti 引擎逾 2.4万–2.9万 Stars）代表了“高密度向量提取派”与“双时态图谱推理派”的直接交锋（[GeniOS, 2026](https://thegenios.com/blog/open-source-memory-layers-2026/)；[Zenn, 2026](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)）。

**Mem0 与 Mem0g 的极致 Token 效率与两阶段冲突消解。** 根据发表于 ECAI 2025 并在 2026 年持续演进的论文 [《Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory》(Chhikara et al., arXiv:2504.19413)](https://arxiv.org/pdf/2504.19413)，Mem0 的核心目标是**用最少的 Token 保存最高信息密度的事实**。其管线分为五个精密步骤：从原始对话中提取候选事实、跨会话去重（若用户在会话 2 与会话 5 表达了相同偏好，仅保留一份副本但将其重要性权重标记为“已确认 Confirmed”）、合并关联事实、按重要性评分裁剪边缘信息，以及建立混合索引。其图增强变体 **Mem0g** 进一步引入实体提取器（Entity Extractor）、关系生成器（Relations Generator）与冲突检测器（Conflict Detector），通过 LLM 判定新摄入事实与既有图谱节点的重叠关系并执行合并、更新或废弃。得益于 2026 年 4 月更新的 Token 高效去重算法，Mem0 在 [LongMemEval 基准](https://niteagent.com/blog/agent-memory-research-2026/)上的准确率从早期的 72.3% 跃升至 **89.8%（部分配置达 93.4%）**，同时将存储与注入开销严格控制在原始会话体积的 **3%–8%**（较全量上下文方法削减 90% Token 消耗与 91% P95 延迟）。

**Zep 与 Graphiti 的双时态（Bi-Temporal）知识图谱引擎。** Mem0 的弱点在于缺乏原生的时间有效期建模——当用户更改收货地址或技术栈时，向量检索仍可能因语义相似而召回旧地址。Zep 在其核心论文（arXiv:2501.13956）中开源的 **Graphiti** 引擎（基于 Neo4j）通过三层子图结构（情景子图 Episode Subgraph、语义实体子图 Semantic Entity Subgraph、社区聚类子图 Community Subgraph）与**双时态边标注（Bi-Temporal Edge Annotation）**完美攻克了这一难题（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)）。在 Graphiti 中，每一条关系边同时记录两个时间维度：
* **事件真实时间（Event Time / Validity Window，$[t_{valid}, t_{invalid})$）**：该事实在客观世界中成立与失效的时间区间；
* **系统摄入时间（Ingestion / Transaction Time，$[t_{created}, t_{expired})$）**：智能体首次观察到该事实及后续修正该认知的系统时间戳。

当发生事实变更或事后纠错（Retroactive Correction）时，Graphiti 不会物理抹除旧边，而是将其 $t_{invalid}$ 闭合，并插入带有新 $t_{valid}$ 的有效边。更值得称道的是其检索层的工程优化：Zep 结合了语义向量相似度、BM25 关键词检索与图谱广度优先遍历，**在检索阶段完全不需要调用 LLM**，从而将 P95 检索延迟稳定控制在 **300ms** 以内，并在 Deep Memory Retrieval (DMR) 基准上取得 94.8% 准确率（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)）。

### 3.3 Agent 原生上下文树与自演化引擎：ByteRover、Cognee、Supermemory 与 LangMem

除了上述三大巨头，2026 年开源生态还涌现出四类极具创新性的专用架构：

**1. ByteRover 2.1.5（Agent 原生层级上下文树 Context Tree）**：发表于 2026 年 4 月的论文 [《ByteRover: Agent-Native Memory Through LLM-Curated Hierarchical Context》(arXiv:2604.01599)](https://niteagent.com/blog/agent-memory-research-2026/) 颠覆了“将记忆作为外部数据库 API”的传统思路，提出由执行任务的同一个 LLM 在空闲轮次自主维护一套基于本地文件系统的层级知识树（包含 `sessions/` 原始日志、`facts/` 提炼事实、`patterns/` 编程风格与工具调用模式，以及 `index.json` 检索索引）。在检索阶段，ByteRover 设计了从零成本到高精度的**五级渐进式检索管线（Five-Tier Retrieval Pipeline）**：（Tier 1）当前工作记忆直接读取（0ms）→（Tier 2）最近 3 次会话时间戳快速扫描 →（Tier 3）基于 `index.json` 的 TF-IDF 关键词索引（~50ms，前三级即可拦截并解决约 **85%** 的日常查询）→（Tier 4）面向提炼事实文件的语义向量检索（~200ms）→（Tier 5）全树深度遍历（~2s，仅在前四级未命中时触发）。这一设计使 ByteRover 在零外部数据库依赖的前提下，将中位检索延迟压低至 80ms，并在 LoCoMo 与 LongMemEval 上分别创下 **96.1%** 与 **94.4%** 的最高准确率记录（[NiteAgent, 2026](https://niteagent.com/blog/agent-memory-research-2026/)）。

**2. Cognee（六阶段 ECL 管线与仿生 `memify` / `Dreamify` 自精炼）**：作为本地优先（Local-First）与数据主权场景的首选开源引擎（~29k GitHub Stars），Cognee 构建了“提取-认知化-加载（Extract-Cognify-Load）”六阶段管线（文档分类、权限校验、文本切块、LLM 三元组提取、摘要生成、向量与图谱双写），并提供 14 种检索模式。其核心创新在于 `memify` 与 `Dreamify` 算子：系统定期遍历知识图谱，自动剪除长期未激活的陈旧节点、根据访问频率重加权关系边（模拟生物神经突触的长时程增强 LTP），并合成高阶派生事实（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)；[GeniOS, 2026](https://thegenios.com/blog/open-source-memory-layers-2026/)）。

**3. Supermemory 与 LangMem（MCP 跨平台记忆总线与 LangGraph 原生状态存储）**：针对开发者同时使用 Cursor、Claude Code 与 OpenCode 导致的“产品级记忆孤岛”，[Supermemory](https://thegenios.com/blog/open-source-memory-layers-2026/)（~28k Stars）与 Memorix 通过 **Model Context Protocol (MCP)** 协议将原子记忆单元与知识图谱暴露为本地服务（如 `localhost:6767`），在 LongMemEval 多会话与时序维度上分别达到 71.43% 与 76.69% 准确率。而 [LangMem](https://vectorize.io/articles/letta-vs-langchain-memory) 则作为深度绑定 LangGraph 检查点（Checkpointing）机制的轻量级 Python SDK，不仅提供基于命名空间（`user/session/agent`）的 JSON 语义去重存储，还支持智能体在运行时调用 `update_system_prompt` 函数动态重写自身的程序性指令块。

| 开源记忆框架 / 系统 | 核心架构范式与存储后端 | LoCoMo / LongMemEval 表现 | P50 / P95 检索延迟与 Token 开销 | 核心优势与最佳适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **ByteRover 2.1.5** ([arXiv:2604.01599](https://niteagent.com/blog/agent-memory-research-2026/)) | Agent 原生层级文件系统 Context Tree + 五级渐进检索 (零外部 DB 依赖) | LoCoMo: **96.1%**<br>LongMemEval: **94.4%**<br>BEAM 1M: 62.0% | P50: ~80ms (Tier 1–3)<br>P95: ~1.2s (Tier 4–5)<br>Token 开销: 会话体积 15–25% | 召回率行业最高、零基础设施运维；适合追求极致精度的研发与多会话智能体 |
| **Mem0 / Mem0g** ([arXiv:2504.19413](https://arxiv.org/pdf/2504.19413)) | 向量优先 (Postgres/pgvector, Qdrant) + 可选图谱 (Neo4j) + LLM 冲突消解管线 | LoCoMo: 62.1% / 68.4%<br>LongMemEval: **89.8%–93.4%** (2026.04 更新) | P50: ~45ms<br>P95: ~350ms<br>Token 开销: **仅占会话 3–8%** | 极致 Token 压缩率、生态集成最广 (AWS Agent SDK 官方选型)；适合高并发严格预算场景 |
| **Zep / Graphiti** ([arXiv:2501.13956](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)) | 基于 Neo4j 的双时态知识图谱 (情景/实体/社区三子图) + BM25/向量/图混合检索 | LongMemEval: 63.8%<br>DMR 基准: **94.8%** | P95: **~300ms** (检索期零 LLM 调用) | 原生双时态有效期窗口 ($t_{valid}/t_{invalid}$)；适合金融、法务、CRM 等状态频繁演化的时序推理 |
| **Letta (MemGPT 2.0)** ([arXiv:2504.13171](https://www.letta.com/blog/sleep-time-compute/)) | OS 虚拟内存三层架构 (Core RAM + Recall SQL + Archival Vector) + 睡眠后台代理 | LoCoMo: 74%–83%<br>LongMemEval: ~83.2% | 在线读取 Core 0ms；后台 Sleep-Time 异步重构无阻塞 | 完整有状态 Agent 运行时、内存块修改完全可审计；适合长生命周期自主演进智能体 |
| **Cognee** ([topoteretes/cognee](https://thegenios.com/blog/open-source-memory-layers-2026/)) | 六阶段 `cognify` ECL 管线 + 向量与图双存储 + `memify` / `Dreamify` 自演化 | 专有基准领先；支持 14 种图与向量混合检索模式 | 本地/私有化部署低延迟；后台异步图谱修剪与边重加权 | 本地优先 (Local-First)、支持根据业务数据免训练调优；适合医疗、政企等严苛隐私合规场景 |
| **Supermemory & LangMem** | Supermemory: 原子记忆 + 图 + MCP 服务；LangMem: LangGraph Checkpoint 原生存储 | Supermemory LongMemEval: 71.4% (多会话) / 76.7% (时序) | Supermemory 本地 API 毫秒级响应；LangMem 离线搜索较慢 | Supermemory 擅长打通多 IDE/CLI 跨工具记忆；LangMem 适合已深度采用 LangGraph 的团队 |

## 四、 2025–2026 智能体记忆模块六大前沿演进方向

如果说前文剖析的声明式文件、上下文压缩与混合向量图谱构成了 2026 年生产级智能体的“标准基建”，那么学术界与工业界顶尖实验室正在探索的六大前沿方向，则正在将智能体记忆从**“被动存储容器（Passive Storage Substrate）”推向“主动认知与自演化器官（Active Cognitive & Self-Evolving Organ）”**。

### 4.1 前沿方向一：睡眠期计算（Sleep-Time Compute）与“做梦（Dreaming）”离线巩固层

传统智能体仅在用户发起提问的测试期（Test-Time）消耗算力进行推理与记忆检索，而在两次交互之间长达数小时的空闲期内完全处于休眠状态。2025 至 2026 年最重要的范式跃迁，便是受人类神经科学中快速眼动睡眠（REM Sleep）海马体-新皮层记忆巩固机制启发，将算力扩展至空闲期的**睡眠期计算（Sleep-Time Compute）与“做梦（Dreaming）”机制**：

**1. Letta 与 UC Berkeley 的睡眠期计算理论（arXiv:2504.13171）及 GPU KV Cache 预填充。** 在由 Letta 与 UC Berkeley Sky Computing Lab（Charles Packer、Sarah Wooders、Ion Stoica 等）联合发表的论文 [《Sleep-time Compute: Beyond Inference Scaling at Test-time》(Lin et al., 2025)](https://www.letta.com/blog/sleep-time-compute/) 中，研究者证明了智能体可以利用交互间隙的空闲算力，将杂乱的“原始上下文（Raw Context）”预先推理、重组并蒸馏为高信息密度的“习得上下文（Learned Context）”。在数学与编程基准（AIME、GSM、SWE 类任务）上，睡眠期计算实现了显著的帕累托改进（Pareto Improvement）：在同等测试期延迟预算下大幅提升准确率，或在保持同等准确率的前提下将在线推理开销降低数倍。此外，在底层 GPU 基础设施层（[Spheron Network, 2026](https://www.spheron.network/blog/sleep-time-compute-gpu-cloud/)），睡眠期计算被用于后台异步预计算并填充共享键值缓存（Shared KV Store Prefill）——当用户的真实查询到达且上下文指纹匹配时，模型可直接跳过昂贵的 Prefill 阶段直接进入 Decode 解码阶段，使首字延迟（TTFT）降低 **5 倍以上**，并将大型代码库的上下文预热成本分摊至原来的 1/1000。

**2. Anthropic Claude AutoDream 与“Dreaming（做梦）”记忆重组层。** 2026 年初至 5 月（Code with Claude 2026 大会），Anthropic 先后在 Claude Code 及企业级智能体平台中推出了 **AutoDream** 与 **Dreaming（做梦）**研究预览功能（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)；[Zenn, 2026](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)）。与只关注“如何写入新记忆”的传统工具不同，Dreaming 是一个独立于在线会话之外的定时后台巩固进程（主要依托 Claude Opus 4.7 与 Sonnet 4.6 运行）。它周期性地跨多个历史会话与多个并行智能体轨迹进行全局回放（Replay），执行三项高阶认知操作：（1）**跨会话去重与冲突消解**（清除过期的局部指令与重复笔记）；（2）**跨轨迹隐性模式提炼**（识别在单次会话中不可见、但在十余次会话中反复出现的错误模式、多个子智能体独立收敛出的高效脚本命令以及团队隐性偏好）；（3）**记忆层级晋升**（将高频验证有效的临时经验自动升格写入核心规则文件）。据法律 AI 独角兽 Harvey 披露的早期生产数据，引入 Claude Dreaming 记忆巩固层后，其复杂多步法律工作流的端到端任务完成率提升了约 **6 倍**（[Zenn, 2026](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)）。与此同时，GeniOS 等新一代上下文智能平台也引入了基于变点检测算法（PELT、BinSeg）的持续后台推理循环，在客户状态或项目进度发生偏移时主动向智能体推送预警，终结了“智能体不主动提问就无法感知记忆变化”的被动检索困局（[GeniOS, 2026](https://thegenios.com/blog/open-source-memory-layers-2026/)）。

### 4.2 前沿方向二：基于强化学习（RL）的原生端到端记忆管理策略

无论是 Letta 的工具调用还是 Mem0 的提取管线，早期系统大多依赖人工编写的提示词启发式规则（Prompt Heuristics）来决定何时写入、何时更新或遗忘。然而，预训练基础模型并未在训练阶段学习过如何最优地管理有限容量的外部记忆库。2025 下半年至 2026 年，以 **AgeMem**、**Memory-R1** 与 **MemRL** 为代表的强化学习原生记忆框架，彻底将记忆操作内化为策略网络（Policy Network）的可学习动作：
* **AgeMem（统一长短期记忆的渐进式 GRPO 训练，ACL 2026 SAC Highlight）**：在论文 [《Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents》(Yu et al., arXiv:2601.01885v3, 2026)](https://arxiv.org/abs/2601.01885) 中，作者打破了长短期记忆分治的传统，将长期记忆（LTM）与短期工作记忆（STM）的存储、检索、更新、摘要与丢弃（Discard）统一暴露为智能体策略空间中的工具动作。为了解决记忆操作与最终任务成败之间奖励信号极度稀疏且不连续的难题，AgeMem 设计了**三阶段渐进式强化学习策略（Three-Stage Progressive RL）**与**步级群组相对策略优化算法（Step-wise GRPO）**，使模型学会为了长远任务收益而在当下主动清理工作记忆噪声并精准归档关键线索，在五大长程基准与多种模型底座上全面超越启发式基线。
* **Memory-R1（双智能体协同 RL 记忆微调，arXiv:2508.19828）**：[Yan et al. (arXiv:2508.19828v5, 2026)](https://arxiv.org/abs/2508.19828) 提出的 Memory-R1 构建了由两个专门化 RL 微调智能体组成的两阶段协同架构：第一阶段由 **Memory Manager（记忆管理器）**接收新信息并自主决策执行 `ADD`（新增）、`UPDATE`（更新）、`DELETE`（删除）或 `NOOP`（空操作）四种结构化动作；第二阶段由 **Answer Agent（作答智能体）**对候选记忆条目进行预筛选、蒸馏与推理。令人瞩目的是，通过结合结果导向的 PPO 与 GRPO 算法，Memory-R1 **仅需 152 对训练问答样本**便可在 3B 至 14B 模型规模上习得稳健的记忆增删改策略，在 LoCoMo、MSC 与 LongMemEval 基准上实现高达 +15.76 分的复合收益。
* **MemRL（运行时非参数化情景记忆强化学习，arXiv:2601.03192）**：针对在线更新模型权重容易引发灾难性遗忘的困境，[Zhang et al. (arXiv:2601.03192v2, 2026)](https://arxiv.org/abs/2601.03192) 提出将“稳定的冻结推理模型（Stable Reasoning Backbone）”与“可塑的非参数化情景记忆库（Plastic Episodic Memory）”解耦。MemRL 在运行时根据环境反馈的强化学习价值信号（Value Estimation / Q-score）动态更新每条情景记忆的效用权重，并配合两阶段检索机制过滤高噪低效记忆，实现了无需修改模型权重即可在交互中持续自我演化的运行时强化学习。

### 4.3 前沿方向三：卡片盒笔记自组织网络（A-Mem）与记忆架构元进化（MemEvolve）

在知识结构化组织方面，传统的预定义图谱本体（Ontology Schema）往往过于僵化，难以适应开放域智能体千变万化的任务形态。发表于 NeurIPS 2025 的代表作 [《A-MEM: Agentic Memory for LLM Agents》(Xu et al., arXiv:2502.12110v11)](https://arxiv.org/abs/2502.12110) 引入了卢曼**卡片盒笔记法（Zettelkasten Method）**的自底向上自组织机制：每当新经验写入时，A-Mem 首先生成一张包含原始记录、上下文语义描述（Contextual Description）、关键词（Keywords）与分类标签（Tags）的结构化原子笔记卡片；随后，系统结合向量检索与智能体主动推理，扫描历史笔记库以建立跨卡片的双向语义链接（Links）。更关键的是其**记忆演化（Memory Evolution）**机制——新笔记的接入会反向触发对既有历史笔记上下文表征与标签属性的动态重写与修正，使整个记忆网络像有机生命体一样随经验积累不断重塑认知结构。

在此基础上，[《MemEvolve: Meta-Evolution of Agent Memory Systems》(arXiv:2512.18746, 2025–2026)](https://arxiv.org/abs/2512.18746) 更进一步：它不再局限于演化记忆库中的“内容”，而是利用元学习与进化搜索算法（Meta-Evolution），根据下游任务簇的反馈自动搜索、变异并演化智能体记忆系统本身的**架构拓扑与管理算子（写入策略、压缩函数、检索重排逻辑）**，开启了针对特定垂直领域自动合成定制化记忆底座的先河。

### 4.4 前沿方向四：程序性记忆（Procedural Memory）的轨迹蒸馏与跨模型迁移

长期以来，智能体在复杂环境（如软件排障、多步旅行规划、家庭具身导航）中完成一次任务后，其宝贵的试错轨迹往往被丢弃，下次遇到同类任务仍需从零试错。发表于 ACL 2026 Findings 的 [《Mem^p: Exploring Agent Procedural Memory》(Fang et al., arXiv:2508.06433v4, 2026)](https://arxiv.org/abs/2508.06433) 系统性建立了**可学习、可更新、终身演进的程序性记忆（Procedural Memory）**框架。

$\text{Mem}^p$ 的核心洞察在于：程序性记忆不应以静态数据片段形式存储，而必须将过往成功与失败的执行轨迹（Trajectories）双向蒸馏为两类互补形态——一是**细粒度的分步操作指南（Fine-grained Step-by-Step Instructions）**，二是**高层级的脚本化抽象工作流（Higher-level Script-like Abstractions）**。围绕“构建（Build）、检索（Retrieval）、更新（Update）”生命周期，$\text{Mem}^p$ 引入了动态修正与废弃机制（Deprecation Regimen）：当环境或 API 发生变化导致旧工作流失效时，系统自动根据新轨迹修正或淘汰过时的程序性记忆。尤为惊艳的实证发现是**程序性记忆的跨模型可迁移性（Cross-Model Transferability）**：由高成本强模型（如 GPT-4o 或 Claude 4 Opus）在探索过程中蒸馏构建出的 $\text{Mem}^p$ 程序性记忆库，可以直接挂载迁移给低成本轻量级小模型使用，在不微调小模型参数的情况下使其在 TravelPlanner 与 ALFWorld 等复杂基准上获得大幅度的成功率与执行效率跃升。

### 4.5 前沿方向五：测试时参数化记忆（Test-Time Parametric Memory）与神经长期记忆模块

上述所有外部记忆系统均属于 Token 级或潜空间级外挂存储，其检索结果最终仍需转化为上下文窗口中的 Token 供 Transformer 注意力层读取。而在基础模型底层架构层面，Google Research 发表于 NeurIPS 2025 的开创性工作 [《Titans: Learning to Memorize at Test Time》(Behrouz, Zhong, & Mirrokni, arXiv:2501.00663)](https://arxiv.org/abs/2501.00663) 及其配套的 **MIRAS 系统**，代表了将长期记忆直接内化入模型推理期权重更新的**测试时参数化记忆（Test-Time Parametric Memory）**最前沿（[Zylos Continual Learning Report, 2026](https://zylos.ai/research/2026-04-09-continual-learning-catastrophic-forgetting-ai-agents/)）。

Titans 架构从认知科学视角重新定义了神经网络组件的分工：将标准自注意力机制（Attention）因受限于有限窗口但擅长精确建模局部依赖的特性定位为**短期工作记忆**；同时设计了一个在测试推理阶段（Test Time）通过在线梯度下降实时更新权重的深度多层感知机——**神经长期记忆模块（Neural Long-Term Memory Module）**；再辅以一组与输入无关的可学习任务元参数作为**持久记忆（Persistent Memory）**。为了决定在推理流中“什么信息值得被写入长期记忆权重”，Titans 引入了受人类海马体启发的三大数学机制：
* **基于梯度的惊奇度指标（Surprise Metric）**：系统通过计算当前输入相对于内部神经记忆模块的关联损失梯度范数（$\nabla \mathcal{L}(M_{t-1}; x_t)$）来度量信息的“惊奇度”。与模型预期高度吻合的平庸信息梯度接近于零，不会被写入长期记忆；而偏离预期的突发事实、异常报错或新定义规则会产生高额梯度惊奇度，立即触发权重写入。
* **动量机制（Momentum of Surprise）**：为了防止只记住单个孤立惊奇词元而漏掉紧随其后的关键上下文，Titans 将瞬时惊奇度与跨越时间步的过去惊奇动量（Past Surprise Momentum）结合，确保重大事件前后的完整因果片段被连贯编码。
* **自适应权重衰减 / 遗忘门（Adaptive Weight Decay / Forgetting Gate）**：随着序列无限延长，神经记忆模块容量终会饱和。Titans 引入自适应衰减因子，在推理过程中动态擦除不再相关的陈旧权重表示，防止内存溢出（Memory Overflow）。

通过将神经长期记忆以三种架构变体——记忆作为上下文（Memory as a Context, **MAC**）、记忆作为门控分支（Memory as a Gate, **MAG**）和记忆作为网络层（Memory as a Layer, **MAL**）——与注意力层融合，Titans 不仅保持了训练期的高度可并行化与推理期的线性级高效吞吐，更成功在单模型内将有效上下文窗口扩展至 **200 万 Token 以上**，在大海捞针（Needle-in-a-Haystack）、常识推理与长程序列预测上全面超越标准 Transformer 及现代线性循环网络（如 Mamba2、DeltaNet）。

### 4.6 前沿方向六：记忆安全攻防（Memory Poisoning）、防投毒免疫与隐私合规架构

当智能体具备了跨会话持久记忆与自主写入能力后，记忆模块本身便成为了企业部署中最脆弱的新型攻击面。一旦攻击者通过网页内容、邮件正文、外部 API 响应或共享文档发起**间接提示注入（Indirect Prompt Injection）**，恶意指令便可能被智能体的摘要管线提取并持久化写入长期记忆库，形成潜伏期极长的**记忆投毒攻击（Memory Poisoning Attack）**（[A Survey on the Security of Long-Term Memory in LLM Agents, arXiv:2604.16548, 2026](https://arxiv.org/html/2604.16548v1)；[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)）：

**1. 记忆注入攻击的严峻现实（MINJA 与 EchoLeak 事件）。** 真实世界的 **EchoLeak 安全事件**已经敲响警钟：攻击者在发送给用户的普通外部邮件中隐藏了一段不可见的诱导指令，当邮件助手读取并总结该邮件时，恶意指令被写入长期记忆，导致智能体在数天后的后续正常会话中将用户的历史商业机密自动外泄。学术界提出的 **MINJA（Memory Injection Attack，Dong et al., 2025）** 证明，仅通过与智能体进行看似正常的查询交互，攻击者即可在不直接访问底层数据库的情况下，对天真记忆存储系统实现超过 **95% 的记忆注入成功率（ISR）与 70% 的攻击触发成功率（ASR）**。更棘手的是，**A-MemGuard** 团队的研究表明，如果仅对单条记忆条目进行静态的独立 LLM 安全审查，会漏检高达 **66% 的投毒条目**——因为高水平攻击者会将恶意逻辑拆解为在孤立状态下看似完全无害的陈述性碎片，唯有当未来某个特定触发查询将多个记忆碎片同时召回并组合进上下文时，恶意行为才会显现。

**2. 纵深防御体系与 GDPR vs. 欧盟 AI 法案的合规冲突。** 针对这一威胁，2026 年的前沿安全架构正在构建四层记忆免疫机制：（1）**写入源血缘追踪与复合信任评分（Provenance & Composite Trust Scoring）**：为每条记忆强制绑定数据源可信度标签（人类用户在终端直接输入的指令信任度最高，而从外部不可信网页或第三方 MCP 工具抓取的内容标记为低信任隔离态，严禁自动升格为程序性系统规则）；（2）**上下文联合审计（In-Context Joint Auditing）**：在睡眠巩固期（Sleep-Time / Dreaming）模拟多跳检索组合，在上下文中联合审查记忆簇的潜在诱导风险；（3）**TTL 自动过期与人类审批门控（Human-in-the-Loop Gate）**：对涉及权限变更、外部网络请求或核心配置修改的程序性记忆写入实施强制人工确认（例如 Claude Code 对外部路径导入弹窗审批）。

与此同时，在法律合规层面，自 2026 年 8 月全面生效的**《欧盟人工智能法案》（EU AI Act）**要求高风险 AI 系统必须保留长达 **10 年的可追溯审计日志**，这与 **GDPR 第 17 条“被遗忘权（Right to Erasure）”**及“目的限制原则（Purpose Limitation）”产生了直接的架构级冲突（[Zylos Research, 2026](https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/)）。当用户要求删除某条个人隐私事实，但该事实已被睡眠巩固进程（如 Cognee `memify` 或 Claude Dreaming）蒸馏合并进高阶语义图谱甚至参数化权重中时，如何实现可证明的“靶向记忆卸载（Targeted Memory Unlearning）”与命名空间硬隔离，已成为 2026 年企业级记忆底座必须跨越的合规门槛。

| 前沿研究方向 | 核心代表作与发布机构 (2025–2026) | 核心机制突破 | 解决的传统架构痛点与实证收益 |
| :--- | :--- | :--- | :--- |
| **1. 睡眠期计算与离线巩固 (Sleep-Time & Dreaming)** | Letta [arXiv:2504.13171](https://www.letta.com/blog/sleep-time-compute/)、Anthropic Claude Dreaming / AutoDream、Spheron KV Prefill | 利用空闲算力在后台异步重写 Core Memory、跨会话去重提炼隐性模式、预填充共享 KV Cache | 消除在线记忆编辑延迟；首字延迟 (TTFT) 降低 5 倍；Harvey 法律复杂任务完成率提升约 6 倍 |
| **2. 强化学习原生记忆策略 (RL-Driven Memory)** | [AgeMem (arXiv:2601.01885)](https://arxiv.org/abs/2601.01885)、[Memory-R1 (arXiv:2508.19828)](https://arxiv.org/abs/2508.19828)、[MemRL (arXiv:2601.03192)](https://arxiv.org/abs/2601.03192) | 将 LTM/STM 的增删改查摘要作为 RL 策略动作；通过分阶段 Step-wise GRPO / PPO 或运行时值估计优化 | 告别脆弱的人工启发式规则；Memory-R1 仅凭 152 对样本微调即在三大基准提升 +15.76 分 |
| **3. 卡片盒自组织与元进化 (Zettelkasten & Meta-Evolution)** | [A-Mem (NeurIPS 2025, arXiv:2502.12110)](https://arxiv.org/abs/2502.12110)、[MemEvolve (arXiv:2512.18746)](https://arxiv.org/abs/2512.18746) | 原子笔记双向动态链接、新记忆触发旧记忆上下文演化；元进化算法自动搜索记忆系统拓扑 | 突破静态图谱 Schema 僵化限制，实现开放域多跳知识的动态重组与架构自适配 |
| **4. 程序性技能蒸馏与迁移 (Procedural Memory)** | [Mem^p (ACL 2026, arXiv:2508.06433)](https://arxiv.org/abs/2508.06433)、LangMem Dynamic Prompt Update | 将历史轨迹蒸馏为细粒度分步指令与高层脚本抽象，支持失败自动修正与过期废弃 | 强模型构建的程序性记忆库可直接迁移赋能轻量小模型，显著减少重复试错轮次 |
| **5. 测试时参数化神经记忆 (Test-Time Parametric)** | Google Research [Titans (NeurIPS 2025, arXiv:2501.00663)](https://arxiv.org/abs/2501.00663) + MIRAS | 推理期基于梯度“惊奇度 (Surprise Metric)”、动量与自适应权重衰减实时更新深度神经记忆模块 | 突破注意力 $O(N^2)$ 瓶颈，单模型有效上下文突破 200 万 Token 且大海捞针精度领先 |
| **6. 记忆防投毒安全与合规 (Security & Governance)** | [arXiv:2604.16548](https://arxiv.org/html/2604.16548v1)、MINJA、A-MemGuard、Unit 42 威胁情报 | 数据源血缘信任分级、上下文联合安全审计、非破坏性双时态归档与命名空间硬隔离 | 拦截单独审查漏检率达 66% 的组合型记忆投毒；兼顾 GDPR 擦除权与 AI 法案审计要求 |

## 五、 架构选型指南与生产级 Agent Harness 记忆设计蓝图

综合上述对主流工业级 Agent Harness、开源记忆中间件及学术前沿的全面调研，工程团队在为自身智能体系统设计或选型记忆模块时，应摒弃“一刀切”地堆砌复杂图数据库或盲目信赖超大上下文窗口的做法，转而遵循以下四条经过 2026 年生产检验的工程决策准则：

1. **以“50 轮会话长度”与“跨会话状态可变性”作为第一道架构分水岭。** 若智能体服务于单次短周期任务（交互轮次稳定在 50 轮以内，且无需跨天继承复杂状态），切勿过度设计外部记忆数据库——直接采用**“Prompt Cache 感知的滑动窗口 + 零开销旧工具输出占位符替换（如 Claude Code Layer 1）+ 轻量级静态配置（`CLAUDE.md` / `AGENTS.md`）”**即可获得最低的延迟与最高的投入产出比。一旦单次任务超过 50 轮或需要跨越多日会话，纯上下文窗口的召回率将不可避免地跌破 50%，此时必须引入独立的压缩与外部记忆层（[NiteAgent, 2026](https://niteagent.com/blog/agent-memory-research-2026/)）。
2. **严格区分“程序性指令注入”与“语义/情景数据检索”。** 永远不要将智能体的操作规范、编码风格与工具调用禁忌（程序性记忆）混入向量数据库靠相似度检索——因为向量检索的不确定性会导致智能体间歇性“忘掉”核心安全规范。程序性记忆必须以**确定性文件层级（如根目录 `AGENTS.md` + 按路径 Glob 触发的 `.claude/rules/*.md` + 按需加载的 `SKILL.md`）**注入系统提示词；同时，将单文件严格控制在 150–200 行以内，并在每次上下文压缩（Compaction）后从磁盘强制重载根指令文件与最近修改的核心工作文件（[GuildBuild, 2026](https://www.guildbuild.ca/insights/agent-memory-after-compaction)）。
3. **根据“Token 预算约束”与“时序状态演化需求”精准选型外部记忆底座。** 在跨会话长期记忆选型上：（1）若追求极致的多会话召回精度且希望保持零外部数据库依赖，优先采用 **ByteRover 的层级文件系统 Context Tree + 五级渐进检索架构**（LoCoMo 96.1%）；（2）若处于高并发、严苛单会话成本上限（如单会话记忆预算低于 $0.01）的商业环境，优先选择 **Mem0 的高密度提取与跨会话去重管线**（仅占 3%–8% Token 开销，LongMemEval ~90%）；（3）若业务核心涉及随时间频繁变更的合同状态、客户画像或医疗记录，且经常需要回答“在时间点 $T$ 之前事实是什么”，则必须采用 **Zep / Graphiti 的双时态知识图谱（Bi-Temporal Graph）**；（4）若开发团队同时混用 Cursor、Claude Code 与 Codex CLI，应通过 **Supermemory 或 Memorix 等基于 MCP 协议的独立记忆服务**构建跨工具共享记忆层（[Zenn, 2026](https://zenn.dev/taka4/articles/7221be19b93603?locale=en)）。
4. **将记忆维护与整合彻底异步化（Async Background Consolidation）。** 无论是 Mem0 的冲突消解、ByteRover 的知识树修剪，还是 Letta 的 Sleep-Time Agent 与 Anthropic 的 Claude Dreaming，2026 年所有顶尖架构均达成了一致共识：**绝不在用户等待响应的主推理循环（Hot Path）中同步执行耗时的记忆整理**。生产级 Harness 应当通过异步消息队列，在会话空闲期或夜间批处理窗口唤醒高推理能力的后台巩固模型（Consolidation Worker），执行跨会话去重、矛盾事实调解、程序性技能蒸馏以及组合型记忆投毒安全审计，从而在不牺牲丝毫在线响应速度的前提下，赋予智能体随时间复利进化的长期智慧。

## 六、 参考文献与核心信息源 (References)

1. Hu, Y., Liu, S., Yue, Y., Zhang, G., et al. (2026). *Memory in the Age of AI Agents: A Survey*. arXiv preprint arXiv:2512.13564v2. https://arxiv.org/abs/2512.13564
2. Liu, J., Zhao, X., Shang, X., & Shen, Z. (2026). *Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems*. arXiv preprint arXiv:2604.14228v3. https://arxiv.org/abs/2604.14228
3. Anthropic. (2026). *How Claude remembers your project (CLAUDE.md, AGENTS.md, .claude/rules, and Auto Memory)*. Claude Code Documentation. https://code.claude.com/docs/en/memory
4. Justin3go. (2026). *Shedding Heavy Memories: Context Compaction in Codex, Claude Code, and OpenCode*. https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode
5. GuildBuild Inc. (2026). *AI Agent Memory After Compaction: Claude Code & Codex (KN-09 Load and Compaction Map)*. https://www.guildbuild.ca/insights/agent-memory-after-compaction
6. Lee, K. (2026). *Investigating How Codex Context Compaction Works*. https://kangwooklee.com/blogs/codex_context_compaction.html
7. Wang, X., Rosenberg, S., Michelini, J., Neubig, G., et al. (2026). *The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents*. MLSys 2026 / arXiv:2511.03690v2. https://arxiv.org/abs/2511.03690
8. Zylos Research. (2026). *AI Agent Memory Architectures: From Context Windows to Persistent Knowledge*. https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/
9. NiteAgent Editorial. (2026). *Agent Memory in 2026: A Research Survey Breakdown for Practitioners (LoCoMo, LongMemEval, BEAM, ByteRover & Mem0)*. https://niteagent.com/blog/agent-memory-research-2026/
10. GeniOS Intelligence. (2026). *Open-Source Memory Layers for AI Agents: The Complete 2026 Comparison*. https://thegenios.com/blog/open-source-memory-layers-2026/
11. Oppy / Zenn. (2026). *Comparing AI Agent Memory: Built-in, External, and the New Layer of "Dreaming"*. https://zenn.dev/taka4/articles/7221be19b93603?locale=en
12. Lin, K., Snell, C., Wang, Y., Packer, C., Wooders, S., Stoica, I., & Gonzalez, J. E. (2025). *Sleep-time Compute: Beyond Inference Scaling at Test-time*. Letta Research / arXiv:2504.13171. https://www.letta.com/blog/sleep-time-compute/
13. Yu, Y., Yao, L., Xie, Y., et al. (2026). *Agentic Memory (AgeMem): Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents*. ACL 2026 SAC Highlight / arXiv:2601.01885v3. https://arxiv.org/abs/2601.01885
14. Yan, S., Yang, X., et al. (2026). *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning*. arXiv:2508.19828v5. https://arxiv.org/abs/2508.19828
15. Zhang, S., et al. (2026). *MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory*. arXiv:2601.03192v2. https://arxiv.org/abs/2601.03192
16. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). *A-MEM: Agentic Memory for LLM Agents*. NeurIPS 2025 / arXiv:2502.12110v11. https://arxiv.org/abs/2502.12110
17. Fang, R., Liang, Y., Wang, X., Zhang, N., et al. (2026). *Mem^p: Exploring Agent Procedural Memory*. ACL 2026 Findings / arXiv:2508.06433v4. https://arxiv.org/abs/2508.06433
18. Behrouz, A., Zhong, P., & Mirrokni, V. (2025). *Titans: Learning to Memorize at Test Time*. NeurIPS 2025 / arXiv:2501.00663. https://arxiv.org/abs/2501.00663
19. Chhikara, P., et al. (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. ECAI 2025 / arXiv:2504.19413. https://arxiv.org/pdf/2504.19413
20. Saffari, O. (2026). *Windsurf vs Cursor After Devin Desktop: Agent Workflow, Memory & Control Planes*. https://omidsaffari.com/blog/windsurf-vs-cursor-after-devin-desktop
