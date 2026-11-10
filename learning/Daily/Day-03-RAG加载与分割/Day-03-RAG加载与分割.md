# Day 03 · RAG Part 1：加载与分割

> 2026-10-26（周一）｜第 3 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

> ⚠️ **欠账提醒**：打卡表里 Day 1、Day 2 仍是「待完成」，连续 3 个学习日没有验收记录。但 `learning/Daily/` 下已有 `day01.py`、`day02.py`（且有 `__pycache__`，说明跑过），所以问题是**没验收、没留记录**，不是没学。今天安排 15 分钟补验收即可，不回退；若今天结束仍未验收，明天自动把 next_day 回退到 3，用 Day 4 的时间补 Day 1-3。

## 今日主题 + 学习目标

**主题**：RAG Part 1：加载与分割
**目标**：掌握 PDF/MD 等格式文档的加载与文本分块策略

递进关系：Day 2 的 `prompt | model | parser` 是引擎，但它只会张嘴不会查资料——今天开始给它接**外部文档**，这是 RAG 的入口。今天产出的是 `Document` 和 `Node`（一堆带 metadata 的文本块），Day 4 把这些块交给 Embedding + FAISS 变成向量索引，Day 5-6 才能组装成端到端问答。**今天不碰向量库，也不出答案**，只解决「把文件变成干净可检索的小块」。

## 资源清单（3 条）

1. LlamaIndex Node Parser 官方文档（今天主读，只读 Get Started + SentenceSplitter 参数）：<https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/>
2. Docling 官方文档（**读懂即可**，PDF 版面/表格解析，Day 13 才动手）：<https://docling-project.github.io/docling/>
3. MinerU GitHub（**读懂即可**，中文与复杂 PDF 效果好的开源方案）：<https://github.com/opendatalab/MinerU>

> **待确认**：`Unstructured` 的本地开源包文档已整体迁往 Transform 云服务（需 API key），今天不依赖它，走 LlamaIndex 自带的 `SimpleDirectoryReader` + `pypdf`。如果你的 PDF 是扫描件（图片型），今天大概率抽不出文字——那是 OCR 的活，记到「待补知识」里，Day 13 解决。

## 时间安排建议（约 2.5 小时 · 40% 读 / 40% 写 / 20% 记录）

| 时段        | 内容                                                              | 类型   |
| --------- | --------------------------------------------------------------- | ---- |
| 0:00-0:15 | **补验收** Day 1/Day 2（见下方「昨日回顾」）                                  | 动手   |
| 0:15-1:15 | Node Parser 文档 + 想清楚三件事：Loader 产出什么、Splitter 按什么切、metadata 有什么用 | 读懂即可 |
| 1:15-2:10 | 敲 `day03.py`：加载 → 两种参数切分 → 打印对比数字                               | 动手   |
| 2:10-2:30 | 填笔记区，写清你选的 chunk_size 及理由                                       | 记录   |

## 今日手撕打卡项（拆自第 1 周清单第 3 项「30 分钟手撕 Naive RAG」的前半段）

- [ ] **动手**：`SimpleDirectoryReader` 加载 `data/` 下的 1 个 PDF + 1 个 MD，打印每个 `Document` 的 `file_name`、总字符数、前 200 字
- [ ] **动手**：用 `SentenceSplitter` 把同一份文档切两遍（`1024/200` 与 `256/30`），打印两次的**块数**和**平均每块字符数**，并各抽 1 块看是否断在半句话上

准备依赖（项目已有 uv，在 `D:/Study/AgentGuide` 下执行）：

```bash
uv add llama-index-core llama-index-readers-file
mkdir -p "learning/Daily/Day-03-RAG加载与分割/data"
```

参考骨架（自己敲，重点看注释里的三个坑）：

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 1) 加载：产出的是 Document（text + metadata），不是字符串
docs = SimpleDirectoryReader("data", recursive=True).load_data()
for d in docs:
    print(d.metadata.get("file_name"), len(d.text), d.text[:200].replace("\n", " "))
    # 坑1：metadata 里有 file_name / page_label 等，这是后面「答案出自第几页」的唯一来源

def split(chunk_size, chunk_overlap):
    sp = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = sp.get_nodes_from_documents(docs)
    chars = [len(n.text) for n in nodes]
    print(f"size={chunk_size} overlap={chunk_overlap} -> {len(nodes)} 块, 平均 {sum(chars)//len(chars)} 字符")
    print("样例:", nodes[0].text[:120], "\n---")
    return nodes

# 2) 切分对比：chunk_size 的单位是 token，不是字符
big = split(1024, 200)
small = split(256, 30)
# 坑2：中文 1 个 token 常常不到 1 个字，所以 1024 token 的块远没有 1024 个汉字
# 坑3：块数变 4 倍不等于质量变好——小块更容易命中关键词但上下文更碎，这是 Day 10 RAGAs 要量化的东西
```

三个必须弄懂的点：

- **Loader 的产物是 `Document`**：`text` + `metadata` 两个字段。metadata 不是装饰，它是检索时按文件/页码过滤、以及回答里标注引用的地基，Day 12 存 Milvus 时也要一起写进去
- **切分是按「分隔符优先级」递归下降的**：先按段落，超了就按句子，再超了才硬切字符。`SentenceSplitter` 的意义就是尽量不把一句话劈成两半，`chunk_overlap` 则是给被劈开的边界留重叠补偿
- **chunk_size 是精度与上下文的权衡**：太小 → 检索命中率高但模型拿到的是碎片；太大 → 语义完整但一块里混着多个主题，向量被平均掉，检索反而变糊。今天你要亲手看到这个数字差异，明天才知道该调哪个

## 验收标准

能报出「同一份 PDF 在 `1024/200` 下切成 N1 块、在 `256/30` 下切成 N2 块」两组具体数字，并说出 N2 明显多于 N1 的原因（token 与字符的换算 + 分隔符回退），以及你最终选哪个参数、为什么。

## 昨日回顾（Day 2 · LangChain 核心概念）

打卡表 Day 1、Day 2 都还是**待完成**——但我看到你其实已经写了代码：

- `learning/Daily/Day-01-FastAP I快速入门/day01.py`：`/`、`/items/`、`/chat` 三个路由，Pydantic 模型都在
- `learning/Daily/Day-02-LangChain 核心概念/day02.py`：`init_chat_model` 接本地 `127.0.0.1:9527`，`/chat_lcel_invoke` 已经接上 chain；流式那段注释掉了

所以不重做，花 15 分钟**补验收**（这三条过了，Day 1/2 我直接给你标已完成）：

1. （4 分钟）`cd learning/Daily/"Day-02-LangChain 核心概念" && uv run uvicorn day02:app --reload`
2. （4 分钟）`/docs` 里调 `POST /chat_lcel_invoke` 传 `{"message": "用一句话解释 RAG"}`，拿到模型真实生成的回答而不是 `You said: ...`
3. （4 分钟）把 `Chain.stream` 那段注释解开，用 `astream` 版跑通，`curl -N` 能看到逐块吐出（这是第 4 周异步的伏笔）
4. （3 分钟）在对话里回一句「Day1/2 过了」+ 一句话收获

## 进度提示

- 距第 7 个学习日（周度总结，2026-11-03 周二）还剩 **4 个学习日**（Day 4、5、6、7）
- Day 7 要交「Naive RAG 项目 Docker 化跑通」，现在开始攒的素材：`data/` 里**真实可用**的样例文档（今天放进去，别用扫描件）、今天的 loader+splitter 脚本、`pyproject.toml`/`uv.lock`（Docker 化直接照抄依赖）、一份能写进 README 的「参数选择理由」
- 之后依次是 Day 14（混合检索 + Reranker + Milvus）、Day 21（研究助手 Agent），现在不用管

---

## 笔记区

**今日卡点**

-

**关键代码片段**

-

**待补知识**

-
