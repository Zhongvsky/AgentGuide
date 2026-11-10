# Day 04 · RAG Part 2：向量化与存储

> 2026-10-27（周二）｜第 4 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

## 今日主题 + 学习目标

**主题**：RAG Part 2：向量化与存储  
**目标**：理解 Embedding 原理，用 FAISS/Chroma 构建本地向量索引

递进关系：Day 3 你手上已经有了一堆 `Node`（带 metadata 的文本块），但那一堆还是「死的字符串」——只能逐个读，没法按意思找。今天的活是把每个块压成一个固定维度的向量，塞进索引，让「按语义找块」变成一次毫秒级的近邻查询。这是 RAG 能用的分水岭：没有索引，检索就是暴力遍历；有了索引，检索才是产品。

今天**不接 LLM、不出答案**。产出物是明天的输入：Day 5-6 把「索引 + Day 2 的 chain + Day 1 的 FastAPI」拼成端到端问答，Day 7 再 Docker 化。

## 资源清单（3 条）

1. Sentence Transformers 官方文档（今天主读，重点看 `SentenceTransformer` 用法与模型选择表）：<https://www.sbert.net/>
2. FAISS 官方 Wiki（主读 Getting started，只看到能建索引 + 存盘加载为止）：<https://faiss.ai/> ／ Wiki：<https://github.com/facebookresearch/faiss/wiki>
3. Chroma 官方文档（**读懂即可**，今天不用它；它的价值是把「向量 + metadata + 持久化」打包管起来，Day 12 上 Milvus 时你会回来对比）：<https://docs.trychroma.com/>

> **待确认**：本地 `127.0.0.1:9527` 端点是否暴露 `/v1/embeddings`，此前没验证过。今天**不依赖它**，直接用 Sentence Transformers 在本机跑 embedding（你的 5060 Ti 16G 足够），模型文件首次下载约几百 MB。

## 时间安排建议（约 2.5 小时 · 40% 读 / 40% 写 / 20% 记录）

| 时段 | 内容 | 类型 |
|---|---|---|
| 0:00-0:50 | 读 Sentence Transformers 的 Embedding 章节 + FAISS Getting started，只搞清楚三件事：向量怎么来、相似度怎么算、索引怎么存 | 读懂即可 |
| 0:50-2:05 | 敲 `day04.py`：Day 3 的 nodes → 编码 → FAISS 建索引 → 存盘/加载 → 3 个 query 检索 top-3 | 动手 |
| 2:05-2:30 | 填笔记区：记下你选的模型、维度、归一化方式，以及一次「检索失败」的案例分析 | 记录 |

前置 15 分钟（补 Day 3 遗留，见文末「昨日回顾」）。

## 今日手撕打卡项（拆自第 1 周清单第 3 项「30 分钟手撕 Naive RAG」的中段）

- [ ] **动手**：用 `SentenceTransformer` 把 Day 3 的 nodes 编码成向量，打印 `emb.shape`；再取一对**语义相近**的句子和一对**毫不相干**的句子，算出两组余弦相似度，确认前者明显更高
- [ ] **动手**：`faiss.IndexFlatIP` 建索引 → `write_index` 存盘 → 重新加载 → 用**3 个中文 query** 各检索 top-3，打印 `(得分, 块序号, 该块前 80 字)`，其中至少 1 个 query 要故意用「换个说法」而非原文关键词

准备依赖（在 `D:/Study/AgentGuide` 下）：

```bash
uv add sentence-transformers faiss-cpu
mkdir -p "learning/Daily/Day-04-RAG向量化与存储"
```

参考骨架（自己敲，注意三个坑的注释）：

```python
import faiss, numpy as np
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

# 0) 复用 Day 3 的加载 + 切分，直接拿 nodes（chunk_size 沿用你昨天选的那组）
docs = SimpleDirectoryReader("learning/Daily/Day-03-RAG加载与分割/data").load_data()
nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=200).get_nodes_from_documents(docs)
texts = [n.text for n in nodes]

# 1) 编码。bge 系列中文友好、512 维、跑得快；追求效果可换 BAAI/bge-m3（1024 维，更慢）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")
emb = model.encode(texts, normalize_embeddings=True, batch_size=32).astype("float32")
print("shape:", emb.shape)   # (块数, 512)

# 坑1：normalize_embeddings=True 才有「内积 == 余弦」。不归一化时内积被向量模长带偏，分数不可比
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, "learning/Daily/Day-04-RAG向量化与存储/faiss.index")

index2 = faiss.read_index("learning/Daily/Day-04-RAG向量化与存储/faiss.index")  # 验证持久化往返
for q in ["如何提升检索的准确率", "这个系统怎么部署", "文档被切碎了会有什么问题"]:
    # 坑2：bge 做检索时 query 要加指令前缀，效果差别不小
    qv = model.encode([f"为这个句子生成表示以用于检索相关文章：{q}"], normalize_embeddings=True).astype("float32")
    scores, ids = index2.search(qv, 3)
    for s, i in zip(scores[0], ids[0]):
        print(f"{s:.3f}  #{i}  {texts[i][:80]}")
    # 坑3：索引里只有向量和行号，metadata 得自己按行号另存（存一份 texts + metadata 的 json），
    #       这就是 Chroma/Milvus 帮你做的事（Day 12）
```

三个必须弄懂的点：

- **Embedding 是「按意思」而不是「按字面」**：语义近 → 向量夹角小 → 余弦大。代价是它对精确字串很弱，型号、错误码、人名这类专有名词经常检索不到——这正是 Day 9 要上 BM25 混合检索的原因，今天先记住这个弱点
- **归一化决定分数含义**：`normalize_embeddings=True` 后所有向量落在单位球面，内积就是余弦，取值 [-1, 1]；不归一化时内积 = 模长 × 余弦，长文本天然占优，排序会被「块有多长」污染
- **索引只解决「找」，不解决「懂」**：FAISS 只存 float 向量和行号，`metadata`、原文、页码全要你自己另存并按行号对齐。Day 3 说的「metadata 是标注引用的地基」，今天你会亲手体会到为什么要额外维护这份映射

## 验收标准

能报出 3 个 query 各自的 top-3 命中（块前 80 字 + 余弦分），并指出至少一处「检索结果其实不相关」的案例、说出可能原因（切块太粗 / 字面不匹配 / 语义漂移），同时能解释为什么必须先归一化再算内积。

## 昨日回顾（Day 3，状态：待完成）

读了 `learning/Daily/Day-03-RAG加载与分割/day03.py`，进展是这样的：

- ✅ 打卡项①基本达成：`SimpleDirectoryReader` 已加载 `data/` 下的 MD，`SentenceSplitter(1024/200)` 切分并打印了每个 node 的 `id_` / `metadata` / `ref_doc_id`
- ❌ 打卡项②只做了一半：只跑了 `1024/200` 一组，**没有 `256/30` 的对照组，也没有打印块数和平均字符数**——而这组对比正是「参数怎么选」的唯一依据
- ⚠️ 两个小问题顺手改掉：`load_data()[0]` 只取第一份文档（多文件时后面全丢），以及 `import` 了 `TokenTextSplitter` / `MarkdownNodeParser` 却没用到

**15 分钟补做方案**（开工前先做，别拖到明天）：

1. 在 `day03.py` 里把切分部分抽成函数 `split(size, overlap)`，返回 `(块数, 平均字符数)`，打印一行对比即可
2. 跑 `1024/200` 与 `256/30` 两组，把两个数字抄到 Day 3 卡片笔记区
3. 顺手把 `[0]` 去掉，遍历所有 docs

做完在对话里回一句「Day 3 完成，N1 / N2」，我就把它标成已完成。

## 进度提示

- 距第 7 个学习日（周度总结，2026-11-03 周二）还剩 **3 个学习日**（Day 5、6、7）
- Day 7 要交「Naive RAG 项目 Docker 化跑通」，现在开始攒：今天这份 `faiss.index` 的**构建脚本**（Docker 里要能一键重建索引，别把 index 文件当源码提交）、`pyproject.toml`/`uv.lock`（依赖会多出 sentence-transformers / faiss，注意镜像体积）、一份「模型 + chunk_size + 归一化方式」的选型说明（直接写进 README）
- 之后是 Day 14（混合检索 + Reranker + Milvus）、Day 21（研究助手 Agent），暂不用管

---

## 笔记区

**今日卡点**

-

**关键代码片段**

-

**待补知识**

-
