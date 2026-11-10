# Day 05 · 手撕 Naive RAG 系统（前半段）

> 2026-10-28（周三）｜第 5 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

> ⚠️ **欠账提醒：Day 3、Day 4 两个学习日的打卡项仍未验收**（Day 3 缺 `256/30` 对照；Day 4 只做了余弦打分、没建 FAISS 索引）。今天不额外开补课，而是把它们直接焊进今天的开工前置——因为 Day 5 的端到端问答**必须**有一个能落盘、能重载的索引，否则今天根本跑不起来。补完顺手把 Day 3/4 一起标验收。

## 今日主题 + 学习目标

**主题**：手撕 Naive RAG 系统（Day 5-6 合并行，今天做前半段：检索层 + 端到端问答脚本）  
**目标**：整合 FastAPI + LangChain，完成端到端文档问答 API

递进关系：Day 3 把文档切成块，Day 4 把块压成向量塞进索引，Day 2 有一条能调本地模型的 chain，Day 1 有能跑的路由——四块零件到今天第一次真正装成一台机器。今天的产出 `rag.py` 是后面 20 个学习日的**基线系统**：Day 9 的混合检索、Day 10-11 的 RAGAs 评估、Day 12 的 Milvus、Day 23 的 Redis 缓存、Day 28 的压测，全都改这一个文件，而不是每次重写。

今天只做到「脚本跑通端到端」，**先不包 FastAPI**（明天 Day 6 再包成接口）。理由：接口会掩盖逻辑错误，先用 `python rag.py` 把 retrieve → context → LLM 这条链的中间态看清楚。

## 资源清单（3 条）

1. **RAG from Scratch**（LangChain 官方系列，今天看第 1-3 集就够：Indexing / Retrieval / Generation 三段分别对应你手头的零件）：<https://github.com/langchain-ai/rag-from-scratch>
2. **LlamaIndex 官方文档**（只查两块：`VectorStoreIndex` 的持久化、`as_retriever` 的 `top_k`／`similarity_top_k`；别通读）：<https://docs.llamaindex.ai/>
3. **动手学大模型应用开发**（Datawhale，中文，第 4 章 RAG 的实现脉络可对照你的写法）：<https://github.com/datawhalechina/llm-universe>

> 用你自己的栈：切分仍用 `SentenceSplitter`，索引仍用 Day 4 的 `faiss.IndexFlatIP` + `faiss.write_index`，生成走 Day 2 已验证可用的本地 `127.0.0.1:9527`。**不引入新的向量库**，Milvus 是 Day 12 的事。

## 时间安排建议（约 2.5 小时 · 40% 读 / 40% 写 / 20% 记录）

| 时段 | 内容 | 类型 |
|---|---|---|
| 0:00-0:30 | **前置补账**：`day03.py` 加 `split(size, overlap)` 打印块数/均值跑两组；`day04.py` 补 `IndexFlatIP` 建索引 → `write_index` → `read_index` → `search` | 动手 |
| 0:30-1:10 | 读 RAG from Scratch 第 1-3 集，只回答一个问题：retrieve 出来的东西**以什么形式**进 prompt（裸文本？带编号？带 metadata？） | 读懂即可 |
| 1:10-2:20 | 写 `day05.py`：`build_index()` / `load_index()` / `retrieve(q, k)` / `answer(q)`，落盘 `faiss.index` + `chunks.json` | 动手 |
| 2:20-2:30 | 笔记区：记下 3 个问题各自的命中块号、答案是否被引用块支撑、以及一次答错的归因 | 记录 |

## 今日手撕打卡项

- [ ] **动手**：把 Day 3 的分块 + Day 4 的向量化封装成 `build_index()`，落盘两样东西：`faiss.index`（纯向量）和 `chunks.json`（`[{"idx", "text", "file", "node_id"}]`）。再写 `load_index()`，**只读盘、不重新编码**也能完成一次检索（这条是 Day 7 Docker 能跑的前提）
- [ ] **动手**：`answer(q)` = `retrieve(q, k=3)` → 拼 context（每块带 `[1] [2] [3]` 编号）→ 走 Day 2 的本地模型 chain 出答案。用 3 个问题跑通：① 原文可直答 ② 换个说法问 ③ **文中根本没有**。打印答案 + 命中块号 + 引用原文前 60 字

准备（在 `D:/Study/AgentGuide` 下）：

```bash
mkdir -p "learning/Daily/Day-05-手撕NaiveRAG系统"
# sentence-transformers / faiss-cpu 昨天已装，今天不需要新依赖
```

参考骨架（自己敲，注意 4 个注释里的坑）：

```python
import json, faiss
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATA = Path(r"D:\Study\AgentGuide\learning\Daily\Day-03-RAG加载与分割\data")
WORK = Path(r"D:\Study\AgentGuide\learning\Daily\Day-05-手撕NaiveRAG系统")
MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")

def build_index(size=1024, overlap=200):
    docs = SimpleDirectoryReader(input_dir=str(DATA)).load_data()   # 坑1：别用 [0]，多文件全要
    nodes = SentenceSplitter(chunk_size=size, chunk_overlap=overlap).get_nodes_from_documents(docs)
    texts = [n.text for n in nodes]
    emb = MODEL.encode(texts, normalize_embeddings=True, batch_size=32).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(WORK / "faiss.index"))
    # 坑2：向量和原文必须分两份存，且靠行号对齐；这份 json 就是「Chroma 帮你做的事」
    (WORK / "chunks.json").write_text(json.dumps([
        {"idx": i, "text": n.text, "file": n.metadata.get("file_name"), "node_id": n.id_}
        for i, n in enumerate(nodes)
    ], ensure_ascii=False, indent=2), encoding="utf-8")

def load_index():
    index = faiss.read_index(str(WORK / "faiss.index"))
    chunks = json.loads((WORK / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks

def retrieve(q, k=3):
    index, chunks = load_index()
    # 坑3：bge 的 query 要加检索指令前缀，且必须和建库时一样以「向量」对齐
    qv = MODEL.encode([f"为这个句子生成表示以用于检索相关文章：{q}"], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, k)
    return [{"score": float(s), **chunks[i]} for s, i in zip(scores[0], ids[0])]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "只依据给定片段回答；片段不足以回答时，直接说「资料中没有提到」，不要凭常识补充。回答后用 [序号] 标注依据。"),
    ("user", "片段：\n{context}\n\n问题：{question}"),
])

llm = ChatOpenAI(model="<你 day02.py 里那个模型名>", base_url="http://127.0.0.1:9527/v1",
                 api_key="not-needed", temperature=0)
chain = PROMPT | llm | StrOutputParser()

def answer(q, k=3):
    hits = retrieve(q, k)
    ctx = "\n\n".join(f"[{i+1}] ({h['file']}) {h['text']}" for i, h in enumerate(hits))
    out = chain.invoke({"context": ctx, "question": q})
    print(out)
    for i, h in enumerate(hits):
        print(f"  [{i+1}] {h['score']:.3f}  {h['file']}  {h['text'][:60]}")
    return out
```

四个必须弄懂的点：

- **索引文件不是源码**（坑2 的延伸）：`faiss.index` 是二进制派生物，Day 7 的 Docker 镜像里靠 `build_index()` 现场重建，或作为 artifact 挂载。弄混这一步，镜像会莫名其妙几个 G 或者应用启动就崩
- **`chunks.json` 才是系统的真相**：向量只负责排序，答案里引用的原文、文件名、块号全来自这份 json。它和 index 的行号一旦错位（重新切分却忘了重建索引），检索会「分数很高但内容对不上」——这是 RAG 最难查的一类 bug
- **`top_k` 是第一个要调的旋钮**：k=3 时若引用块里压根没答案，问题在检索不在模型；k 加到 5-8 常能救回来，代价是 context 变长、模型更容易抓错重点。今天只记数字，Day 10 用 RAGAs 量化
- **「文中没有」要能答出来**：Naive RAG 最典型的翻车是硬编。system prompt 里那句「不足以回答就说没有」是基线，Day 11 的 Faithfulness 指标就是在量这件事

## 验收标准

`python day05.py` 一条命令跑完 3 个问题，**第三个（文中没有的）能明确回答「资料中没有提到」而不编造**，前两个答案的每句关键结论都能在打印出的引用块里找到对应文字；重启进程后（不重新 `build_index`）仍能检索，说明落盘/重载链路通了。

## 昨日回顾（Day 4，状态：待完成）

读了 `learning/Daily/Day-04-RAG向量化与存储/day04.py`（21:56 最后修改，10777 字的 `Gemini-Study.md` 说明文档是认真读过的）：

- ✅ 已完成：`bge-small-zh-v1.5` + `device="cuda"` 编码通了，`normalize_embeddings=True` 用对了，逐块余弦打分也打出来了
- ❌ 未完成：**没有 `faiss.IndexFlatIP`、没有 `write_index`/`read_index`、没有 `top-3`**——即打卡项②整条没做，打卡项①的「相近 vs 不相关」对照也没显式跑（现在是对一个 query 打全部分数，不是两组对照）
- ⚠️ 顺手改：`load_data()[0]` 仍只取第一份文档（`data/` 里现在只有 1 个 MD 所以没暴露，但 Day 7 换成多文档就会静默丢数据）；`os.environ["HF_ENDPOINT"]` 放在 import 之前是对的，保持
- 📌 观察：手动写 `np.dot / norm` 算余弦没问题，但归一化后内积就等于余弦，`index.search` 会直接给你这个数，不用再算一遍

**15 分钟补做**（就是今天时间表 0:00-0:30 的前半段）：

1. 把 day04 的 `document_embeddings` 塞进 `faiss.IndexFlatIP(512)`，`write_index` 存盘、`read_index` 读回、`search(qv, 3)` 打三行 `(分数, 块号, 前 80 字)`
2. Day 3 那边补一行 `split(1024,200)` 与 `split(256,30)` 的块数/平均字符数对比
3. 两个数字回我一句「Day 3 完成 N1/N2，Day 4 完成 top3 分数 a/b/c」，我把两行都标已完成

## 进度提示

- 距第 7 个学习日（周度总结，2026-11-03 周二）还剩 **2 个学习日**（Day 6、Day 7）
- Day 7 要交的是「Naive RAG 项目 Docker 化跑通」，从今天起就要攒三样：① `build_index()` 能在容器里一键重建（今天就在写）② `pyproject.toml` + `uv.lock` 里 `sentence-transformers` 会拖进 torch，镜像体积预计 3-5 G，要考虑 `--no-cache` 装 CPU 版或换 `faiss-cpu` 之外的裁剪 ③ README 里那段「模型 / chunk_size / top_k / 归一化」选型说明——Day 14 对比优化前后就靠它
- 第 2 周起（Day 8）全部建立在这套基线系统上，今天这套封装越干净，后面改造成本越低

---

## 笔记区

**今日卡点**

-

**关键代码片段**

-

**待补知识**

-

---

## 验收记录（2026-11-01 周日，实跑通过）

- 索引规模：153 块，块字符数 min/avg/max = 49 / 275.1 / 667，单文件 `agent_harness_memory_research_2026.md`
- Q1「Claude Code 的记忆是怎么分层实现的」→ 双轨架构 + 四级作用域，引用 `[Doc 2] [Doc 3]`；top-3 分数 0.712 / 0.711 / 0.702
- Q2「换个说法」→ 结论一致，引用 `[Doc 2] [Doc 3]`
- Q3「Docker 镜像多大（文中没有）」→ **「资料中未提及。」** 兜底正确，top-1 分数仅 0.560
- 结论：Day 5 验收通过。遗留 5 项改进（打印命中块、query 指令前缀、参数统一、索引新鲜度、`-1` 越界）并入 Day 6。
