# Day 06 · 手撕 Naive RAG 系统（后半段：FastAPI 接口封装）

> 2026-11-03（周二）｜第 6 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

## 今日主题 + 学习目标

**主题**：手撕 Naive RAG 系统（Day 5-6 合并行，今天做后半段：把昨天跑通的脚本包成 HTTP 接口）
**目标**：整合 FastAPI + LangChain，完成端到端文档问答 **API**

递进关系：Day 1 学会了路由 + Pydantic 请求体，Day 5 把 retrieve → context → LLM 跑通成脚本。今天这两件事第一次合体：**把 `day05.py` 里函数式的 `build_index` / `retrieve` / `answer` 换个入口暴露出去**，逻辑一行不改，只是从「`python day05.py` 打印到 stdout」变成「`POST /ask` 返回 JSON」。

今天不新增算法，只做三件事：接口化、让引用能自证、把 Day 5 留下的三处硬伤补掉。产出直接是 Day 7 Docker 化的镜像内容——Day 7 要 `docker run` 一个能问的容器，今天必须有 `uvicorn` 能起的入口。

## 资源清单（3 条）

1. **FastAPI · Request Body**（今天最需要的一节：`BaseModel` 嵌套、`Field` 校验，用来定义 `/ask` 的响应结构）：<https://fastapi.tiangolo.com/tutorial/body/>
2. **FastAPI · Response Model**（`response_model` 与返回 dict 的区别，决定你的 `citations` 字段会不会被悄悄裁掉）：<https://fastapi.tiangolo.com/tutorial/response-model/>
3. **FastAPI · Lifespan Events**（索引只在启动时加载一次，不要每个请求都读盘；替代已弃用的 `@app.on_event`）：<https://fastapi.tiangolo.com/advanced/events/>

> 栈不变：`fastapi[standard]` 已在 `pyproject.toml` 里，`uvicorn` 随 standard 一起装了，**今天不新增依赖**。向量仍用 `faiss.IndexFlatIP` + `data.index`/`data.json`，模型仍是本地 `127.0.0.1:9527`。

## 时间安排建议（约 2.5 小时 · 40% 读 / 40% 写 / 20% 记录）

| 时段 | 内容 | 类型 |
|---|---|---|
| 0:00-0:15 | **前置补账**：把 Day 5 的三处遗留改掉（见打卡项 2），不改逻辑只加防御与打印 | 动手 |
| 0:15-1:10 | 读上面三节文档，只回答三个问题：① 索引该放哪（lifespan 还是每个请求）② `response_model` 里 `list[Citation]` 怎么声明 ③ 同步函数 `def` 和 `async def` 在 FastAPI 里对阻塞 IO 的区别 | 读懂即可 |
| 1:10-2:15 | 写 `day06.py`：`POST /ingest`、`POST /ask`，起 `uvicorn` 用 curl / `httpx` 打 3 个问题 | 动手 |
| 2:15-2:30 | 笔记区：记下 3 次请求的 `score` 与 `citations` 条数，以及文外问题的实际最高分 | 记录 |

## 今日手撕打卡项

- [ ] **动手**：新建 `learning/Daily/Day-06-NaiveRAG接口封装/day06.py`，用 `lifespan` 在启动时 `load`（无索引则 `build`）一次，暴露两个端点：
  - `POST /ingest` → `{"built": bool, "chunks": int, "seconds": float}`，`force=true` 时重建索引
  - `POST /ask` → `{"answer": str, "citations": [{"doc_id", "file", "score", "snippet"}], "hit": bool}`
  起服务：`uvicorn day06:app --port 8000`，浏览器打开 `http://127.0.0.1:8000/docs` 能看到两个端点并能直接 Try it out。
- [ ] **动手**：把 Day 5 的三处硬伤补掉，并让引用能自证：
  - `chunks[i]` 遇 FAISS 返回 `-1`（索引条目少于 top_k 时）会 `IndexError` → 先过滤 `i < 0 or i >= len(chunks)` 再取
  - 索引无新鲜度校验 → 把 `chunk_size`/`chunk_overlap`/源文件 mtime 写进 `meta.json`，启动时对不上就重建（别再用「文件存在就复用」）
  - 建库参数三套不一致（Day 3 用 256/30、Day 4 用 512/128、Day 5 用 256/16）→ 今天定一个，写进 `meta.json`，以后所有实验都用它
  - `snippet` 取命中块前 80 字，`score` 原样返回——引用不再靠模型自己标 `[Doc n]`，而是**响应里带真凭据**

自测三条（写在笔记区，别只在终端刷过去）：

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"message":"Claude Code 的记忆是怎么分层实现的"}' | python -m json.tool
```

1. 原文可直答的问题 → `hit=true`，`citations` 有分数、`snippet` 与答案对得上
2. 换说法问同一件事 → 命中的 `file` 与第 1 条大体一致（换个问法不该换到另一篇）
3. 文中没有的问题（例：这套系统部署要多大 Docker 镜像）→ `hit=false`，`answer` 为「资料中未提及」，`citations` 为空

阈值建议 `HIT_THRESHOLD = 0.35`（bge-small-zh + 归一化内积的经验值），**必须把你实测的最高分记下来**——低于它就是没命中，这个数 Day 10-11 做 RAGAs 时会再校准。

## 验收标准

`uvicorn` 起服务后，三个 `curl` 全部返回 200 且 `/ask` 响应同时含 `answer` 与带分数、带 snippet 的 `citations`；文外问题返回 `hit=false` + 「资料中未提及」+ 空 `citations`；`/docs` 页面上两个端点可直接 Try it out。

## 昨日回顾（Day 5 · 2026-10-28）

Day 5 已于 11-01 验收通过：153 块建索引、3 问带 `[Doc n]` 引用、文外问题正确兜底，`build`/`load`/`retrieve`/`answer` 四层已分开——这正是今天能只包一层的原因。

留给今天的 5 条遗留，只补 3 条（另 2 条今天不碰）：

| 遗留 | 今天处理 |
|---|---|
| 不打印命中块与分数，引用无法自证 | 补 → 走 `citations` 返回 |
| 建库参数三套不一致 | 补 → 定一套写 `meta.json` |
| 索引无新鲜度校验 | 补 → `meta.json` 比对，不匹配就重建 |
| `chunks[i]` 未处理 FAISS 的 `-1` | 补 → 越界过滤 |
| query 缺 bge 指令前缀 | **已修，不再动**（`day05.py` 第 56 行已有「为这个句子生成表示…」） |

15 分钟补做方案（如果昨天的验收细节记不清）：只跑 `day05.py` 的三问，把每问 top-1 的 `score` 与 `file` 抄进笔记区，然后直接进今天的 `/ask`——同一个分数口径，Day 7 换 Docker 后才有对照。

## 进度提示

- **距 Day 7（周度总结 + Docker 部署）只剩 1 个学习日，就是明天 11-04 周三。**今天必须攒下两样东西：① 一条**能起服务**的命令（`uvicorn day06:app --host 0.0.0.0 --port 8000`）② 一份完整依赖清单（`uv pip freeze` 输出，Dockerfile 的 `requirements.txt` 直接从这儿抄）。Day 7 只有一天，没这两样就做不出镜像。
- 距 Day 14（混合检索 + Reranker + Milvus）还剩 8 个学习日（11-23），需要提前攒「同一批 query 在纯向量检索下的 top-3 基线」——今天这三条自测记录就是基线第一版，别丢。
- 距 Day 21（研究助手 Agent）还剩 15 个学习日（12-08），暂无动作。
- 节奏说明：本周一（11-02）那次自动下发没触发，所以本周只发 2 个学习日（今天 Day 6、明天 Day 7）。进度表里的里程碑日期已按实际下发重算。

---

## 验收结果（2026-11-03 22:30 实测 · 基本达成，未通过）

实跑 `uvicorn day06:app --port 8011` 逐项打点：

| 项 | 结果 |
|---|---|
| 服务启动 / `/docs` | ✅ 200 |
| `/ask` 可直答 | ✅ 200，top-3 = 0.713 / 0.678 / 0.678 |
| `/ask` 换说法 | ✅ 200，top-3 = 0.734 / 0.707 / 0.702，命中块 7/1/3 与第一问重叠 |
| `/ask` 文外问题 | ✅ `hit=false`、`citations=[]`、答「资料中未提及」 |
| 分数分离度 | ✅ 域内 0.678-0.734 vs 域外 0.202-0.282，`hit_thresh=0.5` 落在宽间隙中，定得对 |
| `def` 非 `async def` | ✅ encode 阻塞走线程池，未卡事件循环（今日"待补知识"自答了） |
| FAISS `-1` 越界 | ✅ 已 `continue` 跳过 |
| **`/ingest` 默认路径** | ❌ **HTTP 500** |
| **`force=true` 建索引** | ⚠️ 200 但产出 **2323 块、平均 14 字** |
| **`meta.json` 新鲜度** | ❌ 未做 |

### 20 分钟补做清单

1. `/ingest` 的 `chunks` 只在 `force` 分支赋值，`return` 里 `len(chunks)` 在 else 分支必炸 → 统一从 `request.app.state` 取：

```python
n = len(request.app.state.chunks)
return {"built": force, "chunks": n, "seconds": time.time() - t0}
```

2. `force=true` 的 64/32 是第四套参数，且 64 token 切出 2323 个 14 字碎片 → 提到模块级 `CHUNK_SIZE, CHUNK_OVERLAP = 512, 128`，lifespan 与 /ingest 共用。
3. `meta.json` 写 `{"chunk_size", "overlap", "files": {name: mtime}, "n_chunks"}`，lifespan 与 /ingest 加载前比对，不一致就重建（现在是「文件在就复用」，会把坏索引当好的）。
4. `citations[0].score` 前加 `if not citations: hit=False`，否则 `top_k=0` 就 500。
5. `hit=false` 时保留 `citations`（只 `hit=false`），否则拿不到文外问题的真实最高分，阈值没法校准。

---

## 笔记区

### 今日卡点

- 

### 关键代码片段

```python
# lifespan 里加载一次索引
```

### 待补知识

- FastAPI `def` vs `async def`：`encode()` 是 CPU 阻塞，放 `def` 走线程池，放 `async def` 会卡住事件循环 —— 待实测确认
-

### 自测记录

| # | 问题 | hit | top-1 score | citations 条数 |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
