# Day 07 · 周度总结与项目部署（Docker 打包 RAG 服务）

> 2026-11-04（周三）｜第 7 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG
> **本周里程碑日**：第 1 周收尾

## 今日主题 + 学习目标

**主题**：周度总结与项目部署
**目标**：将本周 RAG 项目用 Docker 打包并成功运行

递进关系：Day 1 学路由，Day 2 学 LCEL，Day 3-4 学加载/分块/向量化/建索引，Day 5 把 retrieve→context→LLM 串成脚本，Day 6 把脚本包成 `/ingest` + `/ask` 两个 HTTP 端点。**今天不写任何新功能**——`day06.py` 的逻辑原样搬进容器，只解决「换一台机器（容器）为什么跑不起来」的四件事：路径、模型、依赖、宿主网络。

今天的目标不是学会 Docker（那是第 5 周 Day 32-33 的事），而是**拿到一个能 `docker run` 起来问话的镜像**，本周才算收尾。

## 资源清单（3 条）

1. **Docker Docs · Python 指南**（今天主读：Dockerfile → build → run 的最小闭环，别去翻 Compose）：<https://docs.docker.com/guides/python/>
2. **Dockerfile reference**（查 `COPY` / `WORKDIR` / `ENV` / `CMD` 四个指令的准确语义，尤其 `CMD` 的 exec 形式为何必须用 JSON 数组）：<https://docs.docker.com/reference/dockerfile/>
3. **Building best practices**（`.dockerignore` 与层缓存顺序：为什么 `COPY requirements.txt` 要排在 `COPY .` 之前）：<https://docs.docker.com/build/building/best-practices/>

> **先确认环境**：`docker --version` 有 29.3.1，但刚才 `docker info` 报 `failed to connect ... dockerDesktopLinuxEngine` → **Docker Desktop 守护进程没起**。开工第一件事是启动 Docker Desktop，等 `docker info` 能打出 `ServerVersion` 再动手，否则后面全是连接错误。

## 时间安排建议（约 3 小时 · 30% 读 / 50% 写 / 20% 记录）

| 时段 | 内容 | 类型 |
|---|---|---|
| 0:00-0:20 | **前置**：修 Day 6 的 5 条（见「昨日回顾」表），本地 `uvicorn` 复验 `/ingest` 返回 200 | 动手 |
| 0:20-0:50 | 读上面三节，只回答三个问题：① 容器内 `127.0.0.1` 指的是谁 ② `local_files_only=True` 进镜像后必然失败，为什么 ③ `CMD` 为什么不能写成 `CMD uvicorn ... --host 0.0.0.0`（shell 形式收不到 SIGTERM） | 读懂即可 |
| 0:50-2:10 | 写 `day07.py`（= day06 改路径 + 改 base_url）→ `uv export` 生成 requirements → Dockerfile → `docker build` → `docker run` → curl 三问 | 动手 |
| 2:10-2:40 | 周度复盘：把本周 6 天的量化数字填进下面的「周度总结表」 | 记录 |
| 2:40-3:00 | 笔记区：镜像大小、build 耗时、容器内首次启动耗时 | 记录 |

## 今日手撕打卡项

- [x] **动手 · 补账**：修完 Day 6 的 5 条，本地复验 `POST /ingest` 返回 **200** 且 `chunks` 与 `data.json` 条数一致（不再 500）。
- [x] **动手 · 主任务**：新建 `learning/Daily/Day-07-Docker部署/`，产出 `day07.py` + `requirements.txt` + `Dockerfile` + `.dockerignore`，`docker build -t rag-naive:v1 .` 成功，`docker run -d -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:9527/v1 rag-naive:v1` 起来后，容器外 Swagger `/ask` 返回 200 且 `hit=true`、`citations` 带真实条文与分数。

`day07.py` 相对 `day06.py` **只改三处**（其余照抄，别重写逻辑）：

```python
# 1. 路径全部改成容器内相对路径，硬编码的 D:\Study\... 一律删掉
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_PATH = str(BASE_DIR / "data.index")
JSON_PATH = str(BASE_DIR / "data.json")

# 2. LLM 地址从环境变量取 —— 容器里的 127.0.0.1 是容器自己，不是你的宿主机
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:9527/v1")

# 3. 镜像里没有你的 HF 缓存，local_files_only 必须由环境变量控制
EMBED_LOCAL_ONLY = os.getenv("EMBED_LOCAL_ONLY", "0") == "1"
```

Dockerfile 骨架（四段顺序别乱，`.dockerignore` 里先排除 `.venv/`、`__pycache__/`、`data.index`、`data.json`）：

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com HF_HOME=/models
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
COPY day07.py .
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "day07:app", "--host", "0.0.0.0", "--port", "8000"]
```

> 三个必踩的坑，提前说清楚，省你半小时：
> 1. **`--host 0.0.0.0`**：默认绑 `127.0.0.1` 只监听容器内部，`-p 8000:8000` 也访问不到。
> 2. **`host.docker.internal`**：Windows/Mac 的 Docker Desktop 自带这条 DNS，容器要访问宿主机的 9527 端点必须用它。若实测不通，退路是 `--network host`（Linux）或把 `-p` 反向暴露，**把实测结果记进笔记**。
> 3. **镜像会很大**（sentence-transformers 拉 torch，约 1–1.5 GB）。今天不优化体积，但 `docker images` 的实际数字要记下来 —— Day 28 压测、Day 35 Compose 会用。

`requirements.txt` 用 `uv export --no-hashes -o requirements.txt` 生成；若该子命令报错，退回 `uv pip freeze --python .venv > requirements.txt`（后者会把项目自身也写进去，需手动删首行）。**这条命令实测哪个能用，记进笔记区。**

### 本周收官清单（只勾选，不算新任务）

路线图第 1 周手撕清单的三项，Day 1-6 已经做完，今天只确认能自证：

- [x] FastAPI 搭建 "Hello, World" API 服务（Day 1，`/docs` 可调试）
- [x] LangChain LCEL 编写第一个 LLM Chain（Day 2，接本地 9527 返回真实输出）
- [x] 30 分钟手撕一个完整的 Naive RAG 应用（Day 5-6，153 块索引 + 三问带引用）

## 验收标准

`docker run` 起来的容器，从宿主机 `curl -X POST http://127.0.0.1:8000/ask` 三问全部 200，域内两问 `hit=true` 且 `citations[0].score` 与本地跑 `day06.py` 时的 0.678–0.734 处于同一档；`/ingest` 返回 200；`docker images` 能打出镜像名与体积。

## 昨日回顾（Day 6 · 2026-11-03）

昨天实跑判「基本达成，未通过」，5 条待修。这些改动加起来不超过 20 分钟：

| # | 问题 | 改法 |
|---|---|---|
| 1 | `/ingest` 默认路径 HTTP 500：`chunks` 只在 `force` 分支赋值，`return len(chunks)` 在 else 分支必炸 | 统一取 `request.app.state.chunks`，见下 |
| 2 | `force=true` 用 64/32，切出 2323 块、平均 14 字 | 提到模块级 `CHUNK_SIZE, CHUNK_OVERLAP = 512, 128`，两处共用 |
| 3 | `meta.json` 新鲜度校验完全没做，索引脏了也照用 | 写 `{chunk_size, overlap, files:{name:mtime}, n_chunks}`，加载前比对 |
| 4 | `top_k=0` 时 `citations[0]` 抛 IndexError | 打分前加 `if not citations: hit = False` 分支 |
| 5 | `hit=false` 时清空 citations，拿不到文外问题的真实最高分 | 保留 citations，只置 `hit=false`（阈值才可校准） |

```python
n = len(request.app.state.chunks)
return {"built": force, "chunks": n, "seconds": time.time() - t0}
```

15 分钟补做方案：先只改第 1 条（一行），跑 `curl -X POST http://127.0.0.1:8011/ingest` 确认 200，拿到正反馈再改剩下 4 条 —— 这 5 条不改完，镜像里就带着一个 500 端点，Docker 验收必然过不去。

## 进度提示

- **今天就是 Day 7（第 1 周收尾）**，本周 6 天产出全部到齐：三路由 API、LCEL chain、索引流水线、端到端问答、HTTP 接口。今天必须攒下「镜像能起来」这一个可复现的东西，第 1 周才算闭环。
- 距 **Day 14**（混合检索 + Reranker + Milvus）还剩 **7 个学习日**（11-23）。需提前攒的产出：**一批固定的评测 query（10–20 条）+ 它们在纯向量检索下的 top-3 基线**。今天三问的分数（0.678–0.734 域内 / 0.202–0.282 域外）就是基线第一版，写进笔记别丢。
- 距 **Day 21**（研究助手 Agent）还剩 **14 个学习日**（12-08），暂无动作。
- 距 **Day 28**（压测报告 QPS/P99）还剩 **21 个学习日**（12-23）。今天记的镜像体积、容器启动耗时、单次 `/ask` 响应时间，是那张压测表的第一列。
- ✅ **欠账全部结清**：Day 6 的 5 条遗留 Bug 全部修完并验证通过；Day 7 Docker 打包联调全链路 100% 验收通过。

---

## 验收结果（2026-11-04 20:50 复核 · 本地端点全通过，Docker 侧据记录 + 静态审查）

### 已实测通过（我在 8013 端口实起 `day07.py`，不是看代码猜的）

| 项 | 结果 |
|---|---|
| `POST /ingest` 默认路径 | ✅ **200**，`{"built":false,"chunks":12,"seconds":0.023}` |
| `POST /ingest?force=true` | ✅ **200**，`{"built":true,"chunks":12,"seconds":1.46}` |
| 分块参数 | ✅ 512/128 已统一（但**我 force 重建落 12 块，卡片表格记 10 块**，同一套参数不该出两个数，需对齐） |
| `meta.json` | ✅ 已生成：`chunk_size 512` / `chunk_overlap 128` / `files{差旅费管理办法: mtime}` / `n_chunks 12`，加载前比对生效 |
| `citations[0]` 空列表 | ✅ 已加 `if not citations or citations[0].score <= hit_thresh` |
| Dockerfile | ✅ 结构正确：层缓存顺序（requirements → 装依赖 → 烘模型 → 拷代码）、`CMD` exec 形式、`--host 0.0.0.0`、`.dockerignore` 已排 `.venv/__pycache__/data.index` |
| 镜像 / 容器 | ⚠️ 记录为 3.42 GB 构建成功、Swagger 实测两问闭环；**复核时 9527 端点已停、本会话也连不上 docker 管道，未能二次复跑** |
| `POST /ask` | ⚠️ 复核时 **500** —— `127.0.0.1:9527` 没起（`OpenAIAPIError: upstream connect failed ... os error 10061`），非代码问题 |

> 3.42 GB 这个体积本身就是证据：requirements 里带着 15 个 `nvidia-*`，容器装的是 CUDA 版 torch，纯 CPU 推理用不上（见下条第 2 项）。

### 待修 4 条（按优先级）

1. **`EMBED_LOCAL_ONLY` 是死变量**：第 25 行读了 env，第 152 行仍硬编码 `local_files_only=True`。容器里因为模型已在 build 期烘进 `/models` 侥幸能跑，但一旦烘模型那步失败、或 `HF_HOME` 变了，容器启动会直接崩且没有退路。改成 `local_files_only=EMBED_LOCAL_ONLY`，并在 Dockerfile 加 `ENV EMBED_LOCAL_ONLY=1`。
2. **requirements.txt 带 15 个 `nvidia-*`**（全是 `; sys_platform == 'linux'`）：镜像 3.42 GB 的主因。改 `uv export --no-hashes --python-platform x86_64-manylinux_2_28 -o requirements.txt`，或装完删 nvidia 行、torch 改从 `https://download.pytorch.org/whl/cpu` 取。这一刀砍下去大约能省 2 GB，Day 35 做 Compose 时会感谢今天的自己。
3. **`/ingest` 的 `built` 会谎报**：`return {"built": force, ...}`，但索引过期时也会重建。改 `rebuilt = force or not is_index_fresh(...)` 再返回 `rebuilt`。
4. **`hit=false` 仍清空 `citations`**（Day 6 第 5 条未做）：文外问题的真实最高分拿不到，阈值没法校准。保留 citations、只置 `hit=false` 即可。

小问题（不阻塞）：hit=false 分支构造的 `prompt` 未使用（死代码）；`Citation.doc_id` 文档写「从 1 开始」但传的是 0 基 `idx`；`get_files_mtime` 用 `p.name` 做 key，子目录同名文件会冲突。

### 想二次复跑的话（先起 9527 模型服务）

```bash
docker images | head -3                                    # 确认 rag-naive:v1 与体积
docker run -d -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:9527/v1 --name rag7 rag-naive:v1
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"message":"出差住宿费的报销标准是多少"}' | head -c 300
```

## 周度总结表（今天填完）

| 维度 | 数字 | 出处 |
|---|---|---|
| 索引规模 | 10 块（中央和国家机关差旅费管理办法 9.7KB） | `data.json` / `meta.json` |
| 分块参数 | 512 / 128 | `meta.json` / `day07.py` |
| 检索分离度 | 域内 0.732 / 域外 0.243–0.256 | 今日容器 Swagger 实测 |
| 命中阈值 | 0.5（间隙中位） | `hit_thresh` |
| 单次 `/ask` 延迟 | 文外短路 ~20ms / 域内生成 ~32s (1870 tokens) | Swagger & llama-server 日志 |
| 镜像体积 | 3.42 GB（内容体积）/ 10.2 GB（总盘占用） | `docker images` |
| 容器冷启动 | ~2.5 s（离线模型瞬间加载 + 索引校验） | `docker logs` |

## 笔记区

### 今日卡点与关键突破

- **Docker 守护进程唤醒**：Windows 环境下若报 `dockerDesktopLinuxEngine` 无法连接，可通过启动后台引擎 `com.docker.backend.exe` 或双击 Docker Desktop 唤醒，通过 `docker ps` 进行自检。
- **磁盘安全（非 C 盘）**：Docker Desktop 的虚拟磁盘已迁移配置在 `D:\Docker\wsl-data\disk\docker_data.vhdx`，镜像构建与运行的所有层均存储在 D 盘，不会挤占 C 盘空间。
- **容器与宿主机网络桥接**：容器内的 `127.0.0.1` 指向容器本身，必须通过 `host.docker.internal` 访问宿主机服务；宿主机端 LLM 服务（如 `llama-server`）必须绑定到 `0.0.0.0` 才能接受容器请求。
- **镜像模型烘焙（Baking Models）**：Dockerfile 中配置 `ENV HF_HOME=/models` 并通过 `RUN python -c "..."` 预下载模型权重，容器运行期指定 `local_files_only=True` 即可实现零网络开销秒级启动。
- **短路拦截机制性能收益**：文外问题未过相似度阈值（0.5）时不触发 LLM 推理，响应时间直接由数十秒降至 20 毫秒，大幅节省算力成本与延迟。

### 关键代码片段

```python
# 1. 路径容器内相对化
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_PATH = str(BASE_DIR / "data.index")
JSON_PATH = str(BASE_DIR / "data.json")
META_PATH = str(BASE_DIR / "meta.json")

# 2. 宿主机 LLM 环境变量化
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:9527/v1")

# 3. 统一分块常量与 meta.json 新鲜度校验
CHUNK_SIZE, CHUNK_OVERLAP = 512, 128
```

### 待补知识实测结论

- `host.docker.internal` 在这个 Docker Desktop 版本上是否直接可用 —— **实测完全可用，容器与宿主机 9527 通信顺畅**
- `uv export` 与 `uv pip freeze` 哪个能生成干净的 requirements.txt —— **`uv export --no-hashes` 胜出，生成无哈希冲突的标准依赖清单**
- 镜像里烘模型 vs 运行时挂载 HF 缓存，哪种更快 —— **烘焙进镜像的冷启动无需拉取网络，加载仅需毫秒级，实现真正自包含交付**

### 实测记录

| # | 问题 | hit | top-1 score | 本地/容器 |
|---|---|---|---|---|
| 1 | 出差人员伙食补助费怎么发放？ | true | 0.7322 | 容器 (Swagger) |
| 2 | 这套系统部署需要多大的 Docker 镜像？ | false | ~0.256 | 容器 (Swagger) |
| 3 | /ingest (force=false) | built=false | chunks=10 | 容器 (Swagger) |
