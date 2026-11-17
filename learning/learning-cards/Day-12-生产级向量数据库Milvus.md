# Day 12 · Advanced RAG：生产级向量数据库 (Milvus) 与元数据混合过滤实战

> 2026-11-18（周三）｜第 12 / 60 个学习日｜第 2 周 · Advanced RAG 与生产级向量数据库  
> **周度里程碑倒计时**：距离第 2 周里程碑（Day 14 · 混合检索+Reranker+Milvus 升级）还差 **2 个学习日**。

---

## 一、 承前启后与痛点剖析

在 Day 4 ~ Day 7 中，我们使用本地 FAISS 构建了 Naive RAG 的向量存储与检索底座。

但随着业务复杂度上升与生产级落地推进，**FAISS 暴露出了三大不可调和的工程硬伤**：
1. **标量元数据混合过滤无力（Metadata Filtering Nightmare）**：
   企业级检索从来不是纯向量比对，而是“向量相似度 + 标量条件约束”（例如：`department == '财务部' AND doc_type == '规章' AND status == '生效中'`）。FAISS 是纯向量算法库，无法原生做前置标量过滤；如果采用“后过滤”（先检索 Top-100 再手动筛选），一旦符合条件的数据排在 100 名开外，检索结果将直接空报（False Negative）！
2. **缺乏动态 CRUD 与实时可见性**：
   FAISS 是单机内存索引文件，新增数据往往需要重建索引，不支持按主键点查、就地更新或物理/逻辑删除。
3. **缺少生产级架构支撑**：
   无法支持多租户（Multi-Tenancy）、动态分区（Partition Key）、分布式水平伸缩与灾备高可用。

今天 Day 12，我们正式引入业界顶级的云原生分布式向量数据库 —— **Milvus**，打通企业级 RAG 的生产级存储基石！

---

## 二、 今日主题与学习目标

- **主题**：生产级向量数据库 (Milvus) 与 PyMilvus 实操 (Vector Database & Scalar Filtering)
- **目标**：
  1. 掌握 Milvus 的云原生架构心智模型（Coordinator、Worker Node、etcd、MinIO/S3）与 **Milvus-Lite（无外部依赖嵌入式秒级启动）** 研发模式；
  2. 掌握 Collection Schema 强类型契约设计（Primary Key、VarChar 动态标量字段、FloatVector 稠密向量）；
  3. 掌握核心向量索引算法（HNSW、IVF_FLAT）及相似度度量（COSINE、IP、L2）参数配置；
  4. 手撕 **生产级标量元数据前置过滤检索（Hybrid Vector + Scalar Search）**，彻底攻克 FAISS 元数据过滤痛点。

---

## 三、 精选参考资源（≤3 条）

1. **Milvus 官方文档 · Quick Start**：<https://milvus.io/docs/quickstart.md>
2. **PyMilvus 现代 API 指南 · MilvusClient**：<https://milvus.io/docs/pymilvus-install-client.md>
3. **Milvus 核心向量索引详解 (HNSW vs IVF_FLAT)**：<https://milvus.io/docs/index.md>

---

## 四、 时间规划建议（约 2~2.5 小时）

- **原理理解与架构心智模型（30% · 40分钟）**：
  - 深入掌握 Milvus 存储与计算分离的分布式架构；
  - 搞懂“前过滤（Pre-filtering）”与“后过滤（Post-filtering）”的本质差异与 Bitset 掩码原理。
- **核心代码动手实现（50% · 65分钟）**：
  - 【关卡 1】：手撕 `define_and_create_collection`（集合 Schema 契约定义与生命周期管理）；
  - 【关卡 2】：手撕 `build_index_and_load`（HNSW 索引构建与内存加载）；
  - 【关卡 3】：手撕 `search_with_scalar_filter`（基于布尔表达式的标量元数据混合过滤检索）。
- **实测对比与复盘（20% · 25分钟）**：
  - 运行 `uv run python learning/Daily/Day-12-生产级向量数据库Milvus/day12.py`；
  - 验证“无过滤检索” vs “特定部门标量过滤”对 Top-K 召回结果的精准截断效果；
  - 记录踩坑点并归档复盘。

---

## 五、 今日动手清单（Checklist）

- [ ] **必做**：在 `day12.py` 中手撕基于 `MilvusClient` 的集合生命周期与强类型 Schema 定义；
- [ ] **必做**：实现 HNSW 索引构建与集合内存加载（Load）；
- [ ] **必做**：实现带复杂 Boolean 标量表达式（如 `department == '财务部' and doc_type in ['规章', '制度']`）的向量混合检索；
- [ ] **进阶（读懂即可）**：理解生产环境中 Docker Compose 独立部署 Milvus 时的资源开销与网络拓扑。

---

## 六、 量化验收标准

在终端执行：
```powershell
uv run python learning/Daily/Day-12-生产级向量数据库Milvus/day12.py
```
**判定标准**：
1. 脚本完整跑通，本地 `milvus_travel.db` 数据存储与初始化正常无报错退出；
2. 打印出完整的 **Collection Schema 字段契约表** 与插入的 5 条含结构化元数据的真实差旅记录；
3. 执行无过滤检索：召回 Top-3 包含全部部门的高相似度片段；
4. 执行标量前置过滤检索（如限定 `department == '财务部'`）：召回结果中 **非财务部片段被 100% 绝对拦截**，实现精准的元数据与向量联合召回！

---

## 七、 周度里程碑倒计时

- 距离本周里程碑日（**Day 14 · 2026-11-23 周一**）还剩 **2 个学习日**；
- 后续我们将在 Day 13 攻克富文本与复杂表格 PDF 解析，并在 Day 14 将混合检索、Reranker、Milvus 与 RAG 自动化评测整合成完整生产级架构！
