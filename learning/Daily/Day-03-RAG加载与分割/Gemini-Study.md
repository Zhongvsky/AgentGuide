# RAG 文档解析与切分学习笔记：LlamaIndex Node Parser、Docling 与 MinerU

> **学习导航与今日重点**
> - **[必读 & 动手] 1. LlamaIndex Node Parser**：重点掌握 `Get Started` 三种调用范式（独立使用、IngestionPipeline、Index 全局/局部配置）以及 `SentenceSplitter` 的完整参数、四级递归切分机制与元数据感知逻辑。
> - **[读懂即可 · Day 13 动手] 2. Docling 官方文档**：理解其基于 `DoclingDocument` 的统一文档抽象、PDF 版面分析（DocLayNet）与表格结构识别（TableFormer）流水线，以及与 LlamaIndex 的无缝衔接方式。
> - **[读懂即可 · 复杂中文 PDF 方案] 3. MinerU (OpenDataLab)**：理解其针对中文与复杂排版（多栏、跨页表格、公式、扫描件）的四档解析体系（Flash / Basic / Standard / Advanced）、引擎配置与 Agent 渐进式阅读协议。

---

## 一、 今日主读：LlamaIndex Node Parser（Get Started + SentenceSplitter 参数详解）

**官方文档来源**：[LlamaIndex Node Parser Usage Pattern](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) | [Node Parser Modules](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/) | [SentenceSplitter API Reference](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/sentence_splitter/)

### 1.1 核心概念：什么是 Node Parser？

在 LlamaIndex 中，**Node Parser（节点解析器/切分器）** 是将原始长文档（`Document` 对象列表）切分为更小的数据块（`Node` 对象，通常是 `TextNode`）的核心抽象组件：

- **属性自动继承**：当一个父级 `Document` 被切分为多个子 `Node` 时，父文档的所有核心属性（包括 `metadata` 元数据字典、文本与元数据格式化模板 `text_template` / `metadata_template` 等）都会自动继承到每一个子 `Node` 中。
- **节点关系维护**：默认情况下，Node Parser 会自动在切分出的相邻节点之间建立前后指针关系（`NodeRelationship.PREVIOUS` 和 `NodeRelationship.NEXT`），并记录指向源文档的引用（`NodeRelationship.SOURCE`）以及该切片在原文中的起止字符索引（`start_char_idx` / `end_char_idx`）。

### 1.2 Get Started：三种标准使用模式（Usage Patterns）

官方文档在 `Getting Started` 部分明确给出了 Node Parser 在实际开发中的 **3 种标准接入方式**：

#### 模式 1：独立直接调用（Standalone Usage）
最基础的用法，直接实例化 `SentenceSplitter`，调用 `.get_nodes_from_documents()` 将 `Document` 列表转化为 `Node` 列表，适合在构建索引前手动检查切分效果或自定义后处理：

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 1. 初始化切分器：设置每个块最多 1024 个 Token，相邻块重叠 20 个 Token
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# 2. 传入 Document 列表，返回切分后的 Node 列表
nodes = node_parser.get_nodes_from_documents(
    [Document(text="long text")], show_progress=False
)
```

#### 模式 2：作为转换流水线的一环（Transformation Usage）
由于 `NodeParser` 继承自 `TransformComponent` 基类，它可以和元数据提取器（Metadata Extractors）、Embedding 模型一起放入 `IngestionPipeline`（数据摄取流水线）中串联执行，支持缓存与并发处理：

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

documents = SimpleDirectoryReader("./data").load_data()

# 将切分器放入 transformations 列表中按序执行
pipeline = IngestionPipeline(transformations=[TokenTextSplitter(), ...])
nodes = pipeline.run(documents=documents)
```

#### 模式 3：构建索引时自动调用（Index Usage：全局配置 vs. 单索引配置）
如果你直接用 `VectorStoreIndex.from_documents()` 构建索引，可以通过全局 `Settings` 或局部 `transformations` 参数指定切分器：

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

documents = SimpleDirectoryReader("./data").load_data()

# 方式 A：全局配置（Global Settings），后续所有索引构建默认生效
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# 方式 B：单索引局部配置（Per-Index），仅对当前索引生效
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
)
```

---

### 1.3 SentenceSplitter 参数全解与底层切分机制

`SentenceSplitter`（继承自 `MetadataAwareTextSplitter`）是 LlamaIndex 默认且最常用的文本切分器。与按硬性 Token 数截断的 `TokenTextSplitter` 不同，**`SentenceSplitter` 优先保持完整的句子和段落边界**，极大减少了块末尾出现“半截残句”的情况。

#### 完整参数速查表

| 参数名                            | 类型                           | 默认值                               | 核心作用与调优说明                                                                                                              |
| :----------------------------- | :--------------------------- | :-------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **`chunk_size`**               | `int`                        | `1024`                            | 每个切片（Chunk）的最大 **Token 数量**上限（必须 `> 0`）。注意：这里包含注入该节点的 Metadata 占用的 Token 数。                                            |
| **`chunk_overlap`**            | `int`                        | `200`                             | 相邻两个切片之间的重叠 **Token 数量**（必须 `>= 0` 且 `< chunk_size`）。注意：在官方示例代码中常手动传入 `20`，但类构造函数默认值为 `200`（`SENTENCE_CHUNK_OVERLAP`）。 |
| **`paragraph_separator`**      | `str`                        | `'\n\n\n'`                        | **一级段落分隔符**。切分器最优先按此分隔符将文本拆成大段落。如果你的 Markdown/文本使用双换行分段，建议显式修改为 `'\n\n'`。                                              |
| **`chunking_tokenizer_fn`**    | `Callable[[str], List[str]]` | `split_by_sentence_tokenizer()`   | **二级句子切分函数**。当单个段落超过 `chunk_size` 时，调用此函数按句子边界拆分（默认基于 NLTK / Punkt 英文分句器）。                                             |
| **`secondary_chunking_regex`** | `str \| None`                | `'[^,.;。？！]+[,.;。？！]?\|[,.;。？！]'` | **三级标点正则回退**。当单个长句依然超过 `chunk_size` 时，按中英文标点符号（逗号、句号、分号、感叹号、问号：`, . ; 。 ？ ！`）拆分为子句。                                    |
| **`separator`**                | `str`                        | `' '`                             | **四级单词/字符兜底分隔符**。当标点切分后的子句仍超过 `chunk_size` 时，按空格拆分为单词（若依然超长则按底层字符/Token 强制切开）。                                         |
| **`tokenizer`**                | `Callable`                   | 全局默认 Tokenizer (`tiktoken`)       | 用于计算文本和元数据 Token 长度的分词器函数。若不传，默认使用全局 `Settings.tokenizer`（通常为 OpenAI `cl100k_base`）。                                   |
| **`include_metadata`**         | `bool`                       | `True`                            | 是否将父级 `Document` 的 `metadata` 继承到切分出的 `Node` 中，并在切分时启用**元数据长度感知**（扣除元数据占用的 Token）。                                     |
| **`include_prev_next_rel`**    | `bool`                       | `True`                            | 是否在同源文档切分出的相邻 `Node` 之间自动建立 `PREVIOUS` 和 `NEXT` 双向链表关系。                                                                |
| **`id_func`**                  | `Callable`                   | `default_id_func`                 | 自定义生成每个 `Node` 唯一 ID 的函数。                                                                                              |
| **`callback_manager`**         | `CallbackManager`            | `None`                            | 用于追踪 `CBEventType.NODE_PARSING` 和 `CBEventType.CHUNKING` 事件的回调管理器。                                                     |

#### 必须掌握的 2 个底层机制（面试与实战高频踩坑点）

1. **四级递归降级切分 + 贪心合并机制（`_split` & `_merge`）**
   `SentenceSplitter` 并不是一上来就把文章拆成碎句子，而是遵循**自顶向下递归拆分、自底向上贪心合并**的策略：
   - **Step 1（检查总长）**：如果整段文本的 Token 数 `<= effective_chunk_size`，直接作为一个完整切片返回，不做任何拆分。
   - **Step 2（四级拆分优先级）**：若超长，按以下顺序逐级尝试切分，直到所有最小单元（`_Split`）的 Token 数均不超过上限：
     1. 按 `paragraph_separator`（默认 `\n\n\n`）切分段落；
     2. 若某段落仍超长，按 `chunking_tokenizer_fn`（默认句子分词器）切分句子；
     3. 若某句子仍超长（或中文文本未被英文分句器切开），按 `secondary_chunking_regex`（支持中文 `。？！` 及英文 `,.;`）切分子句；
     4. 若仍超长，按 `separator`（默认空格 `' '`）及字符级回退切分。
   - **Step 3（合并与重叠 `_merge`）**：将拆出的细粒度句子按顺序贪心拼接，直到逼近 `effective_chunk_size` 生成一个 `Node`，同时根据 `chunk_overlap` 将当前块尾部的句子保留到下一个块的开头，确保上下文连贯。

2. **元数据感知切分（Metadata-Aware Splitting）**
   因为 `SentenceSplitter` 继承自 `MetadataAwareTextSplitter`，当 `include_metadata=True` 时，调用 `split_text_metadata_aware` 会先计算该节点在送入 Embedding 或 LLM 时序列化后的 Metadata 字符串长度（取 `MetadataMode.EMBED` 和 `MetadataMode.LLM` 中较长者）：
   $$\text{effective\_chunk\_size} = \text{chunk\_size} - \text{len}(\text{tokenizer}(\text{metadata\_str}))$$
   - **实战排错**：如果你的 `Document.metadata` 塞入了过长的字段（例如把整篇摘要、长 URL 列表全放进 metadata），导致 `effective_chunk_size <= 0`，LlamaIndex 会直接抛出 `ValueError: Metadata length (...) is longer than chunk size (...)`；若 `effective_chunk_size < 50`，控制台会打印警告提示正文可用窗口不足 50 Tokens。

---

## 二、 读懂即可：Docling 官方文档（PDF 版面与表格解析，Day 13 动手）

**官方文档来源**：[Docling Official Documentation](https://docling-project.github.io/docling/) | [Docling Quickstart](https://docling-project.github.io/docling/getting_started/quickstart/) | [Docling Architecture](https://docling-project.github.io/docling/concepts/architecture/)

### 2.1 Docling 是什么？为什么在 RAG 中需要它？

如果直接用传统的 `pypdf` 提取 PDF 再交给 LlamaIndex 的 `SentenceSplitter`，双栏排版会左右串行、跨行表格会变成乱序文本。**Docling**（由 IBM Research Zurich 发起，现托管于 LF AI & Data 基金会）是一个专门将各类复杂文档转化为生成式 AI 就绪格式（GenAI-ready）的开源解析工具：

- **多格式输入支持**：原生支持 PDF、DOCX、PPTX、XLSX、HTML、EPUB、图片（PNG/JPEG/TIFF）、LaTeX、音频（WAV/MP3 经 Whisper ASR 转写）甚至视频文件。
- **深度 PDF 理解能力**：
  - **页面版面分析（Page Layout Analysis）**：基于 **DocLayNet** 视觉模型识别标题、正文段落、页眉页脚、图表、公式区域，并还原正确的**阅读顺序（Reading Order）**。
  - **表格结构识别（Table Structure Recognition）**：内置 IBM 研发的 **TableFormer** 模型，能精准识别有线表、无线表、合并单元格（跨行/跨列），并完整还原为 Markdown 表格、HTML 或 DataFrame。
  - **VLM 视觉大模型流水线与 OCR**：支持本地运行轻量级视觉模型（如 **GraniteDocling**）以及多种 OCR 引擎（Tesseract、RapidOCR、SuryaOCR），支持扫描件解析与代码块/数学公式富化（Enrichment）。
  - **本地离线与隐私安全**：所有模型均可本地运行（支持 CPU、NVIDIA GPU 及 Apple Silicon MLX 加速），适合企业敏感数据与内网（Air-gapped）环境。

### 2.2 核心架构：三大核心概念

读懂 Docling 只需抓住其架构图中的 **3 个核心抽象**：

1. **`DocumentConverter`（文档转换器入口）**：根据输入文件格式，自动路由到对应的 Format-specific Backend（格式解析后端）和 Pipeline（如标准视觉流水线 `StandardPipeline` 或视觉大模型流水线 `VlmPipeline`）。
2. **`DoclingDocument`（统一文档表示模型）**：Docling 最核心的数据结构。无论输入是 PDF 还是 Word，转换后都会生成一个保留完整版面层级（章节树、表格矩阵、图片坐标、阅读顺序）的 `DoclingDocument` 对象。
3. **导出与切分（Export & HybridChunker）**：
   - **多样化导出**：可直接调用 `doc.export_to_markdown()`、`export_to_html()`、`export_to_dict()`（无损 JSON）或 `DocTags`。
   - **结构感知切分（`HybridChunker`）**：Docling 自带 `HybridChunker`，并原生集成了 LlamaIndex（`DoclingReader` + `DoclingNodeParser`）。它在切分时直接利用 `DoclingDocument` 的版面树（保证同一个表格不被拦腰截断、标题自动附带到对应段落），比纯文本切分更精准。

### 2.3 极简代码印象（留作 Day 13 动手参考）

```python
from docling.document_converter import DocumentConverter

# 可以传入本地 PDF 路径或在线 URL
source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()

# 执行版面分析与表格识别，返回 DoclingDocument
doc = converter.convert(source).document

# 一键导出为结构清晰的 Markdown（表格与标题层级完整保留）
print(doc.export_to_markdown())
```

---

## 三、 读懂即可：MinerU GitHub（中文与复杂 PDF 开源解析方案）

**官方文档来源**：[MinerU GitHub Repository](https://github.com/opendatalab/MinerU) | [MinerU 中文 README](https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md) | [MinerU 4.0 Documentation](https://opendatalab.github.io/MinerU/)

### 3.1 MinerU 是什么？核心优势在哪里？

**MinerU** 是由上海人工智能实验室 **OpenDataLab** 开源的高质量文档解析工具（GitHub 超 82.5k Stars，目前已迭代至 **MinerU 4.0**）。它专门解决大模型语料清洗与 Agent 工作流中的“硬骨头”文档：

- **中文与学术/复杂 PDF 表现优异**：针对中文排版、双栏/多栏学术论文、复杂跨页表格、LaTeX 数学公式、扫描版古籍/教材做了深度优化，自动剔除页眉、页脚、页码与边注干扰。
- **统一文档模型与 9 种渲染目标**：底层结合原生解析引擎（DocVortex）与多模态视觉模型，支持将 PDF、图片、Office（DOC/DOCX/PPT/XLS）、EPUB、中国政企常用的 **OFD** 格式统一转化为 **Markdown、HTML、LaTeX、DOCX、EPUB、PDF、Structured Content、Content List V1/V2** 等 9 种结构化输出。
- **Agent 原生文档库与定位符（Locator）**：MinerU 4.0 不仅是一个无状态转换器（`mineru-kit parse`），还内置了面向 Agent 的本地文档库（`mineru` CLI 与 `DoclibClient`），支持按页/按块（Page/Block）渐进式读取，并生成稳定的引用定位符（如 `doc:ab12cd3/tier:standard/page:12/block:5`），方便 RAG 溯源与精准引用。

### 3.2 MinerU 4.0 的四档解析体系（Four Parsing Tiers）

MinerU 4.0 将解析质量与计算开销解耦为 **4 个档位（Tiers）**，可根据业务场景在速度与精度之间灵活权衡：

| 解析档位 (Tier)    | 中文名称 | 底层引擎与硬件要求                                                           | 速度与质量特点                  | 适用场景                                            |
| :------------- | :--- | :------------------------------------------------------------------ | :----------------------- | :---------------------------------------------- |
| **`flash`**    | 极速解析 | 原生解析（DocVortex），仅需 CPU                                              | 速度最快，质量最基础（不跑重型视觉模型）     | 快速预览、目录发现、全文初步索引，以及 Office/EPUB/OFD/HTML 原生本地解析 |
| **`basic`**    | 基础解析 | 小模型（ONNX CPU 仅需 2GB 内存，或 PyTorch）                                   | 速度较快，提供基础 OCR 与版面模型解析    | 低算力设备（纯 CPU 笔记本）上的本地隐私阅读与常规扫描件解析                |
| **`standard`** | 标准解析 | 小模型 (ONNX/PyTorch) + VLM (`llama.cpp` Vulkan / `vLLM` / `LMDeploy`) | **默认推荐档位**，高画质版面、表格与公式还原 | 日常深度阅读、含复杂表格/公式的中文财报、技术手册与学术论文                  |
| **`advanced`** | 高级解析 | 与 Standard 共享模型栈，但投入更多推理步数与精细校验                                     | 速度最慢，对疑难杂症版面质量最高         | 排版极其复杂的扫描件、密集公式与高价值语料精洗                         |

### 3.3 三款工具在 RAG 链路中的定位与横向对比

为了帮你把今天看的 3 个资源串联成一个完整的知识体系，下面用一张表总结它们在 RAG（检索增强生成）工程中的分工与选型逻辑：

| 维度 | LlamaIndex Node Parser (`SentenceSplitter`) | Docling (IBM) | MinerU (OpenDataLab) |
| :--- | :--- | :--- | :--- |
| **在 RAG 中的位置** | **下游：文本切分（Chunking）** | **上游：文档解析（Parsing）+ 结构切分** | **上游：文档解析（Parsing）+ Agent 阅读库** |
| **核心解决痛点** | 将长文本按语义/句子边界切分为固定 Token 窗口的 `Node`，维护上下文重叠与元数据继承 | 将多格式文档（特别是英文/通用 PDF、Office）解析为结构化 `DoclingDocument` 并还原表格 (`TableFormer`) | 攻克**中文复杂排版 PDF**、双栏论文、密集公式、扫描件与 OFD，转化为干净的 Markdown/JSON |
| **输入 → 输出** | `Document` (纯文本) → `List[TextNode]` | PDF/DOCX/图片/音视频 → `DoclingDocument` / Markdown / JSON | PDF/图片/Office/OFD → Markdown / LaTeX / JSON / `doc:...` 定位符 |
| **与 LlamaIndex 配合方式** | LlamaIndex 核心内置组件（`llama_index.core.node_parser`） | 官方提供 `DoclingReader` 与 `DoclingNodeParser` 插件直接接入 | 先用 `mineru-kit parse` 导出为高质量 Markdown，再用 LlamaIndex 的 `MarkdownNodeParser` + `SentenceSplitter` 二次切分 |

---

## 四、 今日学习自测清单（Check Your Understanding）

读完以上内容后，如果能回答出下面 3 个问题，说明今天的任务已经彻底掌握：

1. **参数辨析**：在 LlamaIndex 中，`SentenceSplitter(chunk_size=1024, chunk_overlap=20)` 切分超长段落时，默认优先按什么字符切分？如果切出来的单句依然超过 1024 Tokens，它会按什么顺序回退？
   - *参考答案*：最优先按 `paragraph_separator`（默认 `\n\n\n`）切分段落；若超长则按 `chunking_tokenizer_fn`（分句器）切分句子；若单句仍超长，按 `secondary_chunking_regex`（中英文句逗标点 `,.;。？！`）切分子句；最后按 `separator`（空格 `' '`）兜底。
2. **元数据踩坑**：为什么有时候明明把 `chunk_size` 设成了 `512`，调用 `SentenceSplitter` 却报错说 `Metadata length is longer than chunk size`？
   - *参考答案*：因为 `SentenceSplitter` 继承自 `MetadataAwareTextSplitter`（默认 `include_metadata=True`），它在切分正文前会自动用 `chunk_size` 减去序列化后的 `metadata` 占用的 Token 数（得出 `effective_chunk_size`），以保证最终拼上 Metadata 送入 Embedding 模型时不会超出模型上下文窗口。
3. **工程组合拳**：以后（比如 Day 13）面对一份包含大量跨页财务表格和中文说明的复杂 PDF 招股书，如果直接用普通 PDF 读取器 + `SentenceSplitter` 会有什么问题？应该如何把今天学的三个工具组合起来？
   - *参考答案*：普通读取器会丢失表格行列结构、混入页眉页脚，导致 `SentenceSplitter` 把表格数据切碎成毫无意义的乱码句子。最佳实践是：**先用上游解析器（中文复杂版面优先选 MinerU 的 `standard` 档，通用多格式可选 Docling）将 PDF 解析为结构完整的 Markdown（保留表格与标题层级），再用 LlamaIndex 的 `MarkdownNodeParser` 或 `SentenceSplitter` 进行下游切分与入库**。
