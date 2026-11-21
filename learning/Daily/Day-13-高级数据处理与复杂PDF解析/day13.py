"""
Day 13: Advanced RAG - 复杂文档版面分析与表格结构化解析实战
=====================================================================
本脚本为 Day 13 的实战闯关代码骨架。
请根据契约定义（Pydantic & Type Hints）与关卡提示（Docstring），
手撕基于空间几何的表格 Markdown 重构、双栏排版顺序还原与噪音清洗、以及表格感知分块（Table-Aware Chunking）管线。

【闯关打靶规则】：
- 严禁直接翻看文末参考答案！
- 主执行逻辑中各核心关卡已预设 raise NotImplementedError，请依次实现各关卡函数，
  实现完成后删除 NotImplementedError 行，逐步点亮各个关卡；
- 如遇卡壳超过 20 分钟，可展开文末折叠区 `# region 🙈 官方参考实现` 对照思路。
"""

import sys
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

# Windows 控制台 UTF-8 防御
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


# ==========================================
# 1. 契约定义 (Data Models & Contracts)
# ==========================================


class TableCell(BaseModel):
    """表格单个单元格契约（坐标与内容）"""

    row_idx: int = Field(..., description="行序号，从 0 开始")
    col_idx: int = Field(..., description="列序号，从 0 开始")
    text: str = Field(..., description="单元格文本内容")
    is_header: bool = Field(default=False, description="是否属于表头行")


class LayoutBoundingBox(BaseModel):
    """版面分析检测出的目标块契约（带空间几何坐标）"""

    element_id: str = Field(..., description="元素唯一标识")
    text: str = Field(..., description="提取出的文字内容")
    box_type: str = Field(
        ...,
        description="元素类型：'title' | 'narrative' | 'table' | 'header' | 'footer'",
    )
    # Bounding Box 空间绝对坐标: (x0, y0, x1, y1), 页面左上角为原点 (0, 0)
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="四元坐标 (left, top, right, bottom)"
    )


class CleanDocumentElement(BaseModel):
    """完成清洗与排版还原后的有序文档元素契约"""

    element_id: str
    element_type: str  # 'title' | 'narrative' | 'table'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """最终交付给 Embedding 模块的 RAG 结构化切片契约"""

    chunk_id: str
    text: str
    is_table: bool = False
    section_context: str = ""
    char_length: int = 0


# ==========================================
# 2. 核心任务手撕区：Day 13 闯关打靶区
# ==========================================


def reconstruct_table_to_markdown(cells: List[TableCell]) -> str:
    """
    【关卡 1】：二维表格单元格矩阵还原与 Markdown 序列化

    痛点：
    PDF 解析器提取出的表格往往是离散的 Cell 坐标与文字，如果直接平铺拼接，
    行列关系荡然无存。必须将其还原为标准 GitHub Flavored Markdown (GFM) 格式表格。

    实现要求：
    1. 统计表格的最大行数 (max_rows) 与最大列数 (max_cols)；
    2. 构建二维二维网格矩阵 grid[row][col]，默认填充空字符串 ""；
    3. 将 cells 填入对应行列，清理内部多余换行符与前后首尾空白（将换行替换为空格）；
    4. 构造 Markdown 字符串：
       - 第 1 行：表头内容 `| col1 | col2 | ... |`
       - 第 2 行：分隔对齐线 `| :--- | :--- | ... |`
       - 后续行：数据内容 `| val1 | val2 | ... |`
    5. 返回格式化后的 Markdown 字符串。

    提示：若某单元格缺失，使用空字符串占位保持列数整齐。
    """
    # TODO 1: 请实现表格网格重构与 Markdown 序列化
    max_rows, max_cols = 0, 0
    for cell in cells:
        max_rows = max(max_rows, cell.row_idx + 1)
        max_cols = max(max_cols, cell.col_idx + 1)

    grid = [["" for _ in range(max_cols)] for _ in range(max_rows)]

    for cell in cells:
        row_idx = cell.row_idx
        col_idx = cell.col_idx
        text = (cell.text).strip("").replace("\n", " ")
        is_header = cell.is_header

        if is_header:
            assert (
                row_idx == 0
            ), f"If cell.is_header is True, cell.row_idx({cell.row_idx}) must be zero."
        assert (
            grid[row_idx][col_idx] == ""
        ), f"grid cell at (row_idx, col_idx)=({row_idx}, {col_idx}) has been inserted"

        grid[row_idx][col_idx] = text

    lines = []
    # 1. 表头行
    lines.append("| " + " | ".join(grid[0]) + " |")
    # 2. 分隔线
    lines.append("| " + " | ".join([":---"] * max_cols) + " |")
    # 3. 数据行
    for r in range(1, max_rows):
        lines.append("| " + " | ".join(grid[r]) + " |")
    return "\n".join(lines)


def reorder_multi_column_and_filter_noise(
    elements: List[LayoutBoundingBox],
    page_width: float = 595.0,  # 标准 A4 宽度点数
    page_height: float = 842.0,  # 标准 A4 高度点数
    header_threshold_ratio: float = 0.08,  # 顶部 8% 视为页眉区
    footer_threshold_ratio: float = 0.92,  # 底部 8% 视为页脚区
) -> List[LayoutBoundingBox]:
    """
    【关卡 2】：双栏版面阅读顺序重构与公文页眉页脚过滤

    痛点：
    金融研报与学术论文的 PDF 常采用双栏排版。若直接按抽取流排序，
    左栏与右栏的文字会横向交叉乱序；同时页眉页脚周期性重复，严重污染向量检索。

    实现要求：
    1. 过滤噪音块：
       - 凡是 `box_type` 为 'header'、'footer' 的元素直接剔除；
       - 凡是 `y0 < page_height * header_threshold_ratio`（落在顶部页眉区）或
         `y1 > page_height * footer_threshold_ratio`（落在底部页脚区）的元素直接剔除；
    2. 双栏几何重构：
       - 页面中线为 `mid_x = page_width / 2.0`；
       - 计算元素几何中心 `center_x = (bbox[0] + bbox[2]) / 2.0`；
       - `center_x < mid_x` 归为左栏（Left Column）；
       - `center_x >= mid_x` 归为右栏（Right Column）；
    3. 顺序排序：
       - 左栏元素按照顶部 y0 坐标由小到大排序（自上而下阅读）；
       - 右栏元素按照顶部 y0 坐标由小到大排序（自上而下阅读）；
    4. 合并并返回：左栏所有元素 + 右栏所有元素。
    """
    # TODO 2: 请实现基于空间几何的双栏重构与噪音过滤

    eles_left, eles_right = [], []
    mid_page_x = page_width / 2

    for element in elements:
        text = element.text
        box_type = element.box_type
        x0, y0, x1, y1 = element.bbox

        if (
            (box_type in ("header", "footer"))
            or (y0 < page_height * header_threshold_ratio)
            or (y1 > page_height * footer_threshold_ratio)
        ):
            continue

        mid_x = (x0 + x1) / 2
        if mid_x <= mid_page_x:
            eles_left.append(element)
        else:
            eles_right.append(element)

    eles_left = sorted(eles_left, key=lambda ele: ele.bbox[1])
    eles_right = sorted(eles_right, key=lambda ele: ele.bbox[1])

    return eles_left + eles_right


def table_aware_chunking(
    elements: List[CleanDocumentElement],
    max_chunk_chars: int = 300,
) -> List[DocumentChunk]:
    """
    【关卡 3】：表格感知语义分块 (Table-Aware Chunking Pipeline)

    痛点：
    通用 SentenceSplitter 会机械地将一个大表格切成两半，导致下半截表格完全丢失表头信息，
    在向量空间中变成无意义的孤儿数据。

    实现要求：
    1. 维护一个当前正文缓冲区 `text_buffer = []` 与当前章节上下文 `current_section = ""`；
    2. 遍历文档元素 elements：
       - 若遇到 'title' 元素：
         - 更新 `current_section = element.content`；
         - 将当前 title 记入正文流中；
       - 若遇到 'table' 元素（原子保护！）：
         - 检查当前 `text_buffer`，若有未落盘的正文，先将其封装并 flush 为一个普通 Chunk；
         - 将该表格作为一个【原子级独立 Chunk】处理：
           - `is_table = True`
           - 文本前置追加当前上下文标题，例如：`f"【所属章节：{current_section}】\n{element.content}"`
           - 独立落盘，严禁与普通正文混合截断！
       - 若遇到 'narrative' 普通正文：
         - 累积长度：若 `当前buffer总长 + 新增内容长 > max_chunk_chars`：
           - 将当前 buffer 拼接为一个 Chunk 落盘；
           - 清空 buffer 并将新增内容放入新 buffer；
         - 否则直接追加至 buffer；
    3. 遍历结束后，若 buffer 中还有剩余文本，flush 最后一个普通 Chunk；
    4. 返回 List[DocumentChunk]。
    """

    # TODO 3: 请实现表格感知的原子分块管线
    def flush(is_table=False):

        nonlocal current_length
        nonlocal chunk_id
        nonlocal text_buffer

        if not is_table:
            text = "\n\n".join(text_buffer)
        else:
            text = f"【所属章节：{current_section}】\n{element.content}"

        document_chunks.append(
            DocumentChunk(
                chunk_id=str(chunk_id),
                text=text,
                is_table=is_table,
                section_context=current_section,
                char_length=len(text),
            )
        )

        current_length = 0
        chunk_id += 1
        text_buffer = []

        return

    document_chunks = []

    text_buffer = []
    current_section = ""
    last_element_type = ""

    chunk_id = 0
    current_length = 0

    for element in elements:
        if element.element_type == "title":
            current_section = element.content
            text_buffer.append(element.content)
            current_length += len(element.content)
            last_element_type = "title"

        elif element.element_type == "narrative":
            if current_length + len(element.content) > max_chunk_chars:
                flush()

            text_buffer.append(element.content)
            current_length += len(element.content)
            last_element_type = "narrative"

        elif element.element_type == "table":
            if (
                text_buffer
            ):  # 遇到新的 table 时，之前的内容只要不为空，都单独输出为一个 chunk，新的 table 要算入新的 chunk
                flush()

            flush(is_table=True)

    if text_buffer:
        flush()

    return document_chunks


# ==========================================
# 3. 主执行入口与 Benchmark 测试用例
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("【Day 13 实战】：复杂文档版面分析与表格结构化解析测试")
    print("=" * 80)

    # ----------------------------------------------------
    # 测试关卡 1：离散单元格还原为 Markdown 表格
    # ----------------------------------------------------
    print("\n🔍 【测试 1：手撕表格矩阵还原为标准 Markdown】")
    raw_cells = [
        TableCell(row_idx=0, col_idx=0, text="费用类别", is_header=True),
        TableCell(row_idx=0, col_idx=1, text="报销标准上限（元/天）", is_header=True),
        TableCell(row_idx=0, col_idx=2, text="核销凭据要求", is_header=True),
        TableCell(row_idx=1, col_idx=0, text="城市内交通费"),
        TableCell(row_idx=1, col_idx=1, text="80.00"),
        TableCell(row_idx=1, col_idx=2, text="滴滴出行发票或\n出租车票据"),
        TableCell(row_idx=2, col_idx=0, text="住宿伙食补贴"),
        TableCell(row_idx=2, col_idx=1, text="120.00"),
        TableCell(row_idx=2, col_idx=2, text="包干核算，无需凭据"),
    ]

    table_md = reconstruct_table_to_markdown(raw_cells)
    print("生成的 Markdown 表格结构：")
    print(table_md)

    assert (
        "| 费用类别 | 报销标准上限（元/天） | 核销凭据要求 |" in table_md
    ), "表头行生成异常"
    assert "| :--- | :--- | :--- |" in table_md, "对齐线分隔行生成异常"
    assert "滴滴出行发票或 出租车票据" in table_md, "未正确清洗单元格内部多余换行"
    print("✅ 关卡 1 验收通过：表格保真度与 Markdown 语法完全合规！")

    # ----------------------------------------------------
    # 测试关卡 2：双栏版面自然阅读序重构与公文噪音清洗
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 2：双栏版面自然阅读序重构与页眉页脚过滤】")
    PAGE_W, PAGE_H = 600.0, 800.0

    # 模拟 PDF 抽取出的交叉混淆原始流（包含页眉、页脚及左右双栏交叉文本）
    messy_boxes = [
        # 页眉（绝对坐标 y 位于顶部 5% 区间）
        LayoutBoundingBox(
            element_id="h1",
            text="集团机密 · 2026年度差旅合规手册（仅供内部传阅）",
            box_type="header",
            bbox=(50.0, 10.0, 550.0, 30.0),
        ),
        # 左栏第 1 段
        LayoutBoundingBox(
            element_id="l1",
            text="第一条 本规范旨在规范集团各子公司员工日常差旅行为。",
            box_type="narrative",
            bbox=(50.0, 100.0, 280.0, 150.0),
        ),
        # 右栏第 1 段（物理抽取时横向扫描误插在左栏之间）
        LayoutBoundingBox(
            element_id="r1",
            text="第五条 财务审核专员须在收到纸质单据后3个工作日内完成复核。",
            box_type="narrative",
            bbox=(320.0, 100.0, 550.0, 160.0),
        ),
        # 左栏第 2 段
        LayoutBoundingBox(
            element_id="l2",
            text="第二条 差旅费用报销遵循'事前申请、按标执行、据实报销'原则。",
            box_type="narrative",
            bbox=(50.0, 180.0, 280.0, 240.0),
        ),
        # 右栏第 2 段
        LayoutBoundingBox(
            element_id="r2",
            text="第六条 凡涉及境外差旅的汇率折算，统一以出发当日中国银行基准价为准。",
            box_type="narrative",
            bbox=(320.0, 190.0, 550.0, 250.0),
        ),
        # 页脚（绝对坐标 y 位于底部 95% 区间）
        LayoutBoundingBox(
            element_id="f1",
            text="-- 第 12 页 / 共 48 页 --",
            box_type="footer",
            bbox=(250.0, 760.0, 350.0, 780.0),
        ),
    ]

    reordered_boxes = reorder_multi_column_and_filter_noise(
        messy_boxes, page_width=PAGE_W, page_height=PAGE_H
    )
    reordered_ids = [b.element_id for b in reordered_boxes]
    print(f"清洗与重排后的阅读序列: {reordered_ids}")

    # 验证判定
    assert "h1" not in reordered_ids, "页眉噪音未被过滤"
    assert "f1" not in reordered_ids, "页脚噪音未被过滤"
    assert reordered_ids == [
        "l1",
        "l2",
        "r1",
        "r2",
    ], f"双栏阅读顺序错误！期望 ['l1', 'l2', 'r1', 'r2'], 实际得到 {reordered_ids}"
    print("✅ 关卡 2 验收通过：双栏阅读序彻底恢复，页眉页脚公文噪音 100% 滤除！")

    # ----------------------------------------------------
    # 测试关卡 3：表格感知语义分块 (Table-Aware Chunking)
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 【测试 3：表格感知语义分块 (Table-Aware Chunking Pipeline)】")

    doc_elements = [
        CleanDocumentElement(
            element_id="d1",
            element_type="title",
            content="第三章 差旅伙食及交通补贴细则",
        ),
        CleanDocumentElement(
            element_id="d2",
            element_type="narrative",
            content="所有因公出差人员的市内交通费均采用限额据实报销制。出差期间若由接待单位统一安排车辆的，不得再行报销市内交通费。各部门应当严格把关。",
        ),
        CleanDocumentElement(
            element_id="d3",
            element_type="table",
            content=table_md,
            metadata={"table_caption": "差旅交通与补贴限额对照表"},
        ),
        CleanDocumentElement(
            element_id="d4",
            element_type="narrative",
            content="第四章 报销审批权限。部门经理审批额度为单次两万元以内，超过两万元须经财务总监审批，凡弄虚作假者将追究当事人纪律责任。",
        ),
    ]

    chunks = table_aware_chunking(doc_elements, max_chunk_chars=150)
    print(f"分块完成，产出 Chunk 数量: {len(chunks)}")
    for idx, c in enumerate(chunks, 1):
        print(f"\n[Chunk {idx}] (is_table={c.is_table}, len={c.char_length})")
        print(f"内容预览:\n{c.text}")

    # 验证判定
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1, "表格未被作为独立原子 Chunk 保留"
    assert (
        "【所属章节：第三章 差旅伙食及交通补贴细则】" in table_chunks[0].text
    ), "表格 Chunk 未成功注入所属章节上下文"
    assert (
        "| 费用类别 | 报销标准上限（元/天） |" in table_chunks[0].text
    ), "表格内容损坏"
    print("\n" + "=" * 80)
    print("🏁 【全流程验收通过】:")
    print("✅ [VERIFICATION PASSED]: 高级文档版面分析与表格结构化处理全链路通过验收！")
    print("=" * 80)


# ==============================================================================
# region 🙈 官方参考实现（建议卡壳20分钟或写完后再展开对照）
# ==============================================================================
"""
def reconstruct_table_to_markdown(cells: List[TableCell]) -> str:
    if not cells:
        return ""

    max_row = max(c.row_idx for c in cells) + 1
    max_col = max(c.col_idx for c in cells) + 1

    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    for c in cells:
        clean_text = " ".join(c.text.split())
        grid[c.row_idx][c.col_idx] = clean_text

    lines = []
    # 1. 表头行
    header_line = "| " + " | ".join(grid[0]) + " |"
    lines.append(header_line)

    # 2. 分隔线
    sep_line = "| " + " | ".join([":---"] * max_col) + " |"
    lines.append(sep_line)

    # 3. 数据行
    for r in range(1, max_row):
        row_line = "| " + " | ".join(grid[r]) + " |"
        lines.append(row_line)

    return "\\n".join(lines)


def reorder_multi_column_and_filter_noise(
    elements: List[LayoutBoundingBox],
    page_width: float = 595.0,
    page_height: float = 842.0,
    header_threshold_ratio: float = 0.08,
    footer_threshold_ratio: float = 0.92,
) -> List[LayoutBoundingBox]:
    filtered = []
    y_min = page_height * header_threshold_ratio
    y_max = page_height * footer_threshold_ratio
    mid_x = page_width / 2.0

    for elem in elements:
        if elem.box_type in ("header", "footer"):
            continue
        x0, y0, x1, y1 = elem.bbox
        if y0 < y_min or y1 > y_max:
            continue
        filtered.append(elem)

    left_col = []
    right_col = []
    for elem in filtered:
        center_x = (elem.bbox[0] + elem.bbox[2]) / 2.0
        if center_x < mid_x:
            left_col.append(elem)
        else:
            right_col.append(elem)

    left_col.sort(key=lambda b: b.bbox[1])
    right_col.sort(key=lambda b: b.bbox[1])

    return left_col + right_col


def table_aware_chunking(
    elements: List[CleanDocumentElement],
    max_chunk_chars: int = 300,
) -> List[DocumentChunk]:
    chunks = []
    current_section = ""
    text_buffer = []
    chunk_counter = 1

    def flush_text_buffer():
        nonlocal chunk_counter
        if not text_buffer:
            return
        combined_text = "\\n\\n".join(text_buffer).strip()
        if combined_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_{chunk_counter:03d}",
                    text=combined_text,
                    is_table=False,
                    section_context=current_section,
                    char_length=len(combined_text),
                )
            )
            chunk_counter += 1
        text_buffer.clear()

    for elem in elements:
        if elem.element_type == "title":
            current_section = elem.content
            text_buffer.append(f"### {elem.content}")
        elif elem.element_type == "table":
            flush_text_buffer()
            table_payload = f"【所属章节：{current_section}】\\n{elem.content}" if current_section else elem.content
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_{chunk_counter:03d}",
                    text=table_payload,
                    is_table=True,
                    section_context=current_section,
                    char_length=len(table_payload),
                )
            )
            chunk_counter += 1
        elif elem.element_type == "narrative":
            tentative_len = sum(len(t) for t in text_buffer) + len(elem.content)
            if tentative_len > max_chunk_chars:
                flush_text_buffer()
            text_buffer.append(elem.content)

    flush_text_buffer()
    return chunks
"""
# endregion
