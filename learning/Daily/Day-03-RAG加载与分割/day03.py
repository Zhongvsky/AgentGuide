from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

DATA = r"D:\Study\AgentGuide\learning\Daily\Day-03-RAG加载与分割\data"

# 坑1：load_data() 返回 list，别加 [0]，多文件时会静默丢数据
docs = SimpleDirectoryReader(input_dir=DATA).load_data()
print(f"装载文档 {len(docs)} 份：{[d.metadata.get('file_name') for d in docs]}")
print(f"总字符数 {sum(len(d.text) for d in docs)}")


def split(size: int, overlap: int):
    """返回 (nodes, 块数, 平均字符数)"""
    nodes = SentenceSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        paragraph_separator="\n\n",
        include_metadata=True,
    ).get_nodes_from_documents(docs)
    total = sum(len(n.text) for n in nodes)
    return nodes, len(nodes), total / len(nodes)


# 坑2：chunk_size 不是「字符数」而是 tokenizer 的 token 数，所以 1024 切出来的块远不止 1024 字符
print("\n=== 参数对照 ===")
for size, overlap in [(1024, 200), (256, 30)]:
    _, cnt, avg = split(size, overlap)
    print(f"chunk_size={size:>5} overlap={overlap:>4} -> 块数={cnt:>3}  平均字符数={avg:7.1f}")

# 明细看 1024/200 这一组（沿用 Day 3 的调试输出）
nodes, _, _ = split(1024, 200)
for k, node in enumerate(nodes):
    print(f"Node ID {k}: {node.id_}")
    print(f"Node Text: {node.text[:120]}...")
    print(f"Parent Document ID: {node.ref_doc_id}")
    print(f"Node Metadata: {node.metadata}")
    print("-" * 50)
