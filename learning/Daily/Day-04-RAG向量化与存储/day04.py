import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import faiss
import numpy as np
import torch
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

DATA = r"D:\Study\AgentGuide\learning\Daily\Day-03-RAG加载与分割\data"
WORK = r"D:\Study\AgentGuide\learning\Daily\Day-04-RAG向量化与存储"

# 坑0：当前环境 torch 是 CPU 版，写死 "cuda" 会直接崩；自动探测，装上 CUDA 版后会自动用上
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")

docs = SimpleDirectoryReader(input_dir=DATA).load_data()
nodes = SentenceSplitter(
    chunk_size=512, chunk_overlap=128, paragraph_separator="\n\n", include_metadata=True
).get_nodes_from_documents(docs)
texts = [node.text for node in nodes]

model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=DEVICE)

# ============ 打卡项①：编码 + 相近/不相关两组对照 ============
document_embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32).astype("float32")
print("shape:", document_embeddings.shape)

pairs = [
    (
        "语义相近",
        "Claude Code 使用双轨制的声明式记忆文件",
        "Claude Code 的记忆采用声明式文件，分为两个轨道",
    ),
    (
        "毫不相干",
        "Claude Code 使用双轨制的声明式记忆文件",
        "今天午饭吃了一碗牛肉面，味道不错",
    ),
]
for label, a, b in pairs:
    va, vb = model.encode([a, b], normalize_embeddings=True)
    print(f"{label}: cos = {float(np.dot(va, vb)):.4f}")

# ============ 打卡项②：IndexFlatIP 建索引 → 存盘 → 重载 → 3 query top-3 ============
# 归一化后内积 == 余弦，所以用 IP 索引检索出来的分数直接就是余弦相似度
INDEX_PATH = os.path.join(WORK, "faiss.index")


def save_index(index: faiss.Index, path: str) -> None:
    # 坑3：faiss.write_index 是 C++ 窄字符接口，遇到中文路径直接报
    #   "could not open ...: No such file or directory"，所以先序列化成 bytes 再用 Python 落盘
    with open(path, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())


def load_index(path: str) -> faiss.Index:
    # 注意 deserialize_index 要的是 uint8 数组，不是 bytes，得先 frombuffer
    data = np.frombuffer(open(path, "rb").read(), dtype="uint8")
    return faiss.deserialize_index(data)


index = faiss.IndexFlatIP(document_embeddings.shape[1])
index.add(document_embeddings)
print("ntotal:", index.ntotal)
save_index(index, INDEX_PATH)
print(f"index 已落盘: {os.path.getsize(INDEX_PATH)} bytes")

# 重载：只读盘，不重新编码
index2 = load_index(INDEX_PATH)
print("reloaded ntotal:", index2.ntotal)

queries = [
    "Claude Code 的记忆是怎么分层实现的",          # 与原文用词高度重合
    "上下文窗口再大也替代不了专门的记忆系统，原因是什么",  # 换个说法
    "这套系统部署需要多大的 Docker 镜像",          # 文中根本没有
]

for q in queries:
    print(f"\n=== {q}")
    # 坑2：bge 建库/检索都用同一套「归一化 + 指令前缀」约定
    qv = model.encode([f"为这个句子生成表示以用于检索相关文章：{q}"], normalize_embeddings=True).astype("float32")
    # qv = model.encode([f"{q}"], normalize_embeddings=True).astype("float32")
    scores, ids = index2.search(qv, 3)
    for s, i in zip(scores[0], ids[0]):
        print(f">>>>>>> {s:.3f}  #{i:>2} \n {texts[i][:80]} \n")
