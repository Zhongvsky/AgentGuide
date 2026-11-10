# 1. 基础镜像：官方 Python 3.12 精简版
FROM python:3.12-slim

# 2. 环境变量：
#    PYTHONUNBUFFERED=1：让日志实时输出到 stdout，不被缓冲区扣留（方便看 docker logs）
#    HF_ENDPOINT=https://hf-mirror.com：配置国内 HuggingFace 镜像源加速
#    HF_HOME=/models：指定模型权重下载到容器内的 /models 目录
#    EMBED_LOCAL_ONLY=1：容器内优先使用已烘焙的本地模型
ENV PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com HF_HOME=/models EMBED_LOCAL_ONLY=1

# 3. 设置容器内的工作目录
WORKDIR /app

# 4. 【层缓存关键设计】先只拷贝 requirements.txt
COPY requirements.txt .

# 5. 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 6. 【模型烘焙】在构建期执行一次加载，把 bge-small-zh-v1.5 提前下到 /models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 7. 最后才拷贝业务代码和数据（代码改动不会让前面的 pip install 和模型下载重新跑一遍！）
COPY day07.py .
COPY data ./data

# 8. 声明暴露端口
EXPOSE 8000

# 9. 启动命令（必须用 JSON 数组的 exec 形式，保证能接收到 SIGTERM 优雅退出）
CMD ["uvicorn", "day07:app", "--host", "0.0.0.0", "--port", "8000"]
