
> 2026-10-20（周二）｜第 1 / 60 个学习日｜第 1 周 · 大模型应用开发基础 + 手撕 Naive RAG

## 今日主题 + 学习目标

**主题**：FastAPI 快速入门
**目标**：掌握 FastAPI 基础，能创建路由、处理请求

今天是从 0 开始的第一格，不建立在任何前置内容上。这一段 FastAPI 骨架会一路用到 Day 5-6 的 Naive RAG API、Day 21 的研究助手、Day 43 的智能客服项目——不要写一次就丢。

## 资源清单（3 条）

1. FastAPI 官方教程（可切中文 `/zh/`）：https://fastapi.tiangolo.com/tutorial/
2. First Steps（15 分钟读完就够）：https://fastapi.tiangolo.com/tutorial/first-steps/
3. 请求体与 Pydantic：https://fastapi.tiangolo.com/tutorial/body/

## 时间安排建议（约 2 小时，40% 读 / 40% 写 / 20% 记录）

| 时段 | 内容 | 类型 |
|---|---|---|
| 0:00-0:45 | 官方教程前 4 节：路由、路径参数、查询参数、Pydantic 请求体 | 读懂即可 |
| 0:45-1:35 | 敲代码：Hello World → 升级成 `/chat` 接口 | 动手 |
| 1:35-2:00 | 用 `/docs` 自测 + 补笔记区 | 动手 |

## 今日手撕打卡项（来自第 1 周手撕清单）

- [ ] **动手**：搭一个 Hello World 接口，`uvicorn main:app --reload` 能跑起来，浏览器访问 `/` 拿到 JSON
- [ ] **动手**：升级成 `POST /chat` 接口，接收 `{"question": str, "history": list[str]}`，返回 mock 回答

参考骨架（自己敲一遍，别复制）：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    history: list[str] = []

@app.get("/")
def hello():
    return {"msg": "hello agent"}

@app.post("/chat")
async def chat(req: ChatRequest):
    # Day 2 接 LangChain，今天先 mock
    return {"answer": f"收到问题：{req.question}", "turns": len(req.history)}
```

三个必须弄懂的点（说得出即算过）：
- 路径参数 vs 查询参数：函数参数名和 `{}` 占位符一致 → 路径参数；否则自动变查询参数（`?q=x&limit=3`）
- Pydantic 模型当参数类型 → 自动从 JSON body 解析 + 校验，失败返回 422
- I/O 密集接口一律 `async def`，这是第 4 周异步优化的伏笔

## 验收标准

打开 `http://127.0.0.1:8000/docs` 能成功调用 `/chat`；故意把 `question` 传成数字时返回 422 而不是 500——说明 Pydantic 校验生效了。

## 昨日回顾

（第 1 个学习日，无历史打卡）

## 进度提示

- 距第 7 个学习日（周度总结，2026-10-29 周四）还剩 **6 个学习日**
- 那天要交的产出是「Naive RAG 项目 Docker 化跑通」，所以现在就要开始攒：今天的 `week1-rag/` 目录 + 能跑的 FastAPI 壳
- 第 1 周只有 3 个学习日（10-20 ~ 10-22），Day 4 从 10-26 周一开始，节奏偏紧，别拖

---

## 笔记区

**今日卡点**

-

**关键代码片段**

-

**待补知识**

-
