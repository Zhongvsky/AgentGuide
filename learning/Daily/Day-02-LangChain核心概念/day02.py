from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model  
import uvicorn

# ============== Day1: FastAPI App ==================

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class ChatRequest(BaseModel):
    message: str
    history: list[str] | None = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.post("/chat")
async def chat(req: ChatRequest):
    return {"message": f"You said: {req.message}"}

# ====================================================

# ============== Day2: LangChain LCEL===================

model = init_chat_model(
    model = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
    model_provider = "openai",
    base_url = "http://127.0.0.1:9527/v1",
    api_key="not-needed",               
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")

chain = prompt | model | StrOutputParser()

@app.post("/chat_lcel_invoke")
async def chat_lcel_invoke(req: ChatRequest):
    return {"message": chain.invoke({"question": req.message})}

# # @app.post("/chat_lcel_stream")
# # async def chat_lcel_stream(req: ChatRequest):
# #     return {"message": chain.stream({"question": req.message})}

# # 2. 修改流式接口
# from fastapi.responses import StreamingResponse
# @app.post("/chat_lcel_stream")
# async def chat_lcel_stream(req: ChatRequest):
#     async def event_generator():
#         # 用 astream 处理异步流式输出
#         async for chunk in chain.astream({"question": req.message}):
#             # 过滤掉模型推理前期输出的空字符串 ""
#             if chunk:
#                 # 遵循标准 SSE (Server-Sent Events) 协议格式
#                 yield f"data: {chunk}\n\n"

#     return StreamingResponse(
#         event_generator(), 
#         media_type="text/event-stream"
#     )



if __name__ == "__main__":
    uvicorn.run("day02:app", host="0.0.0.0", port=8000, reload=True)