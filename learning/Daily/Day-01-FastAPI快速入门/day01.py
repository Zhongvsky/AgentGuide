from fastapi import FastAPI
from pydantic import BaseModel

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