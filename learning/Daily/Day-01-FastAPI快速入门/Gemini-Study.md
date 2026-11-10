FastAPI [官方教程](https://fastapi.tiangolo.com/tutorial/)的前 4 节系统搭建了 Web API 开发的核心逻辑，依次递进讲解了服务启动、URL 动态参数捕获、URL 查询过滤以及基于 Pydantic 的结构化数据提交。

### 1. 基础路由与服务启动（[First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)）

FastAPI 将路由定义拆解为「路径操作装饰器」与「路径操作函数」：
  
- **定义基本组件**：通过 `app = FastAPI()` 建立应用实例，使用 HTTP 动作装饰器（如 `@app.get("/")`、`@app.post()`）绑定路径与请求方法。

- **支持异步与同步**：函数既可以声明为 `async def`（推荐用于异步 I/O），也可以直接写成标准的 `def` 函数。

- **本地热重载调试**：使用命令行工具执行 `fastapi dev main.py`（内部依托 Uvicorn 驱动服务），修改代码后自动热重载。

- **内置交互式文档**：访问 `/docs` 可查看基于 Swagger UI 的交互测试界面，访问 `/redoc` 可查看备选文档，底层均由自动生成的标准 OpenAPI 与 JSON Schema 驱动。

``` Python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### 2. 路径参数与类型校验（[Path Parameters](https://fastapi.tiangolo.com/tutorial/path-params/)）

路径参数用于从 URL 的特定层级提取动态标识符（例如 `/items/{item_id}`）：


- **声明与类型转换**：使用 Python 格式化字符串声明占位符，函数参数标注类型（如 `item_id: int`）。FastAPI 会将提取到的字符串自动解析为对应类型。

- **数据校验与错误反馈**：当访问 `/items/foo` 时，因类型不匹配，接口会自动拦截并返回 `422 Unprocessable Entity`，包含具体的定位字段与错误原因。

- **自顶向下匹配顺序**：路由匹配按代码声明顺序执行。静态路径必须声明在同层级的参数化路径之前（例如 `/users/me` 必须写在 `/users/{user_id}` 之前，否则 `me` 会被识别为 `{user_id}`）。  

- **枚举预定义选项**：继承 `str` 和 `Enum` 可限定取值范围（如模型名称），API 文档会自动呈现为可点击的下拉列表。 

- **通配子路径转换器**：使用 `:path` 标记（如 `/files/{file_path:path}`）可允许参数直接匹配包含斜杠的深层路径。


```Python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"

app = FastAPI()

@app.get("/users/me")
async def read_current_user():
    return {"user_id": "current_user"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name.value}
```

### 3. 查询参数与默认行为（[Query Parameters](https://fastapi.tiangolo.com/tutorial/query-params/)）

凡是未出现在 URL 路径占位符中的基础类型函数参数，FastAPI 都会自动判定为查询参数（即 URL 中 `?` 后的键值对）：

- **配置默认值与可选性**：赋默认值即为可选参数（如 `skip: int = 0`）；设为 `None` 则表示允许不传（如 `q: str | None = None`）；若不给任何默认值（如 `needy: str`），该参数即为必填项。

- **布尔参数多态解析**：标注为 `bool` 的查询参数能够智能解析多种输入，包括 `1`、`True`、`true`、`on`、`yes` 均被视作 `True`，其余常见形式视作 `False`。

- **多参数同源解析**：路径参数与查询参数可同函数自由混用，FastAPI 按参数名自动映射，无需按特定顺序排布。

```Python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(
    item_id: str,
    needy: str,
    skip: int = 0,
    limit: int | None = None,
    short: bool = False,
):
    item = {"item_id": item_id, "needy": needy, "skip": skip, "limit": limit}
    if not short:
        item.update({"description": "Full item description"})
    return item
```

### 4. Pydantic 请求体（[Request Body](https://fastapi.tiangolo.com/tutorial/body/)）

当客户端向接口提交 JSON 数据（通常搭配 `POST`、`PUT` 或 `PATCH`）时，使用 Pydantic 的 `BaseModel` 来定义数据结构：

- **声明数据模型**：继承 `pydantic.BaseModel` 并声明字段类型。带默认值的字段为可选，不带默认值的为必填。

- **自动映射与校验**：在路径操作函数中将参数类型声明为该模型（如 `item: Item`）。FastAPI 会读取请求体、校验 JSON 数据、生成对应 JSON Schema，并为代码提供完整的编辑器自动补全支持。

- **模型方法调用**：在函数内部可通过属性点访问字段（如 `item.price`），也可使用 `item.model_dump()` 转换为标准 Python 字典。

|**参数分类**|**函数参数类型特征**|**是否在路径中定义**|**数据来源与请求形式**|
|---|---|---|---|
|**路径参数**|基础标量类型（`int`, `str` 等）|是（如 `/items/{item_id}`）|URL 路径本身：`/items/42`|
|**查询参数**|基础标量类型（`int`, `str`, `bool` 等）|否|URL 问号后拼接：`?skip=10&short=true`|
|**请求体**|继承 `BaseModel` 的复合模型|否|HTTP 请求体（JSON）：`{"name": "Laptop", "price": 999.0}`|

### 综合实战示例

下面的接口整合了第 1 至 4 节的核心规则，展示了 FastAPI 如何在一个端点中同时解析路径参数、查询参数与请求体：

```Python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    q: str | None = None,
):
    result = {"item_id": item_id, **item.model_dump()}
    if q:
        result.update({"q": q})
    return result
```

FastAPI 按照内置规则自动识别各参数：`item_id` 匹配路径占位符解析为路径参数；`item` 为 `BaseModel` 子类解析为请求体；`q` 为基础标量且未在路径中出现，解析为可选查询参数。你可以直接运行并在 `/docs` 界面中发起请求体验完整的数据校验与交互流程。