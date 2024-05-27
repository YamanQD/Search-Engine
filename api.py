from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from online import SearchEngine

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/search")
async def search(q: str, d: str):
    return SearchEngine.search(d, q)

# @app.get("/process")
# async def process():
#     return SearchEngine.search(q)

# @app.get("/load")
# async def load():
#     return SearchEngine.search(q)
