from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import config as C

C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
state = {}


@asynccontextmanager
async def lifespan(app):
    from rag import MultimodalRAG
    state["rag"] = MultimodalRAG()
    yield


app = FastAPI(title="Multimodal RAG", lifespan=lifespan)
app.mount("/kb-images", StaticFiles(directory=C.IMAGES_DIR), name="kb-images")


def _read(file: UploadFile) -> bytes:
    data = file.file.read()
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")
    try:
        Image.open(BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "Invalid image")
    return data


class TextQuery(BaseModel):
    query: str
    k: int = C.TOP_K


class TextIn(BaseModel):
    animal: str
    text: str


@app.get("/")
def index():
    return FileResponse(C.BASE / "static" / "index.html")


@app.get("/api/kb")
def kb():
    r = state["rag"].kb
    return {"texts": r.texts, "images": [{**i, "url": f"/kb-images/{i['file']}"} for i in r.images],
            "device": C.DEVICE}


@app.post("/api/search/image")
def search_image(file: UploadFile = File(...), k: int = Form(C.TOP_K), alpha: float = Form(C.ALPHA)):
    img = Image.open(BytesIO(_read(file))).convert("RGB")
    return state["rag"].by_image(img, k, alpha)


@app.post("/api/search/text")
def search_text(q: TextQuery):
    return state["rag"].by_text(q.query, q.k)


@app.post("/api/kb/text")
def add_text(t: TextIn):
    return state["rag"].kb.add_text(t.animal, t.text)


@app.post("/api/kb/image")
def add_image(file: UploadFile = File(...), label: str = Form(...)):
    return state["rag"].kb.add_image(_read(file), label)


@app.delete("/api/kb/{kind}/{iid}")
def delete(kind: str, iid: str):
    if kind not in ("texts", "images") or not state["rag"].kb.delete(kind, iid):
        raise HTTPException(404, "Not found")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=C.HOST, port=C.PORT)
