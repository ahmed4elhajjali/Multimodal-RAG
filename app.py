from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import torch
import config as C

C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
state = {}


@asynccontextmanager
async def lifespan(app):
    from rag import MultimodalRAG
    state["rag"] = MultimodalRAG()
    yield


app = FastAPI(title="Multimodal RAG (CLIP + BLIP)", lifespan=lifespan)
app.mount("/kb-images", StaticFiles(directory=C.IMAGES_DIR), name="kb-images")


def _read(file: UploadFile) -> bytes:
    data = file.file.read()
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")
    try:
        Image.open(BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "Invalid image format")
    return data


class TextQuery(BaseModel):
    query: str
    k: int = C.TOP_K


class TextIn(BaseModel):
    animal: str
    text: str


class ImageUpdateIn(BaseModel):
    label: str


@app.get("/")
def index():
    return FileResponse(C.BASE / "static" / "index.html")


@app.get("/api/system")
def get_system_info():
    rag = state.get("rag")
    kb = rag.kb if rag else None
    return {
        "device": C.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "clip_model": C.CLIP_MODEL,
        "blip_model": C.BLIP_MODEL,
        "clip_dim": rag.clip.dim if rag else 512,
        "total_texts": len(kb.texts) if kb else 0,
        "total_images": len(kb.images) if kb else 0,
        "alpha_default": C.ALPHA,
        "top_k_default": C.TOP_K,
        "logit_scale": C.LOGIT_SCALE
    }


@app.get("/api/kb")
def kb():
    r = state["rag"].kb
    return {
        "texts": r.texts,
        "images": [{**i, "url": f"/kb-images/{i['file']}"} for i in r.images],
        "device": C.DEVICE,
        "stats": {
            "texts_count": len(r.texts),
            "images_count": len(r.images),
            "vector_dim": state["rag"].clip.dim
        }
    }


@app.get("/api/samples")
def get_samples():
    """Preset samples for quick testing."""
    sample_queries = [
        "Which animal can survive long periods in the desert without water?",
        "What is the largest land animal with a long trunk?",
        "Loyal companion animal descended from wolves that guards homes",
        "Social big cats living in savannas in groups called prides",
        "Fast herbivore domesticated for riding and racing",
        "Feathered creatures that build nests and fly"
    ]
    r = state["rag"].kb
    sample_images = [{**i, "url": f"/kb-images/{i['file']}"} for i in r.images[:6]]
    return {
        "queries": sample_queries,
        "images": sample_images
    }


@app.post("/api/search/image")
def search_image(
    file: UploadFile = File(...),
    k: int = Form(C.TOP_K),
    alpha: float = Form(C.ALPHA)
):
    img = Image.open(BytesIO(_read(file))).convert("RGB")
    return state["rag"].by_image(img, k, alpha)


@app.post("/api/search/text")
def search_text(q: TextQuery):
    return state["rag"].by_text(q.query, q.k)


@app.post("/api/kb/text")
def add_text(t: TextIn):
    return state["rag"].kb.add_text(t.animal, t.text)


@app.put("/api/kb/text/{iid}")
def update_text(iid: str, t: TextIn):
    updated = state["rag"].kb.update_text(iid, t.animal, t.text)
    if not updated:
        raise HTTPException(404, "Text entry not found")
    return updated


@app.post("/api/kb/image")
def add_image(file: UploadFile = File(...), label: str = Form(...)):
    return state["rag"].kb.add_image(_read(file), label)


@app.put("/api/kb/image/{iid}")
def update_image(iid: str, body: ImageUpdateIn):
    updated = state["rag"].kb.update_image(iid, body.label)
    if not updated:
        raise HTTPException(404, "Image entry not found")
    return updated


@app.delete("/api/kb/{kind}/{iid}")
def delete(kind: str, iid: str):
    if kind not in ("texts", "images") or not state["rag"].kb.delete(kind, iid):
        raise HTTPException(404, "Not found")
    return {"ok": True}


@app.post("/api/kb/reset")
def reset_kb():
    return state["rag"].kb.reset_to_seed()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=C.HOST, port=C.PORT)

