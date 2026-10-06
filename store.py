import json, re, threading, uuid
from io import BytesIO
import numpy as np
from PIL import Image
import config as C

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_id = lambda: uuid.uuid4().hex[:8]


class KnowledgeBase:
    """Texts + images with CLIP embeddings, persisted as data/kb.json + data/images/."""

    def __init__(self, clip):
        self.clip, self.lock = clip, threading.RLock()
        C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        if C.KB_PATH.exists():
            d = json.loads(C.KB_PATH.read_text("utf-8"))
        else:
            seed = json.loads(C.SEED_PATH.read_text("utf-8"))
            d = {"texts": [{"id": _id(), **t} for t in seed], "images": []}
        self.texts = d["texts"]
        self.images = [i for i in d["images"] if (C.IMAGES_DIR / i["file"]).exists()]
        known = {i["file"] for i in self.images}
        # auto-register images dropped into data/images (label = file name)
        for p in sorted(C.IMAGES_DIR.iterdir()):
            if p.suffix.lower() in EXTS and p.name not in known:
                label = re.sub(r"[\d_\-\s]+$", "", p.stem).replace("_", " ").strip() or p.stem
                self.images.append({"id": _id(), "file": p.name, "label": label})
        empty = np.zeros((0, clip.dim), np.float32)
        self.t_emb = clip.texts([t["text"] for t in self.texts]) if self.texts else empty
        self.i_emb = (np.vstack([clip.image(Image.open(C.IMAGES_DIR / i["file"]).convert("RGB"))
                                 for i in self.images]) if self.images else empty)
        self.save()

    def save(self):
        C.KB_PATH.write_text(json.dumps({"texts": self.texts, "images": self.images},
                                        ensure_ascii=False, indent=1), "utf-8")

    def add_text(self, animal, text):
        e = self.clip.texts([text])
        with self.lock:
            item = {"id": _id(), "animal": animal.strip(), "text": text.strip()}
            self.texts.append(item)
            self.t_emb = np.vstack([self.t_emb, e])
            self.save()
        return item

    def add_image(self, data: bytes, label: str):
        img = Image.open(BytesIO(data)).convert("RGB")
        e = self.clip.image(img)
        with self.lock:
            iid = _id()
            img.save(C.IMAGES_DIR / f"{iid}.jpg", quality=92)
            item = {"id": iid, "file": f"{iid}.jpg", "label": label.strip()}
            self.images.append(item)
            self.i_emb = np.vstack([self.i_emb, e])
            self.save()
        return item

    def delete(self, kind, iid):
        with self.lock:
            items = self.texts if kind == "texts" else self.images
            idx = next((n for n, i in enumerate(items) if i["id"] == iid), None)
            if idx is None:
                return False
            item = items.pop(idx)
            if kind == "texts":
                self.t_emb = np.delete(self.t_emb, idx, 0)
            else:
                self.i_emb = np.delete(self.i_emb, idx, 0)
                (C.IMAGES_DIR / item["file"]).unlink(missing_ok=True)
            self.save()
        return True
