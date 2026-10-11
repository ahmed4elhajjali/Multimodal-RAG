import json, re, threading, uuid
from io import BytesIO
from pathlib import Path
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
        self.reload()

    def reload(self):
        with self.lock:
            if C.KB_PATH.exists():
                try:
                    d = json.loads(C.KB_PATH.read_text("utf-8"))
                except Exception:
                    d = {"texts": [], "images": []}
            else:
                seed = json.loads(C.SEED_PATH.read_text("utf-8")) if C.SEED_PATH.exists() else []
                d = {"texts": [{"id": _id(), **t} for t in seed], "images": []}

            self.texts = d.get("texts", [])
            self.images = [i for i in d.get("images", []) if (C.IMAGES_DIR / i["file"]).exists()]
            known = {i["file"] for i in self.images}

            # Auto-register images present in data/images
            for p in sorted(C.IMAGES_DIR.iterdir()):
                if p.suffix.lower() in EXTS and p.name not in known:
                    label = re.sub(r"[\d_\-\s]+$", "", p.stem).replace("_", " ").strip().title() or p.stem.title()
                    self.images.append({"id": _id(), "file": p.name, "label": label})

            empty = np.zeros((0, self.clip.dim), np.float32)
            self.t_emb = self.clip.texts([t["text"] for t in self.texts]) if self.texts else empty
            
            img_embs = []
            for i in self.images:
                try:
                    im = Image.open(C.IMAGES_DIR / i["file"]).convert("RGB")
                    img_embs.append(self.clip.image(im))
                except Exception:
                    pass
            self.i_emb = np.vstack(img_embs) if img_embs else empty
            self.save()

    def save(self):
        C.KB_PATH.write_text(
            json.dumps({"texts": self.texts, "images": self.images}, ensure_ascii=False, indent=2),
            "utf-8",
        )

    def add_text(self, animal: str, text: str):
        animal = animal.strip()
        text = text.strip()
        e = self.clip.texts([text])
        with self.lock:
            item = {"id": _id(), "animal": animal, "text": text}
            self.texts.append(item)
            self.t_emb = np.vstack([self.t_emb, e]) if len(self.texts) > 1 else e
            self.save()
        return item

    def update_text(self, iid: str, animal: str, text: str):
        with self.lock:
            idx = next((n for n, i in enumerate(self.texts) if i["id"] == iid), None)
            if idx is None:
                return None
            animal = animal.strip()
            text = text.strip()
            self.texts[idx]["animal"] = animal
            self.texts[idx]["text"] = text
            # Re-encode this text
            e = self.clip.texts([text])[0]
            self.t_emb[idx] = e
            self.save()
            return self.texts[idx]

    def add_image(self, data: bytes, label: str):
        label = label.strip()
        img = Image.open(BytesIO(data)).convert("RGB")
        e = self.clip.image(img)
        with self.lock:
            iid = _id()
            filename = f"{iid}.jpg"
            img.save(C.IMAGES_DIR / filename, quality=92)
            item = {"id": iid, "file": filename, "label": label}
            self.images.append(item)
            self.i_emb = np.vstack([self.i_emb, e]) if len(self.images) > 1 else e.reshape(1, -1)
            self.save()
        return item

    def update_image(self, iid: str, label: str):
        with self.lock:
            idx = next((n for n, i in enumerate(self.images) if i["id"] == iid), None)
            if idx is None:
                return None
            self.images[idx]["label"] = label.strip()
            self.save()
            return self.images[idx]

    def delete(self, kind: str, iid: str):
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

    def reset_to_seed(self):
        with self.lock:
            seed = json.loads(C.SEED_PATH.read_text("utf-8")) if C.SEED_PATH.exists() else []
            self.texts = [{"id": _id(), **t} for t in seed]
            if C.KB_PATH.exists():
                C.KB_PATH.unlink(missing_ok=True)
            self.reload()
            return {"texts": len(self.texts), "images": len(self.images)}
