import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipForConditionalGeneration, BlipProcessor
)
import config as C


def _load_hf(cls, model_name, **kwargs):
    """Load from local Hugging Face cache first to prevent blocking network checks."""
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_name, **kwargs)


def _feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


class ClipEmbedder:
    def __init__(self):
        self.model = _load_hf(CLIPModel, C.CLIP_MODEL).to(C.DEVICE).eval()
        self.proc = _load_hf(CLIPProcessor, C.CLIP_MODEL)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def texts(self, texts, batch=32):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch):
            inp = self.proc(
                text=texts[i : i + batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(C.DEVICE)
            out.append(_norm(_feat(self.model.get_text_features(**inp))))
        return np.vstack(out)

    @torch.no_grad()
    def image(self, img: Image.Image):
        inp = self.proc(images=img, return_tensors="pt").to(C.DEVICE)
        return _norm(_feat(self.model.get_image_features(**inp)))[0]


class BlipCaptioner:
    def __init__(self):
        self.model = _load_hf(BlipForConditionalGeneration, C.BLIP_MODEL).to(C.DEVICE).eval()
        self.proc = _load_hf(BlipProcessor, C.BLIP_MODEL)

    @torch.no_grad()
    def __call__(self, img: Image.Image, prompt: str = None) -> str:
        if prompt:
            inp = self.proc(images=img, text=prompt, return_tensors="pt").to(C.DEVICE)
        else:
            inp = self.proc(images=img, return_tensors="pt").to(C.DEVICE)
        ids = self.model.generate(**inp, max_new_tokens=40)
        return self.proc.decode(ids[0], skip_special_tokens=True).strip()
