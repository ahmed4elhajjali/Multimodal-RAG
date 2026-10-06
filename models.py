import numpy as np
import torch
from PIL import Image
from transformers import (CLIPModel, CLIPProcessor,
                          BlipForConditionalGeneration, BlipProcessor)
import config as C


def _feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


class ClipEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(C.CLIP_MODEL).to(C.DEVICE).eval()
        self.proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def texts(self, texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            inp = self.proc(text=texts[i:i + batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=77).to(C.DEVICE)
            out.append(_norm(_feat(self.model.get_text_features(**inp))))
        return np.vstack(out)

    @torch.no_grad()
    def image(self, img: Image.Image):
        inp = self.proc(images=img, return_tensors="pt").to(C.DEVICE)
        return _norm(_feat(self.model.get_image_features(**inp)))[0]


class BlipCaptioner:
    def __init__(self):
        self.model = BlipForConditionalGeneration.from_pretrained(C.BLIP_MODEL).to(C.DEVICE).eval()
        self.proc = BlipProcessor.from_pretrained(C.BLIP_MODEL)

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> str:
        inp = self.proc(images=img, return_tensors="pt").to(C.DEVICE)
        ids = self.model.generate(**inp, max_new_tokens=30)
        return self.proc.decode(ids[0], skip_special_tokens=True)
