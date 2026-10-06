import numpy as np
import config as C
from models import ClipEmbedder, BlipCaptioner
from store import KnowledgeBase


def _softmax(s):
    z = np.exp(C.LOGIT_SCALE * (s - s.max()))
    return z / z.sum()


def _top(items, probs, k):
    return [{**items[i], "score": round(float(probs[i]), 4)} for i in np.argsort(-probs)[:k]]


class MultimodalRAG:
    def __init__(self):
        self.clip, self.blip = ClipEmbedder(), BlipCaptioner()
        self.kb = KnowledgeBase(self.clip)

    def _rank(self, text_probs_fn, img_vec, k):
        kb = self.kb
        with kb.lock:
            texts = _top(kb.texts, text_probs_fn(kb.t_emb), k) if len(kb.texts) else []
            images = _top(kb.images, _softmax(kb.i_emb @ img_vec), k) if len(kb.images) else []
        for i in images:
            i["url"] = f"/kb-images/{i['file']}"
        return texts, images

    def by_image(self, img, k=C.TOP_K, alpha=C.ALPHA):
        caption = self.blip(img)
        ie, ce = self.clip.image(img), self.clip.texts([caption])[0]
        fuse = lambda T: alpha * _softmax(T @ ie) + (1 - alpha) * _softmax(T @ ce)
        texts, images = self._rank(fuse, ie, k)
        return {"caption": caption, "texts": texts, "images": images}

    def by_text(self, query, k=C.TOP_K):
        qe = self.clip.texts([query])[0]
        texts, images = self._rank(lambda T: _softmax(T @ qe), qe, k)
        return {"caption": None, "texts": texts, "images": images}
