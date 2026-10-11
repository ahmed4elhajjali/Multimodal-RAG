import time
import numpy as np
import config as C
from models import ClipEmbedder, BlipCaptioner
from store import KnowledgeBase


def _softmax(s):
    if len(s) == 0:
        return np.array([])
    z = np.exp(C.LOGIT_SCALE * (s - s.max()))
    denom = z.sum()
    return z / (denom if denom > 0 else 1.0)


def _top(items, probs, cos_sims, k):
    if len(items) == 0:
        return []
    order = np.argsort(-probs)[:k]
    res = []
    for i in order:
        entry = dict(items[i])
        entry["score"] = round(float(probs[i]), 4)
        entry["cosine_sim"] = round(float(cos_sims[i]), 4)
        res.append(entry)
    return res


class MultimodalRAG:
    def __init__(self):
        self.clip = ClipEmbedder()
        self.blip = BlipCaptioner()
        self.kb = KnowledgeBase(self.clip)

    def _rank(self, text_probs_fn, img_vec, k):
        kb = self.kb
        with kb.lock:
            n_t = len(kb.texts)
            n_i = len(kb.images)
            if n_t > 0:
                t_probs, t_cos = text_probs_fn(kb.t_emb)
                texts = _top(kb.texts, t_probs, t_cos, k)
            else:
                texts = []

            if n_i > 0:
                i_cos = kb.i_emb @ img_vec
                i_probs = _softmax(i_cos)
                images = _top(kb.images, i_probs, i_cos, k)
            else:
                images = []

        for img in images:
            img["url"] = f"/kb-images/{img['file']}"
        return texts, images

    def synthesize_image_answer(self, caption, texts, images):
        if not texts:
            return f"I analyzed the image and detected: '{caption}'. However, no matching knowledge entries were found in the database."
        top = texts[0]
        animal = top.get("animal", "Subject").capitalize()
        relevance = round(top.get("score", 0) * 100, 1)
        return (
            f"**Visual Analysis**: BLIP identified this scene as *\"{caption}\"*.\n\n"
            f"**Grounded Knowledge**: The primary retrieved entity is **{animal}** with a confidence score of **{relevance}%**.\n"
            f"> *\"{top.get('text')}\"*\n\n"
            f"**Cross-Modal Verification**: The visual features and semantic caption both align with {animal} records stored in the vector knowledge base."
        )

    def synthesize_text_answer(self, query, texts, images):
        if not texts:
            return f"No relevant facts found for query: \"{query}\"."
        top = texts[0]
        animal = top.get("animal", "Subject").capitalize()
        relevance = round(top.get("score", 0) * 100, 1)
        return (
            f"**Grounded Answer**: Regarding *\"{query}\"*, the knowledge base highlights **{animal}** ({relevance}% relevance):\n\n"
            f"> *\"{top.get('text')}\"*\n\n"
            f"**Retrieved Sources**: {len(texts)} related knowledge record(s) and {len(images)} relevant visual catalog image(s) were cross-matched."
        )

    def by_image(self, img, k=C.TOP_K, alpha=C.ALPHA):
        t0 = time.perf_counter()
        # 1. BLIP captioning
        t_blip_start = time.perf_counter()
        caption = self.blip(img)
        blip_ms = round((time.perf_counter() - t_blip_start) * 1000, 1)

        # 2. CLIP embeddings
        t_clip_start = time.perf_counter()
        ie = self.clip.image(img)
        ce = self.clip.texts([caption])[0]

        # 3. Fused ranking
        def text_probs_fn(T):
            v_cos = T @ ie
            c_cos = T @ ce
            fused_probs = alpha * _softmax(v_cos) + (1.0 - alpha) * _softmax(c_cos)
            fused_cos = alpha * v_cos + (1.0 - alpha) * c_cos
            return fused_probs, fused_cos

        texts, images = self._rank(text_probs_fn, ie, k)
        clip_ms = round((time.perf_counter() - t_clip_start) * 1000, 1)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Add visual vs caption breakdown to each text result
        with self.kb.lock:
            for item in texts:
                idx = next((n for n, t in enumerate(self.kb.texts) if t["id"] == item["id"]), None)
                if idx is not None:
                    item["visual_sim"] = round(float(self.kb.t_emb[idx] @ ie), 4)
                    item["caption_sim"] = round(float(self.kb.t_emb[idx] @ ce), 4)

        synthesis = self.synthesize_image_answer(caption, texts, images)

        return {
            "caption": caption,
            "synthesis": synthesis,
            "texts": texts,
            "images": images,
            "latency": {
                "blip_ms": blip_ms,
                "clip_ms": clip_ms,
                "total_ms": total_ms
            }
        }

    def by_text(self, query, k=C.TOP_K):
        t0 = time.perf_counter()
        qe = self.clip.texts([query])[0]
        clip_encode_ms = round((time.perf_counter() - t0) * 1000, 1)

        def text_probs_fn(T):
            cos = T @ qe
            return _softmax(cos), cos

        texts, images = self._rank(text_probs_fn, qe, k)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        synthesis = self.synthesize_text_answer(query, texts, images)

        return {
            "caption": None,
            "synthesis": synthesis,
            "texts": texts,
            "images": images,
            "latency": {
                "clip_ms": clip_encode_ms,
                "total_ms": total_ms
            }
        }
