"""Embedding-based ranker: pre-trained GloVe word vectors averaged over each
item's text (title/genres) give item embeddings; a user's vector is the mean
of their history's item vectors, ranked by cosine similarity.
"""
import gensim.downloader as gensim_api
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_glove = None  # lazy singleton: ~66MB download, cached by gensim after first use


def _get_glove():
    global _glove
    if _glove is None:
        _glove = gensim_api.load("glove-wiki-gigaword-50")
    return _glove


def _embed_text(text: str, glove) -> np.ndarray:
    tokens = [t for t in text.lower().split() if t in glove]
    if not tokens:
        return np.zeros(glove.vector_size)
    return np.mean([glove[t] for t in tokens], axis=0)


class EmbeddingRanker:
    def fit(self, interactions: pd.DataFrame, item_text: dict) -> "EmbeddingRanker":
        glove = _get_glove()
        self.item_ids = sorted(set(interactions["item_id"]) & set(item_text))
        self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.item_embeddings = np.stack([_embed_text(item_text[i], glove) for i in self.item_ids])
        return self

    def rank(self, seen_item_ids, k: int = 10):
        vecs = [self.item_embeddings[self.item_id_to_idx[i]] for i in seen_item_ids if i in self.item_id_to_idx]
        if not vecs:
            return []  # cold start: caller should fall back to popularity

        user_vec = np.mean(vecs, axis=0, keepdims=True)
        sims = cosine_similarity(user_vec, self.item_embeddings)[0]
        ranked_idx = np.argsort(-sims)
        seen = set(seen_item_ids)
        return [self.item_ids[i] for i in ranked_idx if self.item_ids[i] not in seen][:k]
