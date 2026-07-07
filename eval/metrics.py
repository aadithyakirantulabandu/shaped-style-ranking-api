"""NDCG@k and precision@k for a ranked list against a set of relevant items."""
import numpy as np


def ndcg_at_k(ranked_items, relevant_items, k: int = 10) -> float:
    relevant = set(relevant_items)
    gains = [1.0 if item in relevant else 0.0 for item in ranked_items[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_items, relevant_items, k: int = 10) -> float:
    relevant = set(relevant_items)
    top_k = ranked_items[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / len(top_k)


if __name__ == "__main__":
    perfect = ["a", "b", "c"]
    assert ndcg_at_k(perfect, {"a", "b", "c"}, k=3) == 1.0
    assert ndcg_at_k(perfect, {"z"}, k=3) == 0.0
    assert ndcg_at_k(["b", "a", "c"], {"a"}, k=3) < 1.0  # relevant item not first -> discounted
    assert precision_at_k(perfect, {"a"}, k=3) == 1 / 3
    assert precision_at_k(perfect, {"z"}, k=3) == 0.0
    print("metrics self-check passed")
