"""Compare the popularity baseline against the embedding ranker with leave-last-out
evaluation: each user's most recent interaction is held out as the relevance target,
the rest is training data.

Run with: python -m eval.run_eval
"""
import pandas as pd

from eval.metrics import ndcg_at_k, precision_at_k
from models.embedding import EmbeddingRanker
from models.movielens import load_movielens_events, load_movielens_item_text
from models.popularity import PopularityRanker

K = 10


def leave_last_out_split(interactions: pd.DataFrame):
    interactions = interactions.sort_values(["user_id", "timestamp"])
    is_last = interactions.groupby("user_id")["timestamp"].transform("max") == interactions["timestamp"]
    return interactions[~is_last], interactions[is_last]


def evaluate(interactions: pd.DataFrame, item_text: dict, k: int = K) -> pd.DataFrame:
    """Leave-last-out NDCG@k / precision@k for popularity vs. embedding, averaged
    across users. Reused by both the eval script and the demo's aggregate view.
    """
    train, test = leave_last_out_split(interactions)
    popularity = PopularityRanker().fit(train)
    embedding = EmbeddingRanker().fit(train, item_text)

    train_by_user = train.groupby("user_id")["item_id"].apply(list)
    test_by_user = test.groupby("user_id")["item_id"].apply(list)

    rows = []
    for user_id, held_out in test_by_user.items():
        seen = train_by_user.get(user_id, [])
        if not seen:
            continue
        for name, ranker in [("popularity", popularity), ("embedding", embedding)]:
            ranked = ranker.rank(seen, k=k)
            rows.append({
                "ranker": name,
                f"ndcg@{k}": ndcg_at_k(ranked, held_out, k=k),
                f"precision@{k}": precision_at_k(ranked, held_out, k=k),
            })

    return pd.DataFrame(rows).groupby("ranker").mean().round(4)


def main():
    interactions = load_movielens_events()
    item_text = load_movielens_item_text()
    results = evaluate(interactions, item_text, k=K)
    print(results)

    report_path = "reports/eval_results.md"
    with open(report_path, "w") as f:
        f.write("Ranker Evaluation (leave-last-out, MovieLens ml-latest-small)\n\n")
        f.write(results.to_string())
        f.write("\n")
    results.to_csv("reports/eval_results.csv")
    print(f"\nWrote {report_path} and reports/eval_results.csv")


if __name__ == "__main__":
    main()
