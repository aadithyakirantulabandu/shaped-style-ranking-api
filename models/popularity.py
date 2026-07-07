"""Popularity baseline: rank by global interaction frequency, minus items the user has seen."""
import pandas as pd


class PopularityRanker:
    def fit(self, interactions: pd.DataFrame) -> "PopularityRanker":
        self.ranked_all = list(interactions["item_id"].value_counts().index)
        return self

    def rank(self, seen_item_ids, k: int = 10):
        seen = set(seen_item_ids)
        return [item for item in self.ranked_all if item not in seen][:k]
