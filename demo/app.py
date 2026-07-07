"""Streamlit demo: customer onboarding flow — load an event dataset, call the
live /rank API, and compare ranked results plus NDCG/precision@k against the
popularity baseline.

Requires the API running separately: uvicorn api.main:app
Run with: streamlit run demo/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # let `models`/`eval` import when run standalone

import pandas as pd
import requests
import streamlit as st

from eval.metrics import ndcg_at_k, precision_at_k
from eval.run_eval import evaluate
from models.movielens import load_movielens_events, load_movielens_item_text

REQUIRED_COLUMNS = {"user_id", "item_id", "timestamp", "event_type"}

st.title("Ranking API — Onboarding Demo")
api_url = st.text_input("API base URL", value="http://localhost:8000")

st.header("1. Load your event dataset")
uploaded = st.file_uploader("Upload events CSV (user_id, item_id, timestamp, event_type)", type="csv")
if uploaded is not None:
    interactions = pd.read_csv(uploaded)
    missing = REQUIRED_COLUMNS - set(interactions.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        st.stop()
    item_text = {}  # a custom upload has no title/genre text for the embedding ranker to use
else:
    st.info("No file uploaded — using MovieLens sample data.")
    interactions = load_movielens_events()
    item_text = load_movielens_item_text()

interactions["user_id"] = interactions["user_id"].astype(str)
interactions["item_id"] = interactions["item_id"].astype(str)

st.header("2. Pick a user")
user_id = st.selectbox("User", sorted(interactions["user_id"].unique()))
k = st.slider("Number of results (k)", min_value=1, max_value=20, value=10)

user_events = interactions[interactions["user_id"] == user_id].sort_values("timestamp")
if len(user_events) < 2:
    st.warning("This user needs at least 2 events to hold one out as a ground-truth check.")
    st.stop()

held_out_item = user_events.iloc[-1]["item_id"]
history = user_events.iloc[:-1]
st.caption(f"Sending {len(history)} events as history; holding out the most recent (`{held_out_item}`) as ground truth.")

payload_events = history[["user_id", "item_id", "timestamp", "event_type"]].to_dict(orient="records")


def call_rank(ranker: str):
    resp = requests.post(f"{api_url}/rank", json={"events": payload_events, "k": k, "ranker": ranker}, timeout=10)
    resp.raise_for_status()
    return resp.json()["ranked_items"]


st.header("3. Ranked results")
if st.button("Rank via API"):
    try:
        popularity_items = call_rank("popularity")
        embedding_items = call_rank("embedding")
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the API at {api_url}: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Popularity baseline")
        st.write(popularity_items)
        hit = held_out_item in popularity_items
        st.caption(f"Held-out item hit: {'yes' if hit else 'no'}")
    with col2:
        st.subheader("Embedding ranker")
        st.write(embedding_items)
        hit = held_out_item in embedding_items
        st.caption(f"Held-out item hit: {'yes' if hit else 'no'}")

    st.header("4. Metric improvement over baseline")
    st.caption("Aggregate NDCG/precision@k across all users in this dataset (leave-last-out) — "
               "the single-user hit above is one noisy sample, this is the statistically meaningful comparison.")
    with st.spinner("Computing aggregate eval across all users..."):
        results = evaluate(interactions, item_text, k=k)
    st.dataframe(results)

    ndcg_col = f"ndcg@{k}"
    baseline_ndcg = results.loc["popularity", ndcg_col]
    embedding_ndcg = results.loc["embedding", ndcg_col]
    delta = embedding_ndcg - baseline_ndcg
    lift_pct = (delta / baseline_ndcg * 100) if baseline_ndcg > 0 else 0.0
    st.metric(f"Embedding {ndcg_col} vs popularity", f"{embedding_ndcg:.4f}", f"{lift_pct:+.1f}%")
