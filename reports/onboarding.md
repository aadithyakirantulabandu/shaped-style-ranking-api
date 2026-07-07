# Ranking API — Technical Onboarding

This is a short walkthrough for a customer's engineering team integrating
with the ranking API: what data it expects, how it ranks, what the numbers
in `eval_results.md` actually mean, and what's missing before this is
production-ready for your traffic.

## 1. Event schema

Everything the API consumes is a stream of interaction events, each shaped
like (`api/schemas.py`):

```json
{"user_id": "1", "item_id": "318", "timestamp": 964982224, "event_type": "like"}
```

- `user_id`, `item_id` — strings, your own IDs, opaque to the ranker.
- `timestamp` — unix epoch int. Used to order a user's history and, in eval,
  to hold out the most recent event as a relevance target.
- `event_type` — free-text label (`view`, `like`, `purchase`, ...). Currently
  informational only — see §4 for what wiring it in would take.

`POST /rank` takes a list of these events (a user's recent history) plus `k`
and a `ranker` choice, and returns a ranked list of `item_id`s.

## 2. How ranking works

Two ranking strategies are implemented, both fit once at API startup and kept
in memory (`api/main.py`):

- **`popularity`** — ranks all items by global interaction frequency, minus
  whatever the requesting user has already seen. No personalization; it's
  the "what's trending" baseline every ranker should be measured against.
- **`embedding`** — content-based. Each item's text (its title and genre
  tags) is embedded with pre-trained GloVe word vectors; a user's vector is
  the mean of their history's item vectors; items are ranked by cosine
  similarity, again excluding what they've already seen. No training step —
  it works purely off item content, so a brand-new item with no interaction
  history yet still gets ranked correctly, as long as it has a title/tags.

If a user has no items the embedding ranker recognizes (a true cold start),
`/rank` falls back to `popularity` automatically and reports which ranker
actually served the response.

## 3. What the metric improvement actually shows

`eval/run_eval.py` runs leave-last-out evaluation on the MovieLens sample
dataset: each user's most recent interaction is held out, the rest is used
as their "seen" history, and both rankers are scored on whether/how high
they place the held-out item.

```
            ndcg@10  precision@10
embedding    0.0006        0.0002
popularity   0.0258        0.0056
```

Read plainly: **popularity wins by a wide margin here.** That's a real
result, not a bug — we spot-checked it (see `README.md`). The embedding
ranker's recommendations are thematically correct (a user who watched Toy
Story and Braveheart gets other Action/Adventure/Family titles back), but
leave-last-out asks a much narrower question: *did we guess the one exact
next item?* Popularity is a famously hard baseline to beat on that specific
question, because a small set of blockbuster items absorb a large share of
everyone's next interaction regardless of taste. Content similarity is
answering a different, arguably more useful question — "what's like what
they already engaged with" — which this eval doesn't reward.

**Takeaway for your team:** don't take NDCG@10 here as "embedding is worse,
ship popularity." It means leave-last-out on implicit, popularity-skewed
data isn't the metric that will tell you if personalization is working.
Once you're integrated, closing the loop with an online metric (click-through
or dwell time on ranked results, A/B tested against popularity) will give
you the real answer for your catalog and your users.

## 4. What a real integration needs next

This service currently proves the shape of the API and the two ranking
strategies end to end. Before it can carry real customer traffic:

- **Real event ingestion.** Right now the API fits both rankers once, at
  startup, on a static MovieLens snapshot. You'd need a pipeline that lands
  your actual event stream (Kafka topic, batch export, whatever you have)
  into the same schema, and a retraining/refit cadence — hourly batch is a
  reasonable starting point for the popularity ranker; the embedding ranker
  only needs refitting when your item catalog's text changes.
- **Item metadata service.** The embedding ranker needs title/tag text per
  item. That has to come from your catalog, not be assumed present — items
  without text should either get a fallback (e.g. popularity-only) or route
  through a content-enrichment step before they're rankable.
- **A true personalization signal.** GloVe-over-title is a reasonable
  zero-training baseline, but it doesn't use *your* users' actual co-purchase
  or co-view patterns. A collaborative-filtering or learned two-tower model
  trained on your interaction data would likely close the gap seen in §3 and
  is the natural next ranker to add behind the same `ranker` interface.
- **event_type-aware weighting.** The schema already carries `event_type`,
  but neither ranker currently distinguishes a `view` from a `purchase`. If
  those carry different intent strength for you, that's a small, high-value
  change (weight interactions in `fit()` rather than treating them all as
  equal signal).
- **Auth, multi-tenancy, rate limiting.** The API currently has none. If this
  serves more than one customer or is reachable outside a trusted network,
  it needs API keys per tenant and per-tenant model isolation (today all
  users share one global catalog and one set of rankers).
- **Latency/scale testing.** Fine for a demo; not load-tested. Item
  embeddings and popularity counts are held in a single process's memory —
  horizontal scaling would need either a shared store (Redis, a vector DB)
  or a stateless refit-on-deploy model per instance.
- **Online evaluation.** As covered in §3, wire up click-through/conversion
  logging on ranked results and run an A/B test against the popularity
  baseline before trusting any offline metric as the ship/no-ship signal.
