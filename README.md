# recsys-engine

> End-to-end movie recommender with **Neural Collaborative Filtering** on **MovieLens 25M**, served via FastAPI in Docker.

**Live demo:** _coming soon (Week 4)_

---

## What this is

An end-to-end recommendation system, not a notebook. It trains multiple models on 25M real movie ratings, precomputes top-N recommendations for 162K users, and serves them through a REST API with a live web frontend.

The headline model is **Neural Collaborative Filtering (NeuMF)** — a PyTorch implementation of He et al. (2017), fusing a Generalized Matrix Factorization branch with a Multi-Layer Perceptron branch via a learned NeuMF layer.

## Models

| Model | Type | Notes |
|---|---|---|
| Popularity + genre-weighted | Baseline | The floor every model must beat |
| SVD | Collaborative filtering | via scikit-surprise |
| **NeuMF (NCF)** | Collaborative filtering | **PyTorch, from paper** — GMF + MLP fusion |
| Content-based | Cold-start | TF-IDF on tags + genres, cosine sim |
| Hybrid | Blended | Rank-normalized NCF + content scores |

### Results (MovieLens latest-small, NDCG@10)

| Model | NDCG@10 | Precision@10 | Recall@10 | Hit Ratio@10 |
|---|---|---|---|---|
| Popularity | 0.0733 | 0.0679 | 0.0335 | 0.2857 |
| SVD | 0.0588 | 0.0571 | 0.0177 | 0.3571 |
| **NeuMF (NCF)** | **0.0892** | **0.0857** | **0.0321** | **0.3214** |
| Content-based | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid (NCF + content) | 0.0850 | 0.0857 | 0.0321 | 0.3214 |

NCF (Neural Collaborative Filtering) outperforms both the popularity baseline and SVD on NDCG@10 and Precision@10 on the MovieLens latest-small dataset (100K ratings, 522 users, 7867 items). Full 25M evaluation pending GPU training.

## Architecture

```
MovieLens 25M ─► data/ ─► train models ─► precompute top-N ─► Parquet
                                                                   │
                                                                   ▼
                                                         FastAPI (Docker)
                                                                   │
                                                                   ▼
                                                         Streamlit frontend
```

## Quick start

```bash
make venv && source .venv/bin/activate
make install
make download-data
make data
make train
make evaluate
make serve
```

## Key design decisions

- **Time-based train/test split** (not random K-fold). Real recommenders train on the past to predict the future; random splits leak future interactions and inflate metrics.
- **Implicit feedback formulation for NCF.** Ratings binarized (≥ 4.0 = positive) with negative sampling. Production recommenders usually optimize clicks/views, not stars.
- **Precomputed top-N per user** served from Parquet at startup. Real-time NeuMF inference over 162K users × 62K items is too slow for a free-tier API; we trade memory for latency and document the tradeoff.
- **NCF training on Google Colab GPU** (free T4). The `.pth` artifact is downloaded at Docker build time from a GitHub Release asset. Training harness lives in `notebooks/03_ncf_training.ipynb` and runs identically on Colab.
- **Hybrid weight tuned on validation set**, not eyeballed.

## Stack

Python 3.11 · PyTorch · scikit-surprise · FastAPI · Streamlit · Docker · Render · GitHub Actions

## Status

- [x] Repo scaffold
- [x] Data download + processing
- [x] Metrics module (from scratch)
- [x] Baseline + SVD
- [x] NCF (NeuMF in PyTorch) + Colab training notebook
- [x] Content-based (cold-start) + Hybrid blender
- [ ] Docker + deploy + frontend + polish (Week 4)
