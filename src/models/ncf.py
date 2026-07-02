"""Neural Collaborative Filtering (NeuMF) — PyTorch implementation.

Implements the architecture from He et al. (2017), "Neural Collaborative
Filtering", WWW 2017. https://arxiv.org/abs/1708.05031

The model fuses two branches:
  - GMF (Generalized Matrix Factorization): element-wise product of user/item
    embeddings -> linear. Captures LINEAR interactions (same as classic MF
    but with learned per-dimension weights instead of a plain dot product).
  - MLP (Multi-Layer Perceptron): concat of user/item embeddings -> stacked
    dense layers with ReLU. Captures NONLINEAR interactions MF cannot.

The NeuMF layer concatenates both branch outputs and feeds them through a
final linear + sigmoid, producing P(user interacts with item) in [0, 1].

Training uses binary cross-entropy on implicit feedback: ratings are
binarized (>= threshold -> positive), and for each positive (user, item)
pair we sample N random items the user did NOT interact with as negatives.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.base import Recommender
from src.utils.config import IMPLICIT_RATING_THRESHOLD, RANDOM_SEED

# Paper defaults for the MLP layer sizes (input dim doubles per layer down).
DEFAULT_MLP_LAYERS = [64, 32, 16, 8]
DEFAULT_GMF_DIM = 64
DEFAULT_MLP_EMBED_DIM = 32  # each side; concat -> 64 = first MLP layer input
DEFAULT_NEG_RATIO = 4
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-3


class GMF(nn.Module):
    """Generalized Matrix Factorization branch.

    Embeds users and items into vectors of dim `gmf_dim`, takes their
    element-wise product, then applies a linear layer (no activation; the
    final sigmoid lives in NeuMF). This is classic MF with a learned weight
    per latent dimension rather than a plain dot product.
    """

    def __init__(self, n_users: int, n_items: int, gmf_dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, gmf_dim)
        self.item_emb = nn.Embedding(n_items, gmf_dim)
        self.out = nn.Linear(gmf_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return self.out(u * i).squeeze(-1)


class MLP(nn.Module):
    """Multi-Layer Perceptron branch.

    Embeds users and items (dim `mlp_embed_dim` each), concatenates them
    (-> 2*mlp_embed_dim), then passes through stacked dense+ReLU layers.
    Captures nonlinear interactions the dot product cannot.
    """

    def __init__(
        self, n_users: int, n_items: int, mlp_embed_dim: int, layer_sizes: list[int]
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, mlp_embed_dim)
        self.item_emb = nn.Embedding(n_items, mlp_embed_dim)

        layers: list[nn.Module] = []
        in_dim = 2 * mlp_embed_dim
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        h = self.net(torch.cat([u, i], dim=-1))
        return self.out(h).squeeze(-1)


class NeuMF(nn.Module):
    """Neural Matrix Factorization: fuses GMF + MLP via a final layer + sigmoid.

    The fusion concatenates the scalar outputs of GMF and MLP, applies a
    linear, then sigmoid -> P(user interacts with item). The two branches
    have SEPARATE embeddings (paper requirement: pretraining them separately
    helps, and fusing requires independent representations).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = DEFAULT_GMF_DIM,
        mlp_embed_dim: int = DEFAULT_MLP_EMBED_DIM,
        mlp_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        mlp_layers = mlp_layers or DEFAULT_MLP_LAYERS
        self.gmf = GMF(n_users, n_items, gmf_dim)
        self.mlp = MLP(n_users, n_items, mlp_embed_dim, mlp_layers)
        self.fusion = nn.Linear(2, 1, bias=False)
        nn.init.normal_(self.fusion.weight, mean=0.0, std=0.1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        gmf_logit = self.gmf(user_idx, item_idx)
        mlp_logit = self.mlp(user_idx, item_idx)
        logit = self.fusion(torch.cat([gmf_logit.unsqueeze(-1), mlp_logit.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)


class NCFDataset(Dataset):
    """Builds (user_idx, item_idx, label) training tuples with negative sampling.

    For every positive interaction (user liked item), we sample `neg_ratio`
    random items the user did NOT interact with as negatives (label=0).
    This is the implicit-feedback formulation: we don't try to predict the
    star rating, we predict the probability of interaction.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        user2idx: dict[int, int],
        item2idx: dict[int, int],
        threshold: float = IMPLICIT_RATING_THRESHOLD,
        neg_ratio: int = DEFAULT_NEG_RATIO,
        n_items: int | None = None,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.neg_ratio = neg_ratio
        self.n_items = n_items or len(item2idx)
        rng = np.random.default_rng(seed)

        positives = train_df[train_df["rating"] >= threshold]
        self.user_idx = positives["userId"].map(user2idx).to_numpy(dtype=np.int64)
        self.item_idx = positives["movieId"].map(item2idx).to_numpy(dtype=np.int64)
        self.labels = np.ones(len(self.user_idx), dtype=np.float32)

        # Precompute per-user seen-item sets for fast negative sampling.
        seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.user_seen_idx = {
            user2idx[u]: {item2idx[i] for i in seen.get(u, set()) if i in item2idx}
            for u in user2idx
        }

        # Sample negatives.
        neg_users: list[int] = []
        neg_items: list[int] = []
        for u_idx in self.user_idx:
            seen_set = self.user_seen_idx[int(u_idx)]
            sampled = 0
            while sampled < neg_ratio:
                cand = int(rng.integers(0, self.n_items))
                if cand not in seen_set:
                    neg_users.append(int(u_idx))
                    neg_items.append(cand)
                    sampled += 1

        all_users = np.concatenate([self.user_idx, np.array(neg_users, dtype=np.int64)])
        all_items = np.concatenate([self.item_idx, np.array(neg_items, dtype=np.int64)])
        all_labels = np.concatenate(
            [self.labels, np.zeros(len(neg_users), dtype=np.float32)]
        )

        self.samples = list(zip(all_users, all_items, all_labels, strict=False))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.int64, np.int64, np.float32]:
        u, i, lbl = self.samples[idx]
        return np.int64(u), np.int64(i), np.float32(lbl)


class NCFRecommender(Recommender):
    """Recommender wrapper around NeuMF.

    Training is done in `notebooks/03_ncf_training.ipynb` on Google Colab GPU.
    This class supports both training locally (slow on CPU) and loading a
    pretrained `.pth` checkpoint via `load()`.
    """

    name = "ncf"

    def __init__(
        self,
        gmf_dim: int = DEFAULT_GMF_DIM,
        mlp_embed_dim: int = DEFAULT_MLP_EMBED_DIM,
        mlp_layers: list[int] | None = None,
        neg_ratio: int = DEFAULT_NEG_RATIO,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
        device: str | None = None,
    ) -> None:
        self.gmf_dim = gmf_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.mlp_layers = mlp_layers or list(DEFAULT_MLP_LAYERS)
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: NeuMF | None = None
        self._user2idx: dict[int, int] = {}
        self._item2idx: dict[int, int] = {}
        self._idx2item: dict[int, int] = {}
        self._seen: dict[int, set[int]] = {}

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        from src.data.loader import build_id_maps

        user2idx, _, item2idx, idx2item = build_id_maps(train)
        self._user2idx = user2idx
        self._item2idx = item2idx
        self._idx2item = idx2item
        self._seen = train.groupby("userId")["movieId"].apply(set).to_dict()

        n_users = len(user2idx)
        n_items = len(item2idx)
        print(f"NCF: {n_users:,} users, {n_items:,} items, device={self.device}")

        self._model = NeuMF(
            n_users, n_items, self.gmf_dim, self.mlp_embed_dim, self.mlp_layers
        ).to(self.device)

        dataset = NCFDataset(
            train, user2idx, item2idx,
            neg_ratio=self.neg_ratio, n_items=n_items,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_u, batch_i, batch_lbl in loader:
                u = batch_u.to(self.device)
                i = batch_i.to(self.device)
                y = batch_lbl.to(self.device)
                optimizer.zero_grad()
                preds = self._model(u, i)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(u)
            print(f"  epoch {epoch + 1}/{self.epochs}  loss={total_loss / len(dataset):.4f}")

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._model is None or user_id not in self._user2idx:
            return []
        self._model.eval()
        u_idx = self._user2idx[user_id]
        n_items = len(self._idx2item)
        seen = self._seen.get(user_id, set())

        with torch.no_grad():
            u_tensor = torch.full((n_items,), u_idx, dtype=torch.long, device=self.device)
            i_tensor = torch.arange(n_items, dtype=torch.long, device=self.device)
            scores = self._model(u_tensor, i_tensor).cpu().numpy()

        if exclude_seen:
            seen_idx = [self._item2idx[m] for m in seen if m in self._item2idx]
            for si in seen_idx:
                scores[si] = -np.inf

        top_k_idx = np.argsort(-scores)[:k]
        return [self._idx2item[int(i)] for i in top_k_idx]

    def save(self, path) -> None:
        """Save model state_dict + id maps to a .pth file."""
        if self._model is None:
            raise RuntimeError("Nothing to save: model not trained.")
        import torch as _torch
        payload = {
            "state_dict": self._model.state_dict(),
            "config": {
                "gmf_dim": self.gmf_dim,
                "mlp_embed_dim": self.mlp_embed_dim,
                "mlp_layers": self.mlp_layers,
            },
            "user2idx": self._user2idx,
            "idx2item": {int(k): int(v) for k, v in self._idx2item.items()},
            "seen": {int(k): list(v) for k, v in self._seen.items()},
        }
        _torch.save(payload, path)
        print(f"Saved NCF checkpoint -> {path}")

    def load(self, path, train: pd.DataFrame | None = None) -> None:
        """Load a pretrained .pth checkpoint.

        Needs `train` only if the checkpoint didn't store `seen` maps.
        """
        import torch as _torch

        payload = _torch.load(path, map_location=self.device, weights_only=False)
        cfg = payload["config"]
        self.gmf_dim = cfg["gmf_dim"]
        self.mlp_embed_dim = cfg["mlp_embed_dim"]
        self.mlp_layers = cfg["mlp_layers"]
        self._user2idx = payload["user2idx"]
        self._idx2item = {int(k): int(v) for k, v in payload["idx2item"].items()}
        self._item2idx = {v: k for k, v in self._idx2item.items()}

        if "seen" in payload:
            self._seen = {int(k): set(v) for k, v in payload["seen"].items()}
        elif train is not None:
            self._seen = train.groupby("userId")["movieId"].apply(set).to_dict()
        else:
            self._seen = {}

        n_users = len(self._user2idx)
        n_items = len(self._idx2item)
        self._model = NeuMF(
            n_users, n_items, self.gmf_dim, self.mlp_embed_dim, self.mlp_layers
        ).to(self.device)
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        print(f"Loaded NCF checkpoint ({n_users:,} users, {n_items:,} items) from {path}")
