"""Transformer autoencoder and training utilities.

Provenance: logic taken unchanged from the Transformer_denoise training script of the
original analysis (Step2_train_Transformer_denoise.py); this module is self-contained.
    - TransformerAutoencoder            (lines 35-121)
    - _transformed_zero / _get_zero_nonzero_masks / weighted_mse_loss (124-181)
    - train_one_epoch / eval_loss       (185-256)
    - EarlyStopping                     (310-332)
    - make_loader                       (347-349)
    - _fit (inner early-stopping pass / fixed-epoch refit)  (527-582)
The only change to _fit is that it is a top-level function that
receives the training tensor explicitly instead of closing over it.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, n_tokens=64, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.chunk_size = math.ceil(input_dim / n_tokens)
        self.padded_input_dim = self.chunk_size * n_tokens
        self.input_proj = nn.Linear(self.chunk_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(n_tokens * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
            nn.ReLU(),  # non-negative outputs (as in the original code)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        if self.input_dim < self.padded_input_dim:
            pad = torch.zeros(b, self.padded_input_dim - self.input_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x = x.view(b, self.n_tokens, self.chunk_size)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.reshape(b, -1)
        return self.output_proj(x)


def _transformed_zero(trans):
    table = {
        'no_trans': (0.0, 1e-8),
        'sqrt': (0.0, 1e-8),
        'sqrt+1': (np.sqrt(0 + 1), 1e-6),
        'log2': (np.log2(0 + 1), 1e-8),
        'log2(count+2)': (np.log2(0 + 2), 1e-6),
        'count+1': (1.0, 1e-6),
    }
    return table.get(trans, (0.0, 1e-8))


def _get_zero_nonzero_masks(x, trans):
    zv, tol = _transformed_zero(trans)
    if zv == 0.0 and tol <= 1e-8:
        return (x == 0).float(), (x > 0).float()
    return (torch.abs(x - zv) < tol).float(), (x > zv + tol).float()


def weighted_mse_loss(recon_x, x, weight_strategy='fixed', zero_weight=1.0, nonzero_weight=1.0, trans=None):
    if weight_strategy != 'fixed':
        raise ValueError("only weight_strategy='fixed' is used in this analysis")
    zero_mask, nonzero_mask = _get_zero_nonzero_masks(x, trans)
    weights = zero_mask * zero_weight + nonzero_mask * nonzero_weight
    return torch.mean(weights * (recon_x - x) ** 2)


def train_one_epoch(model, loader, optimizer, device, zero_weight, nonzero_weight, trans):
    model.train()
    total, n = 0.0, 0
    for bi, (x,) in enumerate(loader):
        x = x.to(device)
        if not torch.isfinite(x).all():
            raise ValueError(f"NaN/Inf in input batch {bi}, trans={trans}")
        optimizer.zero_grad()
        recon = model(x)
        if not torch.isfinite(recon).all():
            raise ValueError(f"NaN/Inf in reconstruction at batch {bi}, trans={trans}")
        loss = weighted_mse_loss(recon, x, 'fixed', zero_weight, nonzero_weight, trans)
        if not torch.isfinite(loss):
            raise ValueError("NaN/Inf training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, zero_weight, nonzero_weight, trans):
    model.eval()
    total, n = 0.0, 0
    for bi, (x,) in enumerate(loader):
        x = x.to(device)
        recon = model(x)
        if not torch.isfinite(recon).all():
            raise ValueError(f"NaN/Inf in eval reconstruction at batch {bi}, trans={trans}")
        loss = weighted_mse_loss(recon, x, 'fixed', zero_weight, nonzero_weight, trans)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.stopped = False
        self.best_state = None

    def step(self, metric, model=None, epoch=None):
        if metric < self.best - self.min_delta:
            self.best = metric
            if epoch is not None:
                self.best_epoch = int(epoch)
            self.bad_epochs = 0
            if model is not None:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped


def make_loader(X, idx, batch_size, shuffle):
    return DataLoader(Subset(TensorDataset(X), idx), batch_size=batch_size, shuffle=shuffle)


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(X, train_idx, hp, trans, device, max_epochs, val_idx=None):
    """Train one model from a fresh, identically seeded initialisation.

    X         : float32 tensor (cells x genes) holding ONLY the rows this call may see.
    val_idx   : given  -> early stopping on val MSE; returns best epoch (pass 1).
                None   -> train exactly max_epochs, no validation (refit).
    Returns (model, best_epoch, best_val_loss, history).
    """
    _seed_all(hp["seed"])
    model = TransformerAutoencoder(
        input_dim=X.shape[1], n_tokens=hp["n_tokens"], d_model=hp["d_model"],
        nhead=hp["nhead"], num_layers=hp["num_layers"],
        dim_feedforward=hp["dim_feedforward"], dropout=hp["dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
    train_loader = make_loader(X, train_idx, hp["batch_size"], shuffle=True)
    zw, nzw = hp["zero_weight"], hp["nonzero_weight"]
    history = []

    if val_idx is None:
        for ep in range(1, int(max_epochs) + 1):
            tl = train_one_epoch(model, train_loader, optimizer, device, zw, nzw, trans)
            history.append({"epoch": ep, "train_loss": tl})
        return model, int(max_epochs), None, history

    val_loader = make_loader(X, val_idx, hp["batch_size"], shuffle=False)
    es = EarlyStopping(patience=hp["patience"], min_delta=hp["min_delta"])
    for ep in range(1, int(max_epochs) + 1):
        tl = train_one_epoch(model, train_loader, optimizer, device, zw, nzw, trans)
        rec = {"epoch": ep, "train_loss": tl}
        if ep % hp["check_every"] == 0:
            vl = eval_loss(model, val_loader, device, zw, nzw, trans)
            rec["val_loss"] = vl
            history.append(rec)
            if es.step(vl, model, epoch=ep):
                break
        else:
            history.append(rec)
    return model, int(max(1, es.best_epoch)), float(es.best), history


@torch.no_grad()
def predict(model, X, batch_size, device):
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        r = model(X[i:i + batch_size].to(device))
        if not torch.isfinite(r).all():
            raise ValueError("NaN/Inf in prediction")
        out.append(r.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, model.input_dim), dtype=np.float32)
