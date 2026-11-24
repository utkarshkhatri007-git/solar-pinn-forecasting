"""Physics-regularized neural network and training utilities."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SolarPINN(nn.Module):
    """MLP with a sigmoid output, constraining capacity factor to [0, 1]."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class TrainConfig:
    epochs: int = 140
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 22
    lambda_night: float = 0.20
    lambda_envelope: float = 0.08
    lambda_prior: float = 0.025
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _loss_components(
    pred: torch.Tensor,
    y: torch.Tensor,
    night: torch.Tensor,
    envelope: torch.Tensor,
    prior: torch.Tensor,
    daylight: torch.Tensor,
    lambda_night: float,
    lambda_envelope: float,
    lambda_prior: float,
):
    """Return total objective and its unweighted components."""
    mse = torch.mean((pred - y) ** 2)
    night_loss = torch.mean((pred * night) ** 2)
    envelope_loss = torch.mean(torch.relu(pred - envelope) ** 2)
    prior_loss = torch.sum(((pred - prior) ** 2) * daylight) / torch.clamp(
        daylight.sum(), min=1.0
    )
    total = (
        mse
        + lambda_night * night_loss
        + lambda_envelope * envelope_loss
        + lambda_prior * prior_loss
    )
    return total, mse, night_loss, envelope_loss, prior_loss


def _validate_physics_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(weights) != 3:
        raise ValueError("physics_weights must contain (night, envelope, prior) weights.")
    result = tuple(float(value) for value in weights)
    if any(value < 0 for value in result):
        raise ValueError("Physics-loss weights cannot be negative.")
    return result


def fit_network(
    X_train,
    y_train,
    phys_train: Mapping[str, np.ndarray],
    X_val,
    y_val,
    phys_val: Mapping[str, np.ndarray],
    config: TrainConfig,
    use_physics: bool | None = None,
    device: str | None = None,
    physics_weights: tuple[float, float, float] | None = None,
):
    """Train a model and select its checkpoint using validation MSE.

    Physics penalties affect gradient updates only. All model variants,
    including the unconstrained baseline, use the same validation-MSE
    checkpoint criterion and scheduler criterion for a fair comparison.

    ``use_physics`` is retained for compatibility with the original project.
    New experiments should pass ``physics_weights`` explicitly.
    """
    if physics_weights is None:
        if use_physics is None:
            use_physics = True
        physics_weights = (
            (config.lambda_night, config.lambda_envelope, config.lambda_prior)
            if use_physics
            else (0.0, 0.0, 0.0)
        )
    weights = _validate_physics_weights(physics_weights)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be two-dimensional feature matrices.")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Training and validation feature dimensions do not match.")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Feature and target row counts do not match.")
    if len(X_train) < 2 or len(X_val) < 2:
        raise ValueError("At least two training and validation rows are required.")

    required_physics = {"night", "envelope", "prior", "daylight"}
    for split_name, arrays, expected_len in (
        ("training", phys_train, len(X_train)),
        ("validation", phys_val, len(X_val)),
    ):
        missing = required_physics.difference(arrays)
        if missing:
            raise ValueError(f"{split_name} physics arrays missing keys: {sorted(missing)}")
        for key in required_physics:
            if len(arrays[key]) != expected_len:
                raise ValueError(f"{split_name} physics array '{key}' has the wrong length.")

    set_seed(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarPINN(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
        torch.as_tensor(phys_train["night"], dtype=torch.float32),
        torch.as_tensor(phys_train["envelope"], dtype=torch.float32),
        torch.as_tensor(phys_train["prior"], dtype=torch.float32),
        torch.as_tensor(phys_train["daylight"], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        train_ds,
        batch_size=min(config.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    pv = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in phys_val.items()
        if key in required_physics
    }

    best_state = None
    best_val_mse = float("inf")
    best_val_total = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        seen = 0
        for xb, yb, nb, eb, pb, db in loader:
            xb, yb, nb, eb, pb, db = [
                value.to(device) for value in (xb, yb, nb, eb, pb, db)
            ]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, *_ = _loss_components(pred, yb, nb, eb, pb, db, *weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_total += float(loss.detach().item()) * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            vp = model(xv)
            val_total, val_mse, val_night, val_env, val_prior = _loss_components(
                vp, yv, pv["night"], pv["envelope"], pv["prior"], pv["daylight"], *weights
            )

        current_val_mse = float(val_mse.item())
        current_val_total = float(val_total.item())
        scheduler.step(current_val_mse)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / max(seen, 1),
                "val_total_loss": current_val_total,
                "val_mse": current_val_mse,
                "val_night_loss": float(val_night.item()),
                "val_envelope_loss": float(val_env.item()),
                "val_physics_prior_loss": float(val_prior.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        # Critical fairness correction: every neural variant is checkpointed
        # and early-stopped against the same data-only validation MSE.
        if current_val_mse < best_val_mse - 1e-7:
            best_val_mse = current_val_mse
            best_val_total = current_val_total
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    train_meta = {
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        # Backward-compatible alias; this is now explicitly the validation MSE.
        "best_validation_loss": best_val_mse,
        "best_validation_total_loss_at_selected_epoch": best_val_total,
        "checkpoint_selection_metric": "validation_mse",
        "physics_loss_weights": {
            "night": weights[0],
            "envelope": weights[1],
            "prior": weights[2],
        },
        "device": device,
    }
    return model, history, train_meta


def predict_network(model: SolarPINN, X, device: str | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device.type
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=device)
        preds = model(xt).detach().cpu().numpy()
    return np.clip(preds, 0.0, 1.0)
