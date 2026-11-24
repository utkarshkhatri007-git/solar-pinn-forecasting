"""Controlled ablation experiment, metrics, checkpoints and visual reporting."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .data import (
    DATASET_REPO,
    OBSERVATION_FEATURES,
    PHYSICS_DERIVED_FEATURES,
    RATED_CAPACITY_KW,
    load_site_csv,
    make_supervised_frame,
)
from .model import SolarPINN, TrainConfig, fit_network, predict_network


def chronological_split(frame: pd.DataFrame):
    """Split chronologically into 70% train, 15% validation and 15% test."""
    n = len(frame)
    if n < 300:
        raise ValueError(f"Only {n} supervised rows remain. At least 300 are required.")
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    return (
        frame.iloc[:i_train].copy(),
        frame.iloc[i_train:i_val].copy(),
        frame.iloc[i_val:].copy(),
    )


def phys_arrays(frame: pd.DataFrame):
    return {
        "night": frame["night_mask"].to_numpy(dtype=np.float32),
        "envelope": frame["physics_envelope"].to_numpy(dtype=np.float32),
        "prior": frame["pv_physics_prior"].to_numpy(dtype=np.float32),
        "daylight": frame["daylight_mask"].to_numpy(dtype=np.float32),
    }


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, daylight_mask: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 1)
    daylight_mask = np.asarray(daylight_mask, dtype=bool)

    def calc(a, b):
        if len(a) == 0:
            return {
                "MAE_CF": np.nan,
                "RMSE_CF": np.nan,
                "R2": np.nan,
                "MAE_kW": np.nan,
                "RMSE_kW": np.nan,
                "sMAPE_pct": np.nan,
            }
        mae = mean_absolute_error(a, b)
        rmse = np.sqrt(mean_squared_error(a, b))
        smape = np.mean(2 * np.abs(b - a) / (np.abs(a) + np.abs(b) + 1e-4)) * 100
        return {
            "MAE_CF": mae,
            "RMSE_CF": rmse,
            "R2": r2_score(a, b) if len(a) > 1 else np.nan,
            "MAE_kW": mae * RATED_CAPACITY_KW,
            "RMSE_kW": rmse * RATED_CAPACITY_KW,
            "sMAPE_pct": smape,
        }

    overall = calc(y_true, y_pred)
    day = calc(y_true[daylight_mask], y_pred[daylight_mask])
    row = {"Model": name}
    row.update({f"Overall_{key}": value for key, value in overall.items()})
    row.update({f"Daylight_{key}": value for key, value in day.items()})
    return row


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _save_figures(test, predictions, histories, metrics, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(test["target_timestamp"], utc=True)
    y = test["target_cf"].to_numpy()
    full_pinn_name = "PINN (full constraints)"
    best_name = str(metrics.iloc[0]["Model"])

    # Continuous test-period trace: measurement, full PINN and best test model.
    n_show = min(240, len(test))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(times.iloc[:n_show], y[:n_show] * RATED_CAPACITY_KW,
            label="Measured AC power", linewidth=1.7)
    ax.plot(times.iloc[:n_show], predictions[full_pinn_name][:n_show] * RATED_CAPACITY_KW,
            label=full_pinn_name, alpha=0.9, linewidth=1.2)
    if best_name != full_pinn_name:
        ax.plot(times.iloc[:n_show], predictions[best_name][:n_show] * RATED_CAPACITY_KW,
                label=f"Best test model: {best_name}", alpha=0.85, linewidth=1.2)
    ax.set_title("One-hour-ahead solar power forecast (test period)")
    ax.set_ylabel("AC power (kW)")
    ax.set_xlabel("Target timestamp (UTC)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "forecast_trace.png", dpi=180)
    plt.close(fig)

    # PINN scatter plot.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sample_idx = np.linspace(0, len(y) - 1, min(2500, len(y))).astype(int)
    ax.scatter(
        y[sample_idx] * RATED_CAPACITY_KW,
        predictions[full_pinn_name][sample_idx] * RATED_CAPACITY_KW,
        s=10,
        alpha=0.35,
    )
    ax.plot([0, RATED_CAPACITY_KW], [0, RATED_CAPACITY_KW],
            linestyle="--", linewidth=1.5, label="Ideal prediction")
    ax.set_xlim(0, RATED_CAPACITY_KW)
    ax.set_ylim(0, RATED_CAPACITY_KW)
    ax.set_xlabel("Measured power (kW)")
    ax.set_ylabel("Predicted power (kW)")
    ax.set_title("Full PINN: predicted vs measured output")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prediction_scatter.png", dpi=180)
    plt.close(fig)

    # Full-PINN training curve: validation MSE is the selection criterion.
    hist = pd.DataFrame(histories[full_pinn_name])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["epoch"], hist["train_loss"], label="Training objective")
    ax.plot(hist["epoch"], hist["val_mse"], label="Validation MSE (checkpoint criterion)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Full PINN training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_curve.png", dpi=180)
    plt.close(fig)

    err = (predictions[full_pinn_name] - y) * RATED_CAPACITY_KW
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(err, bins=45, edgecolor="white")
    ax.axvline(0, linestyle="--", linewidth=1.4)
    ax.set_xlabel("Forecast error (kW; predicted minus measured)")
    ax.set_ylabel("Count")
    ax.set_title("Full PINN test residual distribution")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "residuals.png", dpi=180)
    plt.close(fig)

    hour = times.dt.hour.to_numpy()
    per_hour = []
    for hour_value in range(24):
        mask = hour == hour_value
        per_hour.append(
            np.mean(np.abs(predictions[full_pinn_name][mask] - y[mask])) * RATED_CAPACITY_KW
            if np.any(mask)
            else np.nan
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(24), per_hour)
    ax.set_xlabel("Target hour (UTC)")
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Full PINN test error by hour of day")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "error_by_hour.png", dpi=180)
    plt.close(fig)

    # Compact model-comparison chart.
    ordered = metrics.sort_values("Daylight_MAE_kW", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(ordered))))
    ax.barh(ordered["Model"], ordered["Daylight_MAE_kW"])
    ax.invert_yaxis()
    ax.set_xlabel("Daylight MAE (kW; lower is better)")
    ax.set_title("Test-set model comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "model_comparison.png", dpi=180)
    plt.close(fig)


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig | None = None,
    site_name: str = "site_a",
):
    """Run controlled feature and physics-loss ablations on one fixed split."""
    config = config or TrainConfig()
    output_dir = Path(output_dir)
    model_dir = output_dir / "model"
    scaler_dir = model_dir / "scalers"
    plot_dir = output_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    raw = load_site_csv(data_path)
    frame, all_feature_cols, data_meta = make_supervised_frame(raw)
    missing_obs = set(OBSERVATION_FEATURES).difference(all_feature_cols)
    missing_phys = set(PHYSICS_DERIVED_FEATURES).difference(all_feature_cols)
    if missing_obs or missing_phys:
        raise ValueError(
            f"Expected features are missing. observed/time={sorted(missing_obs)}, "
            f"physics-derived={sorted(missing_phys)}"
        )
    train, val, test = chronological_split(frame)

    y_train = train["target_cf"].to_numpy(dtype=np.float32)
    y_val = val["target_cf"].to_numpy(dtype=np.float32)
    y_test = test["target_cf"].to_numpy(dtype=np.float32)
    p_train, p_val = phys_arrays(train), phys_arrays(val)
    test_night_gate = (test["night_mask"].to_numpy(dtype=float) < 0.5).astype(float)
    daylight = test["daylight_mask"].to_numpy() > 0

    feature_sets = {
        "observed_only": list(OBSERVATION_FEATURES),
        "with_physics_features": list(OBSERVATION_FEATURES) + list(PHYSICS_DERIVED_FEATURES),
    }
    scaled = {}
    scalers = {}
    for set_name, columns in feature_sets.items():
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[columns]).astype(np.float32)
        Xv = scaler.transform(val[columns]).astype(np.float32)
        Xte = scaler.transform(test[columns]).astype(np.float32)
        scaled[set_name] = (Xtr, Xv, Xte)
        scalers[set_name] = scaler
        joblib.dump(scaler, scaler_dir / f"{set_name}.joblib")

    predictions: dict[str, np.ndarray] = {}
    histories: dict[str, list[dict]] = {}
    training_metadata: dict[str, dict] = {}

    # Two feature ablations. Same neural architecture, split, optimizer,
    # training budget, seed and validation-MSE checkpoint criterion.
    neural_experiments = [
        ("DNN (observed-only)", "observed_only", (0.0, 0.0, 0.0)),
        ("DNN + physics features", "with_physics_features", (0.0, 0.0, 0.0)),
        (
            "PINN (night only)",
            "with_physics_features",
            (config.lambda_night, 0.0, 0.0),
        ),
        (
            "PINN (envelope only)",
            "with_physics_features",
            (0.0, config.lambda_envelope, 0.0),
        ),
        (
            "PINN (prior only)",
            "with_physics_features",
            (0.0, 0.0, config.lambda_prior),
        ),
        (
            "PINN (full constraints)",
            "with_physics_features",
            (config.lambda_night, config.lambda_envelope, config.lambda_prior),
        ),
    ]

    for model_name, feature_set_name, weights in neural_experiments:
        Xtr, Xv, Xte = scaled[feature_set_name]
        model, history, train_meta = fit_network(
            Xtr,
            y_train,
            p_train,
            Xv,
            y_val,
            p_val,
            config=config,
            physics_weights=weights,
        )
        pred = predict_network(model, Xte) * test_night_gate
        predictions[model_name] = pred
        histories[model_name] = history
        training_metadata[model_name] = {
            **train_meta,
            "feature_set": feature_set_name,
            "feature_columns": feature_sets[feature_set_name],
        }
        model_payload = {
            "state_dict": model.state_dict(),
            "input_dim": len(feature_sets[feature_set_name]),
            "hidden_dim": 64,
            "dropout": 0.10,
            "model_name": model_name,
            "feature_set": feature_set_name,
            "feature_columns": feature_sets[feature_set_name],
            "physics_loss_weights": train_meta["physics_loss_weights"],
        }
        torch.save(model_payload, model_dir / f"{_slug(model_name)}.pt")
        pd.DataFrame(history).to_csv(output_dir / f"{_slug(model_name)}_history.csv", index=False)

    # Matched classical baselines on the same two feature sets.
    for display_name, feature_set_name in [
        ("HistGradientBoosting (observed-only)", "observed_only"),
        ("HistGradientBoosting + physics features", "with_physics_features"),
    ]:
        Xtr, _, Xte = scaled[feature_set_name]
        hgb = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=config.seed,
        )
        hgb.fit(Xtr, y_train)
        predictions[display_name] = np.clip(hgb.predict(Xte), 0, 1) * test_night_gate
        joblib.dump(hgb, model_dir / f"{_slug(display_name)}.joblib")
        if feature_set_name == "with_physics_features":
            # Preserve the original model filename expected by older project utilities.
            joblib.dump(hgb, model_dir / "hist_gradient_boosting.joblib")

    predictions["Persistence"] = (
        np.clip(test["power_now"].to_numpy(dtype=float), 0, 1) * test_night_gate
    )
    predictions["Physics prior"] = np.clip(
        test["pv_physics_prior"].to_numpy(dtype=float), 0, 1
    )

    metric_rows = [
        metric_row(name, y_test, pred, daylight) for name, pred in predictions.items()
    ]
    metrics = pd.DataFrame(metric_rows).sort_values("Daylight_MAE_kW")
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "target_timestamp": test["target_timestamp"].astype(str).to_numpy(),
            "measured_power_kw": y_test * RATED_CAPACITY_KW,
            "measured_capacity_factor": y_test,
            "future_solar_elevation_deg": test["future_solar_elevation_deg"].to_numpy(),
            "daylight_metric_row": daylight.astype(int),
        }
    )
    for name, pred in predictions.items():
        slug = _slug(name)
        prediction_frame[f"{slug}_capacity_factor"] = pred
        prediction_frame[f"{slug}_kw"] = pred * RATED_CAPACITY_KW
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    for name, history in histories.items():
        pd.DataFrame(history).to_csv(output_dir / f"{_slug(name)}_history.csv", index=False)

    # Preserve the original history filenames for tools that expect them.
    pd.DataFrame(histories["PINN (full constraints)"]).to_csv(
        output_dir / "pinn_training_history.csv", index=False
    )
    pd.DataFrame(histories["DNN + physics features"]).to_csv(
        output_dir / "dnn_training_history.csv", index=False
    )

    # Keep legacy, easy-to-find aliases for the full PINN and full-feature scaler.
    shutil.copy2(model_dir / "pinn_full_constraints.pt", model_dir / "solar_pinn.pt")
    joblib.dump(scalers["with_physics_features"], model_dir / "feature_scaler.joblib")

    metadata = {
        "project": "Physics-Informed Neural Network for Short-Term Solar Power Generation Forecasting",
        "site": site_name,
        "data_path": str(Path(data_path).as_posix()),
        "source_dataset": (
            "Synthetic demonstration dataset generated by this project (not field measurements)"
            if site_name == "synthetic_demo"
            else "Two-Site Solar PV Inverter & Weather Dataset (CC BY 4.0)"
        ),
        "source_repository": DATASET_REPO,
        "dataset_type": "synthetic_demo" if site_name == "synthetic_demo" else "public_field_generation_dataset",
        "forecast_horizon": "1 hour ahead",
        "forecast_protocol": (
            "rolling one-step-ahead; latest measured power and weather through forecast origin t "
            "are assumed available at each forecast origin"
        ),
        "rated_capacity_kw": RATED_CAPACITY_KW,
        "feature_columns": all_feature_cols,
        "feature_sets": feature_sets,
        "data_metadata": data_meta,
        "n_raw_hourly_rows": int(len(raw)),
        "n_supervised_rows": int(len(frame)),
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "split_method": "chronological 70/15/15; scaler fit on training split only",
        "train_config": config.__dict__,
        "neural_experiments": training_metadata,
        # Backward-compatible metadata keys used by the original report utilities.
        "pinn_training": training_metadata["PINN (full constraints)"],
        "dnn_training": training_metadata["DNN + physics features"],
        "checkpoint_selection": "validation MSE for every neural experiment",
        "inference_night_gate": "all learned and persistence models forced to zero when future solar elevation <= 0 degrees",
        "daylight_metrics": "target solar elevation > 5 degrees",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "Results are dataset- and site-specific. They must not be represented as universal "
            "performance or as validation on Indian installations. The physics prior is approximate."
        ),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _save_figures(test, predictions, histories, metrics, plot_dir)
    _write_results_report(output_dir, metrics, metadata)
    return metrics, metadata


def _write_results_report(output_dir: Path, metrics: pd.DataFrame, metadata: dict):
    best = metrics.iloc[0]
    lines = [
        "# Corrected Ablation Results — Solar Forecasting",
        "",
        f"Generated (UTC): {metadata['generated_at_utc']}",
        f"Dataset site: {metadata['site']}",
        f"Forecast horizon: {metadata['forecast_horizon']}",
        f"Rows: {metadata['n_supervised_rows']} supervised; split = {metadata['split_counts']}",
        "",
        "## Test-set comparison",
        "",
        "| Model | Daylight MAE (kW) | Daylight RMSE (kW) | Daylight R² | Overall MAE (kW) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['Model']} | {row['Daylight_MAE_kW']:.3f} | "
            f"{row['Daylight_RMSE_kW']:.3f} | {row['Daylight_R2']:.4f} | "
            f"{row['Overall_MAE_kW']:.3f} |"
        )
    lines += [
        "",
        f"Lowest daylight MAE on this test split: **{best['Model']}** "
        f"({best['Daylight_MAE_kW']:.3f} kW).",
        "",
        "## Experimental design and interpretation",
        "",
        "- All neural variants use the same architecture, chronological split, optimizer settings, "
        "training budget and validation-MSE checkpoint-selection criterion.",
        "- The observed-only variants exclude the three physics-derived input features. The "
        "physics-feature variants include estimated irradiance, estimated module temperature and "
        "the approximate PV prior.",
        "- Single-constraint PINNs activate one penalty at a time. The full PINN activates all "
        "three penalties. The ablation isolates their contribution conditional on using the "
        "physics-derived feature set.",
        "- The future solar position is used as deterministic time/site information. All learned "
        "models and persistence use an exact night gate when future solar elevation is at or below 0 degrees.",
        "- Daylight metrics use target solar elevation above 5 degrees to prevent zero-output "
        "night hours from dominating the primary comparison.",
        "- This is a rolling one-step-ahead experiment. At each forecast origin, recent measured "
        "power and weather through that time are assumed available.",
        "- The estimated irradiance uses future solar elevation and cloud cover observed at the "
        "forecast origin. It is not a measured future irradiance value or an operational weather forecast.",
        "- A lower PINN error is not guaranteed. Report the measured ranking, including cases in "
        "which a simpler baseline outperforms the PINN.",
        "- Results are specific to this dataset, site and split; they do not establish performance "
        "on Indian installations or other PV systems.",
        "",
        "## Reproducibility",
        "",
        "Model weights, feature scalers, model metadata, per-model training histories, test "
        "predictions and plots are saved in this output directory. The untouched test split is "
        "used only for final metric calculation and plots, not for checkpoint selection.",
    ]
    (output_dir / "report_results.md").write_text("\n".join(lines), encoding="utf-8")
