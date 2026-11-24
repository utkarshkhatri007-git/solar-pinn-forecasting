"""Data loading and leakage-aware one-hour-ahead feature engineering."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd

DATASET_REPO = "https://github.com/upb-nlp/PV-Generation-Dataset"
RAW_URLS = {
    "site_a": "https://raw.githubusercontent.com/upb-nlp/PV-Generation-Dataset/main/site_a/site_a_hourly_weather.csv",
    "site_b": "https://raw.githubusercontent.com/upb-nlp/PV-Generation-Dataset/main/site_b/site_b_hourly_weather.csv",
}
RATED_CAPACITY_KW = 50.0

# These features use measurements available at forecast origin t and/or
# deterministic timestamp-derived solar/calendar information at t+1.
OBSERVATION_FEATURES = [
    "power_now",
    "power_lag_1h",
    "power_lag_2h",
    "power_lag_5h",
    "power_lag_23h",
    "ghi_now",
    "ambient_temp_now",
    "cloud_now",
    "humidity_now",
    "wind_now",
    "ghi_lag_1h",
    "ghi_lag_2h",
    "future_hour_sin",
    "future_hour_cos",
    "future_doy_sin",
    "future_doy_cos",
    "solar_elevation_future",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
]

PHYSICS_DERIVED_FEATURES = [
    "ghi_physics_estimate",
    "module_temp_estimate",
    "pv_physics_prior",
]


def download_site(site: str, destination: str | Path, timeout: int = 45, retries: int = 3) -> Path:
    """Download and validate the public hourly CSV before saving it locally."""
    if site not in RAW_URLS:
        raise ValueError(f"Unknown site '{site}'. Choose from {list(RAW_URLS)}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        RAW_URLS[site],
        headers={
            "User-Agent": "Solar-PINN-course-project/1.0",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if len(payload) < 10_000:
                raise RuntimeError("Downloaded file is unexpectedly small.")
            preview = pd.read_csv(BytesIO(payload), nrows=5)
            required = {"datetime_utc", "active_power_kw", "solar_elevation_deg", "ghi_wm2"}
            missing = required.difference(preview.columns)
            if missing:
                raise RuntimeError(
                    f"Downloaded CSV has an unexpected schema; missing {sorted(missing)}."
                )
            if len(payload.splitlines()) < 1_000:
                raise RuntimeError("Downloaded CSV has too few rows to be the expected site dataset.")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(
        f"Failed to download/validate {site} after {retries} attempts from {RAW_URLS[site]}. "
        f"Last error: {last_error}"
    ) from last_error


def load_site_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate the published hourly-weather schema."""
    path = Path(path)
    df = pd.read_csv(path)
    required = {"datetime_utc", "active_power_kw", "solar_elevation_deg", "ghi_wm2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            "Use the Two-Site Solar PV dataset hourly CSV described in README.md."
        )

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates("datetime_utc", keep="last").set_index("datetime_utc")

    # Reindex to a continuous hourly grid so a lag of one row always means
    # exactly one hour. Missing target rows remain missing and are later dropped.
    if len(df.index) > 1:
        hourly_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
        df = df.reindex(hourly_index)
        df.index.name = "datetime_utc"

    numeric_defaults = {
        "active_power_kw": np.nan,
        "ghi_wm2": np.nan,
        "temp_c": np.nan,
        "cloud_cover_pct": 50.0,
        "humidity_pct": 50.0,
        "wind_speed_kmh": 0.0,
        "solar_elevation_deg": np.nan,
        "solar_azimuth_deg": 180.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "capacity_factor" in df.columns:
        cf = pd.to_numeric(df["capacity_factor"], errors="coerce")
        # Measured AC power is preferred; capacity_factor is only a fallback.
        df["target_cf"] = df["active_power_kw"] / RATED_CAPACITY_KW
        df.loc[df["active_power_kw"].isna(), "target_cf"] = cf
    else:
        df["target_cf"] = df["active_power_kw"] / RATED_CAPACITY_KW

    # Invalid labels are discarded; small valid deviations are clipped to the
    # physical capacity-factor interval.
    df.loc[(df["target_cf"] < -0.02) | (df["target_cf"] > 1.25), "target_cf"] = np.nan
    df["target_cf"] = df["target_cf"].clip(0.0, 1.0)
    df["active_power_kw"] = df["target_cf"] * RATED_CAPACITY_KW
    return df


def make_supervised_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str | list[str]]]:
    """Create features at time t to predict PV output at time t+1.

    Inputs contain measurements through forecast origin t. The future solar
    position and calendar features are taken from the timestamp at t+1; they
    are treated as deterministic astronomical/time information, not weather
    measurements. No future power or future weather measurement is a feature.
    """
    d = df.copy().sort_index()
    idx = d.index
    future_idx = pd.Series(idx, index=idx).shift(-1)

    x = pd.DataFrame(index=idx)
    x["power_now"] = d["target_cf"]
    for lag in (1, 2, 5, 23):
        x[f"power_lag_{lag}h"] = d["target_cf"].shift(lag)

    for col, name in [
        ("ghi_wm2", "ghi"),
        ("temp_c", "ambient_temp"),
        ("cloud_cover_pct", "cloud"),
        ("humidity_pct", "humidity"),
        ("wind_speed_kmh", "wind"),
    ]:
        x[f"{name}_now"] = d[col]
    for lag in (1, 2):
        x[f"ghi_lag_{lag}h"] = d["ghi_wm2"].shift(lag)

    future_elev = d["solar_elevation_deg"].shift(-1)
    future_az = d["solar_azimuth_deg"].shift(-1)
    future_hour = pd.DatetimeIndex(future_idx).hour
    future_day = pd.DatetimeIndex(future_idx).dayofyear
    x["future_hour_sin"] = np.sin(2 * np.pi * future_hour / 24.0)
    x["future_hour_cos"] = np.cos(2 * np.pi * future_hour / 24.0)
    x["future_doy_sin"] = np.sin(2 * np.pi * future_day / 365.25)
    x["future_doy_cos"] = np.cos(2 * np.pi * future_day / 365.25)
    x["solar_elevation_future"] = future_elev
    az_rad = np.deg2rad(future_az)
    x["solar_azimuth_sin"] = np.sin(az_rad)
    x["solar_azimuth_cos"] = np.cos(az_rad)

    # Approximate prior, not a full PV-system model or weather forecast.
    cloud_fraction = (d["cloud_cover_pct"] / 100.0).clip(0.0, 1.0)
    sin_elev = np.sin(np.deg2rad(future_elev)).clip(lower=0.0)
    ghi_est = 1000.0 * sin_elev * (1.0 - 0.75 * cloud_fraction)
    module_temp_est = d["temp_c"] + 0.03 * ghi_est
    temp_factor = (1.0 - 0.004 * (module_temp_est - 25.0)).clip(0.70, 1.10)
    physics_prior = (0.82 * (ghi_est / 1000.0) * temp_factor).clip(0.0, 1.0)
    envelope = (1.15 * ghi_est / 1000.0).clip(0.0, 1.0)

    x["ghi_physics_estimate"] = ghi_est
    x["module_temp_estimate"] = module_temp_est
    x["pv_physics_prior"] = physics_prior

    y = d["target_cf"].shift(-1).rename("target_cf")
    target_power = (y * RATED_CAPACITY_KW).rename("target_power_kw")
    target_time = pd.Series(idx, index=idx).shift(-1).rename("target_timestamp")
    future_elev_target = future_elev.rename("future_solar_elevation_deg")

    result = x.copy()
    result["target_cf"] = y
    result["target_power_kw"] = target_power
    result["target_timestamp"] = target_time
    result["future_solar_elevation_deg"] = future_elev_target
    result["physics_envelope"] = envelope
    result["night_mask"] = (future_elev <= 0).astype(float)
    result["daylight_mask"] = (future_elev > 5).astype(float)

    feature_cols = list(x.columns)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feature_cols
        + ["target_cf", "target_timestamp", "future_solar_elevation_deg", "physics_envelope"]
    )
    result = result.sort_values("target_timestamp").reset_index(drop=True)

    metadata = {
        "forecast_horizon": "1 hour ahead",
        "feature_availability": "measurements through forecast origin t; future solar position and calendar features at t+1",
        "observed_or_time_features": OBSERVATION_FEATURES,
        "physics_derived_features": PHYSICS_DERIVED_FEATURES,
        "physics_prior_formula": (
            "0.82 * (estimated_GHI/1000) * clipped_temperature_factor; "
            "estimated_GHI = 1000 * max(sin(future_solar_elevation), 0) "
            "* (1 - 0.75 * cloud_fraction_at_t); "
            "module_temperature_estimate = ambient_temperature_at_t + 0.03 * estimated_GHI"
        ),
        "physics_envelope_formula": "clip(1.15 * estimated_GHI/1000, 0, 1)",
        "physics_prior_status": "approximate engineering prior; not an operational weather forecast or validated PV-system model",
        "daylight_metric_definition": "target solar elevation > 5 degrees",
    }
    return result, feature_cols, metadata


def generate_synthetic_demo(path: str | Path, hours: int = 24 * 180, seed: int = 42) -> Path:
    """Generate a clearly labelled synthetic dataset for an offline smoke test."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()

    seasonal = 0.82 + 0.18 * np.cos(2 * np.pi * (doy - 172) / 365.25)
    daylight_shape = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    elev = np.degrees(np.arcsin(np.clip(daylight_shape * seasonal, 0.0, 1.0)))
    azimuth = (180.0 + 15.0 * (hour - 12.0)) % 360.0

    cloud = np.zeros(hours, dtype=float)
    cloud[0] = rng.uniform(15, 85)
    noise = rng.normal(0, 11, size=hours)
    for i in range(1, hours):
        cloud[i] = np.clip(0.88 * cloud[i - 1] + 0.12 * 50 + noise[i], 0, 100)
    clear_ghi = 1000.0 * np.maximum(0.0, np.sin(np.deg2rad(elev)))
    ghi = clear_ghi * (1.0 - 0.75 * cloud / 100.0) + rng.normal(0, 18, size=hours)
    ghi = np.clip(ghi, 0, 1100)

    seasonal_temp = 12 + 9 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    diurnal_temp = 5 * np.sin(2 * np.pi * (hour - 8) / 24.0)
    temp = seasonal_temp + diurnal_temp + rng.normal(0, 1.8, size=hours)
    module_temp = temp + 0.03 * ghi
    temp_factor = np.clip(1 - 0.004 * (module_temp - 25), 0.70, 1.10)
    cf = 0.82 * (ghi / 1000.0) * temp_factor
    cf += rng.normal(0, 0.018, size=hours) * (ghi > 20)
    cf = np.clip(cf, 0.0, 1.0)
    cf[elev <= 0] = 0.0

    demo = pd.DataFrame(
        {
            "datetime_utc": idx.astype(str),
            "active_power_kw": cf * RATED_CAPACITY_KW,
            "capacity_factor": cf,
            "temp_c": temp,
            "humidity_pct": np.clip(65 - 0.25 * temp + rng.normal(0, 5, hours), 20, 100),
            "cloud_cover_pct": cloud,
            "wind_speed_kmh": np.clip(rng.gamma(2.0, 5.0, hours), 0, 50),
            "ghi_wm2": ghi,
            "dni_wm2": np.clip(ghi * 0.8, 0, 1000),
            "dhi_wm2": np.clip(ghi * 0.2, 0, 500),
            "solar_elevation_deg": elev,
            "solar_azimuth_deg": azimuth,
        }
    )
    demo.to_csv(path, index=False)
    return path
