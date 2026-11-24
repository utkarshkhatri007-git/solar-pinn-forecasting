"""Run the complete Solar PINN experiment from the project root."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from solar_pinn.data import RAW_URLS, download_site, generate_synthetic_demo
from solar_pinn.experiment import run_experiment
from solar_pinn.model import TrainConfig

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(
        description="One-hour-ahead solar PV forecasting with a physics-informed neural network."
    )
    p.add_argument("--source", choices=["real", "demo"], default="real",
                   help="real downloads the CC-BY dataset if missing; demo uses generated synthetic data.")
    p.add_argument("--site", choices=["site_a", "site_b"], default="site_a")
    p.add_argument("--data", type=str, default=None, help="Optional path to a compatible hourly CSV.")
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
    elif args.source == "demo":
        data_path = DATA_DIR / "demo_synthetic_site_a.csv"
        if not data_path.exists():
            print("Creating clearly labelled synthetic demonstration data...")
            generate_synthetic_demo(data_path)
    else:
        data_path = DATA_DIR / "real" / f"{args.site}_hourly_weather.csv"
        if not data_path.exists():
            print(f"Downloading real public dataset: {args.site} ...")
            try:
                download_site(args.site, data_path)
            except Exception as exc:
                print("\nCould not download the real dataset:", exc)
                print("\nManual download option:")
                print(f"  {RAW_URLS[args.site]}")
                print(f"Save the CSV as: {data_path}")
                print("Then run: python run_project.py --data data/real/" + f"{args.site}_hourly_weather.csv\n")
                print("Offline software check (synthetic data; not research results):")
                print("  python run_project.py --source demo\n")
                return 2

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    print(f"Using data file: {data_path}")
    if args.source == "demo" and not args.data:
        print("DATA WARNING: this run uses synthetic demonstration data, not field measurements.")

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    metrics, metadata = run_experiment(
        data_path=data_path,
        output_dir=Path(args.output).expanduser().resolve(),
        config=config,
        site_name=args.site if args.source == "real" else "synthetic_demo",
    )
    print("\nTEST-SET MODEL COMPARISON")
    display_cols = ["Model", "Daylight_MAE_kW", "Daylight_RMSE_kW", "Daylight_R2", "Overall_MAE_kW"]
    print(metrics[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nCompleted. Outputs saved to: {Path(args.output).resolve()}")
    print("Read outputs/report_results.md and reports/technical_report_template.md before submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
