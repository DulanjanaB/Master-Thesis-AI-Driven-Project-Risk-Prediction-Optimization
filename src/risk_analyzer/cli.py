from __future__ import annotations

import argparse
from pathlib import Path

from risk_analyzer.pipelines.generate_synthetic import generate_synthetic_dataset
from risk_analyzer.pipelines.train_model import train_and_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-analyzer",
        description=(
            "AI-powered Lessons Learned risk analyzer (synthetic PoC): generate data, "
            "train a baseline model, and write a short report."
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate a synthetic dataset")
    p_gen.add_argument("--out", type=Path, required=True, help="Output CSV path")
    p_gen.add_argument("--n", type=int, default=500, help="Number of synthetic projects")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")

    p_train = sub.add_parser("train", help="Train baseline model and write report")
    p_train.add_argument("--data", type=Path, required=True, help="Input CSV path")
    p_train.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/model.joblib"),
        help="Output path for trained model artifact",
    )
    p_train.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for run report output",
    )
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "generate":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_dataset(out_csv=args.out, n=args.n, seed=args.seed)
        print(f"Wrote synthetic dataset: {args.out}")
        return

    if args.cmd == "train":
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = train_and_report(
            data_csv=args.data,
            model_out=args.model_out,
            report_dir=args.report_dir,
            seed=args.seed,
        )
        print(f"Saved model: {args.model_out}")
        print(f"Wrote report: {report_path}")
        return

    raise SystemExit(f"Unknown command: {args.cmd}")
