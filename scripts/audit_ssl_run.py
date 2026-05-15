#!/usr/bin/env python3
"""Audit SSL W&B runs without retraining anything."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

LOSS_KEYS = (
    "epoch_loss",
    "loss",
    "train/loss",
    "ssl_loss",
    "contrastive_loss",
)


@dataclass
class RunAudit:
    run_id: str
    run_name: str
    created_at: Optional[datetime]
    final_loss: Optional[float]
    best_loss: Optional[float]
    total_steps: Optional[int]
    loss_epoch_1: Optional[float]
    loss_epoch_5: Optional[float]
    loss_epoch_100: Optional[float]
    epoch100_lt_epoch5: Optional[bool]
    loss_series: Optional[pd.Series]


def _to_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _resolve_project_path(api: wandb.Api, entity: Optional[str], project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return f"{default_entity}/{project}"
    return project


def _pick_loss_column(df: pd.DataFrame) -> Optional[str]:
    for key in LOSS_KEYS:
        if key in df.columns:
            return key
    return None


def _extract_loss_series(run: wandb.apis.public.Run) -> Optional[pd.Series]:
    try:
        history_df = run.history(samples=50000, pandas=True)
    except Exception as exc:
        print(f"[warn] Could not load history for run {run.name} ({run.id}): {exc}")
        return None

    if history_df is None or history_df.empty:
        return None

    loss_col = _pick_loss_column(history_df)
    if loss_col is None:
        return None

    work_df = history_df.copy()
    work_df[loss_col] = pd.to_numeric(work_df[loss_col], errors="coerce")

    if "epoch" in work_df.columns:
        work_df["epoch"] = pd.to_numeric(work_df["epoch"], errors="coerce")
        work_df = work_df.dropna(subset=[loss_col, "epoch"]) 
        if work_df.empty:
            return None
        grouped = work_df.groupby(work_df["epoch"].astype(int))[loss_col].mean().sort_index()
        return grouped

    if "_step" in work_df.columns:
        work_df["_step"] = pd.to_numeric(work_df["_step"], errors="coerce")
        work_df = work_df.dropna(subset=[loss_col, "_step"]) 
        if work_df.empty:
            return None
        series = work_df.sort_values("_step")[loss_col]
        series.index = np.arange(1, len(series) + 1)
        return series

    series = work_df[loss_col].dropna()
    if series.empty:
        return None
    series.index = np.arange(1, len(series) + 1)
    return series


def _value_at_epoch(series: Optional[pd.Series], epoch: int) -> Optional[float]:
    if series is None or series.empty:
        return None
    if epoch in series.index:
        return float(series.loc[epoch])
    return None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(val) or np.isinf(val):
        return None
    return val


def _audit_run(run: wandb.apis.public.Run) -> RunAudit:
    series = _extract_loss_series(run)

    final_loss = _safe_float(series.iloc[-1]) if series is not None and not series.empty else None
    best_loss = _safe_float(series.min()) if series is not None and not series.empty else None

    total_steps = None
    if series is not None and not series.empty:
        total_steps = int(len(series))

    loss_e1 = _value_at_epoch(series, 1)
    loss_e5 = _value_at_epoch(series, 5)
    loss_e100 = _value_at_epoch(series, 100)

    epoch100_lt_epoch5 = None
    if loss_e5 is not None and loss_e100 is not None:
        epoch100_lt_epoch5 = loss_e100 < loss_e5

    return RunAudit(
        run_id=run.id,
        run_name=run.name,
        created_at=_to_datetime(getattr(run, "created_at", None)),
        final_loss=final_loss,
        best_loss=best_loss,
        total_steps=total_steps,
        loss_epoch_1=loss_e1,
        loss_epoch_5=loss_e5,
        loss_epoch_100=loss_e100,
        epoch100_lt_epoch5=epoch100_lt_epoch5,
        loss_series=series,
    )


def _fmt_float(value: Optional[float]) -> str:
    return f"{value:.6f}" if value is not None else "N/A"


def _fmt_bool(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "YES" if value else "NO"


def _plot_best_run(run_audit: RunAudit, out_path: Path) -> None:
    if run_audit.loss_series is None or run_audit.loss_series.empty:
        print("[warn] Best run has no plottable history; skipping loss curve plot.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(run_audit.loss_series.index, run_audit.loss_series.values, linewidth=2)
    plt.title(f"SSL Loss Curve — {run_audit.run_name}")
    plt.xlabel("Epoch" if run_audit.loss_epoch_1 is not None else "Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved loss curve: {out_path}")


def _classify_failure(series: Optional[pd.Series]) -> str:
    if series is None or len(series) < 3:
        return "flat"

    values = series.values.astype(float)
    first = values[0]
    last = values[-1]
    if not np.isfinite(first) or not np.isfinite(last):
        return "flat"

    if last > first * 1.05:
        return "diverging"

    deltas = np.diff(values)
    if len(deltas) < 4:
        return "flat"

    signs = np.sign(deltas)
    changes = np.sum(signs[1:] * signs[:-1] < 0)
    change_ratio = changes / max(1, len(deltas) - 1)
    net_drop = (first - last) / max(abs(first), 1e-12)

    if change_ratio > 0.45 and net_drop < 0.20:
        return "oscillating"

    return "flat"


def _diagnosis_and_fix(failure_type: str) -> tuple[str, str]:
    if failure_type == "diverging":
        return (
            "diverging (lr too high)",
            "Set learning rate to 1e-4 (from 3e-4), keep batch_size=256 and temperature=0.07.",
        )
    if failure_type == "oscillating":
        return (
            "oscillating (batch size too small)",
            "Set batch_size to 512 and keep lr=3e-4, temperature=0.07.",
        )
    return (
        "flat (lr too high or temperature wrong)",
        "Set lr=1e-4 and temperature=0.10 (keep batch_size=256) to restore stable contrastive gradients.",
    )


def _print_verdict(best_run: RunAudit) -> None:
    if best_run.loss_series is None or best_run.loss_series.empty:
        print("\nVERDICT: FAIL")
        print("Diagnosis: no usable loss history found for best run.")
        print("Fix: ensure epoch-level loss is logged (e.g., key='epoch_loss') for every epoch.")
        return

    first_loss = _safe_float(best_run.loss_series.iloc[0])
    final_loss = _safe_float(best_run.loss_series.iloc[-1])

    if first_loss is None or final_loss is None or first_loss <= 0:
        print("\nVERDICT: FAIL")
        print("Diagnosis: invalid first/final loss values in history.")
        print("Fix: log numeric finite loss each epoch and rerun audit.")
        return

    drop_pct = ((first_loss - final_loss) / first_loss) * 100.0
    pass_cond = final_loss < 4.0 and drop_pct >= 20.0

    print("\n=== VERDICT (best final-loss run) ===")
    print(f"First loss: {_fmt_float(first_loss)}")
    print(f"Final loss: {_fmt_float(final_loss)}")
    print(f"Drop from epoch 1: {drop_pct:.2f}%")

    if pass_cond:
        print("VERDICT: PASS")
        print("Reason: final loss < 4.0 and loss dropped by at least 20% from epoch 1.")
        return

    failure_type = _classify_failure(best_run.loss_series)
    diagnosis, fix = _diagnosis_and_fix(failure_type)
    print("VERDICT: FAIL")
    print(f"Diagnosis: {diagnosis}.")
    print(f"Exact hyperparameter fix: {fix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SSL W&B runs and loss behavior.")
    parser.add_argument("--project", default="marketmind", help="W&B project name.")
    parser.add_argument("--entity", default=None, help="W&B entity (optional).")
    parser.add_argument(
        "--out",
        default="notebooks/figures/ssl_loss_curve.png",
        help="Output path for best-run loss curve plot.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    project_path = _resolve_project_path(api, args.entity, args.project)

    print(f"Fetching W&B runs from: {project_path}")
    try:
        runs = list(api.runs(project_path))
    except Exception as exc:
        print(f"[error] Could not fetch runs for '{project_path}': {exc}")
        return

    if not runs:
        print("No runs found in project.")
        return

    audits = [_audit_run(run) for run in runs]
    audits.sort(key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc))

    print("\n=== SSL Run Audit (sorted by date) ===")
    for audit in audits:
        date_str = audit.created_at.isoformat() if audit.created_at else "N/A"
        print(
            f"- [{date_str}] {audit.run_name} ({audit.run_id}) | "
            f"final_loss={_fmt_float(audit.final_loss)} | "
            f"best_loss={_fmt_float(audit.best_loss)} | "
            f"steps={audit.total_steps if audit.total_steps is not None else 'N/A'} | "
            f"loss@100 < loss@5: {_fmt_bool(audit.epoch100_lt_epoch5)}"
        )

    candidates = [a for a in audits if a.final_loss is not None]
    if not candidates:
        print("\nNo runs with usable loss history were found.")
        return

    best_run = min(candidates, key=lambda x: x.final_loss)
    print(
        f"\nBest run by final loss: {best_run.run_name} ({best_run.run_id}) "
        f"with final_loss={best_run.final_loss:.6f}"
    )

    _plot_best_run(best_run, Path(args.out))
    _print_verdict(best_run)


if __name__ == "__main__":
    main()
