#!/usr/bin/env python3
"""Audit whether SSL embeddings correlate with regime-relevant proxies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB_PATH = PROJECT_ROOT / "data/embeddings/test_emb.npy"
BTC_PATH = PROJECT_ROOT / "data/raw/BTCUSDT_1h.parquet"
FIG_DIR = PROJECT_ROOT / "notebooks/figures"

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-01-01")
# Treat absolute 30-day returns <= 0.1% as near-zero noise.
NEAR_ZERO_THRESHOLD = 1e-3


def _resolve_timestamps(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index, name="timestamp")
        return ts

    for col in ("timestamp", "datetime", "date", "time"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed.rename("timestamp")

    raise ValueError(
        "Could not find timestamps in BTCUSDT_1h.parquet. "
        "Expected a DatetimeIndex or one of columns: timestamp/datetime/date/time."
    )


def _rolling_period_30d(timestamps: pd.Series) -> int:
    if len(timestamps) < 2:
        return 30
    ts = pd.to_datetime(timestamps).sort_values()
    diffs = ts.diff().dropna()
    if diffs.empty:
        return 30
    median_step = diffs.median()
    if median_step <= pd.Timedelta(hours=2):
        return 30 * 24
    return 30


def _load_and_align() -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(EMB_PATH)
    if emb.ndim != 2 or emb.shape[1] != 128:
        raise ValueError(
            f"Expected embeddings shape (N, 128), got {emb.shape}. "
            "Fix by regenerating data/embeddings/test_emb.npy from encoder output with out_dim=128."
        )
    if emb.shape[0] == 0:
        raise ValueError("Embeddings file is empty. Regenerate test embeddings before running this audit.")

    btc = pd.read_parquet(BTC_PATH)
    if "close" not in btc.columns:
        raise ValueError("BTCUSDT_1h.parquet must contain a 'close' column for return/volatility proxies.")

    ts = _resolve_timestamps(btc)
    mask = (ts >= TEST_START) & (ts < TEST_END)
    btc_test = btc.loc[mask].copy()
    ts_test = ts.loc[mask]

    if btc_test.empty:
        raise ValueError("No BTC rows found in 2024 test period [2024-01-01, 2025-01-01).")

    n_emb = emb.shape[0]
    n_ts = len(ts_test)
    if n_ts < n_emb:
        raise ValueError(
            f"Not enough BTC 2024 timestamps to align embeddings sequentially: "
            f"embeddings={n_emb}, timestamps={n_ts}. "
            "Fix by extracting BTC-only test embeddings for 2024 so lengths match."
        )
    if n_ts > n_emb:
        print(f"[info] Truncating BTC 2024 rows from {n_ts} to {n_emb} to match embeddings.")
        btc_test = btc_test.iloc[:n_emb].copy()
        ts_test = ts_test.iloc[:n_emb]

    btc_test["timestamp"] = pd.to_datetime(ts_test.values)
    return emb.astype(np.float32, copy=False), btc_test


def _build_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    periods = _rolling_period_30d(df["timestamp"])
    closes = pd.to_numeric(df["close"], errors="coerce")

    quarter_all = df["timestamp"].dt.quarter.astype(int)

    rolling_ret = closes.pct_change(periods=periods)
    ret_sign = np.where(
        rolling_ret > NEAR_ZERO_THRESHOLD,
        "positive",
        np.where(rolling_ret < -NEAR_ZERO_THRESHOLD, "negative", "near-zero"),
    )

    bar_ret = closes.pct_change()
    rolling_vol = bar_ret.rolling(window=periods, min_periods=max(periods // 2, 1)).std()
    valid_mask = rolling_ret.notna() & rolling_vol.notna() & quarter_all.notna()
    if valid_mask.sum() < 2:
        raise ValueError(
            "Not enough valid rows after computing rolling 30-day return/volatility proxies. "
            "Need at least 2 rows with non-NaN rolling metrics."
        )

    quarter_labels = quarter_all.loc[valid_mask].to_numpy(dtype=int)
    ret_sign = np.asarray(ret_sign)[valid_mask.to_numpy()]
    rolling_vol = rolling_vol.loc[valid_mask]

    try:
        vol_bins = pd.qcut(rolling_vol, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        vol_bins = pd.cut(rolling_vol, bins=3, labels=["low", "mid", "high"], include_lowest=True)
    if vol_bins.isna().any():
        raise ValueError(
            "Volatility bins contain NaN values after binning. "
            "Check data quality or rolling window configuration."
        )
    vol_labels = vol_bins.astype(str).to_numpy()

    return valid_mask.to_numpy(), quarter_labels, ret_sign, vol_labels


def _run_tsne(emb: np.ndarray) -> np.ndarray:
    print("Running t-SNE (verbose=1); this may take a while...")
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42,
            verbose=1,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=1000,
            random_state=42,
            verbose=1,
        )
    return tsne.fit_transform(emb)


def _plot_quarter(coords: np.ndarray, quarter_labels: np.ndarray) -> None:
    out = FIG_DIR / "tsne_colored_by_quarter.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=quarter_labels,
        cmap="tab10",
        s=6,
        alpha=0.75,
    )
    cbar = plt.colorbar(scatter, ticks=np.unique(quarter_labels))
    cbar.set_label("Calendar Quarter")
    plt.title("t-SNE of SSL Embeddings — Colored by Quarter (2024)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved: {out}")


def _plot_return_sign(coords: np.ndarray, ret_sign: np.ndarray) -> None:
    out = FIG_DIR / "tsne_colored_by_rolling_return.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    color_map = {"positive": "green", "negative": "red", "near-zero": "gray"}
    colors = [color_map.get(v, "gray") for v in ret_sign]

    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, alpha=0.75)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=7)
        for label, color in color_map.items()
    ]
    plt.legend(handles=handles, title="30-day rolling return sign")
    plt.title("t-SNE of SSL Embeddings — Colored by 30-day Rolling Return Sign")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved: {out}")


def _plot_volatility(coords: np.ndarray, vol_labels: np.ndarray) -> None:
    out = FIG_DIR / "tsne_colored_by_volatility.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    color_map = {"low": "#2ca02c", "mid": "#ffbf00", "high": "#d62728"}
    colors = [color_map.get(v, "#ffbf00") for v in vol_labels]

    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, alpha=0.75)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=7)
        for label, color in color_map.items()
    ]
    plt.legend(handles=handles, title="Rolling 30-day volatility bin")
    plt.title("t-SNE of SSL Embeddings — Colored by Rolling Volatility (low/mid/high)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved: {out}")


def _print_silhouette_and_interpretation(coords: np.ndarray, quarter_labels: np.ndarray) -> None:
    unique_quarters = np.unique(quarter_labels)
    if len(unique_quarters) < 2:
        print("Silhouette score (quarter labels): N/A (need at least 2 distinct quarters).")
        print("Interpretation: no structure (insufficient quarter label diversity).")
        return

    score = silhouette_score(coords, quarter_labels)
    print(f"Silhouette score (quarter labels on 2D t-SNE): {score:.4f}")

    # Requested interpretation bands for this audit task.
    if score > 0.2:
        interpretation = "structure detected"
    elif score < 0.1:
        interpretation = "no structure"
    else:
        interpretation = "weak structure"

    print(
        "Interpretation: investigating whether learned embeddings correlate with regime-relevant "
        f"proxies; result indicates {interpretation}."
    )


def main() -> None:
    emb, btc_test = _load_and_align()
    print(f"Embeddings shape: {emb.shape}")
    print(
        "Aligned BTC rows: "
        f"{len(btc_test)} | {btc_test['timestamp'].min()} -> {btc_test['timestamp'].max()}"
    )

    coords_all = _run_tsne(emb)
    valid_mask, quarter_labels, ret_sign, vol_labels = _build_labels(btc_test)
    coords = coords_all[valid_mask]
    dropped = len(coords_all) - len(coords)
    if dropped > 0:
        print(f"[info] Dropped {dropped} warmup rows with undefined rolling proxies before plotting/scoring.")

    _plot_quarter(coords, quarter_labels)
    _plot_return_sign(coords, ret_sign)
    _plot_volatility(coords, vol_labels)
    _print_silhouette_and_interpretation(coords, quarter_labels)


if __name__ == "__main__":
    main()
