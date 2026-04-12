"""
src/visualization/eda_fixed.py
FINAL FIXED VERSION (robust + safe)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.signal import welch
from tqdm import tqdm

# Columns that are label-derived and must never enter feature-space analysis.
from src.ingestion.data_loader import METADATA_ONLY_COLS


COHORT_PALETTE = {
    "Healthy": "#2196F3",
    "HipOA": "#FF9800",
    "KneeOA": "#FF5722",
    "ACL": "#9C27B0",
    "PD": "#F44336",
    "CVA": "#E91E63",
    "CIPN": "#009688",
    "RIL": "#795548",
}

RISK_PALETTE = {0: "#4CAF50", 1: "#F44336"}
RISK_LABELS = {0: "Low risk", 1: "High risk"}

# Seed for any stochastic operations (sampling, t-SNE) so figures are reproducible.
_RANDOM_STATE = 42


class EDAAnalyzer:

    def __init__(self, config: dict):
        self.config = config
        self.proc_dir = Path(config["paths"]["processed_data"])
        self.feat_dir = Path(config["paths"]["features"])
        self.out_dir = Path(config["paths"]["figures_eda"])
        self.graphs_dir = Path(config["paths"]["graphs"])

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        self.fmt = config["reporting"]["figure_format"]
        self.dpi = config["reporting"]["figure_dpi"]
        self.fs = config["dataset"]["sampling_rate"]

        self.generated_graphs: list[str] = []

    # ─────────────────────────────────────────

    def run(self):
        meta = self._load_meta()
        if meta is None:
            return

        logger.info("Running EDA...")

        self._plot_cohort_distribution(meta)
        self._plot_label_distribution(meta)
        self._plot_trial_duration(meta)
        self._plot_signal_examples(meta)
        self._plot_psd_by_cohort(meta)

        feat_path = self.feat_dir / "patient_features.parquet"
        if feat_path.exists():
            self._plot_tsne(feat_path, meta)

        self.save_all_graphs()

    # ─────────────────────────────────────────

    def _load_meta(self) -> pd.DataFrame | None:
        path = self.proc_dir / "trial_metadata.csv"
        if not path.exists():
            logger.warning("Metadata missing — skipping EDA.")
            return None
        return pd.read_csv(path)

    def _save(self, fig: plt.Figure, name: str) -> None:
        """Save figure, avoiding duplicate writes when fmt == 'png'."""
        fig.tight_layout()

        # FIX: deduplicate extensions so we never write the same file twice.
        extensions = list(dict.fromkeys([self.fmt, "png"]))
        for ext in extensions:
            fig.savefig(self.out_dir / f"{name}.{ext}", dpi=self.dpi, bbox_inches="tight")

        plt.close(fig)
        self.generated_graphs.append(name)

    def save_all_graphs(self) -> None:
        """Copy generated figures to graphs_dir (skipped if paths are identical)."""
        # FIX: guard against copying a file over itself.
        if self.out_dir.resolve() == self.graphs_dir.resolve():
            return

        for name in self.generated_graphs:
            extensions = list(dict.fromkeys([self.fmt, "png"]))
            for ext in extensions:
                src = self.out_dir / f"{name}.{ext}"
                dst = self.graphs_dir / f"{name}.{ext}"
                if src.exists():
                    shutil.copy2(src, dst)

    # ─────────────────────────────────────────

    def _plot_cohort_distribution(self, meta: pd.DataFrame) -> None:
        if "cohort" not in meta.columns:
            logger.warning("'cohort' column missing — skipping cohort distribution plot.")
            return

        counts = meta["cohort"].value_counts()
        colors = [COHORT_PALETTE.get(c, "#888") for c in counts.index]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_xlabel("Cohort")
        ax.set_ylabel("Count")
        ax.set_title("Cohort Distribution")
        ax.tick_params(axis="x", rotation=30)

        self._save(fig, "cohort_distribution")

    # ─────────────────────────────────────────

    def _plot_label_distribution(self, meta: pd.DataFrame) -> None:
        if "risk_label" not in meta.columns:
            logger.warning("'risk_label' column missing — skipping label distribution plot.")
            return

        counts = meta["risk_label"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(
            [RISK_LABELS.get(i, str(i)) for i in counts.index],
            counts.values,
            color=[RISK_PALETTE.get(i, "#888") for i in counts.index],
        )
        ax.set_ylabel("Count")
        ax.set_title("Label Distribution")

        self._save(fig, "label_distribution")

    # ─────────────────────────────────────────

    def _plot_trial_duration(self, meta: pd.DataFrame) -> None:
        if "duration_s" not in meta.columns:
            logger.warning("'duration_s' column missing — skipping trial duration plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(meta["duration_s"].dropna(), bins=30, color="#2196F3", edgecolor="white")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.set_title("Trial Duration Distribution")

        self._save(fig, "trial_duration")

    # ─────────────────────────────────────────

    def _plot_signal_examples(self, meta: pd.DataFrame) -> None:
        if "trial_id" not in meta.columns:
            logger.warning("'trial_id' column missing — skipping signal examples plot.")
            return

        # FIX: fixed random_state for reproducibility.
        sample = meta.sample(min(3, len(meta)), random_state=_RANDOM_STATE)

        fig, ax = plt.subplots(figsize=(10, 4))
        plotted = 0

        for _, row in sample.iterrows():
            df = self._load_signal(row["trial_id"], "lower_back")
            if df is None:
                continue

            # Prefer gravity-free resultant if available.
            col = "acc_resultant" if "acc_resultant" in df.columns else None
            if col is None:
                logger.debug(f"No acc_resultant in {row['trial_id']} lower_back — skipping.")
                continue

            ax.plot(df[col].values[:500], label=row["trial_id"], alpha=0.8)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            logger.warning("No valid signals found for signal examples plot.")
            return

        ax.set_xlabel("Sample")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Signal Examples — Lower Back acc_resultant")
        ax.legend(fontsize=8)

        self._save(fig, "signal_examples")

    # ─────────────────────────────────────────

    def _plot_psd_by_cohort(self, meta: pd.DataFrame) -> None:
        if "cohort" not in meta.columns:
            logger.warning("'cohort' column missing — skipping PSD plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        # FIX: track which cohorts have already been added to the legend
        # to avoid 3 duplicate entries per cohort (one per trial).
        labeled: set[str] = set()

        for cohort in meta["cohort"].unique():
            trials = meta[meta["cohort"] == cohort]["trial_id"].values[:3]
            color = COHORT_PALETTE.get(cohort, "#888")

            for tid in trials:
                df = self._load_signal(tid, "lower_back")
                if df is None:
                    continue

                # FIX: prefer gravity-free axis; fall back to raw acc_z.
                sig_col = (
                    "acc_z_grav_free" if "acc_z_grav_free" in df.columns
                    else "acc_z" if "acc_z" in df.columns
                    else None
                )
                if sig_col is None:
                    continue

                f, pxx = welch(df[sig_col].values, fs=self.fs)

                label = cohort if cohort not in labeled else None
                ax.semilogy(f, pxx, color=color, alpha=0.6, label=label)
                labeled.add(cohort)

        if not labeled:
            plt.close(fig)
            logger.warning("No valid signals found for PSD plot.")
            return

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (m²/s⁴/Hz)")
        ax.set_title("Power Spectral Density by Cohort — Lower Back")
        ax.legend(fontsize=8, ncol=2)

        self._save(fig, "psd_by_cohort")

    # ─────────────────────────────────────────

    def _plot_tsne(self, feat_path: Path, meta: pd.DataFrame) -> None:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        df = pd.read_parquet(feat_path)

        # FIX: explicitly drop all label-derived and metadata columns before
        # selecting numeric features. Previously select_dtypes() alone would
        # include risk_label, fall_probability etc., making t-SNE trivially
        # separate by label and the plot meaningless.
        safe_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in METADATA_ONLY_COLS
        ]

        if not safe_cols:
            logger.warning("No non-metadata numeric columns found — skipping t-SNE.")
            return

        X = df[safe_cols].fillna(0).values

        if len(X) < 10:
            logger.warning(f"Too few samples ({len(X)}) for t-SNE — skipping.")
            return

        perplexity = min(30, max(5, len(X) // 3))
        X_scaled = StandardScaler().fit_transform(X)

        # FIX: pass random_state for reproducibility.
        emb = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=_RANDOM_STATE,
        ).fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 7))

        # FIX: color by risk_label if available; fall back to cohort; fall back
        # to a single color. Previously the scatter had no color coding at all.
        if "risk_label" in df.columns:
            labels = df["risk_label"].values
            unique_labels = sorted(set(labels))
            for lbl in unique_labels:
                mask = labels == lbl
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    s=15,
                    color=RISK_PALETTE.get(int(lbl), "#888"),
                    label=RISK_LABELS.get(int(lbl), str(lbl)),
                    alpha=0.7,
                )
            ax.legend(title="Risk label", fontsize=8)

        elif "cohort" in df.columns:
            cohorts = df["cohort"].values
            for cohort in sorted(set(cohorts)):
                mask = cohorts == cohort
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    s=15,
                    color=COHORT_PALETTE.get(cohort, "#888"),
                    label=cohort,
                    alpha=0.7,
                )
            ax.legend(title="Cohort", fontsize=8, ncol=2)

        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=15, alpha=0.7)

        ax.set_title(f"t-SNE (perplexity={perplexity})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        self._save(fig, "tsne")

    # ─────────────────────────────────────────

    def _load_signal(self, trial_id: str, pos: str) -> pd.DataFrame | None:
        """Load cleaned signal, falling back to raw if cleaned is unavailable."""
        for base in ["signals_clean", "signals"]:
            path = self.proc_dir / base / f"{trial_id}_{pos}.parquet"
            if path.exists():
                return pd.read_parquet(path)
        return None