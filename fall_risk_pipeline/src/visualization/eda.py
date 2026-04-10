"""
src/visualization/eda_fixed.py
FINAL FIXED VERSION (robust + safe)
"""

from __future__ import annotations

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

        self.generated_graphs = []

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
            self._plot_tsne(feat_path)

        self.save_all_graphs()

    # ─────────────────────────────────────────

    def _load_meta(self):
        path = self.proc_dir / "trial_metadata.csv"
        if not path.exists():
            logger.warning("Metadata missing")
            return None
        return pd.read_csv(path)

    def _save(self, fig, name):

        for ext in [self.fmt, "png"]:
            fig.savefig(self.out_dir / f"{name}.{ext}", dpi=self.dpi)

        plt.close(fig)
        self.generated_graphs.append(name)

    def save_all_graphs(self):

        import shutil

        for name in self.generated_graphs:
            for ext in [self.fmt, "png"]:
                src = self.out_dir / f"{name}.{ext}"
                dst = self.graphs_dir / f"{name}.{ext}"
                if src.exists():
                    shutil.copy2(src, dst)

    # ─────────────────────────────────────────

    def _plot_cohort_distribution(self, meta):

        if "cohort" not in meta.columns:
            return

        counts = meta["cohort"].value_counts()

        fig, ax = plt.subplots()

        colors = [COHORT_PALETTE.get(c, "#888") for c in counts.index]

        ax.bar(counts.index, counts.values, color=colors)
        ax.set_title("Cohort Distribution")

        self._save(fig, "cohort_distribution")

    # ─────────────────────────────────────────

    def _plot_label_distribution(self, meta):

        if "risk_label" not in meta.columns:
            return

        counts = meta["risk_label"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            [RISK_LABELS.get(i, str(i)) for i in counts.index],
            counts.values,
            color=[RISK_PALETTE.get(i, "#888") for i in counts.index]
        )

        ax.set_title("Label Distribution")

        self._save(fig, "label_distribution")

    # ─────────────────────────────────────────

    def _plot_trial_duration(self, meta):

        if "duration_s" not in meta.columns:
            return

        fig, ax = plt.subplots()

        ax.hist(meta["duration_s"], bins=30, color="#2196F3")

        ax.set_title("Trial Duration")

        self._save(fig, "trial_duration")

    # ─────────────────────────────────────────

    def _plot_signal_examples(self, meta):

        if "trial_id" not in meta.columns:
            return

        sample = meta.sample(min(3, len(meta)))

        fig, ax = plt.subplots()

        for _, row in sample.iterrows():
            df = self._load_signal(row["trial_id"], "lower_back")

            if df is None:
                continue

            if "acc_resultant" in df.columns:
                ax.plot(df["acc_resultant"].values[:500])

        ax.set_title("Signal Examples")

        self._save(fig, "signal_examples")

    # ─────────────────────────────────────────

    def _plot_psd_by_cohort(self, meta):

        if "cohort" not in meta.columns:
            return

        fig, ax = plt.subplots()

        for cohort in meta["cohort"].unique():

            trials = meta[meta["cohort"] == cohort]["trial_id"].values[:3]

            for tid in trials:
                df = self._load_signal(tid, "lower_back")

                if df is None or "acc_z" not in df.columns:
                    continue

                f, pxx = welch(df["acc_z"].values, fs=self.fs)

                ax.plot(f, pxx, label=cohort)

        ax.set_title("PSD")

        self._save(fig, "psd")

    # ─────────────────────────────────────────

    def _plot_tsne(self, feat_path):

        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        df = pd.read_parquet(feat_path)

        X = df.select_dtypes(include=np.number).fillna(0).values

        if len(X) < 10:
            return

        perplexity = min(30, len(X) // 3)

        X = StandardScaler().fit_transform(X)

        emb = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)

        fig, ax = plt.subplots()

        ax.scatter(emb[:, 0], emb[:, 1], s=10)

        ax.set_title("t-SNE")

        self._save(fig, "tsne")

    # ─────────────────────────────────────────

    def _load_signal(self, trial_id, pos):

        for base in ["signals_clean", "signals"]:
            path = self.proc_dir / base / f"{trial_id}_{pos}.parquet"
            if path.exists():
                return pd.read_parquet(path)

        return None