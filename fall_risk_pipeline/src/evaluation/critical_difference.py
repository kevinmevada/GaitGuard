"""
Critical Difference diagrams + Wilcoxon/Holm tests (Ismail Fawaz / Dempster protocol).

Uses leave-one-participant-out jackknife AUROC vectors (paired across models).
CD diagram follows Demšar (2006) / Nemenyi post-hoc on Friedman ranks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

# Nemenyi critical q_α for α=0.05 (two-tailed), k algorithms (Demšar 2006 Table 5).
_NEMENYI_Q_005: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.269,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}


def holm_correction(p_values: list[float], *, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction (statsmodels multipletests)."""
    if not p_values:
        return np.array([]), np.array([], dtype=bool)
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return p_adj, reject


def nemenyi_critical_difference(num_algorithms: int, num_datasets: int, *, alpha: float = 0.05) -> float:
    """Critical difference for average-rank Nemenyi test."""
    if num_algorithms < 2 or num_datasets < 1:
        return float("nan")
    if alpha != 0.05:
        logger.warning("Nemenyi CD table is calibrated for α=0.05 only; using 0.05 q values")
    q = _NEMENYI_Q_005.get(num_algorithms)
    if q is None:
        q = 3.31 + 0.05 * (num_algorithms - 13)
    return float(q * np.sqrt(num_algorithms * (num_algorithms + 1) / (6.0 * num_datasets)))


def rank_matrix_from_scores(
    score_matrix: np.ndarray,
) -> np.ndarray:
    """
    Rank algorithms per dataset (column). Higher score → better → lower rank (1 = best).
    """
    k, n = score_matrix.shape
    ranks = np.zeros((k, n), dtype=float)
    for j in range(n):
        col = score_matrix[:, j]
        # rankdata: low value → rank 1; negate so high AUROC is rank 1
        ranks[:, j] = rankdata(-col, method="average")
    return ranks


def average_ranks(score_matrix: np.ndarray, model_names: list[str]) -> dict[str, float]:
    ranks = rank_matrix_from_scores(score_matrix)
    return {name: float(np.mean(ranks[i])) for i, name in enumerate(model_names)}


def friedman_test(score_matrix: np.ndarray) -> dict[str, float]:
    """Friedman χ² on jackknife-fold scores (rows = models, cols = folds)."""
    if score_matrix.shape[1] < 2 or score_matrix.shape[0] < 2:
        return {"chi2": float("nan"), "p_value": float("nan"), "n_datasets": int(score_matrix.shape[1])}
    stat, p = friedmanchisquare(*[score_matrix[i] for i in range(score_matrix.shape[0])])
    return {"chi2": float(stat), "p_value": float(p), "n_datasets": int(score_matrix.shape[1])}


def wilcoxon_vs_reference(
    aligned_aucs: dict[str, np.ndarray],
    reference: str,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Wilcoxon signed-rank: reference vs each baseline; Holm-corrected."""
    if reference not in aligned_aucs:
        raise ValueError(f"Reference model {reference!r} not in aligned AUROC dict")

    ref = aligned_aucs[reference]
    rows: list[dict[str, Any]] = []
    baselines = [m for m in sorted(aligned_aucs) if m != reference]
    raw_ps: list[float] = []

    for model in baselines:
        other = aligned_aucs[model]
        if len(ref) != len(other) or len(ref) < 3:
            rows.append(
                {
                    "reference": reference,
                    "baseline": model,
                    "n_folds": len(ref),
                    "wilcoxon_stat": float("nan"),
                    "p_value": float("nan"),
                    "mean_delta_auroc": float(np.mean(ref - other)) if len(ref) else float("nan"),
                    "significant_raw": False,
                }
            )
            raw_ps.append(1.0)
            continue
        try:
            stat, p = wilcoxon(ref, other, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            stat, p = float("nan"), 1.0
        rows.append(
            {
                "reference": reference,
                "baseline": model,
                "n_folds": len(ref),
                "wilcoxon_stat": float(stat) if np.isfinite(stat) else float("nan"),
                "p_value": float(p),
                "mean_delta_auroc": float(np.mean(ref - other)),
                "significant_raw": bool(p < alpha),
            }
        )
        raw_ps.append(float(p))

    p_adj, reject = holm_correction(raw_ps, alpha=alpha)
    for i, row in enumerate(rows):
        row["p_holm"] = float(p_adj[i]) if len(p_adj) else float("nan")
        row["significant_holm"] = bool(reject[i]) if len(reject) else False
    return pd.DataFrame(rows)


def pairwise_significance_matrix(
    aligned_aucs: dict[str, np.ndarray],
    model_names: list[str],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Symmetric matrix: True if Holm-corrected Wilcoxon p < α (all pairs)."""
    n = len(model_names)
    p_vals: list[float] = []
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = model_names[i], model_names[j]
            x, y = aligned_aucs[a], aligned_aucs[b]
            try:
                _, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p = 1.0
            pairs.append((a, b))
            p_vals.append(float(p))

    sig = np.zeros((n, n), dtype=bool)
    if p_vals:
        _, reject = holm_correction(p_vals, alpha=alpha)
        for k, (a, b) in enumerate(pairs):
            i, j = model_names.index(a), model_names.index(b)
            if reject[k]:
                sig[i, j] = sig[j, i] = True
    return pd.DataFrame(sig, index=model_names, columns=model_names)


def plot_critical_difference_diagram(
    avg_ranks: dict[str, float],
    cd: float,
    out_path: Path,
    *,
    title: str = "Critical Difference — jackknife AUROC ranks",
    display_names: dict[str, str] | None = None,
    alpha: float = 0.05,
) -> Path:
    """
    Demšar-style CD diagram: algorithms with average ranks within CD are connected.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(avg_ranks, key=avg_ranks.get)
    ranks = [avg_ranks[n] for n in names]
    labels = [(display_names or {}).get(n, n) for n in names]
    n = len(names)

    fig_h = max(3.5, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_pos = np.arange(n)
    ax.scatter(ranks, y_pos, s=60, c="#1f77b4", zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (lower = better AUROC)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if np.isfinite(cd) and cd > 0:
        ax.text(
            0.98,
            0.02,
            f"CD = {cd:.3f} (α={alpha}, Nemenyi)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        # Connect groups not significantly different (Nemenyi: |rank_i - rank_j| ≤ CD)
        bar_y = -0.8
        for i in range(n):
            group = [i]
            for j in range(i + 1, n):
                if ranks[j] - ranks[i] <= cd + 1e-9:
                    group.append(j)
                else:
                    break
            if len(group) > 1:
                y_hi = max(group)
                x_lo = ranks[i]
                x_hi = min(ranks[i] + cd, ranks[y_hi] + cd * 0.1)
                ax.plot([x_lo, x_hi], [bar_y, bar_y], color="#333333", linewidth=2.5, clip_on=False)
                for g in group:
                    ax.plot(
                        [ranks[g], ranks[g]],
                        [bar_y, y_pos[g]],
                        color="#aaaaaa",
                        linewidth=0.8,
                        linestyle=":",
                        zorder=1,
                    )
                bar_y -= 0.55

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Critical Difference diagram → {}", out_path)
    return out_path


def build_cd_summary(
    aligned_aucs: dict[str, np.ndarray],
    model_names: list[str],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    score_matrix = np.vstack([aligned_aucs[m] for m in model_names])
    friedman = friedman_test(score_matrix)
    avr = average_ranks(score_matrix, model_names)
    cd = nemenyi_critical_difference(len(model_names), score_matrix.shape[1], alpha=alpha)
    return {
        "friedman": friedman,
        "average_ranks": avr,
        "critical_difference": cd,
        "alpha": alpha,
        "n_models": len(model_names),
        "n_jackknife_folds": int(score_matrix.shape[1]),
    }
