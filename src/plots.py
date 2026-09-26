"""All figures (matplotlib only). One quiet style: light surface, hairline grid,
single blue for one-series charts, fixed model colors everywhere else."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from evaluate import reliability, top_k_mask  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WASH = "#f0efec"

MODEL_STYLE = {
    "HistGradientBoosting": dict(color=BLUE, lw=2),
    "Logistic regression": dict(color=ORANGE, lw=2),
    "Per-type late rate (baseline)": dict(color=AQUA, lw=2),
    "Majority class (baseline)": dict(color=MUTED, lw=1.5),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
})


def _pct_axis(ax, axis="x"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _save(fig, path: Path, note: str | None = None):
    if note:  # placed just below the lowest drawn element (e.g. the x-axis label)
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())  # inches
        y = (bb.y0 - 0.12) / fig.get_figheight()
        fig.text(0.01, y, note, fontsize=8, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hbar(ax, labels, values, color=BLUE, fmt="{:.0%}"):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=color, zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    xmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt.format(v), va="center", fontsize=8.5, color=INK2)
    return y


def hbar_rates(tbl: pd.DataFrame, overall: float, path: Path, note: str, title: str):
    """tbl: index=group, columns n, late_rate (already filtered/sorted)."""
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(tbl) + 1.5))
    labels = [f"{t}  (n={n:,})" for t, n in zip(tbl.index, tbl["n"])]
    _hbar(ax, labels, tbl["late_rate"].to_numpy())
    ax.axvline(overall, color=INK2, lw=1, zorder=3)
    _pct_axis(ax)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(f"Share of requests that missed their SLA deadline "
                  f"(vertical line = all requests, {overall:.0%})")
    ax.set_title(title)
    _save(fig, path, note)


def late_rate_observed_vs_expected(tbl: pd.DataFrame, path: Path, note: str, title: str,
                                   what: str):
    """tbl: index=group, columns n, late_rate, expected (from type mix)."""
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(tbl) + 1.6))
    y = np.arange(len(tbl))
    ax.barh(y, tbl["late_rate"], height=0.62, color=BLUE, zorder=2, label="Observed late rate")
    ax.scatter(tbl["expected"], y, s=46, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4,
               label=f"Expected from its request-type mix")
    ax.set_yticks(y, [f"{g}  (n={n:,})" for g, n in zip(tbl.index, tbl["n"])])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for yi, (o, e) in enumerate(zip(tbl["late_rate"], tbl["expected"])):
        ax.text(max(o, e) + 0.012, yi, f"{o:.0%} vs {e:.0%}", va="center", fontsize=8, color=INK2)
    _pct_axis(ax)
    ax.set_xlim(0, max(tbl["late_rate"].max(), tbl["expected"].max()) * 1.25)
    ax.set_xlabel("Share of requests that missed their SLA deadline")
    ax.set_title(title, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9,
              handletextpad=0.4, columnspacing=1.5, borderaxespad=0.2)
    _save(fig, path, f"Sorted by observed minus expected: a bar extending past its dot means the {what} "
          f"is late more often than its mix of request types explains.\n{note}")


def late_rate_by_time(dow: pd.DataFrame, hour: pd.DataFrame, path: Path, note: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.8]})
    ax = axes[0]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.bar(np.arange(7), dow["late_rate"], width=0.62, color=BLUE, zorder=2)
    ax.set_xticks(np.arange(7), names)
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(dow["late_rate"]):
        ax.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
    _pct_axis(ax, "y")
    ax.set_ylim(0, dow["late_rate"].max() * 1.2)
    ax.set_title("By day of submission")
    ax.set_ylabel("Late rate")
    ax = axes[1]
    ax.plot(hour.index, hour["late_rate"], color=BLUE, lw=2, marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of submission (local time)")
    _pct_axis(ax, "y")
    ax.set_ylim(0, hour["late_rate"].max() * 1.15)
    ax.set_title("By hour of submission")
    fig.suptitle("When are requests most at risk of missing their SLA?", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, path, note)


def monthly(tbl: pd.DataFrame, path: Path, note: str):
    """tbl: index=Period[M], columns n, late_rate, split."""
    x = np.arange(len(tbl))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for ax in axes:
        for split, lab in [("train", "train (model selection)"), ("val", "validation"),
                           ("test", "test")]:
            idx = np.where(tbl["split"].to_numpy() == split)[0]
            if len(idx) and split != "train":
                ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=WASH, zorder=0,
                           alpha=1.0 if split == "test" else 0.55)
    ax = axes[0]
    ax.plot(x, tbl["late_rate"], color=BLUE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    _pct_axis(ax, "y")
    ax.set_ylim(0, tbl["late_rate"].max() * 1.2)
    ax.set_ylabel("Late rate")
    ax.set_title("Monthly late rate and volume (study population)")
    top = tbl["late_rate"].max() * 1.13
    for split, lab in [("train", "TRAIN  Jan-Oct 2025"), ("val", "VAL  Nov-Dec"),
                       ("test", "TEST  Jan-Jun 2026")]:
        idx = np.where(tbl["split"].to_numpy() == split)[0]
        ax.text(idx.mean(), top, lab, ha="center", fontsize=8.5, color=INK2)
    ax = axes[1]
    ax.bar(x, tbl["n"], width=0.62, color=BLUE, zorder=2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Requests")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xticks(x, [p.strftime("%b\n%Y") if p.month == 1 or i == 0 else p.strftime("%b")
                      for i, p in enumerate(tbl.index)], fontsize=8.5)
    _save(fig, path, note)


def curves(y: np.ndarray, preds: dict, path: Path, note: str):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.3})
    base = y.mean()
    budgets = np.linspace(0.01, 0.5, 50)
    for name, p in preds.items():
        st = MODEL_STYLE[name]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=name, **st)
        if name.startswith("Majority"):
            axes[1].plot([0, 1], [base, base], label=name, **st)
        else:
            prec, rec, _ = precision_recall_curve(y, p)
            axes[1].plot(rec, prec, label=name, **st)
        pb = [y[top_k_mask(p, b)].mean() for b in budgets]
        axes[2].plot(budgets, pb, label=name, **st)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve",
                xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(xlabel="Recall (share of late cases caught)", ylabel="Precision",
                title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.01))
    axes[2].set(xlabel="Alert budget (share of requests flagged, highest risk first)",
                ylabel="Precision of flagged requests", title="Precision at an alert budget",
                ylim=(0, 1.01))
    axes[2].axvline(0.10, color=INK2, lw=1)
    axes[2].text(0.105, 0.04, "10% budget", fontsize=8.5, color=INK2)
    _pct_axis(axes[2]), _pct_axis(axes[2], "y")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Test set: {len(y):,} requests opened Jan-Jun 2026 ({base:.1%} late)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.03)
    _save(fig, path, note)


def calibration(y: np.ndarray, preds: dict, path: Path, note: str):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, zorder=1, label="Perfect calibration")
    for name, p in preds.items():
        if name.startswith("Majority"):
            continue
        mp, fp = reliability(y, p, n_bins=10)
        ax.plot(mp, fp, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=name, **MODEL_STYLE[name])
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed late rate in bin",
           title="Calibration on the test set (10 equal-count bins)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax = axes[1]
    ax.hist(preds["HistGradientBoosting"], bins=40, range=(0, 1), color=BLUE, zorder=2,
            rwidth=0.85)
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="Predicted probability of missing the SLA", ylabel="Requests")
    ax.set_title("HistGradientBoosting: distribution of predictions", fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    _save(fig, path, note)


def permutation(imp: pd.DataFrame, path: Path, note: str):
    """imp: index=feature, columns mean, std (drop in test ROC-AUC)."""
    imp = imp.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(imp) + 1.4))
    y = np.arange(len(imp))
    ax.barh(y, imp["mean"], xerr=imp["std"], height=0.62, color=BLUE, zorder=2,
            error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.set_yticks(y, imp.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for yi, v in zip(y, imp["mean"]):
        ax.text(max(v, 0) + imp["mean"].max() * 0.03 + imp["std"].max(), yi, f"{v:.4f}",
                va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled (mean of 5 repeats)")
    ax.set_title("Permutation importance - HistGradientBoosting")
    _save(fig, path, note)
