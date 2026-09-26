"""End-to-end pipeline: data -> labels -> features -> baselines/models -> metrics -> figures.

    python src/download.py     # once (~470 MB from data.boston.gov)
    python train.py            # ~ a few minutes on a laptop

Outputs: results/results.json, results/*.csv, figures/*.png, models/hgb.joblib (gitignored).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
import evaluate  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402
import plots  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
MODELS = ROOT / "models"
SEED = 42
PARKING = "Parking Enforcement"
# HistGradientBoosting uses OpenMP. On Apple-silicon laptops, using every core
# (incl. efficiency cores) made fits ~5x slower in our tests; 6 threads is a safe default.
N_THREADS = 6
BUDGET = 0.10

HGB_GRID = [
    dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100),
    dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100),
    dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100),
]
HGB_MAX_ITER = 500
LR_GRID = [0.03, 0.3, 3.0]


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = round(time.time() - self.t0, 1)
        print(f"[{self.marks[name]:7.1f}s] {name}", flush=True)


def rate_table(df: pd.DataFrame, by, min_n: int = 1) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({"n": g.size(), "late_rate": g["late"].mean(),
                      "expected": g["expected_late"].mean()})
    return t[t["n"] >= min_n]


def main() -> None:
    T = Timer()
    for d in (FIG, RES, MODELS):
        d.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    cases, snapshot = data.load_legacy()
    agreement = data.label_agreement(cases)
    pop, funnel = data.study_population(cases)
    T.mark(f"data loaded: {len(cases):,} legacy cases, study population {len(pop):,}")

    X = F.build_features(cases, pop)
    y = X["late"].to_numpy()
    T.mark("features built")

    # --------------------------------------------------------------- splits
    # A model deployed on date D can only learn from cases whose outcome was
    # known on D, i.e. whose SLA deadline had passed. Apply that to training rows.
    is_sel_train = (X["split"] == "train") & (X["sla_target_dt"] < data.VAL_START)
    is_val = X["split"] == "val"
    is_final_train = X["split"].isin(["train", "val"]) & (X["sla_target_dt"] < data.TEST_START)
    is_test = X["split"] == "test"
    split_counts = {
        "selection_train (opened 2025-01..10, deadline < 2025-11-01)": int(is_sel_train.sum()),
        "validation (opened 2025-11..12)": int(is_val.sum()),
        "final_train (opened 2025-01..12, deadline < 2026-01-01)": int(is_final_train.sum()),
        "test (opened 2026-01..06)": int(is_test.sum()),
        "2025 cases excluded from final training because their deadline was after 2026-01-01":
            int((X["split"].isin(["train", "val"]) & ~is_final_train).sum()),
    }
    in_window = cases[(cases["open_dt"] >= data.STUDY_START) & (cases["open_dt"] < data.TEST_END)]
    long_sla = in_window[in_window["sla_hours"] > data.MAX_SLA_HOURS].groupby("type").agg(
        n=("sla_hours", "size"), median_sla_days=("sla_hours", lambda h: round(h.median() / 24, 1)))
    long_sla = long_sla.sort_values("n", ascending=False).head(10).to_dict(orient="index")
    audit = data.schema_audit(X)
    print(audit)
    Xs, ys = X[is_sel_train], y[is_sel_train]
    Xv, yv = X[is_val], y[is_val]
    Xf, yf = X[is_final_train], y[is_final_train]
    Xt, yt = X[is_test], y[is_test]
    print(split_counts)

    # ------------------------------------------------------- leakage checks
    F.check_no_leakage(F.FEATURES)
    # Univariate scan on the validation months: any single feature that almost
    # perfectly separates late from on-time would be a red flag for leakage.
    uni = {}
    for c in F.FEATURES:
        if c in F.CATEGORICAL:
            enc = pd.Series(ys, index=Xs.index).groupby(Xs[c]).mean()
            s = Xv[c].map(enc).fillna(ys.mean()).to_numpy()
        else:
            s = Xv[c].fillna(Xv[c].median()).to_numpy()
        a = roc_auc_score(yv, s)
        uni[c] = round(float(max(a, 1 - a)), 4)
    uni = dict(sorted(uni.items(), key=lambda kv: -kv[1]))
    flagged = [c for c, a in uni.items() if a > 0.95]
    T.mark(f"univariate leakage scan done; max single-feature AUC {max(uni.values()):.3f}; flagged={flagged}")

    # ---------------------------------------------------- model selection
    selection = {"logistic_regression": [], "hist_gradient_boosting": []}
    best_lr = None
    for C in LR_GRID:
        m = M.logistic_regression(C).fit(Xs, ys)
        p = m.predict_proba(Xv)[:, 1]
        row = {"C": C, "val_log_loss": log_loss(yv, p), "val_roc_auc": roc_auc_score(yv, p)}
        selection["logistic_regression"].append(row)
        if best_lr is None or row["val_log_loss"] < best_lr["val_log_loss"]:
            best_lr = row
    T.mark(f"LR selection: best C={best_lr['C']} (val AUC {best_lr['val_roc_auc']:.4f})")

    best_hgb = None
    for cfg in HGB_GRID:
        m = M.hist_gradient_boosting(max_iter=HGB_MAX_ITER, **cfg).fit(Xs, ys)
        Zv = m[:-1].transform(Xv)
        for it, pv in enumerate(m[-1].staged_predict_proba(Zv), start=1):
            if it % 25:
                continue
            row = {**cfg, "n_iter": it, "val_log_loss": log_loss(yv, pv[:, 1]),
                   "val_roc_auc": roc_auc_score(yv, pv[:, 1])}
            selection["hist_gradient_boosting"].append(row)
            if best_hgb is None or row["val_log_loss"] < best_hgb["val_log_loss"]:
                best_hgb = row
        T.mark(f"HGB config {cfg} scored")
    print("best HGB:", best_hgb)

    # ------------------------------------------------------ final models
    hgb_params = {k: best_hgb[k] for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf")}
    fitted = {
        "Majority class (baseline)": M.MajorityBaseline().fit(Xf, yf),
        "Per-type late rate (baseline)": M.TypeRateBaseline().fit(Xf, yf),
        "Logistic regression": M.logistic_regression(best_lr["C"]).fit(Xf, yf),
        "HistGradientBoosting": M.hist_gradient_boosting(max_iter=best_hgb["n_iter"],
                                                         **hgb_params).fit(Xf, yf),
    }
    T.mark("final models fitted on all of 2025")
    joblib.dump(fitted["HistGradientBoosting"], MODELS / "hgb.joblib")

    preds = {name: m.predict_proba(Xt)[:, 1] for name, m in fitted.items()}
    test_metrics = {name: evaluate.all_metrics(yt, p) for name, p in preds.items()}

    not_parking = (Xt["type"] != PARKING).to_numpy()
    test_metrics_ex_parking = {name: evaluate.all_metrics(yt[not_parking], p[not_parking])
                               for name, p in preds.items()}
    by_month = {}
    months = Xt["open_dt"].dt.to_period("M").astype(str).to_numpy()
    for mo in sorted(set(months)):
        mm = months == mo
        by_month[mo] = {"n": int(mm.sum()), "late_rate": float(yt[mm].mean()),
                        **{name: round(float(roc_auc_score(yt[mm], p[mm])), 4)
                           for name, p in preds.items() if not name.startswith("Majority")}}
    T.mark("test metrics computed")
    for name, mt in test_metrics.items():
        print(f"  {name:32s} AUC {mt['roc_auc']:.4f}  PR-AUC {mt['pr_auc']:.4f}  "
              f"P@10% {mt['precision_at_10pct']:.4f}  Brier {mt['brier']:.4f}")

    # ------------------------------------------------ leakage demonstration
    # What would happen if a post-submission column slipped in? (`case_status`,
    # Open/Closed at export time). This model is NOT used anywhere else.
    Xf_leak = Xf.assign(source=Xf["source"] + "|" + Xf["case_status"])
    Xt_leak = Xt.assign(source=Xt["source"] + "|" + Xt["case_status"])
    leak_model = M.hist_gradient_boosting(max_iter=best_hgb["n_iter"], **hgb_params).fit(Xf_leak, yf)
    leak_auc = float(roc_auc_score(yt, leak_model.predict_proba(Xt_leak)[:, 1]))
    T.mark(f"leakage demo: AUC with case_status leaked in = {leak_auc:.4f}")

    # ------------------------------------------------------------ ablation
    # Do the past-only workload features help at all? Refit HGB without them.
    no_backlog = [c for c in F.NUMERIC if c not in F.BACKLOG]
    abl = M.hist_gradient_boosting(max_iter=best_hgb["n_iter"], numeric=no_backlog,
                                   **hgb_params).fit(Xf, yf)
    p_abl = abl.predict_proba(Xt)[:, 1]
    ablation = {
        "HGB without workload/backlog features": evaluate.all_metrics(yt, p_abl),
        "HGB without workload/backlog features, excluding Parking Enforcement":
            evaluate.all_metrics(yt[not_parking], p_abl[not_parking]),
    }
    T.mark(f"ablation: HGB without backlog features AUC {ablation['HGB without workload/backlog features']['roc_auc']:.4f}")

    # Winter drift: 2025 had a mild winter, early 2026 did not.
    snow = {}
    for ty in ["Request for Snow Plowing", "Unshoveled Sidewalk"]:
        for yr in (2025, 2026):
            mm = (X["type"] == ty) & (X["open_dt"].dt.year == yr) & (X["open_dt"].dt.month <= 2)
            snow[f"{ty} | Jan-Feb {yr}"] = {"n": int(mm.sum()), "late_rate": round(float(y[mm].mean()), 4)}

    # Parking Enforcement: how much of its "late" is simply "never closed"?
    pe = (X["type"] == PARKING).to_numpy()
    still_open = X["closed_dt"].isna().to_numpy()
    open_stats = {
        "share_of_late_cases_still_open_at_export": round(float(still_open[y == 1].mean()), 4),
        "parking_enforcement": {
            "n": int(pe.sum()), "share_of_study_population": round(float(pe.mean()), 4),
            "late_rate": round(float(y[pe].mean()), 4),
            "share_still_open_at_export": round(float(still_open[pe].mean()), 4),
            "share_of_all_late_cases": round(float(pe[y == 1].mean()), 4),
        },
    }

    # --------------------------------------------- permutation importance
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Xt), size=min(40_000, len(Xt)), replace=False)
    pi = permutation_importance(fitted["HistGradientBoosting"], Xt.iloc[sub][F.FEATURES], yt[sub],
                                scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std},
                       index=F.FEATURES).sort_values("mean", ascending=False)
    T.mark("permutation importance done")

    # ------------------------------------------------ descriptive tables
    # Expected late rate of each case = its type's late rate over the whole study
    # population; group averages of it give "what the type mix alone predicts".
    desc = X[["type", "subject", "neighborhood", "dayofweek", "hour", "open_dt", "split"]].copy()
    desc["late"] = y
    desc["expected_late"] = desc.groupby("type")["late"].transform("mean")
    overall = float(y.mean())
    by_type = rate_table(desc, "type").sort_values("n", ascending=False)
    by_hood = rate_table(desc, "neighborhood", min_n=500)
    by_hood = by_hood[by_hood.index != "NA"]
    by_hood = by_hood.assign(excess=by_hood["late_rate"] - by_hood["expected"]).sort_values(
        "excess", ascending=False)
    by_dept = rate_table(desc, "subject").sort_values("late_rate", ascending=False)
    by_dow = rate_table(desc, "dayofweek")
    by_hour = rate_table(desc, "hour")
    desc["month"] = desc["open_dt"].dt.to_period("M")
    by_month_desc = desc.groupby("month").agg(n=("late", "size"), late_rate=("late", "mean"),
                                              split=("split", "first"))
    for name, t in [("late_rate_by_type", by_type), ("late_rate_by_neighborhood", by_hood),
                    ("late_rate_by_department", by_dept)]:
        t.round(4).to_csv(RES / f"{name}.csv")

    # ------------------------------------------------------------ figures
    src = "Data: Analyze Boston, 311 Service Requests (legacy system), Jan 2025 - Jun 2026."
    top20 = by_type.head(20).sort_values("late_rate", ascending=False)
    plots.hbar_rates(top20, overall, FIG / "late_rate_by_type.png", src,
                     "Late rate by request type (20 most common types)")
    plots.hbar_rates(by_dept[by_dept["n"] >= 100], overall, FIG / "late_rate_by_department.png",
                     src, "Late rate by department (311 subject)")
    plots.late_rate_observed_vs_expected(
        by_hood, FIG / "late_rate_by_neighborhood.png", src,
        "Late rate by neighborhood: observed vs. expected from request mix", "neighborhood")
    plots.late_rate_by_time(by_dow, by_hour, FIG / "late_rate_by_time.png", src)
    plots.monthly(by_month_desc, FIG / "monthly_late_rate.png",
                  src + " Final models are refit on all of 2025 (train + validation) "
                  "before scoring the test months.")
    order = ["HistGradientBoosting", "Logistic regression", "Per-type late rate (baseline)",
             "Majority class (baseline)"]
    plots.curves(yt, {k: preds[k] for k in order}, FIG / "test_curves.png", src)
    plots.calibration(yt, {k: preds[k] for k in order}, FIG / "calibration.png", src)
    plots.permutation(imp, FIG / "permutation_importance.png",
                      "Computed on a random 40,000-request sample of the Jan-Jun 2026 test set.")
    T.mark("figures written")

    # ------------------------------------------------------------ results
    results = {
        "question": "Will a Boston 311 request miss its official SLA deadline? "
                    "(predicted at submission time)",
        "data": {
            "source": "https://data.boston.gov/dataset/311-service-requests",
            "license": "Open Data Commons Public Domain Dedication and License (PDDL)",
            "export_snapshot": str(snapshot),
            "legacy_cases_loaded_since_2024_09_01": int(len(cases)),
            "funnel": funnel,
            "label_agreement_with_portal_on_time": agreement,
            "new_system_file_not_modelled": data.new_system_summary(),
        },
        "splits": split_counts,
        "late_rate_by_split": {"selection_train": float(ys.mean()), "validation": float(yv.mean()),
                               "final_train": float(yf.mean()), "test": float(yt.mean())},
        "schema_audit": audit,
        "largest_long_sla_types_excluded": long_sla,
        "features": {"categorical": F.CATEGORICAL, "numeric": F.NUMERIC,
                     "excluded_as_leakage": F.FORBIDDEN},
        "leakage_checks": {
            "univariate_val_auc_by_feature": uni,
            "features_with_univariate_auc_above_0.95": flagged,
            "demo_test_auc_if_case_status_leaked": leak_auc,
        },
        "model_selection": {"criterion": "validation log loss (Nov-Dec 2025)",
                            "best_logistic_regression": best_lr,
                            "best_hist_gradient_boosting": best_hgb,
                            "all": selection},
        "test_metrics": test_metrics,
        "test_metrics_excluding_parking_enforcement": test_metrics_ex_parking,
        "test_roc_auc_by_month": by_month,
        "ablation": ablation,
        "winter_drift_snow_types": snow,
        "open_case_stats": open_stats,
        "permutation_importance_test_roc_auc_drop": imp.round(5).to_dict(orient="index"),
        "overall_late_rate_study_population": overall,
        "top_types_by_late_rate_min_1000": by_type[by_type["n"] >= 1000]
            .sort_values("late_rate", ascending=False).head(10)[["n", "late_rate"]]
            .round(4).to_dict(orient="index"),
        "neighborhoods_observed_vs_expected": by_hood[["n", "late_rate", "expected", "excess"]]
            .round(4).to_dict(orient="index"),
        "departments": by_dept[["n", "late_rate"]].round(4).to_dict(orient="index"),
        "late_rate_by_dayofweek_mon0": by_dow["late_rate"].round(4).to_dict(),
        "late_rate_by_hour": by_hour["late_rate"].round(4).to_dict(),
    }
    T.mark("done")
    results["runtime_seconds"] = T.marks
    with open(RES / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"wrote {RES / 'results.json'}")


if __name__ == "__main__":
    with threadpool_limits(limits=N_THREADS, user_api="openmp"):
        main()
