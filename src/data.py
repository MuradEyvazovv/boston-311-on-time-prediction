"""Load, clean and label the Boston 311 legacy-system case files.

Target definition (official SLA, not an invented threshold)
-----------------------------------------------------------
Every legacy case carries `sla_target_dt`, the deadline the City's CRM sets from
the case type's Service Level Agreement when the case is opened (it follows a
business-day calendar: a 1-business-day SLA opened on a Friday gets ~72 h). The
portal's own `on_time` column is ONTIME/OVERDUE relative to this deadline.

We recompute the label from timestamps instead of trusting `on_time` directly,
because `on_time` is evaluated "as of the export": an open case whose deadline
is still in the future is shown as ONTIME even though we do not know yet how it
will end. That is right-censoring, and treating it as "on time" would bias the
most recent months toward on-time.

    late = 1   if closed_dt >  sla_target_dt                     (closed late)
    late = 1   if still open and sla_target_dt <  snapshot       (deadline already missed)
    late = 0   if closed_dt <= sla_target_dt                     (closed on time)
    censored   if still open and sla_target_dt >= snapshot       (outcome unknown -> excluded)

Note: a still-open case whose deadline has passed is *known* to be late - no
matter when (or whether) it is eventually closed - so it is labelled, not dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "legacy_cases.parquet"
SNAPSHOT_FILE = PROCESSED.with_suffix(".snapshot.txt")

LEGACY_FILES = ["legacy_2024.csv", "legacy_2025.csv", "legacy_2026.csv"]
NEW_SYSTEM_FILE = "new_system.csv"

# Study window. 2024 rows are loaded ONLY as look-back history for the
# 30-day backlog features of early-2025 cases; they are never trained/evaluated on.
HISTORY_START = pd.Timestamp("2024-09-01")
STUDY_START = pd.Timestamp("2025-01-01")
VAL_START = pd.Timestamp("2025-11-01")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-07-01")  # exclusive; July+ is mid-migration to the new system

# Cases whose SLA is longer than 90 days (tree planting: 1 year, some ISD
# complaints: 720 days, ...) cannot have a known outcome for 2026 cases yet, so
# keeping only their *resolved* cases would bias the test set toward fast
# closures. We restrict the question to SLAs of at most 90 days.
MAX_SLA_HOURS = 90 * 24 + 1

USECOLS = [
    "case_enquiry_id", "open_dt", "sla_target_dt", "closed_dt", "on_time", "case_status",
    "closure_reason", "case_title", "subject", "reason", "type", "queue", "department",
    "submitted_photo", "closed_photo", "pwd_district", "city_council_district",
    "police_district", "neighborhood", "ward", "location_zipcode", "latitude", "longitude",
    "source",
]

NEIGHBORHOOD_MERGE = {
    # the geocoder emits both the combined planning-district name and the parts
    "Allston": "Allston / Brighton",
    "Brighton": "Allston / Brighton",
    "South Boston": "South Boston / South Boston Waterfront",
    "Mattapan": "Greater Mattapan",
    "Boston": "Boston (unspecified)",
}


def _parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="ISO8601", errors="coerce")


def _norm_ward(s: pd.Series) -> pd.Series:
    digits = s.str.extract(r"(\d+)", expand=False)
    return digits.dropna().astype(int).astype(str).str.zfill(2).reindex(s.index)


def _norm_pwd(s: pd.Series) -> pd.Series:
    s = s.str.strip().str.upper()
    return s.where(~s.str.fullmatch(r"\d", na=False), "0" + s)


def _norm_zip(s: pd.Series) -> pd.Series:
    d = s.str.extract(r"(\d{4,5})", expand=False)
    return d.str.zfill(5)


def load_legacy(use_cache: bool = True) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (cases, snapshot). One row per legacy case opened since HISTORY_START."""
    if use_cache and PROCESSED.exists() and SNAPSHOT_FILE.exists():
        return pd.read_parquet(PROCESSED), pd.Timestamp(SNAPSHOT_FILE.read_text().strip())

    frames = []
    for fname in LEGACY_FILES:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python src/download.py` first")
        frames.append(pd.read_csv(path, dtype=str, usecols=USECOLS, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("case_enquiry_id", keep="last")

    for c in ["open_dt", "sla_target_dt", "closed_dt"]:
        df[c] = _parse_dt(df[c])
    df = df[df["open_dt"].notna()]

    # The export snapshot = the latest timestamp anywhere in the files.
    snapshot = max(df["open_dt"].max(), df["closed_dt"].max())
    df = df[df["open_dt"] >= HISTORY_START].copy()

    for c in ["type", "reason", "subject", "source", "neighborhood", "police_district",
              "city_council_district", "queue", "department", "case_status", "on_time"]:
        df[c] = df[c].str.strip()
    df["neighborhood"] = df["neighborhood"].replace(NEIGHBORHOOD_MERGE)
    df["ward"] = _norm_ward(df["ward"].fillna(""))
    df["pwd_district"] = _norm_pwd(df["pwd_district"])
    df["zipcode"] = _norm_zip(df["location_zipcode"].fillna(""))
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.drop(columns=["location_zipcode"])

    # ---- label -------------------------------------------------------------
    df["sla_hours"] = (df["sla_target_dt"] - df["open_dt"]).dt.total_seconds() / 3600
    has_sla = df["sla_target_dt"].notna()
    closed = df["closed_dt"].notna()
    late = np.where(closed, df["closed_dt"] > df["sla_target_dt"], df["sla_target_dt"] < snapshot)
    df["late"] = pd.Series(late, index=df.index).astype("float").where(has_sla)
    df["censored"] = has_sla & ~closed & (df["sla_target_dt"] >= snapshot)
    df.loc[df["censored"], "late"] = np.nan

    df = df.sort_values("open_dt", kind="stable").reset_index(drop=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED, index=False)
    SNAPSHOT_FILE.write_text(str(snapshot))
    return df, snapshot


def label_agreement(df: pd.DataFrame) -> dict:
    """How often does our recomputed label agree with the portal's `on_time` column?"""
    m = df["late"].notna()
    official_late = df.loc[m, "on_time"].eq("OVERDUE")
    return {
        "n_compared": int(m.sum()),
        "agreement": float((official_late == df.loc[m, "late"].astype(bool)).mean()),
        "censored_cases_marked_ONTIME_by_portal": int(
            (df["censored"] & df["on_time"].eq("ONTIME")).sum()),
    }


def study_population(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the population filters, returning the kept rows and a filter funnel."""
    funnel = []
    d = df[(df["open_dt"] >= STUDY_START) & (df["open_dt"] < TEST_END)]
    funnel.append({"step": f"legacy cases opened {STUDY_START.date()} .. {TEST_END.date()} (excl.)",
                   "n": int(len(d))})
    d = d[d["sla_target_dt"].notna()]
    funnel.append({"step": "has an SLA deadline (drops types with no SLA, e.g. Needle Pickup)",
                   "n": int(len(d))})
    d = d[d["sla_hours"] <= MAX_SLA_HOURS]
    funnel.append({"step": "SLA <= 90 days", "n": int(len(d))})
    n_cens = int(d["censored"].sum())
    d = d[~d["censored"]]
    funnel.append({"step": f"outcome known at export (dropped {n_cens} censored open cases)",
                   "n": int(len(d))})
    d = d.copy()
    d["late"] = d["late"].astype(int)
    d["split"] = np.select(
        [d["open_dt"] < VAL_START, d["open_dt"] < TEST_START], ["train", "val"], "test")
    return d, funnel


def schema_audit(pop: pd.DataFrame) -> dict:
    """Evidence for the feature/leakage decisions (see features.FORBIDDEN).

    'purity within type' = share of a type's cases that carry the type's most common
    value, weighted by type size. 100% means the column is fixed once the type is known
    (i.e. set at intake); less than 100% means it varies, e.g. because cases are re-routed.
    """
    n = pop.groupby("type").size()

    def purity(col):
        top = pop.groupby("type")[col].agg(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
        return round(float((top * n).sum() / n.sum()), 4)

    return {
        "n_request_types_in_study_population": int(pop["type"].nunique()),
        "purity_within_type": {c: purity(c) for c in ["subject", "reason", "department", "queue"]},
        "case_title_equals_type_share": round(float((pop["case_title"] == pop["type"]).mean()), 4),
        "submitted_photo_non_empty_share": round(float(pop["submitted_photo"].notna().mean()), 4),
    }


def new_system_summary() -> dict:
    """Descriptive stats for the NEW SYSTEM file (not modelled - see README)."""
    path = RAW_DIR / NEW_SYSTEM_FILE
    if not path.exists():
        return {}
    n = pd.read_csv(path, dtype=str, low_memory=False)
    o = _parse_dt(n["open_date"])
    c = _parse_dt(n["close_date"])
    t = _parse_dt(n["target_close_date"])
    snap = max(o.max(), c.max())
    closed = c.notna()
    known = closed | (t < snap)
    late = np.where(closed, c > t, t < snap)
    return {
        "n_cases": int(len(n)),
        "open_date_min": str(o.min()),
        "open_date_max": str(o.max()),
        "n_service_names": int(n["service_name"].nunique()),
        "share_opened_2026_05_or_later": float((o >= pd.Timestamp("2026-05-01", tz="UTC")).mean()),
        "late_rate_known_outcomes": float(late[known].mean()),
        "n_known_outcomes": int(known.sum()),
    }
