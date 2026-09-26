"""Submission-time features, and the leakage guard.

Everything here must be knowable at the moment a request is submitted:

* what was reported: type -> reason -> subject (the 311 taxonomy; `subject` is
  the responsible department and is 100% determined by `type`, so it is fixed at
  intake), and the SLA length the CRM attaches to that type;
* where: neighborhood, ZIP, public-works / council / police district, ward, lat/lon;
* how: the intake channel (`source`);
* when: hour, day of week, month, weekend, US federal holiday;
* workload "as of midnight before submission", computed ONLY from earlier cases:
  recent volumes, the number of recent cases still unresolved, and the late rate
  of recent cases whose deadline has already passed (their outcome is known then).

Columns that are only filled in (or can change) after submission are forbidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CATEGORICAL = [
    "type", "reason", "subject", "source", "neighborhood", "zipcode",
    "pwd_district", "city_council_district", "police_district", "ward",
]
TIME_NUMERIC = ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]
BACKLOG = [
    "type_vol_7d", "type_backlog_30d", "type_late_rate_30d", "type_deadlines_30d",
    "subject_vol_7d", "subject_backlog_30d", "subject_late_rate_30d",
    "city_vol_1d", "city_backlog_30d",
]
OTHER_NUMERIC = ["sla_hours", "latitude", "longitude"]
NUMERIC = OTHER_NUMERIC + TIME_NUMERIC + BACKLOG
FEATURES = CATEGORICAL + NUMERIC

# Columns that describe what happened AFTER submission (or may be edited later).
FORBIDDEN = {
    "closed_dt": "close timestamp - defines the target",
    "case_status": "Open/Closed at export time - future information",
    "closure_reason": "written when the case is closed",
    "on_time": "the portal's ONTIME/OVERDUE flag - it IS the target",
    "closed_photo": "uploaded by the crew at closure",
    "queue": "current work queue; cases can be re-routed after intake (varies within a type)",
    "department": "current queue's department code; varies within a type, unlike `subject`",
    "case_title": "free text that staff can edit; mostly identical to `type` anyway",
    "late": "target",
    "censored": "derived from the close date",
}

# How many days of look-back the windowed features need.
LOOKBACK_DAYS = 30
SMOOTHING = 10.0  # pseudo-count pulling small-sample recent late rates to the citywide rate


def check_no_leakage(feature_names: list[str]) -> None:
    bad = [f for f in feature_names if f in FORBIDDEN]
    if bad:
        raise ValueError(f"Leaky columns in feature set: {bad}")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["open_dt"]
    out = df.copy()
    out["hour"] = t.dt.hour
    out["dayofweek"] = t.dt.dayofweek
    out["month"] = t.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    hol = USFederalHolidayCalendar().holidays(start="2024-01-01", end="2026-12-31")
    out["is_holiday"] = t.dt.normalize().isin(hol).astype(int)
    return out


def _day_index(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """Whole days since origin (floor); NaT -> very large."""
    d = (ts.dt.normalize() - origin).dt.days
    return d.fillna(10**6).astype(np.int64).to_numpy()


def backlog_features(history: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Workload features for each row of `query`, as of 00:00 of its submission day.

    `history` = every legacy case (any SLA, any type, including cases outside the
    study population), because they all compete for the same crews. For a query
    case submitted on day D we only use:
      * cases opened on days D-7 .. D-1 (volumes),
      * cases opened on days D-30 .. D-1 that were not closed before 00:00 of day D
        (backlog - uses close dates only if they are < midnight of day D),
      * cases whose SLA deadline fell on days D-30 .. D-1 (their late/on-time status
        is fully determined by then: late <=> not closed by the deadline).
    Nothing dated on or after day D is touched, so there is no look-ahead.
    """
    origin = history["open_dt"].min().normalize()
    h_open = _day_index(history["open_dt"], origin)
    h_close = _day_index(history["closed_dt"], origin)
    h_dead = _day_index(history["sla_target_dt"], origin)
    has_dead = history["sla_target_dt"].notna().to_numpy()
    # lateness relative to deadline; for deadlines before the query day this is observable
    h_late = (history["closed_dt"].isna() | (history["closed_dt"] > history["sla_target_dt"])).to_numpy()
    q_day = _day_index(query["open_dt"], origin)
    n_days = int(max(h_open.max(), q_day.max())) + LOOKBACK_DAYS + 3

    out = pd.DataFrame(index=query.index)

    def by_key(key_h: np.ndarray, key_q: np.ndarray, n_keys: int, prefix: str,
               vol_days: int, want_rate: bool):
        # daily openings per key
        opened = np.zeros((n_keys, n_days + 1))
        np.add.at(opened, (key_h, h_open), 1)
        c_open = np.concatenate([np.zeros((n_keys, 1)), np.cumsum(opened, axis=1)], axis=1)
        # c_open[k, D] = openings on days < D
        vol = c_open[key_q, q_day] - c_open[key_q, np.maximum(q_day - vol_days, 0)]
        out[f"{prefix}_vol_{vol_days}d"] = vol

        # backlog snapshot at 00:00 of day D: opened on [D-30, D-1], closed on/after day D.
        # A case contributes to days D in [open+1, min(close, open+30)].
        start = h_open + 1
        end = np.minimum(h_close, h_open + LOOKBACK_DAYS)
        ok = end >= start
        diff = np.zeros((n_keys, n_days + 2))
        np.add.at(diff, (key_h[ok], start[ok]), 1)
        np.add.at(diff, (key_h[ok], np.minimum(end[ok] + 1, n_days + 1)), -1)
        backlog = np.cumsum(diff, axis=1)
        out[f"{prefix}_backlog_30d"] = backlog[key_q, q_day]

        if want_rate:
            dl = np.zeros((n_keys, n_days + 1))
            lt = np.zeros((n_keys, n_days + 1))
            m = has_dead & (h_dead < n_days)
            np.add.at(dl, (key_h[m], h_dead[m]), 1)
            np.add.at(lt, (key_h[m], h_dead[m]), h_late[m].astype(float))
            c_dl = np.concatenate([np.zeros((n_keys, 1)), np.cumsum(dl, axis=1)], axis=1)
            c_lt = np.concatenate([np.zeros((n_keys, 1)), np.cumsum(lt, axis=1)], axis=1)
            lo = np.maximum(q_day - LOOKBACK_DAYS, 0)
            n = c_dl[key_q, q_day] - c_dl[key_q, lo]
            k = c_lt[key_q, q_day] - c_lt[key_q, lo]
            return n, k
        return None

    # citywide (single key) first: gives the prior for the smoothed rates
    zeros_h = np.zeros(len(history), dtype=np.int64)
    zeros_q = np.zeros(len(query), dtype=np.int64)
    n_city, k_city = by_key(zeros_h, zeros_q, 1, "city", 1, True)
    city_rate = np.where(n_city > 0, k_city / np.maximum(n_city, 1), np.nan)
    city_rate = np.where(np.isnan(city_rate), np.nanmean(city_rate), city_rate)

    for col, prefix in [("type", "type"), ("subject", "subject")]:
        codes, uniques = pd.factorize(pd.concat([history[col], query[col]]).fillna("NA"))
        kh, kq = codes[: len(history)], codes[len(history):]
        n, k = by_key(kh, kq, len(uniques), prefix, 7, True)
        out[f"{prefix}_late_rate_30d"] = (k + SMOOTHING * city_rate) / (n + SMOOTHING)
        if prefix == "type":
            out["type_deadlines_30d"] = n
    return out


def build_features(history: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    X = add_time_features(pop)
    X = X.join(backlog_features(history, pop))
    for c in CATEGORICAL:
        X[c] = X[c].fillna("NA").astype(str)
    check_no_leakage(FEATURES)
    return X
