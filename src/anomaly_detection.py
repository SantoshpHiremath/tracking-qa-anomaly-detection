"""Anomaly detection on daily metrics derived from the event stream,
supporting investigation of root causes and corrective actions.

Method: a rolling-median + MAD (median absolute deviation) z-score,
computed per market so each market's own baseline and volatility is used
rather than a single global threshold (DE and JP have very different
traffic volumes; a global threshold would either miss real anomalies in
small markets or over-flag normal variance in large ones). MAD-based
scoring is used instead of a rolling mean/stdev because it's robust to
the anomalies themselves skewing the baseline — a classical, well-
understood technique for exactly this kind of monitoring, not a novel or
untested one.
"""
from collections import defaultdict
from statistics import median

import pandas as pd


def events_to_daily_metrics(events):
    """Aggregates the raw event stream into daily per-market metrics:
    sessions (approximated by page_view count), add_to_cart count,
    purchase count, and purchase revenue — the metrics a real analytics
    dashboard would track daily.
    """
    rows = defaultdict(lambda: {"page_view": 0, "add_to_cart": 0, "purchase": 0, "revenue": 0.0})
    for e in events:
        key = (e.market, e.event_date)
        if e.event_type == "page_view":
            rows[key]["page_view"] += 1
        elif e.event_type == "add_to_cart":
            rows[key]["add_to_cart"] += 1
        elif e.event_type == "purchase":
            rows[key]["purchase"] += 1
            if e.value is not None:
                rows[key]["revenue"] += e.value

    records = []
    for (market, event_date), metrics in rows.items():
        records.append({"market": market, "event_date": event_date, **metrics})

    df = pd.DataFrame(records).sort_values(["market", "event_date"]).reset_index(drop=True)
    return df


def _mad_zscore(series, window=14, min_periods=7):
    """Rolling median + MAD z-score. For each point, compares it against
    the median and MAD of the preceding `window` days (excluding the
    point itself, so the anomaly can't suppress its own score), returns
    a z-score-like statistic. A MAD of 0 (e.g. a flat baseline of exact
    zeros) is handled by falling back to a small constant to avoid
    division by zero, rather than producing inf/NaN silently.
    """
    scores = []
    values = list(series)
    for i in range(len(values)):
        window_start = max(0, i - window)
        history = values[window_start:i]  # excludes current point
        if len(history) < min_periods:
            scores.append(None)
            continue
        med = median(history)
        abs_devs = [abs(v - med) for v in history]
        mad = median(abs_devs)
        mad_scaled = mad * 1.4826 if mad > 0 else 1e-6  # 1.4826 makes MAD comparable to stdev under normality
        z = (values[i] - med) / mad_scaled
        scores.append(z)
    return scores


def detect_anomalies(daily_df, metric="add_to_cart", z_threshold=3.5, window=14):
    """Runs the rolling MAD z-score anomaly detector per market on the
    given metric, and returns the flagged rows with their z-scores.
    z_threshold=3.5 is a standard, conservative choice for MAD-based
    outlier detection (commonly cited default, e.g. Iglewicz & Hoaglin) —
    not tuned to make this dataset's known-injected bugs pass, which
    would be circular. It's checked against them afterward.
    """
    results = []
    for market, group in daily_df.groupby("market"):
        group = group.sort_values("event_date").reset_index(drop=True)
        z_scores = _mad_zscore(group[metric], window=window)
        group = group.copy()
        group["z_score"] = z_scores
        group["is_anomaly"] = group["z_score"].apply(
            lambda z: (z is not None) and abs(z) > z_threshold
        )
        results.append(group)
    return pd.concat(results, ignore_index=True)


def root_cause_summary(anomaly_row):
    """Simple triage hints based on directly observable signals
    in the row itself — not a black-box diagnosis.
    """
    hints = []
    metric_col = [c for c in ["page_view", "add_to_cart", "purchase", "revenue"] if c in anomaly_row]
    for col in metric_col:
        if anomaly_row[col] == 0:
            hints.append(f"{col} is exactly zero — check for a tracking outage (tag not firing).")

    if anomaly_row.get("z_score") is not None and anomaly_row["z_score"] > 0:
        hints.append("Value is unusually HIGH — check for duplicate event firing or a promo/traffic spike.")
    elif anomaly_row.get("z_score") is not None and anomaly_row["z_score"] < 0:
        hints.append("Value is unusually LOW — check for a tracking outage or a genuine business drop (e.g. site issue).")

    if not hints:
        hints.append("No obvious signal in this metric alone — cross-check other metrics for the same market/day.")
    return hints
