"""Test suite for the tracking QA + anomaly detection project."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from events import (
    generate_events, MISSING_CURRENCY_BUG, DUPLICATE_PURCHASE_BUG,
    TRACKING_OUTAGE_BUG, START_DATE,
)
from tracking_qa import (
    check_missing_required_params, missing_param_summary,
    check_duplicate_events, check_funnel_consistency, check_zero_event_days,
)
from anomaly_detection import (
    events_to_daily_metrics, detect_anomalies, _mad_zscore, root_cause_summary,
)


@pytest.fixture(scope="module")
def events():
    return generate_events(seed=42)


@pytest.fixture(scope="module")
def daily(events):
    return events_to_daily_metrics(events)


# --- Event generation -----------------------------------------------------

def test_generate_events_deterministic_given_seed():
    a = generate_events(seed=1)
    b = generate_events(seed=1)
    assert len(a) == len(b)
    assert sum(e.value or 0 for e in a) == sum(e.value or 0 for e in b)


def test_generate_events_covers_all_markets_and_days(events):
    markets = {e.market for e in events}
    dates = {e.event_date for e in events}
    assert markets == {"DE", "US", "UK", "FR", "JP"}
    assert len(dates) == 60


def test_funnel_shape_is_realistic(events):
    from collections import Counter
    counts = Counter(e.event_type for e in events)
    # each funnel stage should be smaller than the previous one, in
    # aggregate across all markets/days
    assert counts["view_item"] < counts["page_view"]
    assert counts["add_to_cart"] < counts["view_item"]
    assert counts["begin_checkout"] <= counts["add_to_cart"]
    assert counts["purchase"] <= counts["begin_checkout"]


# --- Tracking QA: missing required parameters ------------------------------

def test_missing_item_id_detected_on_view_item(events):
    violations = check_missing_required_params(events)
    item_id_violations = [v for v in violations if v["missing_param"] == "item_id" and v["event_type"] == "view_item"]
    assert len(item_id_violations) > 0


def test_missing_currency_bug_detected_exactly(events):
    """Regression check against the specific injected bug: FR purchase
    events should be missing currency for exactly the injected day range,
    and no other market/event-type should show missing currency at all
    (currency is always populated outside the bug window)."""
    violations = check_missing_required_params(events)
    currency_violations = [v for v in violations if v["missing_param"] == "currency"]
    markets_affected = {v["market"] for v in currency_violations}
    assert markets_affected == {MISSING_CURRENCY_BUG["market"]}

    affected_dates = {v["event_date"] for v in currency_violations}
    from datetime import timedelta
    expected_dates = {
        (START_DATE + timedelta(days=d)).isoformat() for d in MISSING_CURRENCY_BUG["days"]
    }
    assert affected_dates == expected_dates


def test_no_false_positive_missing_params_outside_injected_bugs(events):
    # purchase events should always have item_id and value populated —
    # only currency was deliberately made nullable, and only for FR/bug window
    violations = check_missing_required_params(events)
    non_currency_purchase_violations = [
        v for v in violations if v["event_type"] == "purchase" and v["missing_param"] != "currency"
    ]
    assert non_currency_purchase_violations == []


def test_missing_param_summary_aggregates_correctly(events):
    violations = check_missing_required_params(events)
    summary = missing_param_summary(violations)
    total_from_summary = sum(row["count"] for row in summary)
    assert total_from_summary == len(violations)


def test_missing_param_summary_sorted_descending_by_count():
    violations = [
        {"event_id": 1, "event_type": "purchase", "market": "DE", "event_date": "2026-01-01", "missing_param": "currency"},
        {"event_id": 2, "event_type": "purchase", "market": "DE", "event_date": "2026-01-01", "missing_param": "currency"},
        {"event_id": 3, "event_type": "view_item", "market": "US", "event_date": "2026-01-01", "missing_param": "item_id"},
    ]
    summary = missing_param_summary(violations)
    counts = [row["count"] for row in summary]
    assert counts == sorted(counts, reverse=True)


# --- Tracking QA: duplicate events -----------------------------------------

def test_duplicate_purchase_bug_detected(events):
    dupes = check_duplicate_events(events)
    us_dupes = [d for d in dupes if d["event_type"] == "purchase"]
    assert len(us_dupes) > 0
    # every duplicate purchase should fall within the injected bug's
    # market/day window
    from datetime import timedelta
    bug_dates = {(START_DATE + timedelta(days=d)).isoformat() for d in DUPLICATE_PURCHASE_BUG["days"]}
    for d in us_dupes:
        assert d["session_id"].startswith(DUPLICATE_PURCHASE_BUG["market"])
        assert d["event_date"] in bug_dates


def test_no_duplicate_page_views_flagged(events):
    # page_view is expected to repeat within a session (out of scope for
    # this check) — it should never appear in duplicate results, which
    # are scoped to conversion-critical event types only
    dupes = check_duplicate_events(events)
    assert all(d["event_type"] in ("purchase", "begin_checkout") for d in dupes)


# --- Tracking QA: funnel consistency ---------------------------------------

def test_funnel_consistency_check_runs_without_error(events):
    violations = check_funnel_consistency(events)
    assert isinstance(violations, list)


def test_funnel_consistency_detects_synthetic_violation():
    from dataclasses import dataclass
    @dataclass
    class FakeEvent:
        event_id: int
        event_date: str
        market: str
        event_type: str
        session_id: str = ""
        item_id: str = None
        currency: str = None
        value: float = None

    fake_events = (
        [FakeEvent(i, "2026-01-01", "DE", "page_view") for i in range(5)]
        + [FakeEvent(i + 100, "2026-01-01", "DE", "view_item") for i in range(10)]  # more than page_view -> violation
    )
    violations = check_funnel_consistency(fake_events)
    assert any(v["stage_upper"] == "page_view" and v["stage_lower"] == "view_item" for v in violations)


# --- Tracking QA: zero-event days (outage detection) -----------------------

def test_tracking_outage_bug_detected_exactly(events):
    zero_days = check_zero_event_days(events, "add_to_cart")
    from datetime import timedelta
    expected_dates = {(START_DATE + timedelta(days=d)).isoformat() for d in TRACKING_OUTAGE_BUG["days"]}
    flagged_dates = {row["event_date"] for row in zero_days if row["market"] == TRACKING_OUTAGE_BUG["market"]}
    assert flagged_dates == expected_dates


def test_zero_event_days_only_flags_the_outage_market(events):
    zero_days = check_zero_event_days(events, "add_to_cart")
    markets_flagged = {row["market"] for row in zero_days}
    assert markets_flagged == {TRACKING_OUTAGE_BUG["market"]}


# --- Anomaly detection: MAD z-score -----------------------------------------

def test_mad_zscore_returns_none_before_min_periods():
    scores = _mad_zscore([10] * 20, window=14, min_periods=7)
    assert scores[0] is None
    assert scores[5] is None
    assert scores[7] is not None


def test_mad_zscore_flat_series_no_false_anomaly():
    # a perfectly flat series should never produce a spurious huge
    # z-score (guards the mad=0 fallback-constant handling)
    scores = _mad_zscore([50] * 30, window=14, min_periods=7)
    valid_scores = [s for s in scores if s is not None]
    assert all(abs(s) < 1.0 for s in valid_scores)


def test_mad_zscore_detects_obvious_spike():
    series = [50] * 20 + [500] + [50] * 5
    scores = _mad_zscore(series, window=14, min_periods=7)
    spike_score = scores[20]
    assert spike_score is not None and spike_score > 5


def test_events_to_daily_metrics_row_count(daily):
    # 5 markets x 60 days
    assert len(daily) == 300


def test_events_to_daily_metrics_revenue_matches_manual_sum(events, daily):
    us_events = [e for e in events if e.market == "US" and e.event_date == "2026-06-15" and e.event_type == "purchase"]
    manual_revenue = sum(e.value for e in us_events if e.value is not None)
    row = daily[(daily["market"] == "US") & (daily["event_date"] == "2026-06-15")].iloc[0]
    assert row["revenue"] == pytest.approx(manual_revenue, abs=0.01)


# --- Anomaly detection: injected bugs are actually caught -------------------
# These are the honesty-critical tests: the detector must catch the
# injected bugs on REAL, untuned thresholds (z_threshold=3.5, the
# standard Iglewicz & Hoaglin default) -- not thresholds picked to make
# this specific dataset pass, which would be circular and dishonest.

def test_anomaly_detection_catches_tracking_outage(daily):
    """The detector should catch the outage promptly (the first day, and
    most of the window) — but NOT necessarily every single day. This is
    an honest, documented limitation of rolling-baseline anomaly
    detection: by the later days of a sustained outage, the rolling
    14-day window itself starts to include the outage's own earlier
    zero-value days, which pulls the baseline down and can shrink the
    z-score for the tail end of a long-running anomaly below the
    threshold. Asserting 100% detection on every day would either be
    false, or would require tuning the threshold specifically to this
    dataset's bug window, which would be circular. See README for the
    full explanation.
    """
    result = detect_anomalies(daily, metric="add_to_cart", z_threshold=3.5)
    from datetime import timedelta
    outage_dates = {(START_DATE + timedelta(days=d)).isoformat() for d in TRACKING_OUTAGE_BUG["days"]}
    flagged = result[
        (result["market"] == TRACKING_OUTAGE_BUG["market"]) & (result["event_date"].isin(outage_dates))
    ]
    assert flagged.iloc[0]["is_anomaly"], "the first day of the outage should be flagged immediately"
    assert flagged["is_anomaly"].mean() >= 0.75, "most of the outage window should be flagged"


def test_anomaly_detection_catches_duplicate_purchase_spike(daily):
    from datetime import timedelta
    bug_dates = {(START_DATE + timedelta(days=d)).isoformat() for d in DUPLICATE_PURCHASE_BUG["days"]}
    result = detect_anomalies(daily, metric="purchase", z_threshold=3.5)
    flagged_in_window = result[
        (result["market"] == DUPLICATE_PURCHASE_BUG["market"]) & (result["event_date"].isin(bug_dates))
    ]
    assert flagged_in_window["is_anomaly"].any(), "at least one duplicate-bug day should be flagged as a purchase anomaly"


def test_anomaly_detection_does_not_flag_every_day_as_anomaly(daily):
    # sanity guard against a degenerate detector that flags everything
    # (or nothing) -- a real detector on 300 rows should flag a small
    # minority, not most/all/none of them
    result = detect_anomalies(daily, metric="add_to_cart", z_threshold=3.5)
    anomaly_rate = result["is_anomaly"].mean()
    assert 0 < anomaly_rate < 0.15


# --- Root cause hints --------------------------------------------------------

def test_root_cause_summary_flags_zero_metric():
    row = {"page_view": 100, "add_to_cart": 0, "purchase": 0, "revenue": 0.0, "z_score": -5.0}
    hints = root_cause_summary(row)
    assert any("zero" in h.lower() for h in hints)


def test_root_cause_summary_flags_high_spike():
    row = {"page_view": 100, "add_to_cart": 900, "purchase": 50, "revenue": 5000.0, "z_score": 8.0}
    hints = root_cause_summary(row)
    assert any("high" in h.lower() for h in hints)


def test_root_cause_summary_never_returns_empty():
    row = {"page_view": 100, "add_to_cart": 50, "purchase": 10, "revenue": 500.0, "z_score": None}
    hints = root_cause_summary(row)
    assert len(hints) > 0
