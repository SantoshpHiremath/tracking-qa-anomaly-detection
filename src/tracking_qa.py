"""Automated tracking QA checks: validates event data against expected
schema/business rules, and flags missing or broken tracking parameters —
the posting's "Build automated checks for missing or broken tracking
parameters" and "design, document, and execute test cases for new
tracking implementations" tasks.
"""
from collections import defaultdict

REQUIRED_PARAMS = {
    "purchase": ["currency", "value", "item_id"],
    "add_to_cart": ["item_id"],
    "view_item": ["item_id"],
}


def check_missing_required_params(events):
    """Checks every event against REQUIRED_PARAMS and returns a list of
    violations: (event_id, event_type, market, event_date, missing_param).
    This is the direct implementation of 'automated checks for missing
    or broken tracking parameters.'
    """
    violations = []
    for e in events:
        required = REQUIRED_PARAMS.get(e.event_type, [])
        for param in required:
            value = getattr(e, param, None)
            if value is None:
                violations.append({
                    "event_id": e.event_id, "event_type": e.event_type,
                    "market": e.market, "event_date": e.event_date,
                    "missing_param": param,
                })
    return violations


def missing_param_summary(violations):
    """Aggregates violations by (market, event_type, missing_param, date)
    so a QA engineer sees patterns (e.g. 'FR purchase events missing
    currency for 7 consecutive days') rather than a flat list of
    thousands of individual event IDs — the difference between a useful
    audit report and unusable noise.
    """
    by_key = defaultdict(lambda: {"count": 0, "dates": set()})
    for v in violations:
        key = (v["market"], v["event_type"], v["missing_param"])
        by_key[key]["count"] += 1
        by_key[key]["dates"].add(v["event_date"])

    summary = []
    for (market, event_type, param), agg in by_key.items():
        summary.append({
            "market": market, "event_type": event_type, "missing_param": param,
            "count": agg["count"], "n_days_affected": len(agg["dates"]),
            "date_range": (min(agg["dates"]), max(agg["dates"])),
        })
    summary.sort(key=lambda r: r["count"], reverse=True)
    return summary


def check_duplicate_events(events, dedup_key_fields=("session_id", "event_type", "event_date")):
    """Flags events that appear to be duplicate-fired: same session,
    same event type, same day, appearing more than once. A real tracking
    setup allows repeat page_views in a session, but purchase/
    begin_checkout should not double-fire within the same session on the
    same day — this check is scoped to conversion-critical event types
    where duplication directly corrupts revenue/conversion metrics.
    """
    conversion_events = ("purchase", "begin_checkout")
    seen = defaultdict(int)
    for e in events:
        if e.event_type not in conversion_events:
            continue
        key = tuple(getattr(e, f) for f in dedup_key_fields)
        seen[key] += 1

    duplicates = [
        {"session_id": k[0], "event_type": k[1], "event_date": k[2], "fire_count": count}
        for k, count in seen.items() if count > 1
    ]
    duplicates.sort(key=lambda r: r["fire_count"], reverse=True)
    return duplicates


def check_funnel_consistency(events):
    """Sanity-checks the funnel shape per market/day: add_to_cart should
    never exceed view_item, begin_checkout should never exceed
    add_to_cart, purchase should never exceed begin_checkout (allowing
    for the fact that a session's stages aren't always 1:1 sequential in
    aggregate counts, this is checked as a per-market/day COUNT
    ordering, which is a reasonable sanity bound for a healthy funnel).
    Flags market/days where this ordering is violated — usually a sign
    something is over-firing or under-firing.
    """
    from collections import defaultdict as dd
    counts = dd(lambda: dd(int))
    for e in events:
        counts[(e.market, e.event_date)][e.event_type] += 1

    violations = []
    stage_order = ["page_view", "view_item", "add_to_cart", "begin_checkout", "purchase"]
    for (market, event_date), stage_counts in counts.items():
        for i in range(len(stage_order) - 1):
            upper, lower = stage_order[i], stage_order[i + 1]
            if stage_counts.get(lower, 0) > stage_counts.get(upper, 0):
                violations.append({
                    "market": market, "event_date": event_date,
                    "stage_upper": upper, "stage_lower": lower,
                    "count_upper": stage_counts.get(upper, 0),
                    "count_lower": stage_counts.get(lower, 0),
                })
    return violations


def check_zero_event_days(events, event_type, expected_markets=None):
    """Flags market/day combinations where an event type that should
    always have some volume (e.g. add_to_cart on an active webshop) has
    exactly zero events — the direct check for a silent tracking outage
    (a tag failing to fire at all, rather than firing with bad data).
    """
    from collections import defaultdict as dd
    counts = dd(int)
    all_market_days = set()

    for e in events:
        all_market_days.add((e.market, e.event_date))
        if e.event_type == event_type:
            counts[(e.market, e.event_date)] += 1

    markets = expected_markets or {m for m, _ in all_market_days}
    zero_days = [
        {"market": m, "event_date": d}
        for (m, d) in sorted(all_market_days)
        if m in markets and counts[(m, d)] == 0
    ]
    return zero_days
