"""End-to-end demo: generate synthetic tracking events (with injected
bugs), run tracking QA checks, aggregate to daily metrics, run anomaly
detection, and print root-cause hints for flagged anomalies.
"""
from events import generate_events, MISSING_CURRENCY_BUG, DUPLICATE_PURCHASE_BUG, TRACKING_OUTAGE_BUG, START_DATE
from tracking_qa import (
    check_missing_required_params, missing_param_summary,
    check_duplicate_events, check_funnel_consistency, check_zero_event_days,
)
from anomaly_detection import events_to_daily_metrics, detect_anomalies, root_cause_summary


def run():
    print("=== Generating synthetic webshop tracking events ===")
    events = generate_events(seed=42)
    print(f"  {len(events)} events across 5 markets, 60 days (synthetic data)\n")

    print("=== Tracking QA: Missing Required Parameters ===")
    violations = check_missing_required_params(events)
    summary = missing_param_summary(violations)
    print(f"  {len(violations)} total violations")
    for row in summary[:5]:
        print(f"    {row['market']} {row['event_type']}.{row['missing_param']}: "
              f"{row['count']} events across {row['n_days_affected']} days "
              f"({row['date_range'][0]} to {row['date_range'][1]})")

    print("\n=== Tracking QA: Duplicate Conversion Events ===")
    dupes = check_duplicate_events(events)
    print(f"  {len(dupes)} session/event/day combinations fired more than once")
    if dupes:
        print(f"    example: {dupes[0]}")

    print("\n=== Tracking QA: Funnel Consistency ===")
    funnel_violations = check_funnel_consistency(events)
    print(f"  {len(funnel_violations)} market/day funnel-ordering violations")

    print("\n=== Tracking QA: Zero-Event Days (possible outage) ===")
    zero_days = check_zero_event_days(events, "add_to_cart")
    print(f"  {len(zero_days)} market/days with zero add_to_cart events")
    for row in zero_days[:5]:
        print(f"    {row['market']} {row['event_date']}")

    print("\n=== Aggregating to Daily Metrics ===")
    daily = events_to_daily_metrics(events)
    print(f"  {len(daily)} market/day rows\n")

    print("=== Anomaly Detection: add_to_cart ===")
    result = detect_anomalies(daily, metric="add_to_cart", z_threshold=3.5)
    anomalies = result[result["is_anomaly"]]
    print(f"  {len(anomalies)} anomalies flagged out of {len(result)} market/day rows")
    for _, row in anomalies.sort_values("event_date").iterrows():
        print(f"    {row['market']} {row['event_date']}: add_to_cart={row['add_to_cart']}, z={row['z_score']:.2f}")
        for hint in root_cause_summary(row):
            print(f"      -> {hint}")

    print("\n=== Anomaly Detection: purchase ===")
    result_purchase = detect_anomalies(daily, metric="purchase", z_threshold=3.5)
    anomalies_purchase = result_purchase[result_purchase["is_anomaly"]]
    print(f"  {len(anomalies_purchase)} anomalies flagged out of {len(result_purchase)} market/day rows")
    for _, row in anomalies_purchase.sort_values("event_date").iterrows():
        print(f"    {row['market']} {row['event_date']}: purchase={row['purchase']}, z={row['z_score']:.2f}")

    print("\n=== Known Injected Bugs (for validation against the above) ===")
    print(f"  Missing currency: {MISSING_CURRENCY_BUG['market']}, day offsets {list(MISSING_CURRENCY_BUG['days'])}")
    print(f"  Duplicate purchases: {DUPLICATE_PURCHASE_BUG['market']}, day offsets {list(DUPLICATE_PURCHASE_BUG['days'])}")
    print(f"  Tracking outage: {TRACKING_OUTAGE_BUG['market']}, day offsets {list(TRACKING_OUTAGE_BUG['days'])}")


if __name__ == "__main__":
    run()
