"""Forecast hourly Twitch viewers for a big game launch day.

Model: anchor on the known hour-1 print, then scale it by a ratio curve
learned from past rolling-midnight launches (the case's comparables).

For every comparable launch we take its release-day (UTC) hourly viewers,
form log ratios r_h = log(viewers_h / viewers_1), and fit, per hour band,
a pooled log-log slope of the ratio on the launch's hour-1 size: big
launches grow less into the evening than small ones. The forecast centre
for hour h is the trimmed mean of the slope-adjusted residuals, and the
log spread is the cross-launch residual dispersion, shrunk a little.
Quantiles are lognormal around that centre.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

REF_SIZE = 100000.0
SPREAD_SCALE = 0.7
SPREAD_FLOOR = 0.45
SPREAD_CAP = 1.0
SLOPE_MIN = -1.2
SLOPE_MAX = 0.0
X_CLAMP_LO = math.log(20000.0 / REF_SIZE)
X_CLAMP_HI = math.log(400000.0 / REF_SIZE)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def is_rolling_midnight(comp):
    """Keep console-style rolling-midnight launches; drop the rest."""
    unlock = (comp.get("unlock") or "").lower()
    if "early-access" in unlock:
        return False
    if "single global" in unlock or "simultaneous" in unlock:
        return False
    return "midnight" in unlock


def release_day_ratios(comp):
    """Log ratios of release-day hourly viewers to the hour-1 viewers."""
    try:
        day = parse_time(comp["day_start"]).date()
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            when = parse_time(row["from"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if when.date() == day and viewers > 0:
            by_hour[when.hour + 1] = viewers
    first = by_hour.get(1)
    if not first or first <= 0 or len(by_hour) < 20:
        return None
    return first, {h: math.log(v / first) for h, v in by_hour.items()}


def collect_curves(case, strict):
    curves = []
    for comp in case.get("comparables") or []:
        name = (comp.get("name") or "").lower()
        if "grand theft" in name:
            continue
        if strict and not is_rolling_midnight(comp):
            continue
        got = release_day_ratios(comp)
        if got is not None:
            curves.append(got)
    return curves


def pooled_slope(curves, hours):
    """Pooled OLS slope of log ratio on log hour-1 size over given hours."""
    num = 0.0
    den = 0.0
    for hour in hours:
        points = []
        for size, ratios in curves:
            if hour in ratios:
                points.append((math.log(size / REF_SIZE), ratios[hour]))
        if len(points) < 4:
            continue
        mx = sum(p[0] for p in points) / len(points)
        my = sum(p[1] for p in points) / len(points)
        num += sum((x - mx) * (y - my) for x, y in points)
        den += sum((x - mx) ** 2 for x, y in points)
    if den <= 0:
        return 0.0
    slope = num / den
    return max(SLOPE_MIN, min(SLOPE_MAX, slope))


def trimmed_mean(values):
    ordered = sorted(values)
    if len(ordered) >= 4:
        ordered = ordered[1:-1]
    return sum(ordered) / len(ordered)


def hour_centre_spread(curves, hour, slope, x_case):
    residuals = []
    for size, ratios in curves:
        if hour in ratios:
            residuals.append(ratios[hour] - slope * math.log(size / REF_SIZE))
    if len(residuals) < 3:
        return None, None
    centre = trimmed_mean(residuals) + slope * x_case
    if len(residuals) > 1:
        spread = statistics.pstdev(residuals)
    else:
        spread = 0.35
    spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread)) * SPREAD_SCALE
    return centre, spread


def forecast_case(case):
    known = case.get("known") or []
    if not known:
        raise ValueError("case has no known hour 1")
    p1 = float(known[0]["viewers"])
    if p1 < 1:
        p1 = 1.0

    curves = collect_curves(case, strict=True)
    if len(curves) < 4:
        curves = collect_curves(case, strict=False)
    if not curves:
        # No usable comparables: flat persistence with a widening spread.
        rows = []
        for hour in range(2, 25):
            spread = 0.15 * math.sqrt(hour - 1)
            qs = [round(p1 * math.exp(z * spread), 2) for z in Z]
            rows.append({"hour": hour, "q": qs})
        return {"hours": rows}

    x_case = math.log(p1 / REF_SIZE)
    x_case = max(X_CLAMP_LO, min(X_CLAMP_HI, x_case))
    slope_morning = pooled_slope(curves, range(2, 5))
    slope_evening = pooled_slope(curves, range(14, 25))

    centres = {}
    spreads = {}
    for hour in range(2, 25):
        slope = slope_morning if hour < 5 else slope_evening
        centre, spread = hour_centre_spread(curves, hour, slope, x_case)
        if centre is None:
            continue
        centres[hour] = centre
        spreads[hour] = spread
    # Fill any gap from the nearest hour that has a value.
    for hour in range(2, 25):
        if hour in centres:
            continue
        for dist in range(1, 23):
            done = False
            for cand in (hour - dist, hour + dist):
                if cand in centres:
                    centres[hour] = centres[cand]
                    spreads[hour] = spreads[cand]
                    done = True
                    break
            if done:
                break
        else:
            centres[hour] = 0.0
            spreads[hour] = 0.35

    rows = []
    for hour in range(2, 25):
        centre = centres[hour]
        if hour >= 12:
            centre = max(centre, -0.3)
        spread = spreads[hour]
        qs = []
        prev = 0.0
        for z in Z:
            value = p1 * math.exp(centre + z * spread)
            value = max(1.0, min(1e9, value))
            value = round(value, 2)
            if value < prev:
                value = prev
            qs.append(value)
            prev = value
        rows.append({"hour": hour, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
