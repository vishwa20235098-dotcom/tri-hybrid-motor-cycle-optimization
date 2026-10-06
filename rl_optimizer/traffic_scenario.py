"""
traffic_scenario.py

Takes an Indian urban driving cycle and overlays real-life urban constraints
that the paper's cycle does NOT model explicitly:

  1. Traffic-signal stops: at chosen time windows, speed is forced to 0
     for a hold duration (vehicle waits at a red light / junction).
  2. Speed-limit zones: within chosen time windows, speed is clipped to
     a maximum (e.g. school zone, market street, speed breaker stretch).

The result is saved as CMDC_traffic.csv (same 2-column format: Time,Speed)
so that BOTH the Python RL environment and your MATLAB pipeline
(loadDrivingCycle.m -> vehicleDynamics.m) operate on the identical cycle.

This does not change any existing .m file -- it just produces a new CSV
that main_rl.m points to instead of CMDC.csv.
"""

import os
import numpy as np
import pandas as pd


def _resolve_path(path):
    """Look in the current directory first, then the project root (one
    level up) -- so this works whether you run it from rl_optimizer/ or
    from the project root."""
    if os.path.isfile(path):
        return path
    parent = os.path.join("..", path)
    if os.path.isfile(parent):
        return parent
    raise FileNotFoundError(
        f"Could not find '{path}' in the current directory or its parent. "
        f"Run this script from either the project root or rl_optimizer/."
    )


def load_base_cycle(path="CMDC.csv"):
    path = _resolve_path(path)
    df = pd.read_csv(path)
    t = df["Time"].to_numpy(dtype=float)
    v = df["Speed"].to_numpy(dtype=float)
    # de-dup / sort, mirrors loadDrivingCycle.m behaviour
    t = t - t[0]
    order = np.argsort(t)
    t, v = t[order], v[order]
    _, uniq_idx = np.unique(t, return_index=True)
    return t[uniq_idx], np.clip(v[uniq_idx], 0, None)


def apply_traffic_stops(t, v, stops):
    """
    stops: list of dicts {"start": s, "hold": h}
        start -> time (s) at which the signal turns red
        hold  -> how long (s) the vehicle must stay stopped

    The vehicle is decelerated to 0 over a short ramp before `start`,
    held at 0 for `hold` seconds, then resumes the original profile
    (time-shifted forward by `hold` seconds for everything after).
    """
    t = t.copy()
    v = v.copy()
    ramp = 3.0  # seconds to bring speed to 0 before the stop line

    shift = 0.0
    for stop in sorted(stops, key=lambda s: s["start"]):
        s_time = stop["start"] + shift
        hold = stop["hold"]

        idx_ramp_start = np.searchsorted(t, s_time - ramp)
        idx_stop = np.searchsorted(t, s_time)

        # ramp speed down to zero approaching the stop
        if idx_ramp_start < idx_stop:
            ramp_len = idx_stop - idx_ramp_start
            v[idx_ramp_start:idx_stop] = np.minimum(
                v[idx_ramp_start:idx_stop],
                np.linspace(v[idx_ramp_start] if idx_ramp_start < len(v) else 0, 0, ramp_len)
            )

        # insert a hold: duplicate the stop instant, hold at 0, then
        # shift every later sample forward by `hold` seconds
        insert_t = np.array([s_time + 0.5 * hold])
        insert_v = np.array([0.0])

        after_mask = t >= s_time
        t = np.concatenate([t[~after_mask], insert_t, t[after_mask] + hold])
        v = np.concatenate([v[~after_mask], insert_v, v[after_mask]])

        shift += hold  # subsequent stop times are given in *original* timeline

    order = np.argsort(t)
    t, v = t[order], v[order]
    _, uniq_idx = np.unique(t, return_index=True)
    return t[uniq_idx], v[uniq_idx]


def apply_speed_limits(t, v, zones):
    """
    zones: list of dicts {"start": s, "end": e, "limit_kmph": L}
    Clips speed to `limit_kmph` for t in [start, end].
    """
    v = v.copy()
    for z in zones:
        mask = (t >= z["start"]) & (t <= z["end"])
        v[mask] = np.minimum(v[mask], z["limit_kmph"])
    return t, v


def build_default_scenario(t, v):
    """A reasonable default real-world overlay, scaled to the cycle's own
    duration so it works for both the ~1450s CMDC and the ~1530s Pune cycle
    (or any other cycle length)."""
    dur = t[-1]
    frac_stops = [
        (0.10, 20), (0.30, 15), (0.54, 25), (0.76, 12),
    ]  # (fraction of duration, hold seconds)
    frac_zones = [
        (0.17, 0.24, 20), (0.41, 0.48, 15), (0.62, 0.68, 25),
    ]  # (start frac, end frac, limit kmph)

    stops = [{"start": f * dur, "hold": h} for f, h in frac_stops]
    zones = [{"start": s * dur, "end": e * dur, "limit_kmph": L}
             for s, e, L in frac_zones]

    t2, v2 = apply_traffic_stops(t, v, stops)
    t2, v2 = apply_speed_limits(t2, v2, zones)
    return t2, v2, stops, zones


# registry of available base cycles: name -> csv filename
CYCLES = {
    "cmdc": "CMDC.csv",
    "pune": "PUNE_DC.csv",
    "delhi": "DELHI_DMDC_reconstructed.csv",
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", choices=list(CYCLES) + ["all"], default="all",
                     help="Which base cycle to overlay with traffic (default: all)")
    args = ap.parse_args()

    names = list(CYCLES) if args.cycle == "all" else [args.cycle]

    for name in names:
        base_path = CYCLES[name]
        t, v = load_base_cycle(base_path)
        t2, v2, stops, zones = build_default_scenario(t, v)

        out_path = f"{name.upper()}_traffic.csv"
        pd.DataFrame({"Time": t2, "Speed": v2}).to_csv(out_path, index=False)

        print(f"[{name}] base: {t[-1]:.1f}s/{len(t)} samples -> "
              f"traffic: {t2[-1]:.1f}s/{len(t2)} samples "
              f"(+{t2[-1]-t[-1]:.1f}s from {len(stops)} stops, "
              f"{len(zones)} speed-limit zones) -> {out_path}")


if __name__ == "__main__":
    main()
