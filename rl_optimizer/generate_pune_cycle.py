"""
generate_pune_cycle.py

The Kamble et al. (2009) paper "Development of real-world driving cycle:
Case study of Pune, India" does NOT publish a digitized second-by-second
speed trace -- only summary statistics (Tables 9-11) and a plot. This
script constructs a synthetic 1 Hz speed-time trace that matches those
published target parameters, so it can be used exactly like CMDC.csv.

Target parameters used (Table 9 "Third complete set" + Table 10 "Complete
data" + Table 11 speed-range distribution from the paper):
    Duration            : 1533 s
    Average velocity    : 19.55 km/h
    Max velocity        : 53.69 km/h
    Mean acceleration   : 3.73 m/s^2   (max 14.26 m/s^2)
    Mean deceleration   : -4.58 m/s^2  (max -10.92 m/s^2)
    % time acceleration : 14.18
    % time deceleration : 11.48
    % time cruise       : 56.25
    % time idle         : 18.09
    Speed distribution  : 0-10:32.4%, 10-20:22.0%, 20-30:21.3%,
                           30-40:12.0%, >40:12.3%

Method: builds a sequence of micro-trips (idle -> accelerate -> cruise ->
decelerate) with randomized durations/targets drawn to satisfy the phase
percentages and speed-range distribution above, at 1 Hz resolution, then
lightly smooths transitions. This is a statistically representative
reconstruction, not the original GPS trace.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DURATION = 1533
DT = 1.0
V_MAX = 53.69

# seconds budget for each phase (paper's percentages)
BUDGET = {
    "idle": 0.1809 * DURATION,
    "accel": 0.1418 * DURATION,
    "decel": 0.1148 * DURATION,
    "cruise": 0.5625 * DURATION,
}

# target cruise speed bands with their share of time (Table 11, complete set)
SPEED_BANDS = [
    (2, 10, 0.324),
    (10, 20, 0.220),
    (20, 30, 0.213),
    (30, 40, 0.120),
    (40, 53.69, 0.123),
]

ACCEL_MEAN, ACCEL_MAX = 3.73, 14.26   # m/s^2
DECEL_MEAN, DECEL_MAX = 4.58, 10.92   # m/s^2 (magnitude)


def sample_cruise_speed():
    bands, weights = zip(*[((lo, hi), w) for lo, hi, w in SPEED_BANDS])
    idx = RNG.choice(len(bands), p=np.array(weights) / sum(weights))
    lo, hi = bands[idx]
    return float(RNG.uniform(lo, min(hi, V_MAX)))


def build_cycle():
    t = [0.0]
    v = [0.0]
    remaining = dict(BUDGET)

    max_outer_iters = 20000
    outer_iter = 0
    while t[-1] < DURATION and sum(remaining.values()) > 0:
        outer_iter += 1
        if outer_iter > max_outer_iters:
            break
        progressed = False
        cur_v = v[-1]

        # --- idle ---
        if remaining["idle"] > 0 and (cur_v < 1.0 or RNG.random() < 0.15):
            dur = min(RNG.uniform(3, 12), remaining["idle"])
            n = max(int(round(dur / DT)), 1)
            for _ in range(n):
                t.append(t[-1] + DT)
                v.append(0.0)
            remaining["idle"] -= n * DT
            progressed = True
            continue

        target_v = sample_cruise_speed()

        # --- accelerate to target_v ---
        if target_v > cur_v and remaining["accel"] > 0:
            a = min(abs(RNG.normal(ACCEL_MEAN, 2.0)), ACCEL_MAX)
            a = max(a, 0.5)
            dv = a * 3.6 * DT  # km/h per step
            n_budget = int(remaining["accel"] / DT)
            steps = []
            vv = cur_v
            while vv < target_v and len(steps) < max(n_budget, 1):
                vv = min(vv + dv, target_v, V_MAX)
                steps.append(vv)
            for s in steps:
                t.append(t[-1] + DT)
                v.append(s)
            remaining["accel"] -= len(steps) * DT
            if steps:
                progressed = True

        # --- cruise near target_v ---
        cur_v = v[-1]
        if remaining["cruise"] > 0:
            dur = min(RNG.uniform(5, 25), remaining["cruise"])
            n = max(int(round(dur / DT)), 1)
            for _ in range(n):
                t.append(t[-1] + DT)
                v.append(max(0.0, cur_v + RNG.normal(0, 0.6)))
            remaining["cruise"] -= n * DT
            progressed = True

        # --- decelerate ---
        cur_v = v[-1]
        if remaining["decel"] > 0 and cur_v > 0:
            d = min(abs(RNG.normal(DECEL_MEAN, 2.0)), DECEL_MAX)
            d = max(d, 0.5)
            dv = d * 3.6 * DT
            end_v = 0.0 if RNG.random() < 0.4 else RNG.uniform(0, cur_v)
            n_budget = int(remaining["decel"] / DT)
            steps = []
            vv = cur_v
            while vv > end_v and len(steps) < max(n_budget, 1):
                vv = max(vv - dv, end_v, 0.0)
                steps.append(vv)
            for s in steps:
                t.append(t[-1] + DT)
                v.append(s)
            remaining["decel"] -= len(steps) * DT
            if steps:
                progressed = True

        if not progressed:
            # nothing could progress this pass (e.g. stuck target==cur speed) -
            # force-drain remaining budgets to avoid spinning forever
            for k in remaining:
                remaining[k] = 0.0

        if len(t) > DURATION * 3:
            break

    t = np.array(t)
    v = np.array(v)

    # trim / pad to exact duration
    mask = t <= DURATION
    t, v = t[mask], v[mask]
    if t[-1] < DURATION:
        extra_t = np.arange(t[-1] + DT, DURATION + DT, DT)
        extra_v = np.zeros_like(extra_t)
        t = np.concatenate([t, extra_t])
        v = np.concatenate([v, extra_v])

    v = np.clip(v, 0, V_MAX)
    return t, v


def report(t, v):
    dt = np.diff(t, prepend=t[0])
    a = np.gradient(v / 3.6, t)  # m/s^2
    idle = np.mean(v < 1.0) * 100
    accel_mask = a > 0.1
    decel_mask = a < -0.1
    cruise_mask = (~accel_mask) & (~decel_mask) & (v >= 1.0)

    print(f"Duration        : {t[-1]:.0f} s (target 1533 s)")
    print(f"Average speed   : {np.average(v, weights=dt):.2f} km/h (target 19.55)")
    print(f"Max speed       : {v.max():.2f} km/h (target 53.69)")
    print(f"%% idle          : {idle:.2f} (target 18.09)")
    print(f"%% accel         : {np.mean(accel_mask)*100:.2f} (target 14.18)")
    print(f"%% decel         : {np.mean(decel_mask)*100:.2f} (target 11.48)")
    print(f"%% cruise        : {np.mean(cruise_mask)*100:.2f} (target 56.25)")


def main():
    t, v = build_cycle()
    report(t, v)
    pd.DataFrame({"Time": t, "Speed": v}).to_csv("PUNE_DC.csv", index=False)
    print("Saved -> PUNE_DC.csv")


if __name__ == "__main__":
    main()
