"""Indian urban driving-cycle helpers.

The project uses the city-specific cycles already supplied by the user:
Chennai (CMDC) and Pune.  Delhi DMDC is included as an additional Indian
urban motorcycle cycle.  The original DMDC paper reports the statistical
properties and a plotted representative trace, but does not provide a
machine-readable trace in the paper download.  Therefore DELHI_DMDC.csv is
explicitly marked as a *reconstructed* trace matching the published summary;
it must not be presented as the authors' raw data.
"""
import numpy as np
import pandas as pd


def generate_delhi_dmdc(out_path="DELHI_DMDC_reconstructed.csv", seed=42):
    rng = np.random.default_rng(seed)
    # Published DMDC summary: 847.8 s, mean speed 34.36 km/h,
    # running speed 36.61 km/h, total distance 8054.7 m, max <= 50 km/h.
    # We create a reproducible stop-go profile with the same duration,
    # approximately the published distance and speed statistics.
    dt = 0.5
    t = np.arange(0, 847.8 + dt, dt)
    v = np.zeros_like(t)
    i = 0
    while i < len(t):
        r = rng.random()
        if r < 0.10:
            dur = rng.uniform(3, 10); target = 0
        elif r < 0.42:
            dur = rng.uniform(8, 28); target = rng.uniform(22, 42)
        elif r < 0.72:
            dur = rng.uniform(5, 18); target = rng.uniform(35, 50)
        else:
            dur = rng.uniform(2, 8); target = rng.uniform(5, 28)
        n = max(1, int(dur/dt))
        j = min(len(t), i+n)
        start = v[i-1] if i else 0
        # smooth transition with small traffic fluctuations
        x = np.linspace(0, 1, j-i)
        seg = start + (target-start)*(3*x*x-2*x*x*x)
        seg += rng.normal(0, 1.1, size=j-i)
        v[i:j] = np.clip(seg, 0, 50)
        i = j
    # add realistic complete stops; then scale to target mean speed/distance
    for center in rng.choice(len(v)-20, size=15, replace=False):
        v[center:center+int(rng.integers(4,16))] = 0
    target_mean = 34.35778043
    v *= target_mean / max(v.mean(), 1e-9)
    v = np.clip(v, 0, 50)
    # final small scaling to target distance without exceeding 50 km/h
    target_dist = 8.054710556
    dist = np.trapz(v/3600, t)
    v *= target_dist/max(dist,1e-9)
    v = np.clip(v, 0, 50)
    # restore mean close to published value after clipping, with bounded scaling
    for _ in range(6):
        dist = np.trapz(v/3600, t)
        v *= target_dist/max(dist,1e-9)
        v = np.clip(v,0,50)
    pd.DataFrame({"Time": t, "Speed": v}).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print(generate_delhi_dmdc())
