"""Shared helper for loading Indian driving cycles only."""
import os
import pandas as pd

DEFAULT_CYCLE_FILES = {
    "cmdc": "CMDC_traffic.csv",          # Chennai Motorcycle Driving Cycle
    "pune": "PUNE_traffic.csv",          # Pune urban cycle supplied in project
    "delhi": "DELHI_traffic.csv",   # Delhi Motorcycle Driving Cycle (reconstructed trace)
}


def load_cycle_csv(path):
    df = pd.read_csv(path)
    return df["Time"].to_numpy(dtype=float), df["Speed"].to_numpy(dtype=float)


def load_cycles(names=None, files=None):
    files = files or DEFAULT_CYCLE_FILES
    names = names or list(files.keys())
    cycles = []
    for name in names:
        t, v = load_cycle_csv(files[name])
        cycles.append({"name": name, "t": t, "v": v})
    return cycles
