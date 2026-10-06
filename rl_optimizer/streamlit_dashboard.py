"""Streamlit dashboard for the Indian tri-hybrid EV project.

Run from this directory:
    streamlit run streamlit_dashboard.py

The dashboard is deliberately separate from MATLAB. It reads the same Indian
cycle CSVs and can evaluate the trained PPO policy directly. It also creates a
non-optimized rule-based baseline in Python so the dashboard can compare
"without optimizer" vs "with optimizer" without requiring MATLAB to be open.

Expected trained model:
    ems_ppo_model.zip

If the PPO model is not present, the dashboard still shows the driving-cycle
and baseline results and clearly marks optimized results as unavailable.
"""

from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cycles import load_cycles, DEFAULT_CYCLE_FILES
from models import (
    make_vehicle, make_fuel_cell, make_battery, make_ultracapacitor,
    vehicle_dynamics, fuel_cell_model, battery_model, ultracapacitor_model,
)

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except Exception:
    PPO = None
    SB3_AVAILABLE = False


st.set_page_config(
    page_title="Indian Tri-Hybrid EV Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYCLE_LABELS = {
    "cmdc": "Chennai — CMDC",
    "pune": "Pune urban cycle",
    "delhi": "Delhi — DMDC (reconstructed)",
}
FAILURES = {
    "none": "Healthy system",
    "fuel_cell": "Fuel-cell failure",
    "ultracapacitor": "Ultracapacitor failure",
}


@st.cache_data(show_spinner=False)
def load_cycle(name):
    cycles = load_cycles(names=[name])
    c = cycles[0]
    P, _ = vehicle_dynamics(c["t"], c["v"], make_vehicle())
    c["P_demand"] = P
    return c


def _backup_model():
    return dict(Vnom=50.0, capacity_Ah=3.0, Pmax_discharge=1.5,
                Pmax_charge=0.75, SOC_initial=0.90, SOC_min=0.20,
                SOC_max=0.95, eff_charge=0.95, eff_discharge=0.95,
                SOC=0.90)


def _init_state():
    b = make_battery()
    u = make_ultracapacitor()
    u["V_current"] = u["V_initial"]
    bk = _backup_model()
    bk["SOC"] = bk["SOC_initial"]
    return b, u, bk


def simulate_baseline(cycle_name, failure):
    """Non-optimized deterministic power-split baseline.

    It prioritizes the fuel cell for the slower/base load, uses the UC for
    transients and regenerative braking, then lets the traction battery and
    emergency backup close the remaining gap.
    """
    c = load_cycle(cycle_name)
    t, v, pdm = c["t"], c["v"], c["P_demand"]
    fc0 = make_fuel_cell()
    batt, uc, backup = _init_state()
    N = len(t)
    out = {k: np.zeros(N) for k in [
        "P_fc", "P_battery", "P_uc", "P_backup", "SOC", "UC_voltage",
        "backup_SOC", "balance_error", "P_demand"
    ]}
    out["SOC"][0] = batt["SOC"]
    out["UC_voltage"][0] = uc["V_current"]
    out["backup_SOC"][0] = backup["SOC"]
    out["P_demand"] = pdm.copy()

    accel = np.gradient(v / 3.6, t)
    accel_norm = np.clip(np.abs(accel) / 2.0, 0, 1)
    fail_i = int(0.50 * (N - 1)) if failure != "none" else N + 1

    for k in range(N - 1):
        dt = max(float(t[k + 1] - t[k]), 1e-6)
        req = float(pdm[k])
        fc_alive = not (failure == "fuel_cell" and k >= fail_i)
        uc_alive = not (failure == "ultracapacitor" and k >= fail_i)

        if req > 0:
            # Base load to FC; transients to UC; battery fills the middle.
            fc_cmd = min(req * (0.70 - 0.20 * accel_norm[k]), fc0["Pmax"]) if fc_alive else 0.0
            rem = max(req - fc_cmd, 0.0)
            uc_cmd = min(rem * (0.75 + 0.20 * accel_norm[k]), 2.5) if uc_alive else 0.0
            rem2 = rem - uc_cmd
            batt_cmd = np.clip(rem2, -1.0, 1.0)
        elif req < 0:
            fc_cmd = 0.0
            # Absorb braking energy with UC first, then battery.
            uc_cmd = max(req, -2.5) if uc_alive else 0.0
            rem = req - uc_cmd
            batt_cmd = np.clip(rem, -1.0, 0.0)
        else:
            fc_cmd = uc_cmd = batt_cmd = 0.0

        fc_params = fc0 if fc_alive else dict(Pmax=0.0, Pmin=0.0, efficiency=0.55)
        pfc, _ = fuel_cell_model(fc_cmd, fc_params)
        pb, batt["SOC"] = battery_model(batt_cmd, dt, batt)

        if uc_alive:
            puc_cmd = req - pfc - pb
            puc_cmd = np.clip(puc_cmd, -2.5, 2.5)
            puc, uc["V_current"] = ultracapacitor_model(puc_cmd, dt, uc)
        else:
            puc = 0.0

        supplied = pfc + pb + puc
        pbackup_cmd = max(req - supplied, 0.0)
        pbackup, backup["SOC"] = battery_model(
            min(pbackup_cmd, backup["Pmax_discharge"]), dt, backup
        )
        supplied += pbackup

        out["P_fc"][k] = pfc
        out["P_battery"][k] = pb
        out["P_uc"][k] = puc
        out["P_backup"][k] = pbackup
        out["SOC"][k + 1] = batt["SOC"]
        out["UC_voltage"][k + 1] = uc["V_current"]
        out["backup_SOC"][k + 1] = backup["SOC"]
        out["balance_error"][k] = req - supplied

    return finalize_result(out, t, v, cycle_name, failure, "Without optimizer")


def simulate_ppo(cycle_name, failure, model_path):
    if not SB3_AVAILABLE or not model_path.exists():
        return None
    from env import TriHybridEnv
    cycles = load_cycles(names=list(DEFAULT_CYCLE_FILES.keys()))
    env = TriHybridEnv(cycles)
    model = PPO.load(str(model_path))
    obs, _ = env.reset(options={"cycle": cycle_name, "failure": failure})
    N = env.N
    out = {k: np.zeros(N) for k in [
        "P_fc", "P_battery", "P_uc", "P_backup", "SOC", "UC_voltage",
        "backup_SOC", "balance_error", "P_demand"
    ]}
    out["SOC"][0] = env.battery["SOC"]
    out["UC_voltage"][0] = env.uc["V_current"]
    out["backup_SOC"][0] = env.backup["SOC"]
    out["P_demand"] = env.P_demand.copy()

    for k in range(N - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        for key in ["P_fc", "P_battery", "P_uc", "P_backup", "SOC",
                    "UC_voltage", "backup_SOC", "balance_error"]:
            out[key][k] = info[key]
        if terminated or truncated:
            break

    out["SOC"][-1] = env.battery["SOC"]
    out["UC_voltage"][-1] = env.uc["V_current"]
    out["backup_SOC"][-1] = env.backup["SOC"]
    return finalize_result(out, env.t, env.cur["v"], cycle_name, failure, "With PPO optimizer")


def finalize_result(out, t, v, cycle, failure, label):
    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    demand_pos = np.maximum(out["P_demand"], 0)
    source_pos = sum(np.maximum(out[k], 0) for k in ["P_fc", "P_battery", "P_uc", "P_backup"])
    distance_km = np.trapezoid(v / 3600.0, t)
    traction_energy = np.trapezoid(demand_pos, t) / 3600.0
    fuel_energy = np.sum(np.maximum(out["P_fc"], 0) * dt) / 3600.0
    backup_energy = np.sum(np.maximum(out["P_backup"], 0) * dt) / 3600.0
    battery_throughput = np.sum(np.abs(out["P_battery"]) * dt) / 3600.0
    uc_throughput = np.sum(np.abs(out["P_uc"]) * dt) / 3600.0
    metrics = {
        "Cycle": CYCLE_LABELS[cycle],
        "Mode": label,
        "Failure": FAILURES[failure],
        "Distance (km)": distance_km,
        "Avg speed (km/h)": float(np.mean(v)),
        "Max speed (km/h)": float(np.max(v)),
        "Positive traction energy (kWh)": traction_energy,
        "Fuel-cell energy (kWh)": fuel_energy,
        "Backup energy (kWh)": backup_energy,
        "Battery throughput (kWh)": battery_throughput,
        "UC throughput (kWh)": uc_throughput,
        "Final battery SOC (%)": 100 * out["SOC"][-1],
        "Final UC voltage (V)": out["UC_voltage"][-1],
        "Final backup SOC (%)": 100 * out["backup_SOC"][-1],
        "Peak battery power (kW)": float(np.max(np.abs(out["P_battery"]))),
        "Peak UC power (kW)": float(np.max(np.abs(out["P_uc"]))),
        "Peak backup power (kW)": float(np.max(np.abs(out["P_backup"]))),
        "Max |power balance| (kW)": float(np.max(np.abs(out["balance_error"]))),
    }
    out.update({"t": t, "speed": v, "metrics": metrics})
    return out


@st.cache_resource(show_spinner=False)
def cached_ppo_result(cycle, failure, model_path_str):
    return simulate_ppo(cycle, failure, Path(model_path_str))


@st.cache_data(show_spinner=False)
def cached_baseline_result(cycle, failure):
    return simulate_baseline(cycle, failure)


def metric_table(results):
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append(m)
    return pd.DataFrame(rows)


def plot_power(result, title_prefix):
    t = result["t"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, result["P_fc"], linewidth=1.2)
    axes[0].set_ylabel("kW")
    axes[0].set_title("Fuel-cell power")
    axes[1].plot(t, result["P_battery"], linewidth=1.2)
    axes[1].set_ylabel("kW")
    axes[1].set_title("Battery power")
    axes[2].plot(t, result["P_uc"], linewidth=1.2)
    axes[2].set_ylabel("kW")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Ultracapacitor power")
    fig.suptitle(title_prefix + " — source power", fontsize=14)
    fig.tight_layout()
    return fig



def plot_power_split(result, title):
    """Show how the requested power is split across the four available sources."""
    t = result["t"]
    fig, ax = plt.subplots(figsize=(12, 4.5))

    p_fc = result["P_fc"]
    p_batt = result["P_battery"]
    p_uc = result["P_uc"]
    p_backup = result["P_backup"]

    # Positive source contribution is stacked; negative values are shown separately
    # as regenerative/charging power below the zero line.
    positive = [
        np.maximum(p_fc, 0),
        np.maximum(p_batt, 0),
        np.maximum(p_uc, 0),
        np.maximum(p_backup, 0),
    ]

    labels = ["Fuel cell", "Battery", "Ultracapacitor", "Backup battery"]
    ax.stackplot(t, positive, labels=labels, alpha=0.85)

    ax.plot(
        t,
        np.maximum(result["P_demand"], 0),
        linewidth=1.2,
        label="Positive power demand",
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(title + " — power split")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def plot_power_balance(result, title):
    """Show demand, total source power, and instantaneous power-balance error."""
    t = result["t"]
    demand = result["P_demand"]
    supplied = (
        result["P_fc"]
        + result["P_battery"]
        + result["P_uc"]
        + result["P_backup"]
    )
    error = demand - supplied

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    axes[0].plot(t, demand, linewidth=1.2, label="Vehicle power demand")
    axes[0].plot(t, supplied, linewidth=1.2, label="Total source power")
    axes[0].axhline(0, linewidth=0.8)
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title("Power demand vs total supplied power")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, error, linewidth=1.2, label="Power-balance error")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (kW)")
    axes[1].set_title("Instantaneous power-balance error")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    fig.suptitle(title + " — power balance", fontsize=14)
    fig.tight_layout()
    return fig

def plot_source_comparison(a, b, title):
    t = a["t"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    pairs = [("P_fc", "Fuel cell"), ("P_battery", "Battery"), ("P_uc", "Ultracapacitor")]
    for ax, (key, label) in zip(axes, pairs):
        ax.plot(t, a[key], label="Without optimizer", linewidth=1.0)
        if b is not None:
            ax.plot(t, b[key], label="With PPO optimizer", linewidth=1.0)
        ax.set_ylabel("kW")
        ax.set_title(label)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_states(result, title):
    t = result["t"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, result["SOC"] * 100)
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_title("Traction battery SOC")
    axes[1].plot(t, result["UC_voltage"])
    axes[1].set_ylabel("V")
    axes[1].set_title("Ultracapacitor voltage")
    axes[2].plot(t, result["backup_SOC"] * 100)
    axes[2].set_ylabel("SOC (%)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Backup battery SOC")
    fig.suptitle(title + " — storage states", fontsize=14)
    fig.tight_layout()
    return fig


def plot_cycle_speed(result, title):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(result["t"], result["speed"], linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(title + " — driving cycle")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def comparison_bar(df, metric, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot = df.pivot(index="Cycle", columns="Mode", values=metric)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


st.title("⚡ Indian Urban Tri-Hybrid EV Dashboard")
st.caption("Chennai + Pune + Delhi | Fuel cell + battery + ultracapacitor + emergency backup")

model_path = ROOT / "ems_ppo_model.zip"

with st.sidebar:
    st.header("Controls")
    cycle_choice = st.selectbox(
        "Driving cycle",
        ["all"] + list(DEFAULT_CYCLE_FILES.keys()),
        format_func=lambda x: "All 3 Indian cycles" if x == "all" else CYCLE_LABELS[x],
    )
    failure = st.selectbox(
        "Operating condition",
        list(FAILURES.keys()),
        format_func=lambda x: FAILURES[x],
    )
    st.divider()
    st.write("**PPO model:**", "Found" if model_path.exists() else "Not found")
    st.write("**SB3 installed:**", "Yes" if SB3_AVAILABLE else "No")
    st.info("Train the PPO model first for the optimized curves. The dashboard can still show the non-optimized baseline without it.")

if cycle_choice == "all":
    st.header("All-three-cycle comparison")
    results = []
    for cycle in DEFAULT_CYCLE_FILES:
        base = cached_baseline_result(cycle, failure)
        results.append(base)
        if model_path.exists() and SB3_AVAILABLE:
            opt = cached_ppo_result(cycle, failure, str(model_path))
            if opt is not None:
                results.append(opt)

    df = metric_table(results)
    st.dataframe(df.round(4), use_container_width=True)

    st.subheader("Driving-cycle statistics")
    stats_cols = ["Cycle", "Distance (km)", "Avg speed (km/h)", "Max speed (km/h)", "Positive traction energy (kWh)"]
    stats = df[df["Mode"] == "Without optimizer"][stats_cols].copy()
    st.dataframe(stats.round(4), use_container_width=True, hide_index=True)

    if (df["Mode"] == "With PPO optimizer").any():
        st.subheader("Optimizer comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(comparison_bar(df, "Fuel-cell energy (kWh)", "Fuel-cell energy by cycle", "Energy (kWh)"), clear_figure=True)
        with c2:
            st.pyplot(comparison_bar(df, "Battery throughput (kWh)", "Battery throughput by cycle", "Throughput (kWh)"), clear_figure=True)
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(comparison_bar(df, "Backup energy (kWh)", "Backup energy by cycle", "Energy (kWh)"), clear_figure=True)
        with c4:
            st.pyplot(comparison_bar(df, "Max |power balance| (kW)", "Power-balance error by cycle", "Max absolute error (kW)"), clear_figure=True)

    st.subheader("Cycle profiles")
    tabs = st.tabs([CYCLE_LABELS[c] for c in DEFAULT_CYCLE_FILES])
    for tab, cycle in zip(tabs, DEFAULT_CYCLE_FILES):
        with tab:
            r = cached_baseline_result(cycle, failure)
            st.pyplot(plot_cycle_speed(r, CYCLE_LABELS[cycle]), clear_figure=True)

            st.pyplot(
                plot_power_split(r, CYCLE_LABELS[cycle] + " — baseline"),
                clear_figure=True,
            )
            st.pyplot(
                plot_power_balance(r, CYCLE_LABELS[cycle] + " — baseline"),
                clear_figure=True,
            )

else:
    st.header(CYCLE_LABELS[cycle_choice])
    base = cached_baseline_result(cycle_choice, failure)
    opt = cached_ppo_result(cycle_choice, failure, str(model_path)) if model_path.exists() and SB3_AVAILABLE else None

    st.pyplot(plot_cycle_speed(base, CYCLE_LABELS[cycle_choice]), clear_figure=True)

    m1, m2, m3, m4 = st.columns(4)
    bm = base["metrics"]
    m1.metric("Distance", f"{bm['Distance (km)']:.2f} km")
    m2.metric("Avg speed", f"{bm['Avg speed (km/h)']:.1f} km/h")
    m3.metric("Max speed", f"{bm['Max speed (km/h)']:.1f} km/h")
    m4.metric("Traction energy", f"{bm['Positive traction energy (kWh)']:.3f} kWh")

    st.subheader("Power supplied by the three primary sources")
    if opt is not None:
        st.pyplot(plot_source_comparison(base, opt, CYCLE_LABELS[cycle_choice]), clear_figure=True)
    else:
        st.pyplot(plot_power(base, CYCLE_LABELS[cycle_choice]), clear_figure=True)

    # Additional energy-management views.
    st.subheader("Power split")
    if opt is not None:
        st.pyplot(
            plot_power_split(opt, CYCLE_LABELS[cycle_choice] + " — PPO"),
            clear_figure=True,
        )
    else:
        st.pyplot(
            plot_power_split(base, CYCLE_LABELS[cycle_choice] + " — baseline"),
            clear_figure=True,
        )

    st.subheader("Power balance")
    if opt is not None:
        st.pyplot(
            plot_power_balance(opt, CYCLE_LABELS[cycle_choice] + " — PPO"),
            clear_figure=True,
        )
    else:
        st.pyplot(
            plot_power_balance(base, CYCLE_LABELS[cycle_choice] + " — baseline"),
            clear_figure=True,
        )

    st.subheader("Storage states")
    st.pyplot(plot_states(opt if opt is not None else base, CYCLE_LABELS[cycle_choice]), clear_figure=True)

    st.subheader("With vs without optimizer")
    comp = pd.DataFrame([bm])
    if opt is not None:
        om = opt["metrics"]
        compare = pd.DataFrame([bm, om], index=["Without optimizer", "With PPO optimizer"])
        selected = [
            "Fuel-cell energy (kWh)", "Backup energy (kWh)",
            "Battery throughput (kWh)", "Peak battery power (kW)",
            "Peak UC power (kW)", "Max |power balance| (kW)",
            "Final battery SOC (%)", "Final UC voltage (V)",
            "Final backup SOC (%)",
        ]
        st.dataframe(compare[selected].round(4), use_container_width=True)
    else:
        st.warning("PPO model not available. Showing the non-optimized baseline only.")

    st.subheader("Detailed statistics")
    st.dataframe(pd.DataFrame([bm]).T.rename(columns={0: "Value"}).round(5), use_container_width=True)

    # Download the selected result as CSV for reporting.
    report = opt if opt is not None else base
    export_df = pd.DataFrame({
        "Time_s": report["t"],
        "Speed_kmph": report["speed"],
        "P_demand_kW": report["P_demand"],
        "P_fuel_cell_kW": report["P_fc"],
        "P_battery_kW": report["P_battery"],
        "P_ultracapacitor_kW": report["P_uc"],
        "P_backup_kW": report["P_backup"],
        "Battery_SOC": report["SOC"],
        "UC_voltage_V": report["UC_voltage"],
        "Backup_SOC": report["backup_SOC"],
        "Power_balance_error_kW": report["balance_error"],
    })
    st.download_button(
        "Download selected case CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{cycle_choice}_{failure}_dashboard_results.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Note: Delhi DMDC is a reconstructed trace in this project, not the authors' raw machine-readable trace. Replace it with the original raw cycle if available.")
