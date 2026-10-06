"""
export_results.py

Runs the (jointly-trained, multi-cycle) RL policy deterministically over a
CHOSEN driving cycle and saves a .mat file whose field names match the
`results` struct your ruleBasedEMS.m already produces (P_fc, P_battery,
P_uc, SOC, UC_voltage, power_balance_error) -- plus t and speed_kmph.

main_rl.m loads this .mat directly (picking the file for whichever cycle
you select) and reuses calculatePerformance.m / the existing plotting
code unchanged.

Usage:
    python export_results.py --model ems_ppo_model.zip --cycle cmdc
    python export_results.py --model ems_ppo_model.zip --cycle pune
    python export_results.py --model ems_ppo_model.zip --cycle delhi --failure fuel_cell
"""

import argparse
import numpy as np
from scipy.io import savemat
from stable_baselines3 import PPO

from env import TriHybridEnv
from cycles import load_cycles, DEFAULT_CYCLE_FILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="ems_ppo_model.zip")
    ap.add_argument("--cycle", choices=list(DEFAULT_CYCLE_FILES), default="cmdc",
                     help="Which cycle to evaluate/export results for")
    ap.add_argument("--failure", choices=["none","fuel_cell","ultracapacitor"], default="none")
    ap.add_argument("--out", type=str, default=None,
                     help="Output .mat path (default: rl_results_<cycle>.mat)")
    args = ap.parse_args()
    out_path = args.out or f"rl_results_{args.cycle}_{args.failure}.mat"

    # Build the env with ALL cycles registered (so obs normalization /
    # action mapping matches what the model was trained with) but force
    # evaluation on the requested one via reset(options=...).
    cycles = load_cycles(names=list(DEFAULT_CYCLE_FILES.keys()))
    env = TriHybridEnv(cycles)
    model = PPO.load(args.model)

    obs, info = env.reset(options={"cycle": args.cycle, "failure": args.failure})
    t = env.t
    N = env.N
    P_fc = np.zeros(N)
    P_battery = np.zeros(N)
    P_uc = np.zeros(N)
    P_backup = np.zeros(N)
    backup_SOC = np.zeros(N)
    SOC = np.zeros(N)
    UC_voltage = np.zeros(N)
    balance_error = np.zeros(N)

    SOC[0] = env.battery["SOC"]
    UC_voltage[0] = env.uc["V_current"]
    backup_SOC[0] = env.backup["SOC"]

    for k in range(N - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)

        P_fc[k] = step_info["P_fc"]
        P_battery[k] = step_info["P_battery"]
        P_uc[k] = step_info["P_uc"]
        P_backup[k] = step_info["P_backup"]
        backup_SOC[k + 1] = step_info["backup_SOC"]
        SOC[k + 1] = step_info["SOC"]
        UC_voltage[k + 1] = step_info["UC_voltage"]
        balance_error[k] = step_info["balance_error"]

        if terminated or truncated:
            break

    v = env.cur["v"]
    savemat(out_path, {
        "t": t.reshape(-1, 1),
        "speed_kmph": v.reshape(-1, 1),
        "P_fc": P_fc.reshape(-1, 1),
        "P_battery": P_battery.reshape(-1, 1),
        "P_uc": P_uc.reshape(-1, 1),
        "P_backup": P_backup.reshape(-1, 1),
        "backup_SOC": backup_SOC.reshape(-1, 1),
        "SOC": SOC.reshape(-1, 1),
        "UC_voltage": UC_voltage.reshape(-1, 1),
        "power_balance_error": balance_error.reshape(-1, 1),
    })

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    distance_km = _trapz(v / 3.6, t) / 1000.0
    E_total = np.sum((np.maximum(P_fc, 0) + np.maximum(P_battery, 0)
                       + np.maximum(P_uc, 0)) * dt) / 3600.0
    print(f"Cycle               : {args.cycle}")
    print(f"Distance            : {distance_km:.3f} km")
    print(f"Energy consumption  : {E_total/distance_km:.6f} kWh/km")
    print(f"Final SOC           : {SOC[-1]*100:.2f} %")
    print(f"Max |balance error| : {np.max(np.abs(balance_error)):.6f} kW")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
