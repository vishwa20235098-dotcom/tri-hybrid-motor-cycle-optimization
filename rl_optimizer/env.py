"""Failure-aware tri-hybrid RL environment.

Sources: fuel cell + traction battery + ultracapacitor, with an emergency
backup battery. The same PPO policy can be trained across all registered
Indian driving cycles and across healthy / fuel-cell-failed / UC-failed cases.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from models import (make_vehicle, make_fuel_cell, make_battery,
                     make_ultracapacitor, vehicle_dynamics,
                     fuel_cell_model, battery_model, ultracapacitor_model)


class TriHybridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cycles, lookahead=5, failure_modes=None):
        super().__init__()
        self.vehicle = make_vehicle()
        self.fc_nom = make_fuel_cell()
        self.Pfc_max = self.fc_nom["Pmax"]
        self.Pbatt_max = make_battery()["Pmax_discharge"]
        self.Puc_max = make_ultracapacitor()["Pmax_discharge"]
        self.backup_max = 1.5
        self.backup_battery = dict(Vnom=50.0, capacity_Ah=3.0,
                                   Pmax_discharge=self.backup_max,
                                   Pmax_charge=0.75, SOC_initial=0.90,
                                   SOC_min=0.20, SOC_max=0.95,
                                   eff_charge=0.95, eff_discharge=0.95,
                                   SOC=0.90)
        self.lookahead = lookahead
        self.failure_modes = failure_modes or ["none", "fuel_cell", "ultracapacitor"]

        if isinstance(cycles, tuple) or (isinstance(cycles, list) and len(cycles) == 2
                                          and isinstance(cycles[0], np.ndarray)):
            cycles = [{"name": "cycle0", "t": cycles[0], "v": cycles[1]}]
        self.cycles = []
        for c in cycles:
            t = np.asarray(c["t"], dtype=np.float64)
            v = np.asarray(c["v"], dtype=np.float64)
            P_demand, _ = vehicle_dynamics(t, v, self.vehicle)
            self.cycles.append(dict(name=c.get("name", "cycle"), t=t, v=v,
                                    P_demand=P_demand, N=len(t),
                                    Pdemand_max=max(np.max(np.abs(P_demand)), 1e-6)))
        self._name_to_idx = {c["name"]: i for i, c in enumerate(self.cycles)}
        self._failure_to_idx = {m: i for i, m in enumerate(self.failure_modes)}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # demand + lookahead + SOC + UC voltage + backup SOC + 3 health flags
        obs_dim = 1 + lookahead + 2 + 1 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    @property
    def t(self): return self.cur["t"]
    @property
    def P_demand(self): return self.cur["P_demand"]
    @property
    def N(self): return self.cur["N"]
    @property
    def Pdemand_max(self): return self.cur["Pdemand_max"]

    def _get_obs(self, k):
        window = np.zeros(self.lookahead)
        end = min(k + 1 + self.lookahead, self.N)
        seg = self.P_demand[k+1:end] / self.Pdemand_max
        window[:len(seg)] = seg
        soc_norm = (self.battery["SOC"] - .4) / .4
        v_norm = (self.uc["V_current"] - 40) / 20
        b_soc = (self.backup["SOC"] - .2) / .75
        health = [self.fc_alive, self.uc_alive, self.backup_alive]
        return np.concatenate([[self.P_demand[k]/self.Pdemand_max], window,
                               [soc_norm, v_norm, b_soc], health]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cycle_name = options.get("cycle") if options else None
        failure = options.get("failure") if options else None
        idx = self._name_to_idx[cycle_name] if cycle_name else int(self.np_random.integers(len(self.cycles)))
        self.cur = self.cycles[idx]
        self.failure_mode = failure or self.failure_modes[int(self.np_random.integers(len(self.failure_modes)))]
        self.failure_index = int(0.50 * (self.N - 1)) if self.failure_mode != "none" else self.N + 1
        self.fc_alive = True
        self.uc_alive = True
        self.backup_alive = True
        self.k = 0
        self.battery = make_battery()
        self.uc = make_ultracapacitor(); self.uc["V_current"] = self.uc["V_initial"]
        self.backup = self.backup_battery.copy(); self.backup["SOC"] = self.backup["SOC_initial"]
        return self._get_obs(0), {"cycle": self.cur["name"], "failure": self.failure_mode}

    def _source_health_update(self):
        if self.k >= self.failure_index:
            if self.failure_mode == "fuel_cell": self.fc_alive = False
            if self.failure_mode == "ultracapacitor": self.uc_alive = False

    def step(self, action):
        k = self.k
        dt = (self.t[1]-self.t[0]) if k == 0 else (self.t[k]-self.t[k-1])
        P_req = self.P_demand[k]
        self._source_health_update()
        a_fc = (float(action[0])+1)/2
        a_batt = float(action[1])

        fc = self.fc_nom.copy() if self.fc_alive else dict(Pmax=0.0, Pmin=0.0, efficiency=.55)
        if abs(P_req) < 1e-8:
            P_fc_cmd = P_batt_cmd = 0.0
        elif P_req > 0:
            P_fc_cmd = np.clip(a_fc*P_req, 0, self.Pfc_max if self.fc_alive else 0)
            rem = P_req - P_fc_cmd
            P_batt_cmd = np.clip(a_batt*rem, -self.Pbatt_max, self.Pbatt_max)
        else:
            P_fc_cmd = 0.0
            P_batt_cmd = np.clip(a_batt*P_req, -self.Pbatt_max, self.Pbatt_max)

        P_fc, P_fuel = fuel_cell_model(P_fc_cmd, fc)
        P_batt, new_soc = battery_model(P_batt_cmd, dt, self.battery)
        self.battery["SOC"] = new_soc

        if self.uc_alive:
            P_uc_cmd = P_req - P_fc - P_batt
            P_uc_cmd = np.clip(P_uc_cmd, -self.Puc_max, self.Puc_max)
            P_uc, new_v = ultracapacitor_model(P_uc_cmd, dt, self.uc)
        else:
            P_uc, new_v = 0.0, self.uc["V_current"]
        self.uc["V_current"] = new_v

        P_supplied = P_fc + P_batt + P_uc
        P_backup_cmd = P_req - P_supplied
        if self.backup_alive:
            # backup is discharge-only during a failure/emergency
            P_backup_cmd = np.clip(P_backup_cmd, 0, self.backup["Pmax_discharge"])
            P_backup, b_soc = battery_model(P_backup_cmd, dt, self.backup)
            self.backup["SOC"] = b_soc
        else:
            P_backup = 0.0
        P_supplied += P_backup
        balance = P_req - P_supplied

        fuel_cost = P_fuel*dt/3600
        emergency_penalty = 0.03*max(P_backup, 0)*dt/3600
        balance_penalty = 15*abs(balance)
        soc_pen = max(0,.4-new_soc)+max(0,new_soc-.8)
        v_pen = max(0,40-new_v)+max(0,new_v-60)
        reward = -(fuel_cost + emergency_penalty + balance_penalty +
                   15*soc_pen + 10*v_pen + .02*abs(P_batt)*dt/3600)

        info = dict(P_fc=P_fc, P_battery=P_batt, P_uc=P_uc, P_backup=P_backup,
                    SOC=new_soc, UC_voltage=new_v, backup_SOC=self.backup["SOC"],
                    balance_error=balance, P_demand=P_req,
                    fc_alive=self.fc_alive, uc_alive=self.uc_alive,
                    failure=self.failure_mode)
        self.k += 1
        terminated = self.k >= self.N-1
        obs = self._get_obs(min(self.k, self.N-1))
        return obs, float(reward), terminated, False, info
