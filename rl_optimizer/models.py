"""
models.py

Faithful Python ports of:
  fuelCellModel.m
  batteryModel.m
  ultracapacitorModel.m
  vehicleDynamics.m

Same equations, same limits, same conventions (P_command > 0 = discharge).
Kept as plain dict-based state so it's trivial to line up against the .m
source when checking correctness.
"""

import numpy as np


def make_vehicle():
    return dict(mass=150.0, rho=1.2, Cd=0.7, A=0.5, Cr=0.01, g=9.81)


def make_fuel_cell():
    return dict(Pmax=3.0, Pmin=0.0, efficiency=0.55)


def make_battery():
    return dict(
        Vnom=50.0, capacity_Ah=2.0,
        Pmax_discharge=1.0, Pmax_charge=1.0,
        SOC_initial=0.60, SOC_min=0.40, SOC_max=0.80,
        eff_charge=0.95, eff_discharge=0.95,
        SOC=0.60,
    )


def make_ultracapacitor():
    return dict(
        Vnom=60.0, Vmin=40.0, Vmax=60.0, R=0.038, C=100.0,
        Pmax_discharge=2.5, Pmax_charge=2.5, efficiency=0.98,
        V_initial=60.0, V_current=60.0,
    )


def vehicle_dynamics(t, speed_kmph, vehicle):
    speed = speed_kmph / 3.6
    accel = np.gradient(speed, t)

    F_inertia = vehicle["mass"] * accel
    F_drag = 0.5 * vehicle["rho"] * vehicle["Cd"] * vehicle["A"] * speed ** 2
    F_roll = vehicle["mass"] * vehicle["g"] * vehicle["Cr"] * np.ones_like(speed)
    F_traction = F_inertia + F_drag + F_roll

    P_demand = F_traction * speed / 1000.0  # kW
    P_demand[np.abs(P_demand) < 1e-8] = 0.0
    return P_demand, accel


def fuel_cell_model(P_command, fc):
    P_fc = min(max(P_command, fc["Pmin"]), fc["Pmax"])
    eta = fc["efficiency"]
    P_fuel = P_fc / eta if eta > 0 else 0.0
    return P_fc, P_fuel


def battery_model(P_command, dt, battery):
    """Returns (P_actual, new_SOC). `battery` dict is not mutated."""
    E_battery = battery["Vnom"] * battery["capacity_Ah"] / 1000.0  # kWh
    SOC = battery["SOC"]

    if P_command >= 0:
        P_actual = min(P_command, battery["Pmax_discharge"])
        E_out = P_actual * dt / 3600.0
        E_from_batt = E_out / battery["eff_discharge"]
        delta_SOC = E_from_batt / E_battery
        new_SOC = SOC - delta_SOC

        if new_SOC < battery["SOC_min"]:
            allowed_energy = (SOC - battery["SOC_min"]) * E_battery
            allowed_out_energy = allowed_energy * battery["eff_discharge"]
            P_actual = max(allowed_out_energy * 3600.0 / dt, 0.0)
            E_out = P_actual * dt / 3600.0
            E_from_batt = E_out / battery["eff_discharge"]
            delta_SOC = E_from_batt / E_battery
            new_SOC = SOC - delta_SOC
    else:
        P_charge_req = abs(P_command)
        P_actual = -min(P_charge_req, battery["Pmax_charge"])
        E_charge = abs(P_actual) * battery["eff_charge"] * dt / 3600.0
        delta_SOC = E_charge / E_battery
        new_SOC = SOC + delta_SOC

        if new_SOC > battery["SOC_max"]:
            allowed_energy = (battery["SOC_max"] - SOC) * E_battery
            P_actual = -allowed_energy * 3600.0 / (battery["eff_charge"] * dt)
            P_actual = -min(abs(P_actual), battery["Pmax_charge"])
            E_charge = abs(P_actual) * battery["eff_charge"] * dt / 3600.0
            delta_SOC = E_charge / E_battery
            new_SOC = SOC + delta_SOC

    new_SOC = min(max(new_SOC, battery["SOC_min"]), battery["SOC_max"])
    return P_actual, new_SOC


def ultracapacitor_model(P_command, dt, uc):
    """Returns (P_actual, new_V). `uc` dict is not mutated."""
    V = uc["V_current"]
    E_initial = 0.5 * uc["C"] * V ** 2

    if P_command >= 0:
        P_actual = min(P_command, uc["Pmax_discharge"])
        E_min = 0.5 * uc["C"] * uc["Vmin"] ** 2
        E_available = E_initial - E_min

        E_required = P_actual * dt / 3600.0 * 3.6e6
        E_required_from_uc = E_required / uc["efficiency"]

        if E_required_from_uc > E_available:
            E_delivered = E_available * uc["efficiency"]
            P_actual = max(E_delivered / 3.6e6 * 3600.0 / dt, 0.0)
            E_required_from_uc = E_available

        E_new = E_initial - E_required_from_uc
    else:
        P_charge = abs(P_command)
        P_actual = -min(P_charge, uc["Pmax_charge"])
        E_max = 0.5 * uc["C"] * uc["Vmax"] ** 2
        E_avail_capacity = E_max - E_initial

        E_input = abs(P_actual) * dt / 3600.0 * 3.6e6
        E_stored = E_input * uc["efficiency"]

        if E_stored > E_avail_capacity:
            E_stored = E_avail_capacity
            P_actual = -E_stored / (uc["efficiency"] * 3.6e6) * 3600.0 / dt

        E_new = E_initial + E_stored

    new_V = np.sqrt(max(2.0 * E_new / uc["C"], 0.0))
    new_V = min(max(new_V, uc["Vmin"]), uc["Vmax"])
    return P_actual, new_V
