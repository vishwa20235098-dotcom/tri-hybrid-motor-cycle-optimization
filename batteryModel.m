function [battery, P_actual] = batteryModel( ...
    P_command, ...
    dt, ...
    battery)

%% ============================================================
% BATTERY MODEL
%
% Convention:
%
% P_command > 0  -> battery discharging
% P_command < 0  -> battery charging
%
% SOC decreases during discharge.
% SOC increases during charging.
%
% =============================================================

%% Battery energy in kWh

E_battery = ...
    battery.Vnom * battery.capacity_Ah / 1000;

%% Requested power

P_requested = P_command;

%% ------------------------------------------------------------
% DISCHARGING
% -------------------------------------------------------------

if P_requested >= 0

    %% Respect power limit

    P_actual = min( ...
        P_requested, ...
        battery.Pmax_discharge);

    %% SOC change

    E_out = ...
        P_actual * dt / 3600;

    E_from_battery = ...
        E_out / battery.eff_discharge;

    delta_SOC = ...
        E_from_battery / E_battery;

    new_SOC = ...
        battery.SOC - delta_SOC;

    %% SOC protection

    if new_SOC < battery.SOC_min

        allowed_energy = ...
            (battery.SOC - battery.SOC_min) ...
            * E_battery;

        allowed_output_energy = ...
            allowed_energy ...
            * battery.eff_discharge;

        P_actual = ...
            allowed_output_energy * 3600 / dt;

        P_actual = max(P_actual, 0);

        %% Recalculate SOC

        E_out = P_actual * dt / 3600;

        E_from_battery = ...
            E_out / battery.eff_discharge;

        delta_SOC = ...
            E_from_battery / E_battery;

        new_SOC = battery.SOC - delta_SOC;
    end

%% ------------------------------------------------------------
% CHARGING
% -------------------------------------------------------------

else

    P_charge_requested = abs(P_requested);

    %% Respect charge limit

    P_actual = ...
        -min( ...
            P_charge_requested, ...
            battery.Pmax_charge);

    %% Energy entering battery

    E_charge = ...
        abs(P_actual) * ...
        battery.eff_charge * ...
        dt / 3600;

    delta_SOC = ...
        E_charge / E_battery;

    new_SOC = ...
        battery.SOC + delta_SOC;

    %% SOC protection

    if new_SOC > battery.SOC_max

        allowed_energy = ...
            (battery.SOC_max - battery.SOC) ...
            * E_battery;

        P_actual = ...
            -allowed_energy * 3600 / ...
            (battery.eff_charge * dt);

        P_actual = ...
            -min(abs(P_actual), ...
            battery.Pmax_charge);

        %% Recalculate SOC

        E_charge = ...
            abs(P_actual) * ...
            battery.eff_charge * ...
            dt / 3600;

        delta_SOC = ...
            E_charge / E_battery;

        new_SOC = battery.SOC + delta_SOC;
    end

end

%% Update SOC

battery.SOC = ...
    max(battery.SOC_min, ...
    min(battery.SOC_max, new_SOC));

end