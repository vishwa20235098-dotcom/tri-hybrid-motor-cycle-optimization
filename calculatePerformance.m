function performance = calculatePerformance( ...
    t, ...
    speed_kmph, ...
    P_demand, ...
    results, ...
    vehicle)

%% ============================================================
% PERFORMANCE CALCULATION
% =============================================================

%% Time step

dt = zeros(size(t));

dt(1) = t(2) - t(1);

for k = 2:length(t)
    dt(k) = t(k) - t(k-1);
end

%% ------------------------------------------------------------
% Distance
% -------------------------------------------------------------

speed_ms = speed_kmph / 3.6;

distance_m = trapz(t, speed_ms);

distance_km = distance_m / 1000;

%% ------------------------------------------------------------
% Average speed
% -------------------------------------------------------------

average_speed_kmph = ...
    distance_km / (t(end) / 3600);

%% ------------------------------------------------------------
% Maximum speed
% -------------------------------------------------------------

max_speed_kmph = max(speed_kmph);

%% ------------------------------------------------------------
% Source energies
% -------------------------------------------------------------

%% Fuel cell

E_fc = ...
    sum(results.P_fc .* dt) / 3600;

%% Battery

% Only positive battery power is treated as discharge energy
E_battery_discharge = ...
    sum(max(results.P_battery,0) .* dt) / 3600;

%% UC

E_uc_discharge = ...
    sum(max(results.P_uc,0) .* dt) / 3600;

%% Total supplied energy

E_total = ...
    E_fc + ...
    E_battery_discharge + ...
    E_uc_discharge;

%% ------------------------------------------------------------
% Energy consumption
% -------------------------------------------------------------

if distance_km > 0

    energy_consumption = ...
        E_total / distance_km;

else

    energy_consumption = NaN;

end

%% ------------------------------------------------------------
% SOC
% -------------------------------------------------------------

initial_SOC = results.SOC(1);

final_SOC = results.SOC(end);

minimum_SOC = min(results.SOC);

maximum_SOC = max(results.SOC);

%% ------------------------------------------------------------
% UC voltage
% -------------------------------------------------------------

initial_UC_voltage = ...
    results.UC_voltage(1);

final_UC_voltage = ...
    results.UC_voltage(end);

minimum_UC_voltage = ...
    min(results.UC_voltage);

maximum_UC_voltage = ...
    max(results.UC_voltage);

%% ------------------------------------------------------------
% Power balance
% -------------------------------------------------------------

max_power_balance_error = ...
    max(abs(results.power_balance_error));

%% ------------------------------------------------------------
% Average powers
% -------------------------------------------------------------

average_P_fc = mean(results.P_fc);

average_P_battery = mean(results.P_battery);

average_P_uc = mean(results.P_uc);

%% ------------------------------------------------------------
% Output
% -------------------------------------------------------------

performance.distance_km = distance_km;

performance.average_speed_kmph = ...
    average_speed_kmph;

performance.max_speed_kmph = ...
    max_speed_kmph;

performance.E_fc = E_fc;

performance.E_battery = ...
    E_battery_discharge;

performance.E_uc = ...
    E_uc_discharge;

performance.E_total = E_total;

performance.energy_consumption = ...
    energy_consumption;

performance.initial_SOC = initial_SOC;

performance.final_SOC = final_SOC;

performance.minimum_SOC = minimum_SOC;

performance.maximum_SOC = maximum_SOC;

performance.initial_UC_voltage = ...
    initial_UC_voltage;

performance.final_UC_voltage = ...
    final_UC_voltage;

performance.minimum_UC_voltage = ...
    minimum_UC_voltage;

performance.maximum_UC_voltage = ...
    maximum_UC_voltage;

performance.max_power_balance_error = ...
    max_power_balance_error;

performance.average_P_fc = ...
    average_P_fc;

performance.average_P_battery = ...
    average_P_battery;

performance.average_P_uc = ...
    average_P_uc;

end