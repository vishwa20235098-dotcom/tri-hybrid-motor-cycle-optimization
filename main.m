%% ============================================================
%  TRI-HYBRID MOTORBIKE MODEL
%  Fuel Cell + Battery + Ultracapacitor
%
%  Version 1:
%  - No APSO
%  - No ML optimization
%  - Rule-based Energy Management
%
%  ============================================================

clear;
clc;
close all;

fprintf('====================================================\n');
fprintf('       TRI-HYBRID MOTORBIKE SIMULATION\n');
fprintf('====================================================\n\n');

%% ------------------------------------------------------------
% 1. VEHICLE PARAMETERS
% -------------------------------------------------------------

vehicle.mass = 150;          % kg
vehicle.rho  = 1.2;          % kg/m^3
vehicle.Cd   = 0.7;          % aerodynamic drag coefficient
vehicle.A    = 0.5;          % frontal area, m^2
vehicle.Cr   = 0.01;         % rolling resistance coefficient
vehicle.g    = 9.81;         % m/s^2

%% ------------------------------------------------------------
% 2. FUEL CELL PARAMETERS
% -------------------------------------------------------------

fc.Pmax = 3.0;               % kW
fc.Pmin = 0.0;               % kW
fc.efficiency = 0.55;        % 55%

%% ------------------------------------------------------------
% 3. BATTERY PARAMETERS
% -------------------------------------------------------------

battery.Vnom = 50;           % V
battery.capacity_Ah = 2.0;   % Ah

battery.Pmax_discharge = 1.0;    % kW
battery.Pmax_charge    = 1.0;    % kW

battery.SOC_initial = 0.60;
battery.SOC_min     = 0.40;
battery.SOC_max     = 0.80;

battery.eff_charge    = 0.95;
battery.eff_discharge = 0.95;

%% ------------------------------------------------------------
% 4. ULTRACAPACITOR PARAMETERS
% -------------------------------------------------------------

uc.Vnom = 60;                 % V
uc.Vmin = 40;                 % V
uc.Vmax = 60;                 % V

uc.R = 0.038;                 % Ohm

% The paper does not specify capacitance.
% Therefore this is an explicit modelling assumption.
uc.C = 100;                   % Farads

uc.Pmax_discharge = 2.5;      % kW
uc.Pmax_charge    = 2.5;      % kW

uc.efficiency = 0.98;

uc.V_initial = uc.Vnom;

%% ------------------------------------------------------------
% 5. LOAD DRIVING CYCLE
% -------------------------------------------------------------

fprintf('Loading CMDC driving cycle...\n');

[t, speed_kmph] = loadDrivingCycle('CMDC.csv');

fprintf('Driving cycle loaded.\n');
fprintf('Duration : %.2f s\n', t(end));
fprintf('Distance : %.3f km\n', ...
    trapz(t, speed_kmph / 3600));

%% ------------------------------------------------------------
% 6. VEHICLE DYNAMICS
% -------------------------------------------------------------

fprintf('\nCalculating vehicle dynamics...\n');

vehicleData = vehicleDynamics(t, speed_kmph, vehicle);

P_demand = vehicleData.P_demand;

fprintf('Maximum power demand : %.3f kW\n', max(P_demand));
fprintf('Minimum power demand : %.3f kW\n', min(P_demand));

%% ------------------------------------------------------------
% 7. RULE-BASED ENERGY MANAGEMENT
% -------------------------------------------------------------

fprintf('\nRunning rule-based energy management...\n');

results = ruleBasedEMS( ...
    t, ...
    P_demand, ...
    fc, ...
    battery, ...
    uc);

%% ------------------------------------------------------------
% 8. PERFORMANCE CALCULATION
% -------------------------------------------------------------

fprintf('\nCalculating performance...\n');

performance = calculatePerformance( ...
    t, ...
    speed_kmph, ...
    P_demand, ...
    results, ...
    vehicle);

%% ------------------------------------------------------------
% 9. DISPLAY RESULTS
% -------------------------------------------------------------

fprintf('\n');
fprintf('====================================================\n');
fprintf('                 SIMULATION RESULTS\n');
fprintf('====================================================\n');

fprintf('Distance                  : %.3f km\n', ...
    performance.distance_km);

fprintf('Maximum speed             : %.2f km/h\n', ...
    performance.max_speed_kmph);

fprintf('Average speed             : %.2f km/h\n', ...
    performance.average_speed_kmph);

fprintf('\n');

fprintf('Total FC energy           : %.6f kWh\n', ...
    performance.E_fc);

fprintf('Total battery energy      : %.6f kWh\n', ...
    performance.E_battery);

fprintf('Total UC energy           : %.6f kWh\n', ...
    performance.E_uc);

fprintf('Total input energy        : %.6f kWh\n', ...
    performance.E_total);

fprintf('\n');

fprintf('Energy consumption        : %.6f kWh/km\n', ...
    performance.energy_consumption);

fprintf('\n');

fprintf('Initial battery SOC       : %.2f %%\n', ...
    performance.initial_SOC * 100);

fprintf('Final battery SOC         : %.2f %%\n', ...
    performance.final_SOC * 100);

fprintf('Minimum battery SOC       : %.2f %%\n', ...
    performance.minimum_SOC * 100);

fprintf('Maximum battery SOC       : %.2f %%\n', ...
    performance.maximum_SOC * 100);

fprintf('\n');

fprintf('Initial UC voltage        : %.2f V\n', ...
    performance.initial_UC_voltage);

fprintf('Final UC voltage          : %.2f V\n', ...
    performance.final_UC_voltage);

fprintf('Minimum UC voltage        : %.2f V\n', ...
    performance.minimum_UC_voltage);

fprintf('\n');

fprintf('Maximum power balance error : %.10f kW\n', ...
    performance.max_power_balance_error);

fprintf('====================================================\n');

%% ------------------------------------------------------------
% 10. PLOTS
% -------------------------------------------------------------

figure('Name','Driving Cycle');

plot(t, speed_kmph, 'LineWidth', 1.5);
grid on;

xlabel('Time (s)');
ylabel('Speed (km/h)');
title('CMDC Driving Cycle');


figure('Name','Power Demand');

plot(t, P_demand, 'LineWidth', 1.5);
grid on;

xlabel('Time (s)');
ylabel('Power (kW)');
title('Motor Power Demand');


figure('Name','Power Split');

plot(t, results.P_fc, 'LineWidth', 1.5);
hold on;

plot(t, results.P_battery, 'LineWidth', 1.5);
plot(t, results.P_uc, 'LineWidth', 1.5);

grid on;

xlabel('Time (s)');
ylabel('Power (kW)');

title('Tri-Hybrid Power Split');

legend( ...
    'Fuel Cell', ...
    'Battery', ...
    'Ultracapacitor', ...
    'Location','best');


figure('Name','Battery SOC');

plot(t, results.SOC * 100, 'LineWidth', 1.5);
grid on;

xlabel('Time (s)');
ylabel('SOC (%)');

title('Battery State of Charge');

yline(battery.SOC_min * 100, '--');
yline(battery.SOC_max * 100, '--');


figure('Name','Ultracapacitor Voltage');

plot(t, results.UC_voltage, 'LineWidth', 1.5);
grid on;

xlabel('Time (s)');
ylabel('Voltage (V)');

title('Ultracapacitor Voltage');

yline(uc.Vmin, '--');
yline(uc.Vmax, '--');


figure('Name','Power Balance');

plot(t, P_demand, 'LineWidth', 1.5);
hold on;

plot(t, ...
    results.P_fc + ...
    results.P_battery + ...
    results.P_uc, ...
    '--', ...
    'LineWidth', 1.5);

grid on;

xlabel('Time (s)');
ylabel('Power (kW)');

title('Power Balance Verification');

legend( ...
    'Power Demand', ...
    'FC + Battery + UC', ...
    'Location','best');


figure('Name','Acceleration');

plot(t, vehicleData.acceleration, 'LineWidth', 1.5);
grid on;

xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

title('Vehicle Acceleration');

fprintf('\nSimulation completed successfully.\n');