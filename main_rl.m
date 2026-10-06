%% TRI-HYBRID MOTORBIKE - FAILURE-AWARE RL EMS
% Indian urban driving cycles only.
% CYCLE: cmdc | pune | delhi
% FAILURE_MODE: none | fuel_cell | ultracapacitor
% The Python PPO policy is trained jointly across all registered Indian
% cycles and failure cases. MATLAB is used for dynamics/performance/plots.

clear; clc; close all;

CYCLE = 'cmdc';
FAILURE_MODE = 'none';

vehicle.mass = 150; vehicle.rho = 1.2; vehicle.Cd = 0.7;
vehicle.A = 0.5; vehicle.Cr = 0.01; vehicle.g = 9.81;
battery.SOC_min = 0.40; battery.SOC_max = 0.80;
uc.Vmin = 40; uc.Vmax = 60;

fprintf('====================================================\n');
fprintf(' FAILURE-AWARE TRI-HYBRID RL SIMULATION\n');
fprintf(' Indian cycle: %s | Failure: %s\n', upper(CYCLE), FAILURE_MODE);
fprintf('====================================================\n');

matFile = fullfile('rl_optimizer', sprintf('rl_results_%s_%s.mat', lower(CYCLE), lower(FAILURE_MODE)));
if ~isfile(matFile)
    error('Missing %s. Run export_results.py for this cycle/failure first.', matFile);
end
rl = load(matFile);
t = rl.t; speed_kmph = rl.speed_kmph;
results.P_fc = rl.P_fc; results.P_battery = rl.P_battery;
results.P_uc = rl.P_uc; results.P_backup = rl.P_backup;
results.SOC = rl.SOC; results.UC_voltage = rl.UC_voltage;
results.backup_SOC = rl.backup_SOC;
results.power_balance_error = rl.power_balance_error;

vehicleData = vehicleDynamics(t, speed_kmph, vehicle);
P_demand = vehicleData.P_demand;
performance = calculatePerformance(t, speed_kmph, P_demand, results, vehicle);

fprintf('Duration                 : %.1f s\n', t(end));
fprintf('Distance                 : %.3f km\n', performance.distance_km);
fprintf('Maximum speed            : %.2f km/h\n', performance.max_speed_kmph);
fprintf('FC energy                : %.5f kWh\n', performance.E_fc);
fprintf('Battery energy            : %.5f kWh\n', performance.E_battery);
fprintf('UC energy                 : %.5f kWh\n', performance.E_uc);
fprintf('Backup energy             : %.5f kWh\n', trapz(t,max(results.P_backup,0))/3600);
fprintf('Final battery SOC         : %.2f %%\n', 100*results.SOC(end));
fprintf('Final backup SOC          : %.2f %%\n', 100*results.backup_SOC(end));
fprintf('Maximum |power error|     : %.6f kW\n', max(abs(results.power_balance_error)));

%% Driving cycle
figure('Name','Indian Driving Cycle','Color','w');
plot(t,speed_kmph,'LineWidth',1.5); grid on;
xlabel('Time (s)'); ylabel('Speed (km/h)');
title(sprintf('%s Indian Urban Driving Cycle',upper(CYCLE)));

%% Three source plots on ONE PAGE
figure('Name','Power Supplied - Three Sources','Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
nexttile; plot(t,results.P_fc,'LineWidth',1.3); grid on;
ylabel('FC (kW)'); title('Fuel Cell Power');
nexttile; plot(t,results.P_battery,'LineWidth',1.3); grid on;
ylabel('Battery (kW)'); title('Battery Power');
nexttile; plot(t,results.P_uc,'LineWidth',1.3); grid on;
ylabel('UC (kW)'); xlabel('Time (s)'); title('Ultracapacitor Power');
sgtitle(sprintf('%s | Failure: %s',upper(CYCLE),strrep(FAILURE_MODE,'_',' ')));

%% Backup and total balance
figure('Name','Backup and Power Balance','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile; plot(t,results.P_backup,'LineWidth',1.4); grid on;
ylabel('Backup (kW)'); title('Emergency Backup Battery Power');
nexttile; plot(t,P_demand,'LineWidth',1.3); hold on;
plot(t,results.P_fc+results.P_battery+results.P_uc+results.P_backup,'--','LineWidth',1.2);
grid on; xlabel('Time (s)'); ylabel('Power (kW)');
legend('Motor demand','Total supplied','Location','best'); title('Power Balance');

%% State variables
figure('Name','Energy Storage States','Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
nexttile; plot(t,100*results.SOC,'LineWidth',1.3); grid on;
ylabel('Battery SOC (%)'); yline(40,'--'); yline(80,'--'); title('Traction Battery SOC');
nexttile; plot(t,results.UC_voltage,'LineWidth',1.3); grid on;
ylabel('UC Voltage (V)'); yline(40,'--'); yline(60,'--'); title('Ultracapacitor Voltage');
nexttile; plot(t,100*results.backup_SOC,'LineWidth',1.3); grid on;
ylabel('Backup SOC (%)'); xlabel('Time (s)'); title('Backup Battery SOC');

fprintf('\nPlots generated. Change CYCLE and FAILURE_MODE at the top of main_rl.m to switch cases.\n');
