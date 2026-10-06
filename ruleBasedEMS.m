function results = ruleBasedEMS( ...
    t, ...
    P_demand, ...
    fc, ...
    battery, ...
    uc)

%% ============================================================
% RULE-BASED ENERGY MANAGEMENT SYSTEM
%
% Initial strategy:
%
%     FC      -> base power
%     Battery -> medium/transient power
%     UC      -> fast transient power
%
% Nominal split:
%
%     60% FC
%     30% Battery
%     10% UC
%
% Power limits and SOC/voltage limits are enforced.
%
% =============================================================

N = length(t);

%% Initialize output arrays

P_fc = zeros(N,1);

P_battery = zeros(N,1);

P_uc = zeros(N,1);

P_balance_error = zeros(N,1);

SOC = zeros(N,1);

UC_voltage = zeros(N,1);

%% Initial states

battery.SOC = battery.SOC_initial;

uc.V_current = uc.V_initial;

SOC(1) = battery.SOC;

UC_voltage(1) = uc.V_current;

%% ------------------------------------------------------------
% Main simulation loop
% -------------------------------------------------------------

for k = 1:N

    %% Time step

    if k == 1
        dt = t(2) - t(1);
    else
        dt = t(k) - t(k-1);
    end

    %% Current power demand

    P_req = P_demand(k);

    %% --------------------------------------------------------
    % ZERO / VERY SMALL POWER
    % ---------------------------------------------------------

    if abs(P_req) < 1e-8

        P_fc_cmd = 0;
        P_bat_cmd = 0;
        P_uc_cmd = 0;

    %% --------------------------------------------------------
    % POSITIVE POWER DEMAND
    % ---------------------------------------------------------

    elseif P_req > 0

        %% Nominal rule-based allocation

        P_fc_cmd = 0.60 * P_req;

        P_bat_cmd = 0.30 * P_req;

        P_uc_cmd = 0.10 * P_req;

        %% ----------------------------------------------------
        % Fuel cell limit
        % -----------------------------------------------------

        if P_fc_cmd > fc.Pmax

            P_fc_cmd = fc.Pmax;
        end

        %% ----------------------------------------------------
        % Battery limit
        % -----------------------------------------------------

        if P_bat_cmd > battery.Pmax_discharge

            P_bat_cmd = battery.Pmax_discharge;
        end

        %% ----------------------------------------------------
        % Calculate remaining power
        % -----------------------------------------------------

        remaining = ...
            P_req - P_fc_cmd - P_bat_cmd;

        %% Give remaining power to UC

        P_uc_cmd = remaining;

        %% UC maximum limit

        if P_uc_cmd > uc.Pmax_discharge

            P_uc_cmd = uc.Pmax_discharge;
        end

        %% ----------------------------------------------------
        % If still not enough, increase FC
        % -----------------------------------------------------

        remaining = ...
            P_req - P_fc_cmd - P_bat_cmd - P_uc_cmd;

        if remaining > 0

            additional_fc = ...
                min(remaining, ...
                fc.Pmax - P_fc_cmd);

            P_fc_cmd = ...
                P_fc_cmd + additional_fc;

            remaining = ...
                P_req - ...
                P_fc_cmd - ...
                P_bat_cmd - ...
                P_uc_cmd;
        end

        %% ----------------------------------------------------
        % Then increase battery
        % -----------------------------------------------------

        if remaining > 0

            additional_bat = ...
                min(remaining, ...
                battery.Pmax_discharge - P_bat_cmd);

            P_bat_cmd = ...
                P_bat_cmd + additional_bat;

            remaining = ...
                P_req - ...
                P_fc_cmd - ...
                P_bat_cmd - ...
                P_uc_cmd;
        end

        %% ----------------------------------------------------
        % Finally UC
        % -----------------------------------------------------

        if remaining > 0

            additional_uc = ...
                min(remaining, ...
                uc.Pmax_discharge - P_uc_cmd);

            P_uc_cmd = ...
                P_uc_cmd + additional_uc;
        end

    %% --------------------------------------------------------
    % REGENERATIVE BRAKING
    % ---------------------------------------------------------

    else

        %% Negative power means regenerative braking

        regen_power = abs(P_req);

        %% First charge UC

        P_uc_cmd = ...
            -min(regen_power, ...
            uc.Pmax_charge);

        remaining_regen = ...
            regen_power - abs(P_uc_cmd);

        %% Then charge battery

        if remaining_regen > 0

            P_bat_cmd = ...
                -min( ...
                remaining_regen, ...
                battery.Pmax_charge);

            remaining_regen = ...
                remaining_regen - ...
                abs(P_bat_cmd);

        else

            P_bat_cmd = 0;

        end

        %% Remaining regenerative energy

        % We do not force the fuel cell to absorb energy.
        % Any unused regenerative energy is considered
        % mechanically dissipated through braking.

        P_fc_cmd = 0;

    end

    %% --------------------------------------------------------
    % FUEL CELL MODEL
    % ---------------------------------------------------------

    fc_out = fuelCellModel(P_fc_cmd, fc);

    P_fc_actual = fc_out.P_fc;

    %% --------------------------------------------------------
    % BATTERY MODEL
    % ---------------------------------------------------------

    [battery, P_bat_actual] = ...
        batteryModel( ...
        P_bat_cmd, ...
        dt, ...
        battery);

    %% --------------------------------------------------------
    % UC MODEL
    % ---------------------------------------------------------

    [uc, P_uc_actual] = ...
        ultracapacitorModel( ...
        P_uc_cmd, ...
        dt, ...
        uc);

    %% --------------------------------------------------------
    % STORE RESULTS
    % ---------------------------------------------------------

    P_fc(k) = P_fc_actual;

    P_battery(k) = P_bat_actual;

    P_uc(k) = P_uc_actual;

    %% Power balance

    P_balance_error(k) = ...
        P_req - ...
        (P_fc_actual + ...
         P_bat_actual + ...
         P_uc_actual);

    %% Store states

    SOC(k) = battery.SOC;

    UC_voltage(k) = uc.V_current;

end

%% ------------------------------------------------------------
% Output structure
% -------------------------------------------------------------

results.P_fc = P_fc;

results.P_battery = P_battery;

results.P_uc = P_uc;

results.SOC = SOC;

results.UC_voltage = UC_voltage;

results.power_balance_error = ...
    P_balance_error;

end