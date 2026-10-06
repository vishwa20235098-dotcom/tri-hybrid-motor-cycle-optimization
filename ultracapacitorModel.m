function [uc, P_actual] = ultracapacitorModel( ...
    P_command, ...
    dt, ...
    uc)

%% ============================================================
% ULTRACAPACITOR MODEL
%
% P_command > 0 -> discharge
% P_command < 0 -> charge
%
% Energy:
%
%       E = 1/2 C V^2
%
% =============================================================

%% Initial energy

E_initial = ...
    0.5 * uc.C * uc.V_current^2;

%% ------------------------------------------------------------
% DISCHARGING
% -------------------------------------------------------------

if P_command >= 0

    %% Respect maximum power

    P_actual = ...
        min(P_command, ...
        uc.Pmax_discharge);

    %% Available energy at minimum voltage

    E_min = ...
        0.5 * uc.C * uc.Vmin^2;

    E_available = ...
        E_initial - E_min;

    %% Energy required from UC

    E_required = ...
        P_actual * dt / 3600 * 1000;

    % Convert kWh -> Joules:
    % kWh * 3.6e6 = Joules

    E_required = ...
        P_actual * dt / 3600 * 3.6e6;

    %% Account for efficiency

    E_required_from_uc = ...
        E_required / uc.efficiency;

    %% Check available energy

    if E_required_from_uc > E_available

        E_delivered = ...
            E_available * uc.efficiency;

        P_actual = ...
            E_delivered / 3.6e6 * ...
            3600 / dt;

        P_actual = max(P_actual,0);

        E_required_from_uc = E_available;
    end

    %% New energy

    E_new = ...
        E_initial - E_required_from_uc;

%% ------------------------------------------------------------
% CHARGING
% -------------------------------------------------------------

else

    P_charge = abs(P_command);

    %% Respect charge limit

    P_actual = ...
        -min(P_charge, ...
        uc.Pmax_charge);

    %% Maximum UC energy

    E_max = ...
        0.5 * uc.C * uc.Vmax^2;

    E_available_capacity = ...
        E_max - E_initial;

    %% Energy entering UC

    E_input = ...
        abs(P_actual) * ...
        dt / 3600 * ...
        3.6e6;

    E_stored = ...
        E_input * uc.efficiency;

    %% Check capacity

    if E_stored > E_available_capacity

        E_stored = E_available_capacity;

        P_actual = ...
            -E_stored / ...
            (uc.efficiency * 3.6e6) ...
            * 3600 / dt;
    end

    %% New energy

    E_new = ...
        E_initial + E_stored;

end

%% ------------------------------------------------------------
% Calculate new voltage
% -------------------------------------------------------------

uc.V_current = ...
    sqrt(2 * E_new / uc.C);

%% Protection

uc.V_current = ...
    max(uc.Vmin, ...
    min(uc.Vmax, uc.V_current));

end