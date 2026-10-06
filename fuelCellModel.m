function output = fuelCellModel(P_command, fc)

%% ============================================================
% FUEL CELL MODEL
%
% Input:
%   P_command -> requested FC power in kW
%
% Output:
%   P_fc       -> actual FC power
%   P_fuel     -> equivalent fuel/input power
%   efficiency
%
% =============================================================

%% Limit power

P_fc = max(P_command, fc.Pmin);

P_fc = min(P_fc, fc.Pmax);

%% Efficiency

eta = fc.efficiency;

%% Fuel/input power

if eta > 0
    P_fuel = P_fc / eta;
else
    P_fuel = 0;
end

%% Output

output.P_fc = P_fc;

output.P_fuel = P_fuel;

output.efficiency = eta;

end