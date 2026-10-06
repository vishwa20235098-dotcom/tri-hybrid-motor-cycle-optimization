function data = vehicleDynamics(t, speed_kmph, vehicle)

%% ============================================================
% VEHICLE DYNAMICS
%
% Calculates:
%   Speed
%   Acceleration
%   Inertial force
%   Aerodynamic drag
%   Rolling resistance
%   Total traction force
%   Power demand
%
% =============================================================

%% Convert speed

speed = speed_kmph / 3.6;       % m/s

%% Acceleration

acceleration = gradient(speed, t);

%% Vehicle parameters

m   = vehicle.mass;
rho = vehicle.rho;
Cd  = vehicle.Cd;
A   = vehicle.A;
Cr  = vehicle.Cr;
g   = vehicle.g;

%% ------------------------------------------------------------
% 1. Inertial force
% -------------------------------------------------------------

F_inertia = m .* acceleration;

%% ------------------------------------------------------------
% 2. Aerodynamic drag
% -------------------------------------------------------------

F_drag = 0.5 * rho * Cd * A .* speed.^2;

%% ------------------------------------------------------------
% 3. Rolling resistance
% -------------------------------------------------------------

F_roll = m * g * Cr * ones(size(speed));

%% ------------------------------------------------------------
% 4. Total traction force
% -------------------------------------------------------------

F_traction = ...
    F_inertia + ...
    F_drag + ...
    F_roll;

%% ------------------------------------------------------------
% 5. Mechanical power demand
% -------------------------------------------------------------

P_demand = F_traction .* speed;

%% Convert W to kW

P_demand = P_demand / 1000;

%% Prevent extremely small numerical values

P_demand(abs(P_demand) < 1e-8) = 0;

%% Store results

data.speed_ms = speed;

data.speed_kmph = speed_kmph;

data.acceleration = acceleration;

data.F_inertia = F_inertia;

data.F_drag = F_drag;

data.F_roll = F_roll;

data.F_traction = F_traction;

data.P_demand = P_demand;

end