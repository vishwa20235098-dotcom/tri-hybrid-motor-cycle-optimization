function [t, speed_kmph] = loadDrivingCycle(filename)

%% ============================================================
% LOAD DRIVING CYCLE
%
% Input:
%   filename
%
% Output:
%   t           -> time in seconds
%   speed_kmph  -> speed in km/h
%
% =============================================================

if ~isfile(filename)
    error('Driving cycle file "%s" was not found.', filename);
end

%% Read CSV

data = readmatrix(filename);

if size(data,2) < 2
    error('CMDC.csv must contain at least two columns: Time and Speed.');
end

%% Extract data

t = data(:,1);
speed_kmph = data(:,2);

%% Remove invalid rows

valid = isfinite(t) & isfinite(speed_kmph);

t = t(valid);
speed_kmph = speed_kmph(valid);

%% Make sure time starts from zero

t = t - t(1);

%% Remove duplicate time points

[t, uniqueIndex] = unique(t);

speed_kmph = speed_kmph(uniqueIndex);

%% Prevent negative speed

speed_kmph(speed_kmph < 0) = 0;

%% Basic validation

if length(t) < 2
    error('Driving cycle must contain at least two data points.');
end

if any(diff(t) <= 0)
    error('Time values must be strictly increasing.');
end

fprintf('Number of samples : %d\n', length(t));

end