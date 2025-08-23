function [AT, V, AP, RH] = environmentalConditionsBlock(time_input)
%ENVIRONMENTALCONDITIONSBLOCK Generate realistic environmental conditions
%#codegen
%
% This Simulink block generates realistic time-varying environmental
% conditions for Combined Cycle Power Plant simulation based on:
% - Daily temperature cycles
% - Weather pattern variations  
% - Seasonal effects
% - Industrial site characteristics
%
% INPUTS:
%   time_input - Current simulation time (seconds)
%
% OUTPUTS:
%   AT - Ambient Temperature (°C)
%   V  - Vacuum Pressure (cm Hg)
%   AP - Atmospheric Pressure (mbar)
%   RH - Relative Humidity (%)

%% Initialize persistent variables for consistency
persistent base_conditions noise_states initialized;

if isempty(initialized)
    % Base environmental conditions (typical for industrial CCPP site)
    base_conditions = struct();
    base_conditions.temp_base = 15;        % °C - annual average
    base_conditions.vacuum_base = 50;      % cm Hg - typical condenser vacuum
    base_conditions.pressure_base = 1013;  % mbar - sea level standard
    base_conditions.humidity_base = 65;    % % - typical industrial site
    
    % Initialize noise states for smooth variations
    noise_states = struct();
    noise_states.temp_noise = 0;
    noise_states.vacuum_noise = 0;
    noise_states.pressure_noise = 0;
    noise_states.humidity_noise = 0;
    
    initialized = true;
end

%% Ensure time_input is valid
if nargin < 1 || isempty(time_input) || ~isscalar(time_input)
    time_input = 0;
end

% Convert to positive time for calculations
time_hours = max(0, time_input) / 3600;  % Convert seconds to hours

%% Daily Temperature Cycle
% Sinusoidal daily temperature variation with peak at 2 PM (14:00)
daily_temp_amplitude = 8;  % °C - typical daily swing
daily_phase = 2 * pi * (time_hours - 14) / 24;  % Phase shift for 2 PM peak
daily_temp_variation = daily_temp_amplitude * cos(daily_phase);

% Long-term temperature drift (simulates weather fronts)
long_term_period = 72;  % hours - 3-day weather cycle
long_term_amplitude = 4;  % °C
long_term_variation = long_term_amplitude * sin(2 * pi * time_hours / long_term_period);

% Calculate ambient temperature
AT = base_conditions.temp_base + daily_temp_variation + long_term_variation;

% Add small random variations (filtered noise)
temp_noise_factor = 0.98;  % Noise filtering
noise_states.temp_noise = temp_noise_factor * noise_states.temp_noise + ...
                          (1 - temp_noise_factor) * randn() * 1.5;
AT = AT + noise_states.temp_noise;

% Bound temperature to realistic range
AT = max(-5, min(40, AT));

%% Vacuum Pressure (inversely related to ambient temperature)
% Better vacuum (lower pressure) in cooler conditions
temp_effect_vacuum = -(AT - base_conditions.temp_base) * 0.3;  % cm Hg per °C
V = base_conditions.vacuum_base + temp_effect_vacuum;

% Add atmospheric pressure influence
pressure_influence = (1013 - base_conditions.pressure_base) * 0.02;  % Slight coupling
V = V + pressure_influence;

% Add smooth noise
vacuum_noise_factor = 0.95;
noise_states.vacuum_noise = vacuum_noise_factor * noise_states.vacuum_noise + ...
                           (1 - vacuum_noise_factor) * randn() * 1.0;
V = V + noise_states.vacuum_noise;

% Bound vacuum to realistic condenser operating range
V = max(25, min(75, V));

%% Atmospheric Pressure 
% Simulate weather system pressure changes
pressure_period = 48;  % hours - 2-day pressure cycle
pressure_amplitude = 8;  % mbar
pressure_cycle = pressure_amplitude * sin(2 * pi * time_hours / pressure_period);

AP = base_conditions.pressure_base + pressure_cycle;

% Add high-frequency pressure variations (wind/turbulence)
pressure_noise_factor = 0.92;
noise_states.pressure_noise = pressure_noise_factor * noise_states.pressure_noise + ...
                             (1 - pressure_noise_factor) * randn() * 2.0;
AP = AP + noise_states.pressure_noise;

% Bound to realistic atmospheric range
AP = max(995, min(1030, AP));

%% Relative Humidity
% Humidity typically inversely related to temperature
temp_humidity_effect = -(AT - base_conditions.temp_base) * 1.2;  % % per °C
RH = base_conditions.humidity_base + temp_humidity_effect;

% Daily humidity cycle (typically highest at dawn, lowest at mid-day)
humidity_daily_amplitude = 15;  % %
humidity_phase = 2 * pi * (time_hours - 6) / 24;  % Phase for 6 AM peak
daily_humidity_variation = humidity_daily_amplitude * cos(humidity_phase);
RH = RH + daily_humidity_variation;

% Weather front humidity changes
humidity_weather_period = 60;  % hours
humidity_weather_amplitude = 8;  % %
weather_humidity = humidity_weather_amplitude * cos(2 * pi * time_hours / humidity_weather_period);
RH = RH + weather_humidity;

% Add noise
humidity_noise_factor = 0.90;
noise_states.humidity_noise = humidity_noise_factor * noise_states.humidity_noise + ...
                             (1 - humidity_noise_factor) * randn() * 3.0;
RH = RH + noise_states.humidity_noise;

% Bound humidity to physical limits
RH = max(25, min(95, RH));

%% Output validation
% Ensure all outputs are finite and within bounds
if ~isfinite(AT), AT = base_conditions.temp_base; end
if ~isfinite(V), V = base_conditions.vacuum_base; end  
if ~isfinite(AP), AP = base_conditions.pressure_base; end
if ~isfinite(RH), RH = base_conditions.humidity_base; end

end