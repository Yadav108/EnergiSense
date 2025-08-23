function [iot_status, alarm_state, maintenance_alert, data_quality] = industrialIoTBlock(power_output, control_signal, environmental_data, performance_metrics)
%INDUSTRIALIOTBLOCK Industrial IoT monitoring and alerting system
%#codegen
%
% This Simulink block simulates industrial IoT capabilities including:
% - Real-time data monitoring and validation
% - Predictive maintenance alerts
% - System health monitoring
% - Data quality assessment
% - Industrial protocol simulation (Modbus, OPC-UA, etc.)
%
% INPUTS:
%   power_output - Current electrical power output (MW)
%   control_signal - Controller output signal
%   environmental_data - [AT, V, AP, RH] environmental conditions  
%   performance_metrics - System performance indicators
%
% OUTPUTS:
%   iot_status - IoT system status (0=offline, 1=degraded, 2=optimal)
%   alarm_state - Active alarm level (0=none, 1=warning, 2=critical)
%   maintenance_alert - Maintenance recommendation (0=none, 1=scheduled, 2=urgent)
%   data_quality - Overall data quality score (0-100%)

%% Input validation and defaults
if nargin < 4
    iot_status = 0;
    alarm_state = 0; 
    maintenance_alert = 0;
    data_quality = 0;
    return;
end

% Ensure inputs are properly sized
if ~isscalar(power_output), power_output = 450; end
if ~isscalar(control_signal), control_signal = 0; end
if length(environmental_data) < 4, environmental_data = [15, 50, 1013, 65]; end
if length(performance_metrics) < 3, performance_metrics = [0, 95, 1]; end

%% Persistent variables for IoT system state
persistent iot_history alarm_counters maintenance_timers system_health;
persistent data_buffer communication_status protocol_errors;

% Initialize on first call
if isempty(iot_history)
    iot_history = struct();
    iot_history.power_trend = zeros(1, 20);
    iot_history.control_trend = zeros(1, 20);
    iot_history.quality_trend = zeros(1, 20);
    
    alarm_counters = struct();
    alarm_counters.power_deviation = 0;
    alarm_counters.control_saturation = 0;
    alarm_counters.environmental_extreme = 0;
    alarm_counters.performance_degradation = 0;
    
    maintenance_timers = struct();
    maintenance_timers.last_maintenance = 0;
    maintenance_timers.operating_hours = 0;
    maintenance_timers.thermal_cycles = 0;
    
    system_health = struct();
    system_health.overall_score = 100;
    system_health.component_health = ones(1, 5) * 100;  % 5 major components
    
    data_buffer = struct();
    data_buffer.samples_received = 0;
    data_buffer.samples_valid = 0;
    data_buffer.last_update_time = 0;
    
    communication_status = 2;  % Start optimal
    protocol_errors = 0;
end

%% Extract environmental data
AT = environmental_data(1);   % Ambient temperature
V = environmental_data(2);    % Vacuum pressure  
AP = environmental_data(3);   % Atmospheric pressure
RH = environmental_data(4);   % Relative humidity

%% Data Quality Assessment
data_buffer.samples_received = data_buffer.samples_received + 1;

% Check data validity ranges
power_valid = (power_output >= 150 && power_output <= 600);
control_valid = (abs(control_signal) <= 200);
env_valid = (AT >= -20 && AT <= 50) && (V >= 20 && V <= 80) && ...
           (AP >= 990 && AP <= 1040) && (RH >= 10 && RH <= 100);

if power_valid && control_valid && env_valid
    data_buffer.samples_valid = data_buffer.samples_valid + 1;
end

% Calculate data quality percentage
if data_buffer.samples_received > 0
    data_quality = (data_buffer.samples_valid / data_buffer.samples_received) * 100;
else
    data_quality = 0;
end

% Apply quality degradation factors
if data_quality < 95
    data_quality = data_quality * 0.95;  % Penalize poor quality
end

%% Update Historical Trends  
iot_history.power_trend = [iot_history.power_trend(2:end), power_output];
iot_history.control_trend = [iot_history.control_trend(2:end), abs(control_signal)];
iot_history.quality_trend = [iot_history.quality_trend(2:end), data_quality];

%% System Health Monitoring
% Component 1: Power Generation System
power_deviation = abs(power_output - 450) / 450;
if power_deviation > 0.15
    system_health.component_health(1) = max(70, system_health.component_health(1) - 0.5);
else
    system_health.component_health(1) = min(100, system_health.component_health(1) + 0.1);
end

% Component 2: Control System
control_effort = mean(iot_history.control_trend);
if control_effort > 80
    system_health.component_health(2) = max(75, system_health.component_health(2) - 0.3);
else
    system_health.component_health(2) = min(100, system_health.component_health(2) + 0.05);
end

% Component 3: Environmental Systems (Heat Exchangers, Cooling)
env_stress = (abs(AT - 15) / 30) + (abs(V - 50) / 25) + (abs(RH - 65) / 35);
if env_stress > 0.6
    system_health.component_health(3) = max(80, system_health.component_health(3) - 0.2);
else
    system_health.component_health(3) = min(100, system_health.component_health(3) + 0.03);
end

% Component 4: Data Acquisition System
if data_quality < 90
    system_health.component_health(4) = max(60, system_health.component_health(4) - 1.0);
else
    system_health.component_health(4) = min(100, system_health.component_health(4) + 0.2);
end

% Component 5: Communication Systems
% Simulate communication reliability
comm_reliability = 98 + 2 * sin(data_buffer.samples_received * 0.1) + randn() * 1;
comm_reliability = max(85, min(100, comm_reliability));
system_health.component_health(5) = 0.9 * system_health.component_health(5) + 0.1 * comm_reliability;

% Overall system health
system_health.overall_score = mean(system_health.component_health);

%% Alarm Generation
alarm_state = 0;  % Start with no alarms

% Power deviation alarms
power_error = abs(power_output - 450);
if power_error > 50
    alarm_counters.power_deviation = alarm_counters.power_deviation + 1;
    if alarm_counters.power_deviation > 3
        alarm_state = max(alarm_state, 2);  % Critical alarm
    else
        alarm_state = max(alarm_state, 1);  % Warning
    end
else
    alarm_counters.power_deviation = max(0, alarm_counters.power_deviation - 1);
end

% Control saturation alarms
if abs(control_signal) > 100
    alarm_counters.control_saturation = alarm_counters.control_saturation + 1;
    if alarm_counters.control_saturation > 5
        alarm_state = max(alarm_state, 1);  % Warning
    end
else
    alarm_counters.control_saturation = max(0, alarm_counters.control_saturation - 1);
end

% Environmental extreme conditions
if AT > 35 || AT < -5 || V > 70 || V < 30 || RH > 90 || RH < 20
    alarm_counters.environmental_extreme = alarm_counters.environmental_extreme + 1;
    if alarm_counters.environmental_extreme > 10
        alarm_state = max(alarm_state, 1);  % Warning
    end
else
    alarm_counters.environmental_extreme = max(0, alarm_counters.environmental_extreme - 1);
end

% System health degradation
if system_health.overall_score < 85
    alarm_counters.performance_degradation = alarm_counters.performance_degradation + 1;
    if system_health.overall_score < 70
        alarm_state = max(alarm_state, 2);  % Critical
    else
        alarm_state = max(alarm_state, 1);  % Warning
    end
else
    alarm_counters.performance_degradation = max(0, alarm_counters.performance_degradation - 1);
end

%% Predictive Maintenance Assessment
maintenance_alert = 0;

% Update operating hours and thermal cycles
maintenance_timers.operating_hours = maintenance_timers.operating_hours + 1/3600;  % Assume 1-second calls

% Detect thermal cycles (significant power changes)
power_change = abs(power_output - mean(iot_history.power_trend(1:10)));
if power_change > 30
    maintenance_timers.thermal_cycles = maintenance_timers.thermal_cycles + 0.1;
end

% Maintenance scheduling logic
hours_since_maintenance = maintenance_timers.operating_hours - maintenance_timers.last_maintenance;

% Scheduled maintenance (time-based)
if hours_since_maintenance > 8760  % 1 year of operation
    maintenance_alert = 1;  % Scheduled maintenance due
end

% Condition-based maintenance triggers
condition_score = system_health.overall_score - (maintenance_timers.thermal_cycles * 0.1);

if condition_score < 75
    maintenance_alert = 2;  % Urgent maintenance needed
elseif condition_score < 85 && hours_since_maintenance > 4380  % 6 months
    maintenance_alert = 1;  % Scheduled maintenance recommended
end

%% IoT System Status Determination
communication_status = 2;  % Assume optimal by default

% Simulate communication issues based on data quality
if data_quality < 80
    communication_status = 1;  % Degraded
    protocol_errors = protocol_errors + 1;
elseif data_quality < 50
    communication_status = 0;  % Offline
    protocol_errors = protocol_errors + 2;
else
    protocol_errors = max(0, protocol_errors - 0.1);  % Gradual recovery
end

% Overall IoT status considers communication and system health
if communication_status == 0 || system_health.overall_score < 60
    iot_status = 0;  % Offline/Failed
elseif communication_status == 1 || system_health.overall_score < 80
    iot_status = 1;  % Degraded
else
    iot_status = 2;  % Optimal
end

%% Output bounds checking
iot_status = max(0, min(2, round(iot_status)));
alarm_state = max(0, min(2, round(alarm_state)));  
maintenance_alert = max(0, min(2, round(maintenance_alert)));
data_quality = max(0, min(100, data_quality));

end