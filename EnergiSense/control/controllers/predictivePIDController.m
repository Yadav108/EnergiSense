function [control_signal, controller_status, performance_metrics] = predictivePIDController(setpoint, predicted_power, actual_power, dt, params)
%#codegen
% Predictive PID Controller for Energisense Digital Twin Power System

%% Input Validation
if nargin < 5
    error('predictivePIDController requires 5 inputs');
end

if ~isscalar(setpoint) || ~isscalar(predicted_power) || ~isscalar(actual_power) || ~isscalar(dt)
    error('All inputs must be scalar values');
end

if dt <= 0
    dt = 0.001;
end

%% Initialize Parameters with Defaults
Kp = 1.8; Ki = 0.15; Kd = 0.12;
u_max = 120.0; u_min = -120.0; I_max = 40.0; I_min = -40.0;
prediction_weight = 0.6; model_quality_threshold = 0.8;
enable_adaptive = true; enable_derivative_filter = true;
derivative_filter_alpha = 0.2; setpoint_weight = 0.9;
deadband = 0.5; adaptive_factor = 0.02;
large_error_threshold = 20.0; small_error_threshold = 2.5;
feedforward_gain = 0.4; disturbance_gain = 0.3;

% Override with user parameters if provided
if nargin >= 5 && ~isempty(params) && isstruct(params)
    if isfield(params, 'Kp') && ~isempty(params.Kp), Kp = params.Kp; end
    if isfield(params, 'Ki') && ~isempty(params.Ki), Ki = params.Ki; end
    if isfield(params, 'Kd') && ~isempty(params.Kd), Kd = params.Kd; end
    if isfield(params, 'u_max') && ~isempty(params.u_max), u_max = params.u_max; end
    if isfield(params, 'u_min') && ~isempty(params.u_min), u_min = params.u_min; end
    if isfield(params, 'prediction_weight') && ~isempty(params.prediction_weight), prediction_weight = params.prediction_weight; end
end

%% Persistent Variables
persistent integral_error previous_actual_error previous_predicted_error;
persistent previous_setpoint previous_control_signal filtered_derivative;
persistent prediction_error_history adaptive_Kp adaptive_Ki adaptive_Kd initialized;

if isempty(initialized)
    integral_error = 0.0; previous_actual_error = 0.0;
    previous_predicted_error = 0.0; previous_setpoint = 0.0;
    previous_control_signal = 0.0; filtered_derivative = 0.0;
    prediction_error_history = zeros(1, 10);
    adaptive_Kp = Kp; adaptive_Ki = Ki; adaptive_Kd = Kd;
    initialized = true;
end

%% Error Calculations
actual_error = setpoint - actual_power;
prediction_error = predicted_power - actual_power;
predictive_error = setpoint - predicted_power;

if abs(actual_error) < deadband
    actual_error = 0.0;
end

%% Update Prediction Error History
prediction_error_history = [prediction_error_history(2:end), prediction_error];

%% Digital Twin Quality Assessment
prediction_mae = mean(abs(prediction_error_history));
model_quality = exp(-prediction_mae / 5.0);
model_quality = max(0.1, min(1.0, model_quality));

%% Adaptive Gains
current_Kp = adaptive_Kp; current_Ki = adaptive_Ki; current_Kd = adaptive_Kd;

if enable_adaptive
    error_magnitude = abs(actual_error);
    if error_magnitude > large_error_threshold
        current_Kp = min(current_Kp * (1 + adaptive_factor), Kp * 2.0);
        current_Ki = min(current_Ki * (1 + adaptive_factor * 0.5), Ki * 1.5);
    elseif error_magnitude < small_error_threshold
        current_Kp = max(current_Kp * (1 - adaptive_factor * 0.5), Kp * 0.5);
        current_Ki = max(current_Ki * (1 - adaptive_factor * 0.3), Ki * 0.5);
    end
end

% Model quality adaptation
if model_quality > model_quality_threshold
    current_Kp = current_Kp * 0.8;
else
    current_Kp = current_Kp * 1.2;
end

%% PID Terms Calculation
P_actual = current_Kp * actual_error;
P_predictive = current_Kp * predictive_error * prediction_weight;
P_term = P_actual * (1.0 - prediction_weight) + P_predictive * prediction_weight;

I_term = current_Ki * integral_error;

D_term = 0.0;
if dt > 0
    actual_derivative = (actual_error - previous_actual_error) / dt;
    predictive_derivative = (predictive_error - previous_predicted_error) / dt;
    combined_derivative = actual_derivative * (1.0 - prediction_weight) + predictive_derivative * prediction_weight;
    
    if enable_derivative_filter
        filtered_derivative = derivative_filter_alpha * combined_derivative + (1.0 - derivative_filter_alpha) * filtered_derivative;
        D_term = current_Kd * filtered_derivative;
    else
        D_term = current_Kd * combined_derivative;
    end
end

%% Feed-forward Control
feedforward_term = 0.0;
if dt > 0
    prediction_ff = (setpoint - predicted_power) * feedforward_gain * model_quality;
    setpoint_rate = (setpoint - previous_setpoint) / dt;
    rate_ff = setpoint_rate * feedforward_gain * 0.5;
    feedforward_term = prediction_ff + rate_ff;
end

%% Disturbance Compensation
disturbance_compensation = 0.0;
recent_errors = prediction_error_history(end-4:end);
persistent_bias = mean(recent_errors);
if abs(persistent_bias) > 2.0 && std(recent_errors) < abs(persistent_bias) * 0.5
    disturbance_compensation = -persistent_bias * disturbance_gain;
end

%% Total Control Signal
raw_control_signal = P_term + I_term + D_term + feedforward_term + disturbance_compensation;
control_signal = max(u_min, min(raw_control_signal, u_max));

%% Anti-Windup
is_saturated = (control_signal ~= raw_control_signal);
integrate_condition = true;
if is_saturated && sign(actual_error) == sign(integral_error)
    integrate_condition = false;
end
if model_quality > model_quality_threshold && sign(predictive_error) ~= sign(actual_error)
    integrate_condition = true;
end
if integrate_condition
    integral_error = integral_error + actual_error * dt;
end
integral_error = max(I_min, min(integral_error, I_max));

%% Update Persistent Variables
previous_actual_error = actual_error;
previous_predicted_error = predictive_error;
previous_setpoint = setpoint;
previous_control_signal = control_signal;
alpha = 0.001;
adaptive_Kp = current_Kp * alpha + adaptive_Kp * (1.0 - alpha);
adaptive_Ki = current_Ki * alpha + adaptive_Ki * (1.0 - alpha);
adaptive_Kd = current_Kd * alpha + adaptive_Kd * (1.0 - alpha);

%% Generate Outputs
controller_status = [current_Kp; current_Ki; current_Kd; model_quality; prediction_error; integral_error; model_quality];
performance_metrics = [actual_error; abs(actual_error); abs(prediction_error); model_quality; abs(control_signal); model_quality * 100; (1.0 - abs(actual_error) / max(abs(setpoint), 1.0)) * 100; 1.0];

end