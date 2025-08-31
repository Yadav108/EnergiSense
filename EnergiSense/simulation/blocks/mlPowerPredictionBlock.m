function [predicted_power, model_confidence, prediction_status] = mlPowerPredictionBlock(AT, V, AP, RH)
%MLPOWERPREDICTIONBLOCK Simulink-compatible ML power prediction block
%#codegen
%
% This function provides a Simulink-compatible interface to the trained
% 95.9% accurate Random Forest model for real-time power prediction.
%
% INPUTS:
%   AT - Ambient Temperature (°C)
%   V  - Vacuum Pressure (cm Hg) 
%   AP - Atmospheric Pressure (mbar)
%   RH - Relative Humidity (%)
%
% OUTPUTS:
%   predicted_power - Predicted electrical power output (MW)
%   model_confidence - Model confidence (0-1 scale)
%   prediction_status - Status indicator (0=error, 1=empirical, 2=ML)

%% Input validation and sanitization
if nargin < 4
    predicted_power = 450.0;  % Default safe value
    model_confidence = 0.5;
    prediction_status = 0;
    return;
end

% Ensure inputs are scalar and within realistic ranges
AT = max(-10, min(45, AT));    % Temperature range: -10°C to 45°C
V = max(25, min(75, V));       % Vacuum: 25-75 cm Hg  
AP = max(990, min(1040, AP));  % Pressure: 990-1040 mbar
RH = max(20, min(100, RH));    % Humidity: 20-100%

%% Try ML prediction first
persistent ml_model ml_validation_results ml_model_loaded;

% Initialize ML model on first call
if isempty(ml_model_loaded)
    ml_model_loaded = false;
    try
        % Try to load from base workspace first (fastest)
        if evalin('base', 'exist(''trained_ml_model'', ''var'')')
            ml_model = evalin('base', 'trained_ml_model');
            ml_validation_results = evalin('base', 'ml_validation_results');
            if ~isempty(ml_model)
                ml_model_loaded = true;
            end
        end
        
        % If not in workspace, load from file
        if ~ml_model_loaded
            model_path = fullfile(pwd, 'core', 'models', 'ccpp_random_forest_model.mat');
            if exist(model_path, 'file')
                model_data = load(model_path);
                ml_model = model_data.model;
                ml_validation_results = model_data.validation_results;
                ml_model_loaded = true;
            end
        end
    catch
        ml_model_loaded = false;
    end
end

%% Attempt ML prediction
if ml_model_loaded && ~isempty(ml_model)
    try
        % Prepare input data in correct format for TreeBagger
        input_data = [AT, V, AP, RH];
        
        % Make prediction using trained Random Forest
        predicted_power_raw = predict(ml_model, input_data);
        predicted_power = predicted_power_raw(1);  % Extract scalar value
        
        % Use validated model confidence
        model_confidence = ml_validation_results.r2_score;  % 0.9594 for 95.9%
        prediction_status = 2;  % ML prediction successful
        
        % Sanity check - ensure prediction is in realistic range
        if predicted_power < 200 || predicted_power > 600
            % Fall back to empirical model for out-of-range predictions
            predicted_power = calculateEmpiricalPower(AT, V, AP, RH);
            model_confidence = 0.85;
            prediction_status = 1;
        end
        
        return;
        
    catch ME
        % ML prediction failed, fall back to empirical model
        % (Don't print errors in Simulink block for performance)
    end
end

%% Fallback: Empirical model (from original implementation)
predicted_power = calculateEmpiricalPower(AT, V, AP, RH);
model_confidence = 0.85;  % Lower confidence for empirical model
prediction_status = 1;    % Empirical prediction

end

%% Helper Functions
function power = calculateEmpiricalPower(AT, V, AP, RH)
%CALCULATEEMPIRICALPOWER Fallback empirical power calculation
% Based on typical CCPP performance characteristics

% Base power calculation using simplified thermodynamic relations
base_power = 480;  % Nominal power (MW)

% Temperature effect (higher temp = lower efficiency)
temp_effect = 1 - (AT - 15) * 0.006;  % 0.6% per degree from 15°C

% Vacuum effect (better vacuum = higher efficiency) 
vacuum_effect = 1 + (50 - V) * 0.004;  % 0.4% per cm Hg from 50 cm Hg

% Atmospheric pressure effect
pressure_effect = 1 + (AP - 1013) * 0.0001;  % 0.01% per mbar from standard

% Humidity effect (higher humidity = slightly lower efficiency)
humidity_effect = 1 - (RH - 60) * 0.0005;  % 0.05% per % RH from 60%

% Combined effect
total_effect = temp_effect * vacuum_effect * pressure_effect * humidity_effect;

% Apply bounds to prevent unrealistic values
total_effect = max(0.7, min(1.15, total_effect));

power = base_power * total_effect;

% Final bounds check
power = max(200, min(550, power));

end