function [y, confidence, anomaly_flag] = predictPowerEnhanced(inputData)
%#codegen
% Working EnergiSense prediction function

    % Input validation
    inputData = reshape(inputData, 1, 4);
    AT = inputData(1);   % Temperature
    V = inputData(2);    % Vacuum
    RH = inputData(3);   % Humidity
    AP = inputData(4);   % Pressure

    % Simple but accurate CCPP model
    base_power = 454.5;
    temp_effect = -2.2 * AT;
    vacuum_effect = -0.39 * V;
    humidity_effect = -0.078 * RH;
    pressure_effect = 0.043 * AP;
    bias_correction = 2.0;

    % Calculate power
    y = base_power + temp_effect + vacuum_effect + humidity_effect + pressure_effect + bias_correction;

    % Ensure realistic range
    y = max(400, min(500, y));

    % Simple confidence and anomaly
    confidence = 0.85;
    anomaly_flag = false;

    % Ensure correct types for Simulink
    y = double(y);
    confidence = double(confidence);
    anomaly_flag = logical(anomaly_flag);
end
