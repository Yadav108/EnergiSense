function analyzeResults(simout)
if nargin < 1
    fprintf('Usage: analyzeResults(simout)\n');
    return;
end

fprintf('=== ENERGISENSE SIMULATION ANALYSIS ===\n');

if isstruct(simout)
    signals = fieldnames(simout);
    fprintf('Available signals:\n');
    for i = 1:length(signals)
        fprintf('  - %s\n', signals{i});
    end
    
    % Check for required signals
    if isfield(simout, 'setpoint') && isfield(simout, 'actual_power') && isfield(simout, 'control_signal')
        performAnalysis(simout);
    else
        fprintf('Missing required signals. Enable signal logging in Simulink.\n');
    end
else
    fprintf('Simulation output format not recognized.\n');
end
end

function performAnalysis(simout)
time = simout.setpoint.Time;
setpoint = simout.setpoint.Data;
actual_power = simout.actual_power.Data;
control_signal = simout.control_signal.Data;

error = setpoint - actual_power;
mae = mean(abs(error));
rmse = sqrt(mean(error.^2));
max_error = max(abs(error));
sse = mean(abs(error(round(0.8*length(error)):end)));

fprintf('\n--- PERFORMANCE METRICS ---\n');
fprintf('Mean Absolute Error: %.2f MW\n', mae);
fprintf('RMS Error: %.2f MW\n', rmse);
fprintf('Maximum Error: %.2f MW\n', max_error);
fprintf('Steady-State Error: %.2f MW\n', sse);
fprintf('Control Range: [%.1f, %.1f]\n', min(control_signal), max(control_signal));

if mae < 3.0 && sse < 1.5
    fprintf('Performance: EXCELLENT\n');
elseif mae < 6.0 && sse < 3.0
    fprintf('Performance: GOOD\n');
else
    fprintf('Performance: NEEDS IMPROVEMENT\n');
end

figure('Name', 'Energisense Performance');
subplot(2,1,1);
plot(time, setpoint, 'r--', time, actual_power, 'b-', 'LineWidth', 1.5);
ylabel('Power (MW)');
legend('Setpoint', 'Actual Power');
title('Power Tracking');
grid on;

subplot(2,1,2);
plot(time, control_signal, 'g-', 'LineWidth', 1.5);
ylabel('Control Signal');
xlabel('Time (s)');
title('Controller Output');
grid on;
end