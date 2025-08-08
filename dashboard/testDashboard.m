%% FILE 3: Save this as "testDashboard.m" (Simple Test Version)

% Simple Dashboard Test Script
% This is a simpler version that doesn't require the To Workspace blocks

clear; clc;

fprintf('Testing Dashboard Setup...\n');

% Create test data (simulates your Simulink outputs)
fprintf('Creating test data...\n');
time_sim = 0:0.1:10;
predicted_power = 400 + 50*sin(time_sim/2) + 10*randn(size(time_sim));
control_signal = 0.1*randn(size(time_sim));
actual_power = predicted_power + 5*randn(size(time_sim));

% Create the dashboard figure
fig = figure('Name', 'Digital Twin Dashboard Test', 'Position', [100, 100, 1000, 600]);

% Create plots
subplot(2,2,1);
plot(time_sim, predicted_power, 'b-', 'LineWidth', 2); hold on;
plot(time_sim, actual_power, 'r--', 'LineWidth', 2);
plot(time_sim, 400*ones(size(time_sim)), 'g:', 'LineWidth', 2);
title('Power Tracking Test');
xlabel('Time (s)');
ylabel('Power (MW)');
legend('Predicted', 'Actual', 'Setpoint');
grid on;

subplot(2,2,2);
plot(time_sim, control_signal, 'k-', 'LineWidth', 2);
title('Control Signal Test');
xlabel('Time (s)');
ylabel('Control Effort');
grid on;

subplot(2,2,3);
metrics = [85, 92, 78]; % Test metrics
bar(metrics, 'FaceColor', [0.3 0.6 0.9]);
title('Performance Metrics Test');
set(gca, 'XTickLabel', {'Efficiency', 'Accuracy', 'Stability'});
ylim([0 100]);

subplot(2,2,4);
% Status information
text(0.1, 0.8, 'System Status (Test):', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.6, sprintf('Power: %.1f MW', predicted_power(end)), 'FontSize', 12);
text(0.1, 0.4, sprintf('Error: %.1f MW', 400 - predicted_power(end)), 'FontSize', 12);
text(0.1, 0.2, sprintf('Control: %.2f', control_signal(end)), 'FontSize', 12);
set(gca, 'XTick', [], 'YTick', []);
xlim([0 1]); ylim([0 1]);

fprintf('✅ Dashboard test completed!\n');
fprintf('If you see 4 plots, the dashboard system is working.\n\n');

fprintf('NEXT STEPS:\n');
fprintf('1. Add To Workspace blocks to your Simulink model\n');
fprintf('2. Run your Simulink model\n');
fprintf('3. Type: runDashboard()\n\n');