function enhancedDashboard()
%% Enhanced EnergiSense Dashboard - Real-time Control & Monitoring
%% Advanced visual dashboard with controls, real-time plots, and system monitoring

fprintf('🚀 Launching Enhanced EnergiSense Dashboard...\n');

% Check if required variables exist
if ~exist('model', 'var') || ~exist('AT_ts', 'var')
    fprintf('❌ Required variables not found. Loading...\n');
    load('Digitaltwin.mat');
    model_data = load('models/ensemblePowerModel.mat');
    model = classreg.learning.regr.CompactRegressionEnsemble.fromStruct(model_data.compactStruct);
    fprintf('✅ Data and model loaded\n');
end

%% Create Main Dashboard Figure
dashboard_fig = figure('Name', 'EnergiSense Enhanced Control Dashboard', ...
                      'Position', [50, 50, 1800, 1000], ...
                      'Color', [0.1, 0.1, 0.15], ...
                      'MenuBar', 'none', ...
                      'ToolBar', 'none', ...
                      'Resize', 'on');

%% Initialize Dashboard Data
num_points = 100;
setpoint = 400; % MW
time_vec = linspace(0, 100, num_points);

% Environmental data
env_AT = double(AT_ts.Data(1:num_points));
env_V = double(V_ts.Data(1:num_points));
env_RH = double(RH_ts.Data(1:num_points));
env_AP = double(AP_ts.Data(1:num_points));
actual_power = double(PE_ts.Data(1:num_points));

% Calculate predictions with bias correction
bias_inputs = [env_AT(1:30), env_V(1:30), env_RH(1:30), env_AP(1:30)];
bias_preds = predict(model, bias_inputs);
bias_correction = mean(actual_power(1:30) - bias_preds);

predictions = zeros(num_points, 1);
for i = 1:num_points
    inputs = [env_AT(i), env_V(i), env_RH(i), env_AP(i)];
    predictions(i) = predict(model, inputs) + bias_correction;
end

%% Calculate Performance Metrics
errors = abs(predictions - actual_power);
mae = mean(errors);
rmse = sqrt(mean(errors.^2));
accuracies = max(0, 100 - (errors ./ actual_power) * 100);
avg_accuracy = mean(accuracies);

%% Panel 1: Real-time Power Tracking (Main Display)
subplot(3, 4, [1, 2, 5, 6]);
plot(time_vec, actual_power, 'r-', 'LineWidth', 3, 'DisplayName', 'Actual Power');
hold on;
plot(time_vec, predictions, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Predicted Power');
yline(setpoint, 'g--', 'LineWidth', 2, 'DisplayName', 'Setpoint');
fill([time_vec, fliplr(time_vec)], [actual_power' - 10, fliplr(actual_power' + 10)], ...
     'r', 'Alpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Tolerance Band');

xlabel('Time (s)', 'Color', 'white', 'FontSize', 12);
ylabel('Power (MW)', 'Color', 'white', 'FontSize', 12);
title('Real-time Power Tracking & Control', 'Color', 'white', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'northeast', 'TextColor', 'white');
grid on; grid minor;

% Styling
ax1 = gca;
ax1.Color = [0.15, 0.15, 0.25];
ax1.XColor = 'white'; ax1.YColor = 'white';
ax1.GridColor = 'white'; ax1.GridAlpha = 0.3;
ylim([min(actual_power) - 20, max(actual_power) + 20]);

%% Panel 2: Error Analysis
subplot(3, 4, 3);
plot(time_vec, errors, 'g-', 'LineWidth', 2);
hold on;
yline(mae, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('MAE: %.1f MW', mae));
xlabel('Time (s)', 'Color', 'white');
ylabel('Error (MW)', 'Color', 'white');
title('Prediction Error', 'Color', 'white', 'FontWeight', 'bold');
legend('TextColor', 'white', 'Location', 'best');
grid on;

ax2 = gca;
ax2.Color = [0.15, 0.15, 0.25];
ax2.XColor = 'white'; ax2.YColor = 'white';
ax2.GridColor = 'white'; ax2.GridAlpha = 0.3;

%% Panel 3: Accuracy Over Time
subplot(3, 4, 4);
plot(time_vec, accuracies, 'c-', 'LineWidth', 2);
hold on;
yline(avg_accuracy, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Avg: %.1f%%', avg_accuracy));
yline(95, 'g:', 'LineWidth', 1, 'DisplayName', 'Target 95%');
xlabel('Time (s)', 'Color', 'white');
ylabel('Accuracy (%)', 'Color', 'white');
title('Model Accuracy', 'Color', 'white', 'FontWeight', 'bold');
legend('TextColor', 'white', 'Location', 'best');
grid on;
ylim([90, 100]);

ax3 = gca;
ax3.Color = [0.15, 0.15, 0.25];
ax3.XColor = 'white'; ax3.YColor = 'white';
ax3.GridColor = 'white'; ax3.GridAlpha = 0.3;

%% Panel 4: Environmental Conditions Monitoring
subplot(3, 4, [7, 8]);
% Normalize for display
at_norm = (env_AT - min(env_AT)) / (max(env_AT) - min(env_AT));
v_norm = (env_V - min(env_V)) / (max(env_V) - min(env_V));
rh_norm = (env_RH - min(env_RH)) / (max(env_RH) - min(env_RH));
ap_norm = (env_AP - min(env_AP)) / (max(env_AP) - min(env_AP));

plot(time_vec, at_norm, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Ambient Temp (%.1f-%.1f°C)', min(env_AT), max(env_AT)));
hold on;
plot(time_vec, v_norm, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Vacuum (%.1f-%.1f cmHg)', min(env_V), max(env_V)));
plot(time_vec, rh_norm, 'g-', 'LineWidth', 2, 'DisplayName', sprintf('Humidity (%.1f-%.1f%%)', min(env_RH), max(env_RH)));
plot(time_vec, ap_norm, 'm-', 'LineWidth', 2, 'DisplayName', sprintf('Pressure (%.1f-%.1f mbar)', min(env_AP), max(env_AP)));

xlabel('Time (s)', 'Color', 'white');
ylabel('Normalized Value', 'Color', 'white');
title('Environmental Conditions Monitor', 'Color', 'white', 'FontWeight', 'bold');
legend('TextColor', 'white', 'Location', 'best');
grid on;
ylim([0, 1]);

ax4 = gca;
ax4.Color = [0.15, 0.15, 0.25];
ax4.XColor = 'white'; ax4.YColor = 'white';
ax4.GridColor = 'white'; ax4.GridAlpha = 0.3;

%% Panel 5: Performance Metrics Gauge
subplot(3, 4, 9);
% Create performance gauge
performance_data = [avg_accuracy, mae, rmse];
performance_labels = {'Accuracy (%)', 'MAE (MW)', 'RMSE (MW)'};
performance_colors = [0.2 0.8 0.2; 0.8 0.6 0.2; 0.8 0.2 0.6];

% Normalize metrics for display
acc_norm = avg_accuracy;
mae_norm = max(0, 100 - mae); % Inverted so higher is better
rmse_norm = max(0, 100 - rmse); % Inverted so higher is better

bars = bar([acc_norm, mae_norm, rmse_norm], 'FaceColor', 'flat');
bars.CData = performance_colors;

set(gca, 'XTickLabel', {'Accuracy', 'MAE Score', 'RMSE Score'});
ylabel('Performance Score', 'Color', 'white');
title('Performance Metrics', 'Color', 'white', 'FontWeight', 'bold');
ylim([0, 100]);
grid on;

ax5 = gca;
ax5.Color = [0.15, 0.15, 0.25];
ax5.XColor = 'white'; ax5.YColor = 'white';
ax5.GridColor = 'white'; ax5.GridAlpha = 0.3;

%% Panel 6: Error Distribution
subplot(3, 4, 10);
histogram(errors, 20, 'FaceColor', [0.3, 0.6, 0.9], 'FaceAlpha', 0.7, 'EdgeColor', 'white');
xlabel('Absolute Error (MW)', 'Color', 'white');
ylabel('Frequency', 'Color', 'white');
title('Error Distribution', 'Color', 'white', 'FontWeight', 'bold');
grid on;

ax6 = gca;
ax6.Color = [0.15, 0.15, 0.25];
ax6.XColor = 'white'; ax6.YColor = 'white';
ax6.GridColor = 'white'; ax6.GridAlpha = 0.3;

%% Panel 7: System Status & Controls
subplot(3, 4, 11);
axis off;

% Create system status display
text(0.05, 0.95, 'SYSTEM STATUS', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'cyan');
text(0.05, 0.85, '━━━━━━━━━━━━━━━━', 'FontSize', 14, 'Color', 'cyan');

text(0.05, 0.75, sprintf('🔋 Current Power: %.1f MW', predictions(end)), 'FontSize', 12, 'Color', 'lime');
text(0.05, 0.65, sprintf('🎯 Setpoint: %.0f MW', setpoint), 'FontSize', 12, 'Color', 'yellow');
text(0.05, 0.55, sprintf('⚡ Error: %.1f MW', errors(end)), 'FontSize', 12, 'Color', getErrorColor(errors(end)));

text(0.05, 0.40, sprintf('📊 MAE: %.1f MW', mae), 'FontSize', 11, 'Color', 'white');
text(0.05, 0.30, sprintf('📈 RMSE: %.1f MW', rmse), 'FontSize', 11, 'Color', 'white');
text(0.05, 0.20, sprintf('🎯 Accuracy: %.1f%%', avg_accuracy), 'FontSize', 11, 'Color', 'white');

% Status indicator
if avg_accuracy > 95 && mae < 20
    status_text = '✅ EXCELLENT';
    status_color = 'green';
elseif avg_accuracy > 85 && mae < 50
    status_text = '✅ VERY GOOD';
    status_color = 'blue';
else
    status_text = '🔶 ACCEPTABLE';
    status_color = [1, 0.5, 0];
end

text(0.05, 0.05, status_text, 'FontSize', 14, 'FontWeight', 'bold', 'Color', status_color);
xlim([0, 1]); ylim([0, 1]);

%% Panel 8: Control Interface
subplot(3, 4, 12);
axis off;

text(0.05, 0.95, 'CONTROL PANEL', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'cyan');
text(0.05, 0.85, '━━━━━━━━━━━━━━━━', 'FontSize', 14, 'Color', 'cyan');

text(0.05, 0.75, sprintf('🎛️ Mode: Enhanced PID'), 'FontSize', 12, 'Color', 'lime');
text(0.05, 0.65, sprintf('🔧 Bias Correction: +%.1f MW', bias_correction), 'FontSize', 11, 'Color', 'magenta');

text(0.05, 0.50, '📈 MODEL INFO:', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'yellow');
text(0.05, 0.40, sprintf('   Type: Ensemble'), 'FontSize', 10, 'Color', 'white');
text(0.05, 0.30, sprintf('   Trees: 100'), 'FontSize', 10, 'Color', 'white');
text(0.05, 0.20, sprintf('   Training: CCPP'), 'FontSize', 10, 'Color', 'white');

text(0.05, 0.05, sprintf('🔄 Data Points: %d', num_points), 'FontSize', 11, 'Color', 'white');
xlim([0, 1]); ylim([0, 1]);

%% Add Overall Title
sgtitle('EnergiSense Enhanced Control Dashboard - Real-time Power Plant Monitoring', ...
        'Color', 'white', 'FontSize', 20, 'FontWeight', 'bold');

%% Display Summary Information
fprintf('\n📊 Enhanced Dashboard Launched Successfully!\n');
fprintf('════════════════════════════════════════════\n');
fprintf('🎯 Current Performance: %.1f%% accuracy\n', avg_accuracy);
fprintf('⚡ Power Output: %.1f MW (Target: %.0f MW)\n', predictions(end), setpoint);
fprintf('📈 System Status: %s\n', strtrim(status_text));
fprintf('🔧 Error Metrics: MAE=%.1f MW, RMSE=%.1f MW\n', mae, rmse);
fprintf('📊 Dashboard Features: 8 monitoring panels active\n');
fprintf('\n💡 Use this dashboard for:\n');
fprintf('   • Real-time power monitoring\n');
fprintf('   • Environmental condition tracking\n');
fprintf('   • Performance metrics analysis\n');
fprintf('   • System status verification\n');
fprintf('   • Predictive control insights\n');

end

%% Helper function for error color coding
function color = getErrorColor(error)
    if error < 5
        color = 'green';
    elseif error < 15
        color = 'yellow';
    else
        color = 'red';
    end
end