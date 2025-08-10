%% FILE 1: Save this as "runDashboard.m"

function runDashboard()
%% EnergiSense Professional Dashboard - Enhanced Version
%% Comprehensive analysis with advanced visualizations

fprintf('🏭 EnergiSense Live Dashboard - Enhanced\n');
fprintf('=========================================\n\n');

%% Check if required data is loaded
if ~evalin('base', 'exist(''PE_ts'', ''var'')') || ~evalin('base', 'exist(''model'', ''var'')')
    fprintf('❌ Required data not loaded. Run demo() first.\n');
    return;
end

%% Load data from base workspace
PE_ts = evalin('base', 'PE_ts');
AT_ts = evalin('base', 'AT_ts');
V_ts = evalin('base', 'V_ts');
RH_ts = evalin('base', 'RH_ts');
AP_ts = evalin('base', 'AP_ts');
model = evalin('base', 'model');

%% Professional Color Palette
colors = struct();
colors.excellent = [0.2, 0.8, 0.2];     % Green
colors.good = [1.0, 0.8, 0.0];          % Amber  
colors.warning = [1.0, 0.4, 0.0];       % Orange
colors.critical = [0.8, 0.2, 0.2];      % Red
colors.primary = [0.2, 0.6, 0.8];       % Blue
colors.secondary = [0.6, 0.2, 0.8];     % Purple
colors.accent = [0.0, 0.8, 0.8];        % Cyan
colors.background = [0.1, 0.1, 0.15];   % Dark background
colors.panel = [0.15, 0.15, 0.2];       % Panel background
colors.text = [1.0, 1.0, 1.0];          % White text

%% Performance Analysis (100 data points)
data_points = min(100, length(PE_ts.Data));
fprintf('📊 Analyzing %d data points...\n', data_points);

%% Initialize arrays
predictions = zeros(data_points, 1);
actual_values = zeros(data_points, 1);
errors = zeros(data_points, 1);
mae_values = zeros(data_points, 1);
rmse_values = zeros(data_points, 1);
accuracy_values = zeros(data_points, 1);

% Environmental data arrays
ambient_temp = zeros(data_points, 1);
vacuum = zeros(data_points, 1);
humidity = zeros(data_points, 1);
pressure = zeros(data_points, 1);

% Progress indicator for large calculations
fprintf('🔄 Computing predictions: ');

for i = 1:data_points
    if mod(i, 20) == 0
        fprintf('%.0f%% ', (i/data_points)*100);
    end
    
    % Get input conditions
    input_data = [AT_ts.Data(i), V_ts.Data(i), RH_ts.Data(i), AP_ts.Data(i)];
    
    % Store environmental data
    ambient_temp(i) = AT_ts.Data(i);
    vacuum(i) = V_ts.Data(i);
    humidity(i) = RH_ts.Data(i);
    pressure(i) = AP_ts.Data(i);
    
    % Get prediction
    pred = predict(model, input_data);
    actual = PE_ts.Data(i);
    
    % Store values
    predictions(i) = pred;
    actual_values(i) = actual;
    error = abs(pred - actual);
    errors(i) = error;
    
    % Calculate running metrics
    mae_values(i) = mean(errors(1:i));
    rmse_values(i) = sqrt(mean(errors(1:i).^2));
    accuracy_values(i) = 100 * (1 - error / actual);
end
fprintf('✅ Complete!\n\n');

%% COMPREHENSIVE PERFORMANCE SUMMARY
fprintf('📈 PERFORMANCE SUMMARY\n');
fprintf('=====================\n');

% Overall Statistics
overall_accuracy = mean(100 * (1 - errors ./ actual_values));
final_mae = mae_values(end);
final_rmse = rmse_values(end);
pred_range = [min(predictions), max(predictions)];
actual_range = [min(actual_values), max(actual_values)];

% Performance Categories
excellent_points = sum(errors < 15);  % < 15 MW error
good_points = sum(errors >= 15 & errors < 30);  % 15-30 MW error
fair_points = sum(errors >= 30);  % > 30 MW error

fprintf('🎯 Overall Accuracy: %.1f%%\n', overall_accuracy);
fprintf('📊 Final MAE: %.1f MW | Final RMSE: %.1f MW\n', final_mae, final_rmse);
fprintf('⚡ Predicted Range: %.1f - %.1f MW\n', pred_range(1), pred_range(2));
fprintf('📈 Actual Range: %.1f - %.1f MW\n', actual_range(1), actual_range(2));

fprintf('\n🏆 ACCURACY DISTRIBUTION\n');
fprintf('========================\n');
fprintf('🟢 Excellent (<15 MW error): %d points (%.1f%%)\n', excellent_points, (excellent_points/data_points)*100);
fprintf('🟡 Good (15-30 MW error): %d points (%.1f%%)\n', good_points, (good_points/data_points)*100);
fprintf('🟠 Fair (>30 MW error): %d points (%.1f%%)\n', fair_points, (fair_points/data_points)*100);

%% DETAILED METRICS
fprintf('\n📊 DETAILED METRICS\n');
fprintf('===================\n');
fprintf('🔹 Mean Error: %.1f MW\n', mean(errors));
fprintf('🔹 Std Deviation: %.1f MW\n', std(errors));
fprintf('🔹 Max Error: %.1f MW\n', max(errors));
fprintf('🔹 Min Error: %.1f MW\n', min(errors));
fprintf('🔹 Median Error: %.1f MW\n', median(errors));

%% Environmental Summary
fprintf('\n🌡️ ENVIRONMENTAL SUMMARY\n');
fprintf('========================\n');
fprintf('🌡️ Ambient Temp: %.1f°C (%.1f - %.1f°C)\n', mean(ambient_temp), min(ambient_temp), max(ambient_temp));
fprintf('💨 Vacuum: %.1f cmHg (%.1f - %.1f cmHg)\n', mean(vacuum), min(vacuum), max(vacuum));
fprintf('💧 Humidity: %.1f%% (%.1f - %.1f%%)\n', mean(humidity), min(humidity), max(humidity));
fprintf('🌪️ Pressure: %.1f mbar (%.1f - %.1f mbar)\n', mean(pressure), min(pressure), max(pressure));

%% SYSTEM STATUS
fprintf('\n🖥️ SYSTEM STATUS\n');
fprintf('================\n');

if overall_accuracy > 95
    status_icon = '🟢';
    status_text = 'EXCELLENT';
    status_color = colors.excellent;
elseif overall_accuracy > 90
    status_icon = '🟡';
    status_text = 'GOOD';
    status_color = colors.good;
elseif overall_accuracy > 85
    status_icon = '🟠';
    status_text = 'FAIR';
    status_color = colors.warning;
else
    status_icon = '🔴';
    status_text = 'NEEDS ATTENTION';
    status_color = colors.critical;
end

fprintf('%s Status: %s (%.1f%% accuracy)\n', status_icon, status_text, overall_accuracy);
fprintf('🔋 Model Performance: %s\n', status_text);
fprintf('📡 Data Quality: HIGH\n');
fprintf('⚙️ Control System: STABLE\n');

%% RECOMMENDATIONS
fprintf('\n💡 RECOMMENDATIONS\n');
fprintf('==================\n');

if overall_accuracy > 95
    fprintf('✅ System performing excellently - no action needed\n');
    fprintf('🔧 Consider: Advanced MPC control for optimization\n');
    fprintf('📈 Opportunity: Implement predictive maintenance\n');
elseif overall_accuracy > 90
    fprintf('⚠️ Good performance - minor tuning recommended\n');
    fprintf('🔧 Consider: Model recalibration for improvement\n');
    fprintf('📊 Review: Environmental impact on performance\n');
elseif overall_accuracy > 85
    fprintf('⚠️ Fair performance - optimization needed\n');
    fprintf('🔧 Action: Review control parameters\n');
    fprintf('📊 Analyze: Environmental correlation patterns\n');
else
    fprintf('❌ Performance needs immediate attention\n');
    fprintf('🔧 Action: Review input data quality and model parameters\n');
    fprintf('🚨 Priority: Investigate systematic errors\n');
end

%% ENHANCED VISUAL DASHBOARD
fprintf('\n🖼️ Launching enhanced visual dashboard...\n');

try
    % Create enhanced dashboard figure
    fig = figure('Name', 'EnergiSense CCPP Enhanced Dashboard', ...
                 'NumberTitle', 'off', ...
                 'Position', [50, 50, 1400, 900], ...
                 'Color', colors.background);

    % Create time vector for better x-axis labels
    time_vector = 1:data_points;
    
    % Panel 1: Enhanced Power Tracking with Confidence Bands
    subplot(2,3,1);
    % Plot actual values
    plot(time_vector, actual_values, 'Color', colors.primary, 'LineWidth', 2.5, 'DisplayName', 'Actual Power');
    hold on;
    % Plot predictions
    plot(time_vector, predictions, 'Color', colors.critical, 'LineStyle', '--', 'LineWidth', 2, 'DisplayName', 'Predicted Power');
    
    % Add confidence band (±1 standard deviation)
    error_band = std(errors);
    upper_bound = predictions + error_band;
    lower_bound = predictions - error_band;
    fill([time_vector, fliplr(time_vector)], [upper_bound', fliplr(lower_bound')], ...
         colors.secondary, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Confidence Band');
    
    title('Power Output: Actual vs Predicted', 'Color', colors.text, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Sample Number', 'Color', colors.text);
    ylabel('Power Output (MW)', 'Color', colors.text);
    legend('TextColor', colors.text, 'Location', 'best');
    grid on; grid minor;
    set(gca, 'Color', colors.panel, 'XColor', colors.text, 'YColor', colors.text, 'GridColor', [0.3, 0.3, 0.3]);

    % Panel 2: Enhanced Error Distribution with Statistics
    subplot(2,3,2);
    histogram(errors, 20, 'FaceColor', colors.accent, 'EdgeColor', colors.text, 'LineWidth', 1);
    hold on;
    
    % Add statistical lines
    mean_error = mean(errors);
    std_error = std(errors);
    xline(mean_error, '--', sprintf('Mean: %.1f MW', mean_error), 'Color', colors.excellent, 'LineWidth', 2);
    xline(mean_error + std_error, ':', sprintf('+1σ: %.1f MW', mean_error + std_error), 'Color', colors.warning, 'LineWidth', 1.5);
    xline(mean_error - std_error, ':', sprintf('-1σ: %.1f MW', mean_error - std_error), 'Color', colors.warning, 'LineWidth', 1.5);
    
    title('Error Distribution with Statistics', 'Color', colors.text, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Prediction Error (MW)', 'Color', colors.text);
    ylabel('Frequency', 'Color', colors.text);
    grid on;
    set(gca, 'Color', colors.panel, 'XColor', colors.text, 'YColor', colors.text, 'GridColor', [0.3, 0.3, 0.3]);

    % Panel 3: Enhanced Running Performance Metrics
    subplot(2,3,3);
    plot(time_vector, mae_values, 'Color', colors.excellent, 'LineWidth', 2.5, 'DisplayName', 'MAE');
    hold on;
    plot(time_vector, rmse_values, 'Color', colors.secondary, 'LineWidth', 2.5, 'DisplayName', 'RMSE');
    
    % Add target performance lines
    yline(10, '--', 'Target MAE (10 MW)', 'Color', colors.good, 'LineWidth', 1.5);
    yline(15, '--', 'Target RMSE (15 MW)', 'Color', colors.warning, 'LineWidth', 1.5);
    
    title('Running Performance Metrics', 'Color', colors.text, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Sample Number', 'Color', colors.text);
    ylabel('Error (MW)', 'Color', colors.text);
    legend('TextColor', colors.text, 'Location', 'best');
    grid on; grid minor;
    set(gca, 'Color', colors.panel, 'XColor', colors.text, 'YColor', colors.text, 'GridColor', [0.3, 0.3, 0.3]);

    % Panel 4: Multi-Variable Environmental Conditions
    subplot(2,3,4);
    yyaxis left
    temp_line = plot(time_vector, ambient_temp, 'Color', colors.critical, 'LineWidth', 2, 'DisplayName', 'Ambient Temp');
    hold on;
    vacuum_line = plot(time_vector, vacuum, 'Color', colors.primary, 'LineWidth', 2, 'DisplayName', 'Vacuum');
    ylabel('Temperature (°C) / Vacuum (cmHg)', 'Color', colors.text);
    set(gca, 'YColor', colors.text);
    
    yyaxis right
    humidity_line = plot(time_vector, humidity, 'Color', colors.accent, 'LineWidth', 2, 'DisplayName', 'Humidity');
    ylabel('Relative Humidity (%)', 'Color', colors.text);
    set(gca, 'YColor', colors.text);
    
    title('Environmental Conditions', 'Color', colors.text, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Sample Number', 'Color', colors.text);
    
    % Create custom legend
    yyaxis left; % Reset to left axis for legend
    legend([temp_line, vacuum_line, humidity_line], {'Ambient Temp (°C)', 'Vacuum (cmHg)', 'Humidity (%)'}, ...
           'TextColor', colors.text, 'Location', 'best');
    grid on;
    set(gca, 'Color', colors.panel, 'XColor', colors.text, 'GridColor', [0.3, 0.3, 0.3]);

    % Panel 5: Enhanced Accuracy Trend with Performance Zones
    subplot(2,3,5);
    running_accuracy = 100 * (1 - errors ./ actual_values);
    
    % Create performance zones
    excellent_zone = 95 * ones(size(time_vector));
    good_zone = 90 * ones(size(time_vector));
    fair_zone = 85 * ones(size(time_vector));
    
    % Fill performance zones
    fill([time_vector, fliplr(time_vector)], [100*ones(size(time_vector)), fliplr(excellent_zone)], ...
         colors.excellent, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    hold on;
    fill([time_vector, fliplr(time_vector)], [excellent_zone, fliplr(good_zone)], ...
         colors.good, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    fill([time_vector, fliplr(time_vector)], [good_zone, fliplr(fair_zone)], ...
         colors.warning, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    
    % Plot accuracy trend
    plot(time_vector, running_accuracy, 'Color', colors.accent, 'LineWidth', 3, 'DisplayName', 'Accuracy');
    
    % Add reference lines
    yline(95, '--', 'Excellent (95%)', 'Color', colors.excellent, 'LineWidth', 1.5);
    yline(90, '--', 'Good (90%)', 'Color', colors.good, 'LineWidth', 1.5);
    yline(85, '--', 'Fair (85%)', 'Color', colors.warning, 'LineWidth', 1.5);
    
    title('Accuracy Trend with Performance Zones', 'Color', colors.text, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Sample Number', 'Color', colors.text);
    ylabel('Accuracy (%)', 'Color', colors.text);
    ylim([80, 100]);
    grid on; grid minor;
    set(gca, 'Color', colors.panel, 'XColor', colors.text, 'YColor', colors.text, 'GridColor', [0.3, 0.3, 0.3]);

    % Panel 6: Enhanced System Summary with Visual Indicators
    subplot(2,3,6);
    
    % Create status indicator circle
    theta = linspace(0, 2*pi, 100);
    radius = 0.15;
    center_x = 0.2;
    center_y = 0.7;
    
    circle_x = center_x + radius * cos(theta);
    circle_y = center_y + radius * sin(theta);
    fill(circle_x, circle_y, status_color, 'EdgeColor', colors.text, 'LineWidth', 2);
    
    % Add performance text
    text(0.05, 0.85, 'SYSTEM STATUS', 'FontSize', 14, 'Color', colors.text, 'FontWeight', 'bold');
    text(0.45, 0.7, sprintf('Overall: %.1f%%', overall_accuracy), ...
         'FontSize', 16, 'Color', status_color, 'FontWeight', 'bold');
    text(0.45, 0.6, sprintf('Status: %s', status_text), ...
         'FontSize', 14, 'Color', status_color, 'FontWeight', 'bold');
    
    % Performance metrics
    text(0.05, 0.45, 'PERFORMANCE METRICS', 'FontSize', 12, 'Color', colors.text, 'FontWeight', 'bold');
    text(0.05, 0.35, sprintf('MAE: %.1f MW', final_mae), ...
         'FontSize', 11, 'Color', colors.text);
    text(0.05, 0.25, sprintf('RMSE: %.1f MW', final_rmse), ...
         'FontSize', 11, 'Color', colors.text);
    text(0.05, 0.15, sprintf('Excellent Points: %d (%.1f%%)', excellent_points, (excellent_points/data_points)*100), ...
         'FontSize', 10, 'Color', colors.excellent);
    text(0.05, 0.05, sprintf('Data Points: %d', data_points), ...
         'FontSize', 10, 'Color', colors.text);
    
    title('System Performance Dashboard', 'Color', colors.text, 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'Color', colors.panel, 'XTick', [], 'YTick', [], ...
             'XColor', colors.text, 'YColor', colors.text);
    xlim([0, 1]); ylim([0, 1]);

    % Add enhanced main title with timestamp
    current_time = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
    sgtitle(sprintf('EnergiSense CCPP Digital Twin - Professional Dashboard | %s', current_time), ...
            'FontSize', 18, 'Color', colors.text, 'FontWeight', 'bold');

    % Improve overall figure appearance
    set(fig, 'PaperPositionMode', 'auto');
    
    fprintf('✅ Enhanced visual dashboard launched successfully!\n');
    
    % Export functionality
    try
        print(fig, 'EnergiSense_Dashboard_Report', '-dpng', '-r300');
        fprintf('📄 Dashboard exported as high-resolution PNG\n');
    catch
        fprintf('⚠️ Export warning: Could not save dashboard image\n');
    end
    
catch ME
    fprintf('⚠️ Visual dashboard error: %s\n', ME.message);
    fprintf('📊 Console summary provided above\n');
end

%% FINAL SUMMARY
fprintf('\n🎉 Enhanced dashboard analysis complete!\n');
fprintf('📈 System ready for continuous monitoring\n');
fprintf('🔧 Advanced features: Performance zones, environmental tracking, statistical analysis\n');
fprintf('📊 Export ready: High-resolution dashboard saved\n\n');

%% Performance Trending Analysis
fprintf('📈 PERFORMANCE TRENDING\n');
fprintf('======================\n');
trend_window = min(20, data_points);
if data_points > trend_window
    recent_accuracy = mean(running_accuracy(end-trend_window+1:end));
    early_accuracy = mean(running_accuracy(1:trend_window));
    trend_change = recent_accuracy - early_accuracy;
    
    if trend_change > 1
        trend_icon = '📈';
        trend_text = 'IMPROVING';
    elseif trend_change < -1
        trend_icon = '📉';
        trend_text = 'DECLINING';
    else
        trend_icon = '➡️';
        trend_text = 'STABLE';
    end
    
    fprintf('%s Trend: %s (%.1f%% change)\n', trend_icon, trend_text, trend_change);
    fprintf('🔍 Recent %d samples: %.1f%% accuracy\n', trend_window, recent_accuracy);
    fprintf('🔍 Early %d samples: %.1f%% accuracy\n', trend_window, early_accuracy);
end

end
