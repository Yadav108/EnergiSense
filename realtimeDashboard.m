function realtimeDashboard()
%% Real-time Interactive EnergiSense Dashboard
%% Live updating dashboard with interactive controls

fprintf('🔄 Launching Real-time Interactive Dashboard...\n');

%% Load required data
if ~exist('model', 'var') || ~exist('AT_ts', 'var')
    load('Digitaltwin.mat');
    model_data = load('models/ensemblePowerModel.mat');
    model = classreg.learning.regr.CompactRegressionEnsemble.fromStruct(model_data.compactStruct);
end

%% Create Interactive Figure
fig = figure('Name', 'EnergiSense Real-time Control Center', ...
            'Position', [100, 100, 1600, 900], ...
            'Color', [0.1, 0.1, 0.2], ...
            'DeleteFcn', @cleanupTimer);

%% Initialize Real-time Data
global rt_data;
rt_data = struct();
rt_data.current_idx = 1;
rt_data.max_points = length(AT_ts.Data);
rt_data.time_window = 50; % Show last 50 points
rt_data.setpoint = 400;
rt_data.model = model;

% Calculate bias correction
bias_inputs = [double(AT_ts.Data(1:30)), double(V_ts.Data(1:30)), ...
              double(RH_ts.Data(1:30)), double(AP_ts.Data(1:30))];
bias_preds = predict(rt_data.model, bias_inputs);
rt_data.bias_correction = mean(double(PE_ts.Data(1:30)) - bias_preds);

%% Create UI Controls
uicontrol('Style', 'text', 'Position', [50, 850, 200, 30], ...
          'String', 'Real-time Control Center', 'FontSize', 14, ...
          'BackgroundColor', [0.1, 0.1, 0.2], 'ForegroundColor', 'white');

% Start/Stop Button
rt_data.start_btn = uicontrol('Style', 'pushbutton', 'Position', [300, 850, 80, 30], ...
                             'String', 'START', 'FontSize', 12, ...
                             'BackgroundColor', 'green', 'ForegroundColor', 'white', ...
                             'Callback', @startRealtime);

% Stop Button
rt_data.stop_btn = uicontrol('Style', 'pushbutton', 'Position', [390, 850, 80, 30], ...
                            'String', 'STOP', 'FontSize', 12, ...
                            'BackgroundColor', 'red', 'ForegroundColor', 'white', ...
                            'Callback', @stopRealtime);

% Speed Control
uicontrol('Style', 'text', 'Position', [500, 860, 80, 20], ...
          'String', 'Speed:', 'FontSize', 10, ...
          'BackgroundColor', [0.1, 0.1, 0.2], 'ForegroundColor', 'white');
          
rt_data.speed_slider = uicontrol('Style', 'slider', 'Position', [580, 850, 100, 30], ...
                                'Min', 0.1, 'Max', 2, 'Value', 1, ...
                                'BackgroundColor', [0.3, 0.3, 0.3]);

% Setpoint Control
uicontrol('Style', 'text', 'Position', [700, 860, 80, 20], ...
          'String', 'Setpoint:', 'FontSize', 10, ...
          'BackgroundColor', [0.1, 0.1, 0.2], 'ForegroundColor', 'white');
          
rt_data.setpoint_edit = uicontrol('Style', 'edit', 'Position', [780, 850, 60, 30], ...
                                 'String', '400', 'FontSize', 12, ...
                                 'BackgroundColor', 'white', ...
                                 'Callback', @updateSetpoint);

%% Create Real-time Plots
% Main Power Plot
rt_data.ax1 = subplot(2, 3, [1, 2]);
rt_data.line_actual = plot(NaN, NaN, 'r-', 'LineWidth', 3, 'DisplayName', 'Actual');
hold on;
rt_data.line_predicted = plot(NaN, NaN, 'b-', 'LineWidth', 2, 'DisplayName', 'Predicted');
rt_data.line_setpoint = yline(rt_data.setpoint, 'g--', 'LineWidth', 2, 'DisplayName', 'Setpoint');
xlabel('Time Step'); ylabel('Power (MW)');
title('Real-time Power Tracking', 'Color', 'white');
legend('TextColor', 'white'); grid on;
set(gca, 'Color', [0.15, 0.15, 0.25], 'XColor', 'white', 'YColor', 'white');

% Error Plot
rt_data.ax2 = subplot(2, 3, 3);
rt_data.line_error = plot(NaN, NaN, 'g-', 'LineWidth', 2);
xlabel('Time Step'); ylabel('Error (MW)');
title('Prediction Error', 'Color', 'white');
grid on;
set(gca, 'Color', [0.15, 0.15, 0.25], 'XColor', 'white', 'YColor', 'white');

% Environmental Conditions
rt_data.ax3 = subplot(2, 3, 4);
rt_data.line_AT = plot(NaN, NaN, 'r-', 'LineWidth', 2, 'DisplayName', 'Temp');
hold on;
rt_data.line_V = plot(NaN, NaN, 'b-', 'LineWidth', 2, 'DisplayName', 'Vacuum');
rt_data.line_RH = plot(NaN, NaN, 'g-', 'LineWidth', 2, 'DisplayName', 'Humidity');
rt_data.line_AP = plot(NaN, NaN, 'm-', 'LineWidth', 2, 'DisplayName', 'Pressure');
xlabel('Time Step'); ylabel('Normalized');
title('Environmental Conditions', 'Color', 'white');
legend('TextColor', 'white'); grid on;
set(gca, 'Color', [0.15, 0.15, 0.25], 'XColor', 'white', 'YColor', 'white');

% Accuracy Plot
rt_data.ax4 = subplot(2, 3, 5);
rt_data.line_accuracy = plot(NaN, NaN, 'c-', 'LineWidth', 2);
hold on;
yline(95, 'r--', 'LineWidth', 1, 'DisplayName', 'Target');
xlabel('Time Step'); ylabel('Accuracy (%)');
title('Model Accuracy', 'Color', 'white');
legend('TextColor', 'white'); grid on; ylim([85, 100]);
set(gca, 'Color', [0.15, 0.15, 0.25], 'XColor', 'white', 'YColor', 'white');

% Status Panel
rt_data.ax5 = subplot(2, 3, 6);
axis off;
rt_data.status_text = text(0.1, 0.8, 'STATUS: READY', 'FontSize', 14, 'Color', 'yellow');
rt_data.current_text = text(0.1, 0.6, 'Current: -- MW', 'FontSize', 12, 'Color', 'white');
rt_data.error_text = text(0.1, 0.4, 'Error: -- MW', 'FontSize', 12, 'Color', 'white');
rt_data.accuracy_text = text(0.1, 0.2, 'Accuracy: --%', 'FontSize', 12, 'Color', 'white');
xlim([0, 1]); ylim([0, 1]);
title('System Status', 'Color', 'white');

%% Initialize Timer (but don't start)
rt_data.timer = timer('TimerFcn', @updateRealtime, 'Period', 0.5, ...
                     'ExecutionMode', 'fixedRate', 'BusyMode', 'drop');

fprintf('✅ Real-time dashboard ready!\n');
fprintf('🎮 Controls: START/STOP buttons, Speed slider, Setpoint control\n');
fprintf('📊 Features: Live updating plots, real-time status, interactive controls\n');

end

%% Timer Update Function
function updateRealtime(~, ~)
    global rt_data;
    
    try
        % Get current data point
        idx = rt_data.current_idx;
        if idx > rt_data.max_points
            rt_data.current_idx = 1; % Loop back
            idx = 1;
        end
        
        % Get current environmental inputs
        AT_val = double(evalin('base', sprintf('AT_ts.Data(%d)', idx)));
        V_val = double(evalin('base', sprintf('V_ts.Data(%d)', idx)));
        RH_val = double(evalin('base', sprintf('RH_ts.Data(%d)', idx)));
        AP_val = double(evalin('base', sprintf('AP_ts.Data(%d)', idx)));
        actual_val = double(evalin('base', sprintf('PE_ts.Data(%d)', idx)));
        
        % Make prediction
        inputs = [AT_val, V_val, RH_val, AP_val];
        predicted_val = predict(rt_data.model, inputs) + rt_data.bias_correction;
        
        % Calculate metrics
        error_val = abs(predicted_val - actual_val);
        accuracy_val = max(0, 100 - (error_val / actual_val) * 100);
        
        % Update data arrays (keep sliding window)
        window_size = rt_data.time_window;
        start_idx = max(1, idx - window_size + 1);
        end_idx = idx;
        time_range = start_idx:end_idx;
        
        % Update plots
        updatePlotData(rt_data.line_actual, time_range, ...
                      double(evalin('base', sprintf('PE_ts.Data(%d:%d)', start_idx, end_idx))));
        updatePlotData(rt_data.line_predicted, time_range, ...
                      arrayfun(@(i) predict(rt_data.model, ...
                      [double(evalin('base', sprintf('AT_ts.Data(%d)', i))), ...
                       double(evalin('base', sprintf('V_ts.Data(%d)', i))), ...
                       double(evalin('base', sprintf('RH_ts.Data(%d)', i))), ...
                       double(evalin('base', sprintf('AP_ts.Data(%d)', i)))]) + rt_data.bias_correction, ...
                      time_range));
        
        % Update error plot
        actual_range = double(evalin('base', sprintf('PE_ts.Data(%d:%d)', start_idx, end_idx)));
        predicted_range = arrayfun(@(i) predict(rt_data.model, ...
                         [double(evalin('base', sprintf('AT_ts.Data(%d)', i))), ...
                          double(evalin('base', sprintf('V_ts.Data(%d)', i))), ...
                          double(evalin('base', sprintf('RH_ts.Data(%d)', i))), ...
                          double(evalin('base', sprintf('AP_ts.Data(%d)', i)))]) + rt_data.bias_correction, ...
                         time_range);
        errors = abs(predicted_range - actual_range);
        updatePlotData(rt_data.line_error, time_range, errors);
        
        % Update environmental plots (normalized)
        AT_range = double(evalin('base', sprintf('AT_ts.Data(%d:%d)', start_idx, end_idx)));
        V_range = double(evalin('base', sprintf('V_ts.Data(%d:%d)', start_idx, end_idx)));
        RH_range = double(evalin('base', sprintf('RH_ts.Data(%d:%d)', start_idx, end_idx)));
        AP_range = double(evalin('base', sprintf('AP_ts.Data(%d:%d)', start_idx, end_idx)));
        
        updatePlotData(rt_data.line_AT, time_range, normalize(AT_range));
        updatePlotData(rt_data.line_V, time_range, normalize(V_range));
        updatePlotData(rt_data.line_RH, time_range, normalize(RH_range));
        updatePlotData(rt_data.line_AP, time_range, normalize(AP_range));
        
        % Update accuracy
        accuracies = max(0, 100 - (errors ./ actual_range) * 100);
        updatePlotData(rt_data.line_accuracy, time_range, accuracies);
        
        % Update status text
        set(rt_data.status_text, 'String', sprintf('STATUS: RUNNING (Point %d)', idx));
        set(rt_data.current_text, 'String', sprintf('Current: %.1f MW', predicted_val));
        set(rt_data.error_text, 'String', sprintf('Error: %.1f MW', error_val));
        set(rt_data.accuracy_text, 'String', sprintf('Accuracy: %.1f%%', accuracy_val));
        
        % Update setpoint line
        rt_data.line_setpoint.Value = rt_data.setpoint;
        
        % Auto-scale axes
        for ax = [rt_data.ax1, rt_data.ax2, rt_data.ax3, rt_data.ax4]
            xlim(ax, [start_idx, end_idx]);
        end
        
        drawnow;
        
        % Increment index
        rt_data.current_idx = idx + 1;
        
    catch ME
        fprintf('Real-time update error: %s\n', ME.message);
    end
end

%% Helper Functions
function updatePlotData(line_handle, x_data, y_data)
    set(line_handle, 'XData', x_data, 'YData', y_data);
end

function norm_data = normalize(data)
    norm_data = (data - min(data)) / (max(data) - min(data));
end

function startRealtime(~, ~)
    global rt_data;
    start(rt_data.timer);
    set(rt_data.status_text, 'String', 'STATUS: STARTING...', 'Color', 'green');
end

function stopRealtime(~, ~)
    global rt_data;
    stop(rt_data.timer);
    set(rt_data.status_text, 'String', 'STATUS: STOPPED', 'Color', 'red');
end

function updateSetpoint(src, ~)
    global rt_data;
    rt_data.setpoint = str2double(get(src, 'String'));
end

function cleanupTimer(~, ~)
    global rt_data;
    if isfield(rt_data, 'timer') && isvalid(rt_data.timer)
        stop(rt_data.timer);
        delete(rt_data.timer);
    end
end