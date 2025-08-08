%% FILE 1: Save this as "runDashboard.m"

function runDashboard()
    % Real-time Dashboard for Digital Twin CCPP System
    
    fprintf('Starting Digital Twin Dashboard...\n');
    fprintf('Make sure your Simulink model is running!\n\n');
    
    % Create dashboard figure
    fig = figure('Name', 'Digital Twin CCPP Dashboard', ...
                 'Position', [100, 100, 1200, 800], ...
                 'Color', [0.1 0.1 0.2]);
    
    % Create subplots
    % Subplot 1: Power Tracking
    subplot(2,3,1);
    h_power = plot(0, 0, 'b-', 'LineWidth', 2); hold on;
    h_actual = plot(0, 0, 'r--', 'LineWidth', 2);
    h_setpoint = plot(0, 400, 'g:', 'LineWidth', 2);
    title('Power Tracking', 'Color', 'white', 'FontSize', 12);
    xlabel('Time (s)', 'Color', 'white');
    ylabel('Power (MW)', 'Color', 'white');
    legend('Predicted', 'Actual', 'Setpoint', 'TextColor', 'white');
    grid on; 
    set(gca, 'Color', [0.15 0.15 0.25], 'XColor', 'white', 'YColor', 'white');
    
    % Subplot 2: Control Signal
    subplot(2,3,2);
    h_control = plot(0, 0, 'k-', 'LineWidth', 2);
    title('Control Signal', 'Color', 'white', 'FontSize', 12);
    xlabel('Time (s)', 'Color', 'white');
    ylabel('Control Effort', 'Color', 'white');
    grid on;
    set(gca, 'Color', [0.15 0.15 0.25], 'XColor', 'white', 'YColor', 'white');
    
    % Subplot 3: Performance Metrics
    subplot(2,3,3);
    h_metrics = bar([0 0 0], 'FaceColor', [0.3 0.6 0.9]);
    title('Performance Metrics', 'Color', 'white', 'FontSize', 12);
    set(gca, 'XTickLabel', {'Efficiency', 'Accuracy', 'Stability'}, ...
             'Color', [0.15 0.15 0.25], 'XColor', 'white', 'YColor', 'white');
    ylim([0 100]);
    
    % Subplot 4: Environmental Conditions  
    subplot(2,3,4);
    h_temp = plot(0, 0, 'r-', 'LineWidth', 2); hold on;
    h_pressure = plot(0, 0, 'b-', 'LineWidth', 2);
    h_humidity = plot(0, 0, 'g-', 'LineWidth', 2);
    h_vacuum = plot(0, 0, 'm-', 'LineWidth', 2);
    title('Environmental Conditions', 'Color', 'white', 'FontSize', 12);
    xlabel('Time (s)', 'Color', 'white');
    ylabel('Normalized Values', 'Color', 'white');
    legend('Temp', 'Pressure', 'Humidity', 'Vacuum', 'TextColor', 'white');
    grid on;
    set(gca, 'Color', [0.15 0.15 0.25], 'XColor', 'white', 'YColor', 'white');
    
    % Subplot 5: System Status
    subplot(2,3,5);
    status_text = text(0.1, 0.8, 'System Status:', 'Color', 'white', 'FontSize', 14, 'FontWeight', 'bold');
    status_power = text(0.1, 0.6, 'Power: -- MW', 'Color', 'cyan', 'FontSize', 12);
    status_error = text(0.1, 0.4, 'Error: -- MW', 'Color', 'yellow', 'FontSize', 12);
    status_control = text(0.1, 0.2, 'Control: --', 'Color', 'green', 'FontSize', 12);
    set(gca, 'Color', [0.15 0.15 0.25], 'XTick', [], 'YTick', [], 'XColor', 'white', 'YColor', 'white');
    xlim([0 1]); ylim([0 1]);
    
    % Subplot 6: Control Panel
    subplot(2,3,6);
    % Create control buttons (simplified for display)
    btn_text = text(0.1, 0.8, 'Control Panel:', 'Color', 'white', 'FontSize', 14, 'FontWeight', 'bold');
    setpoint_text = text(0.1, 0.6, 'Setpoint: 400 MW', 'Color', 'cyan', 'FontSize', 12);
    mode_text = text(0.1, 0.4, 'Mode: PID Control', 'Color', 'green', 'FontSize', 12);
    tuning_text = text(0.1, 0.2, 'Kp=1.5 Ki=0.1 Kd=0.05', 'Color', 'yellow', 'FontSize', 10);
    set(gca, 'Color', [0.15 0.15 0.25], 'XTick', [], 'YTick', [], 'XColor', 'white', 'YColor', 'white');
    xlim([0 1]); ylim([0 1]);
    
    % Main dashboard loop
    fprintf('Dashboard started! Monitoring data from Simulink...\n');
    fprintf('Press Ctrl+C to stop\n\n');
    
    % Initialize data storage
    max_points = 100;
    time_data = zeros(1, max_points);
    power_data = zeros(1, max_points);
    actual_data = zeros(1, max_points);
    control_data = zeros(1, max_points);
    
    % Real-time update loop
    update_count = 0;
    while ishandle(fig)
        try
            % Get data from workspace (check if variables exist)
            if evalin('base', 'exist(''predicted_power'', ''var'')')
                % Get latest data from Simulink
                sim_power = evalin('base', 'predicted_power');
                sim_control = evalin('base', 'control_signal');
                
                if evalin('base', 'exist(''actual_power'', ''var'')')
                    sim_actual = evalin('base', 'actual_power');
                else
                    sim_actual = sim_power + randn*10; % Simulated if not available
                end
                
                % Update data arrays (shift and add new data)
                if length(sim_power) > update_count + 1
                    update_count = update_count + 1;
                    
                    % Shift data left
                    time_data = [time_data(2:end), update_count*0.1];
                    power_data = [power_data(2:end), sim_power(end)];
                    control_data = [control_data(2:end), sim_control(end)];
                    
                    if length(sim_actual) >= update_count
                        actual_data = [actual_data(2:end), sim_actual(min(end, update_count))];
                    else
                        actual_data = [actual_data(2:end), power_data(end) + randn*5];
                    end
                    
                    % Update plots
                    set(h_power, 'XData', time_data, 'YData', power_data);
                    set(h_actual, 'XData', time_data, 'YData', actual_data);
                    set(h_setpoint, 'XData', time_data, 'YData', 400*ones(size(time_data)));
                    set(h_control, 'XData', time_data, 'YData', control_data);
                    
                    % Update status text
                    current_power = power_data(end);
                    current_error = 400 - current_power;
                    current_control = control_data(end);
                    
                    set(status_power, 'String', sprintf('Power: %.1f MW', current_power));
                    set(status_error, 'String', sprintf('Error: %.1f MW', current_error));
                    set(status_control, 'String', sprintf('Control: %.2f', current_control));
                    
                    % Update performance metrics
                    efficiency = max(0, min(100, 100 - abs(current_error)));
                    accuracy = max(0, min(100, 95 - abs(current_error)*2));
                    stability = max(0, min(100, 100 - std(power_data(end-9:end))*5));
                    
                    set(h_metrics, 'YData', [efficiency, accuracy, stability]);
                    
                    % Auto-scale axes
                    subplot(2,3,1);
                    if max(power_data) > min(power_data)
                        ylim([min([power_data, actual_data])-50, max([power_data, actual_data])+50]);
                    end
                    xlim([time_data(1), time_data(end)+1]);
                    
                    subplot(2,3,2);
                    if max(control_data) > min(control_data)
                        ylim([min(control_data)-1, max(control_data)+1]);
                    end
                    xlim([time_data(1), time_data(end)+1]);
                end
            else
                % No data available yet
                fprintf('Waiting for Simulink data... Make sure model is running.\n');
            end
            
            % Refresh display
            drawnow;
            pause(0.5); % Update every 0.5 seconds
            
        catch ME
            fprintf('Dashboard error: %s\n', ME.message);
            pause(1);
        end
    end
    
    fprintf('Dashboard stopped.\n');
end