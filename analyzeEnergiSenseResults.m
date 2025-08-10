function analyzeEnergiSenseResults(simout)
%ANALYZEENERGISENSERESULTS Improved analysis for EnergiSense simulation results

fprintf('=== ENERGISENSE SIMULATION ANALYSIS ===\n');

% Handle different simulation output formats
if isa(simout, 'Simulink.SimulationOutput')
    fprintf('Simulation output type: Simulink.SimulationOutput\n');
    
    % Get available signals
    try
        signal_names = simout.who;
        fprintf('Available logged signals:\n');
        for i = 1:length(signal_names)
            fprintf('  - %s\n', signal_names{i});
        end
        
        % Try to extract common signals
        if any(strcmp(signal_names, 'yout'))
            fprintf('Using yout for analysis\n');
            analyzeTimeSeriesData(simout.yout, simout.tout);
        else
            fprintf('Extracting individual signals...\n');
            extractIndividualSignals(simout, signal_names);
        end
        
    catch ME
        fprintf('Error accessing SimulationOutput: %s\n', ME.message);
        analyzeBasicOutput(simout);
    end
    
elseif isstruct(simout)
    fprintf('Simulation output type: Structure\n');
    
    % Handle structure format
    fields = fieldnames(simout);
    fprintf('Available fields:\n');
    for i = 1:length(fields)
        fprintf('  - %s\n', fields{i});
    end
    
    % Look for standard signal names
    signal_names = {'setpoint', 'predicted_power', 'actual_power', 'control_signal'};
    found_signals = {};
    for i = 1:length(signal_names)
        if isfield(simout, signal_names{i})
            found_signals{end+1} = signal_names{i};
        end
    end
    
    if length(found_signals) >= 2
        fprintf('Found signals for analysis: %s\n', strjoin(found_signals, ', '));
        analyzeStructureSignals(simout, found_signals);
    else
        fprintf('Insufficient signals for analysis\n');
        fprintf('Enable signal logging in Simulink and re-run simulation\n');
    end
    
else
    fprintf('Unknown simulation output format: %s\n', class(simout));
    analyzeBasicOutput(simout);
end

end

function analyzeTimeSeriesData(yout, tout)
    fprintf('\\n--- TIME SERIES ANALYSIS ---\\n');
    fprintf('Time vector length: %d\\n', length(tout));
    fprintf('Output signals: %d\\n', size(yout, 2));
    
    % Basic plot
    figure('Name', 'EnergiSense Simulation Results');
    plot(tout, yout);
    xlabel('Time (s)');
    ylabel('Signals');
    title('Simulation Results');
    grid on;
    legend('show');
    
    fprintf('✅ Basic analysis plot created\\n');
end

function extractIndividualSignals(simout, signal_names)
    fprintf('\\n--- INDIVIDUAL SIGNALS ANALYSIS ---\\n');
    
    for i = 1:length(signal_names)
        signal_name = signal_names{i};
        try
            signal_data = simout.get(signal_name);
            fprintf('Signal %s: %d samples\\n', signal_name, length(signal_data.Values.Data));
        catch
            fprintf('Could not access signal: %s\\n', signal_name);
        end
    end
end

function analyzeStructureSignals(simout, signal_names)
    fprintf('\\n--- STRUCTURE SIGNALS ANALYSIS ---\\n');
    
    % Extract time and data
    first_signal = signal_names{1};
    time = simout.(first_signal).Time;
    
    fprintf('Time vector: %d samples (%.1f to %.1f seconds)\\n', ...
        length(time), time(1), time(end));
    
    % Create analysis plot
    figure('Name', 'EnergiSense Analysis', 'Position', [100, 100, 1200, 600]);
    
    for i = 1:length(signal_names)
        signal_name = signal_names{i};
        signal_data = simout.(signal_name).Data;
        
        subplot(2, 2, i);
        plot(time, signal_data, 'LineWidth', 1.5);
        title(signal_name, 'Interpreter', 'none');
        xlabel('Time (s)');
        ylabel('Value');
        grid on;
        
        % Basic statistics
        fprintf('%s: mean=%.2f, std=%.2f, range=[%.2f, %.2f]\\n', ...
            signal_name, mean(signal_data), std(signal_data), ...
            min(signal_data), max(signal_data));
    end
    
    fprintf('✅ Analysis plots created\\n');
end

function analyzeBasicOutput(simout)
    fprintf('\\n--- BASIC OUTPUT ANALYSIS ---\\n');
    fprintf('Output class: %s\\n', class(simout));
    fprintf('Output size: %s\\n', mat2str(size(simout)));
    
    if isnumeric(simout)
        fprintf('Numeric data statistics:\\n');
        fprintf('  Mean: %.4f\\n', mean(simout(:)));
        fprintf('  Std: %.4f\\n', std(simout(:)));
        fprintf('  Range: [%.4f, %.4f]\\n', min(simout(:)), max(simout(:)));
    end
end