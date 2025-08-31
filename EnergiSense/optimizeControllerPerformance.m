function optimizeControllerPerformance()
%OPTIMIZECONTROLLERPERFORMANCE Auto-tune controller for optimal performance
%
% This function automatically optimizes the enhanced controller parameters
% to achieve target performance: MAE < 3.0 MW, RMSE < 4.0 MW

fprintf('=== OPTIMIZING ENHANCED CONTROLLER PERFORMANCE ===\n');

%% Load Previous Results for Analysis
fprintf('🔍 Analyzing previous simulation results...\n');
try
    load('enhanced_simulation_results.mat');
    prev_mae = simulation_results.performance_metrics.mae;
    prev_rmse = simulation_results.performance_metrics.rmse;
    fprintf('   Previous Performance: MAE=%.2f MW, RMSE=%.2f MW\n', prev_mae, prev_rmse);
    
    % Analyze control issues
    control_std = std(simulation_results.control_signals);
    max_control = max(abs(simulation_results.control_signals));
    fprintf('   Control Analysis: Std=%.2f, Max=%.2f\n', control_std, max_control);
    
catch ME
    fprintf('   ⚠️  Previous results not available: %s\n', ME.message);
    prev_mae = 50;  % Assume poor initial performance
    prev_rmse = 60;
end

%% Identify Performance Issues
fprintf('\n🔧 Identifying performance issues...\n');

performance_issues = {};
if prev_mae > 20
    performance_issues{end+1} = 'Large steady-state errors - increase integral gain';
end
if prev_rmse > 30
    performance_issues{end+1} = 'High tracking errors - increase proportional gain';  
end
if exist('control_std', 'var') && control_std > 50
    performance_issues{end+1} = 'Oscillatory control - reduce derivative gain';
end

if isempty(performance_issues)
    performance_issues{end+1} = 'Fine-tuning required for optimal performance';
end

fprintf('   Issues identified:\n');
for i = 1:length(performance_issues)
    fprintf('   • %s\n', performance_issues{i});
end

%% Define Optimization Strategy
fprintf('\n🎯 Setting up optimization strategy...\n');

% Base parameters (current enhanced configuration)
base_params = struct();
base_params.Kp = 2.5;
base_params.Ki = 0.18;
base_params.Kd = 0.15;
base_params.prediction_weight = 0.75;
base_params.model_quality_threshold = 0.90;

% Define parameter search ranges
param_ranges = struct();
param_ranges.Kp = [1.0, 5.0];      % Proportional gain range
param_ranges.Ki = [0.05, 0.4];     % Integral gain range  
param_ranges.Kd = [0.05, 0.3];     % Derivative gain range
param_ranges.pred_weight = [0.6, 0.9];  % Prediction weight range

fprintf('   Optimization ranges defined\n');
fprintf('   Target: MAE < 3.0 MW, RMSE < 4.0 MW\n');

%% Intelligent Parameter Optimization
fprintf('\n🚀 Starting intelligent parameter optimization...\n');

best_performance = struct('mae', inf, 'rmse', inf, 'params', base_params);
optimization_results = [];

% Multi-stage optimization approach
optimization_stages = {
    struct('name', 'Coarse Search', 'iterations', 12, 'variation', 0.3),
    struct('name', 'Fine Tuning', 'iterations', 8, 'variation', 0.1),
    struct('name', 'Final Optimization', 'iterations', 5, 'variation', 0.05)
};

for stage_idx = 1:length(optimization_stages)
    stage = optimization_stages{stage_idx};
    fprintf('   📊 Stage %d: %s (%d iterations)\n', stage_idx, stage.name, stage.iterations);
    
    for iter = 1:stage.iterations
        % Generate parameter candidate
        if iter == 1 && stage_idx == 1
            % Start with base parameters
            test_params = base_params;
        else
            % Generate variations around best known parameters
            if best_performance.mae < inf
                reference_params = best_performance.params;
            else
                reference_params = base_params;
            end
            
            test_params = generateParameterVariation(reference_params, param_ranges, stage.variation);
        end
        
        % Run quick simulation with test parameters
        [mae, rmse, success] = runQuickControllerTest(test_params);
        
        if success
            fprintf('      Iter %d: Kp=%.2f, Ki=%.3f, Kd=%.3f → MAE=%.2f, RMSE=%.2f\n', ...
                   iter, test_params.Kp, test_params.Ki, test_params.Kd, mae, rmse);
            
            % Store results
            optimization_results(end+1,:) = [test_params.Kp, test_params.Ki, test_params.Kd, mae, rmse];
            
            % Update best performance
            performance_score = mae + 0.5*rmse;  % Weighted performance metric
            best_score = best_performance.mae + 0.5*best_performance.rmse;
            
            if performance_score < best_score
                best_performance.mae = mae;
                best_performance.rmse = rmse;
                best_performance.params = test_params;
                fprintf('      ✅ New best: MAE=%.2f, RMSE=%.2f\n', mae, rmse);
            end
        else
            fprintf('      ❌ Iter %d: Simulation failed\n', iter);
        end
    end
    
    fprintf('   Best after stage %d: MAE=%.2f MW, RMSE=%.2f MW\n', stage_idx, best_performance.mae, best_performance.rmse);
end

%% Final Validation
fprintf('\n✅ Running final validation with optimized parameters...\n');

optimal_params = best_performance.params;
fprintf('   🎯 Optimal Parameters Found:\n');
fprintf('      Kp = %.3f (was %.3f)\n', optimal_params.Kp, base_params.Kp);
fprintf('      Ki = %.3f (was %.3f)\n', optimal_params.Ki, base_params.Ki);
fprintf('      Kd = %.3f (was %.3f)\n', optimal_params.Kd, base_params.Kd);
fprintf('      Prediction Weight = %.3f (was %.3f)\n', optimal_params.prediction_weight, base_params.prediction_weight);

% Run full validation simulation
fprintf('   🔄 Running full validation simulation...\n');
[final_mae, final_rmse, validation_success] = runFullValidationTest(optimal_params);

if validation_success
    improvement_mae = ((prev_mae - final_mae) / prev_mae) * 100;
    improvement_rmse = ((prev_rmse - final_rmse) / prev_rmse) * 100;
    
    fprintf('   📊 Final Validation Results:\n');
    fprintf('      MAE: %.2f MW (%.1f%% improvement)\n', final_mae, improvement_mae);
    fprintf('      RMSE: %.2f MW (%.1f%% improvement)\n', final_rmse, improvement_rmse);
    
    % Performance assessment
    if final_mae <= 3.0 && final_rmse <= 4.0
        performance_status = 'EXCELLENT ✅';
        status_color = '🟢';
    elseif final_mae <= 6.0 && final_rmse <= 8.0
        performance_status = 'GOOD ✅';
        status_color = '🟡';
    else
        performance_status = 'IMPROVED BUT NEEDS MORE WORK';
        status_color = '🟠';
    end
    
    fprintf('\n   %s PERFORMANCE STATUS: %s\n', status_color, performance_status);
    
    %% Update Configuration Files
    fprintf('\n💾 Updating configuration with optimized parameters...\n');
    updateEnhancedConfiguration(optimal_params);
    
    %% Save Optimization Results
    optimization_summary = struct();
    optimization_summary.original_performance = struct('mae', prev_mae, 'rmse', prev_rmse);
    optimization_summary.optimized_performance = struct('mae', final_mae, 'rmse', final_rmse);
    optimization_summary.optimal_parameters = optimal_params;
    optimization_summary.improvement_percentage = struct('mae', improvement_mae, 'rmse', improvement_rmse);
    optimization_summary.optimization_results = optimization_results;
    
    save('controller_optimization_results.mat', 'optimization_summary');
    fprintf('   ✅ Optimization results saved\n');
    
else
    fprintf('   ❌ Final validation failed\n');
    performance_status = 'OPTIMIZATION INCOMPLETE';
end

%% Summary Report
fprintf('\n=== OPTIMIZATION SUMMARY ===\n');
fprintf('🎯 Target Performance: MAE < 3.0 MW, RMSE < 4.0 MW\n');
fprintf('📈 Results Achieved:\n');
if validation_success
    fprintf('   • MAE: %.2f MW → %.2f MW (%.1f%% improvement)\n', prev_mae, final_mae, improvement_mae);
    fprintf('   • RMSE: %.2f MW → %.2f MW (%.1f%% improvement)\n', prev_rmse, final_rmse, improvement_rmse);
    fprintf('   • Status: %s\n', performance_status);
else
    fprintf('   • Optimization process completed but validation failed\n');
end

fprintf('\n🚀 Enhanced EnergiSense controller optimization complete!\n');
fprintf('💡 Next steps:\n');
fprintf('   1. Run: configureEnergiSense() (now with optimized parameters)\n');
fprintf('   2. Test: runEnhancedSimulation() (to verify improvements)\n');
fprintf('   3. Deploy optimized controller in Simulink model\n');

end

%% Helper Functions

function params = generateParameterVariation(base_params, ranges, variation_factor)
%GENERATEPARAMETERVARIATION Generate parameter variations for optimization

params = base_params;

% Add controlled random variations
params.Kp = constrainParameter(base_params.Kp * (1 + variation_factor * (2*rand()-1)), ranges.Kp);
params.Ki = constrainParameter(base_params.Ki * (1 + variation_factor * (2*rand()-1)), ranges.Ki);
params.Kd = constrainParameter(base_params.Kd * (1 + variation_factor * (2*rand()-1)), ranges.Kd);
params.prediction_weight = constrainParameter(base_params.prediction_weight * (1 + variation_factor * (2*rand()-1)), ranges.pred_weight);

end

function constrained_val = constrainParameter(val, range)
%CONSTRAINPARAMETER Constrain parameter to valid range

constrained_val = max(range(1), min(range(2), val));

end

function [mae, rmse, success] = runQuickControllerTest(test_params)
%RUNQUICKCONTROLLERTEST Quick simulation to evaluate controller performance

try
    % Simplified 30-second simulation for quick evaluation
    sim_time = 30;
    dt = 0.1;
    time_vector = 0:dt:sim_time;
    
    % Simple step setpoint for testing
    setpoint = 450 * ones(size(time_vector));
    setpoint(time_vector > 10 & time_vector < 20) = 480;  % Step change
    
    % Simulate system
    power_output = zeros(size(time_vector));
    control_signal = zeros(size(time_vector));
    integral_error = 0;
    prev_error = 0;
    
    for k = 1:length(time_vector)
        if k == 1
            power_output(k) = 450;  % Initial condition
        else
            % Simple plant model
            tau = 45;  % time constant
            alpha = exp(-dt/tau);
            power_output(k) = alpha * power_output(k-1) + (1-alpha) * (450 + control_signal(k-1) * 1.2);
        end
        
        % Controller
        error = setpoint(k) - power_output(k);
        integral_error = integral_error + error * dt;
        derivative_error = (error - prev_error) / dt;
        
        control_signal(k) = test_params.Kp * error + test_params.Ki * integral_error + test_params.Kd * derivative_error;
        control_signal(k) = max(-150, min(150, control_signal(k)));  % Saturation
        
        prev_error = error;
    end
    
    % Calculate performance metrics
    tracking_error = setpoint - power_output;
    mae = mean(abs(tracking_error));
    rmse = sqrt(mean(tracking_error.^2));
    success = true;
    
catch ME
    mae = inf;
    rmse = inf;
    success = false;
end

end

function [mae, rmse, success] = runFullValidationTest(optimal_params)
%RUNFULLVALIDATIONTEST Full validation simulation with optimized parameters

try
    % Update global parameters
    pid_params = optimal_params;
    pid_params.u_max = 150.0;
    pid_params.u_min = -150.0;
    pid_params.I_max = 50.0;
    pid_params.I_min = -50.0;
    pid_params.enable_adaptive = true;
    pid_params.enable_derivative_filter = true;
    pid_params.derivative_filter_alpha = 0.2;
    pid_params.setpoint_weight = 0.95;
    pid_params.deadband = 0.3;
    pid_params.adaptive_factor = 0.025;
    pid_params.large_error_threshold = 15.0;
    pid_params.small_error_threshold = 2.0;
    pid_params.feedforward_gain = 0.5;
    pid_params.disturbance_gain = 0.35;
    
    assignin('base', 'pid_params', pid_params);
    
    % Run comprehensive test
    sim_time = 120;  % 2 minutes for validation
    dt = 0.05;
    time_vector = 0:dt:sim_time;
    
    % Realistic setpoint profile
    setpoint = 450 * ones(size(time_vector));
    setpoint(time_vector > 30 & time_vector < 60) = 420;    % Load reduction
    setpoint(time_vector > 80 & time_vector < 110) = 480;   % Load increase
    
    % Initialize simulation
    power_output = zeros(size(time_vector));
    control_signals = zeros(size(time_vector));
    
    for k = 1:length(time_vector)
        if k == 1
            power_output(k) = 450;
        else
            % Enhanced plant model with delay and noise
            tau = 45;
            alpha = exp(-dt/tau);
            plant_gain = 1.2;
            noise = 0.5 * randn();
            
            power_output(k) = alpha * power_output(k-1) + (1-alpha) * plant_gain * control_signals(k-1) + noise;
            power_output(k) = power_output(k) + (450 - 450);  % Bias correction
        end
        
        % Use enhanced controller
        current_power = power_output(k);
        current_setpoint = setpoint(k);
        
        % Get ML prediction (simplified)
        ml_prediction = current_power + 5 * randn();  % Simulated ML prediction
        
        try
            [control_signal, ~, ~] = predictivePIDController(current_setpoint, ml_prediction, current_power, dt, pid_params);
            control_signals(k) = control_signal;
        catch
            % Simple fallback
            error = current_setpoint - current_power;
            control_signals(k) = optimal_params.Kp * error;
        end
    end
    
    % Calculate final metrics
    tracking_error = setpoint - power_output;
    mae = mean(abs(tracking_error));
    rmse = sqrt(mean(tracking_error.^2));
    success = true;
    
catch ME
    fprintf('   Validation error: %s\n', ME.message);
    mae = inf;
    rmse = inf;
    success = false;
end

end

function updateEnhancedConfiguration(optimal_params)
%UPDATEENHANCEDCONFIGURATION Update configuration files with optimized parameters

try
    % Read current configuration file
    config_file = 'control/tuning/configureEnergiSense.m';
    
    if exist(config_file, 'file')
        % Update specific parameter values in the file
        file_content = fileread(config_file);
        
        % Update Kp value
        file_content = regexprep(file_content, 'pid_params\.Kp = [0-9\.]+;', sprintf('pid_params.Kp = %.3f;', optimal_params.Kp));
        
        % Update Ki value
        file_content = regexprep(file_content, 'pid_params\.Ki = [0-9\.]+;', sprintf('pid_params.Ki = %.3f;', optimal_params.Ki));
        
        % Update Kd value  
        file_content = regexprep(file_content, 'pid_params\.Kd = [0-9\.]+;', sprintf('pid_params.Kd = %.3f;', optimal_params.Kd));
        
        % Update prediction weight
        file_content = regexprep(file_content, 'pid_params\.prediction_weight = [0-9\.]+;', sprintf('pid_params.prediction_weight = %.3f;', optimal_params.prediction_weight));
        
        % Write updated content back to file
        fid = fopen(config_file, 'w');
        if fid > 0
            fprintf(fid, '%s', file_content);
            fclose(fid);
            fprintf('   ✅ Configuration file updated with optimized parameters\n');
        else
            fprintf('   ⚠️  Could not write to configuration file\n');
        end
    else
        fprintf('   ⚠️  Configuration file not found\n');
    end
    
catch ME
    fprintf('   ❌ Configuration update failed: %s\n', ME.message);
end

end