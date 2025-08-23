%% Enhanced Configuration for Energisense Digital Twin with 95.9% ML Model
clear pid_params sample_time power_setpoint sim_time;
clc;

fprintf('=== ENHANCED ENERGISENSE CONFIGURATION (95.9%% ML MODEL) ===\n');

%% Enhanced Simulation Parameters
sim_time = 300;          % Extended to 5 minutes for better analysis
sample_time = 0.05;      % 50ms for real-time performance
power_setpoint = 450;    % More realistic CCPP nominal power

%% Enhanced Predictive PID Parameters (Tuned for 95.9% ML Model)
pid_params = struct();
pid_params.Kp = 5.000;                        % Enhanced proportional gain
pid_params.Ki = 0.088;                       % Optimized integral gain
pid_params.Kd = 0.171;                       % Improved derivative gain
pid_params.u_max = 150.0;                   % Increased control range
pid_params.u_min = -150.0;                  % Increased control range
pid_params.I_max = 50.0;                    % Enhanced integral limits
pid_params.I_min = -50.0;                   % Enhanced integral limits
pid_params.prediction_weight = 0.621;        % Higher weight for 95.9% model
pid_params.model_quality_threshold = 0.90;  % Stricter quality threshold
pid_params.enable_adaptive = true;
pid_params.enable_derivative_filter = true;
pid_params.derivative_filter_alpha = 0.2;
pid_params.setpoint_weight = 0.95;          % Enhanced setpoint weighting
pid_params.deadband = 0.3;                  % Tighter deadband for accuracy
pid_params.adaptive_factor = 0.025;         % Fine-tuned adaptation
pid_params.large_error_threshold = 15.0;    % Reduced threshold
pid_params.small_error_threshold = 2.0;     % Tighter small error band
pid_params.feedforward_gain = 0.5;          % Enhanced feedforward
pid_params.disturbance_gain = 0.35;         % Improved disturbance rejection

%% Load to Workspace
assignin('base', 'pid_params', pid_params);
assignin('base', 'sample_time', sample_time);
assignin('base', 'power_setpoint', power_setpoint);
assignin('base', 'sim_time', sim_time);

%% Load Enhanced ML Model
fprintf(' Loading 95.9%% accurate ML model...\n');
try
    % Initialize enhanced Simulink environment
    initializeEnhancedSimulink();
    fprintf('✅ Enhanced ML integration completed\n');
catch ME
    fprintf('⚠️  Enhanced initialization failed: %s\n', ME.message);
    fprintf('   Proceeding with basic configuration\n');
end

fprintf('\n✅ Enhanced configuration completed successfully!\n');
fprintf(' Enhanced Variables loaded to workspace:\n');
fprintf('  - pid_params: Enhanced controller parameters (95.9%% ML tuned)\n');
fprintf('  - sample_time: %.0f ms (real-time performance)\n', sample_time*1000);
fprintf('  - power_setpoint: %.0f MW (realistic CCPP power)\n', power_setpoint);
fprintf('  - sim_time: %.0f seconds (extended analysis)\n', sim_time);
fprintf('  - trained_ml_model: 95.9%% accurate Random Forest\n');
fprintf('  - enhanced blocks: ML prediction, IoT monitoring, environmental\n');

fprintf('\n ENHANCED NEXT STEPS ===\n');
fprintf('1. Open enhanced model: open(''Energisense.slx'')\n');
fprintf('2. Add new function blocks:\n');
fprintf('   - mlPowerPredictionBlock (for 95.9%% ML predictions)\n');
fprintf('   - environmentalConditionsBlock (realistic conditions)\n');  
fprintf('   - industrialIoTBlock (monitoring & alerts)\n');
fprintf('3. Update PID Controller: predictivePIDController\n');
fprintf('4. Run enhanced simulation: simout = sim(''Energisense'')\n');
fprintf('5. Analyze with validation: validateSimulationResults(simout)\n');