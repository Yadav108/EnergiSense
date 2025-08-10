%% Configuration for Energisense Digital Twin Predictive PID
clear pid_params sample_time power_setpoint sim_time;
clc;

fprintf('=== ENERGISENSE PREDICTIVE PID CONFIGURATION ===\n');

%% Simulation Parameters
sim_time = 120;
sample_time = 0.1;
power_setpoint = 400;

%% Predictive PID Parameters
pid_params = struct();
pid_params.Kp = 1.8;
pid_params.Ki = 0.15;
pid_params.Kd = 0.12;
pid_params.u_max = 120.0;
pid_params.u_min = -120.0;
pid_params.I_max = 40.0;
pid_params.I_min = -40.0;
pid_params.prediction_weight = 0.6;
pid_params.model_quality_threshold = 0.8;
pid_params.enable_adaptive = true;
pid_params.enable_derivative_filter = true;
pid_params.derivative_filter_alpha = 0.2;
pid_params.setpoint_weight = 0.9;
pid_params.deadband = 0.5;
pid_params.adaptive_factor = 0.02;
pid_params.large_error_threshold = 20.0;
pid_params.small_error_threshold = 2.5;
pid_params.feedforward_gain = 0.4;
pid_params.disturbance_gain = 0.3;

%% Load to Workspace
assignin('base', 'pid_params', pid_params);
assignin('base', 'sample_time', sample_time);
assignin('base', 'power_setpoint', power_setpoint);
assignin('base', 'sim_time', sim_time);

fprintf('✓ Configuration completed successfully!\n');
fprintf('Variables loaded to workspace:\n');
fprintf('  - pid_params: Controller parameters\n');
fprintf('  - sample_time: %.3f seconds\n', sample_time);
fprintf('  - power_setpoint: %.0f MW\n', power_setpoint);
fprintf('  - sim_time: %.0f seconds\n', sim_time);

fprintf('\n=== NEXT STEPS ===\n');
fprintf('1. Open Energisense.slxc model\n');
fprintf('2. Update PID Controller function name to: predictivePIDController\n');
fprintf('3. Verify signal connections\n');
fprintf('4. Run: simout = sim(''Energisense'')\n');