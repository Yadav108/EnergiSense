# Installation and Setup Guide

## Overview

This guide provides complete installation instructions for EnergiSense, the 95.9% accurate Combined Cycle Power Plant digital twin system. Follow these steps to set up the enhanced ML-powered system with Simulink integration and industrial IoT capabilities.

## System Requirements

### MATLAB Requirements
- **MATLAB Version**: R2019b or later (R2021a+ recommended)
- **License Type**: Full MATLAB license (Student versions supported)
- **Architecture**: 64-bit Windows, macOS, or Linux

### Required MATLAB Toolboxes
| Toolbox | Essential | Purpose |
|---------|-----------|---------|
| **Statistics and Machine Learning** | ✅ **Required** | Random Forest ML model |
| **Simulink** | ✅ **Required** | Enhanced simulation blocks |
| Control System Toolbox | ⚡ Recommended | Advanced MPC features |
| DSP System Toolbox | ⚡ Recommended | Signal processing |
| Parallel Computing Toolbox | ⚠️ Optional | Training acceleration |

### Hardware Requirements
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 500 MB free space (including datasets)
- **CPU**: Intel i5/AMD Ryzen 5 or better for real-time simulation
- **Graphics**: Dedicated GPU optional for visualization

### Check Your MATLAB Installation
```matlab
% Verify required toolboxes
toolboxes = ver;
required = {'Statistics and Machine Learning Toolbox', 'Simulink'};
for i = 1:length(required)
    if ~any(strcmp({toolboxes.Name}, required{i}))
        error('Missing required toolbox: %s', required{i});
    else
        fprintf('✓ %s installed\n', required{i});
    end
end
```

## Installation Methods

### Method 1: Direct Download (Recommended)

1. **Download the complete EnergiSense package**:
   ```
   [Download from repository/provided archive]
   ```

2. **Extract to MATLAB-accessible location**:
   ```
   C:\Users\[Username]\Documents\MATLAB\EnergiSense\
   ```

3. **Add to MATLAB path**:
   ```matlab
   cd('C:\Users\[Username]\Documents\MATLAB\EnergiSense');
   addpath(genpath(pwd));
   savepath;
   ```

### Method 2: Git Clone (For Developers)

```bash
# Clone repository
git clone [repository-url] EnergiSense
cd EnergiSense

# Initialize in MATLAB
matlab -r "addpath(genpath(pwd)); savepath; setupEnergiSense();"
```

## Quick Setup (5-Minute Start)

### Step 1: Initialize System
```matlab
% Navigate to EnergiSense directory
cd('path/to/EnergiSense');

% Add paths and initialize
addpath(genpath(pwd));
savepath;

% Run automated setup
setup_status = setupEnergiSense();
```

### Step 2: Verify Installation
```matlab
% Test ML model loading (95.9% accuracy)
[power, conf, status] = predictPowerEnhanced([20, 45, 1015, 70]);
if status == 2 && conf > 0.95
    fprintf('✓ ML model loaded successfully: %.2f MW (%.1f%% confidence)\n', power, conf*100);
else
    error('❌ ML model failed to load properly');
end

% Test Simulink integration
if exist('simulation/blocks/mlPowerPredictionBlock.m', 'file')
    fprintf('✓ Simulink blocks available\n');
else
    error('❌ Simulink blocks not found');
end
```

### Step 3: Run Quick Test
```matlab
% 30-second demonstration simulation
demo_results = runEnhancedSimulation(30, 0.1, struct('initial_power', 450));
fprintf('✓ Demo simulation completed: %.1f%% ML accuracy\n', demo_results.ml_accuracy * 100);
```

## Detailed Setup Process

### 1. Directory Structure Setup

Create the following directory structure:
```
EnergiSense/
├── core/                   % Core ML and control functions
│   ├── models/            % Trained ML models (95.9% accuracy)
│   ├── control/           % PID and MPC controllers  
│   └── utils/             % Utility functions
├── simulation/            % Simulink integration
│   ├── blocks/            % 4 specialized Simulink blocks
│   └── models/            % Simulink model files
├── data/                  % Datasets and logs
│   ├── processed/         % Clean CCPP datasets
│   └── logs/              % Runtime data logs
├── docs/                  % Documentation
├── config/                % Configuration files
└── tests/                 % Validation tests
```

### 2. ML Model Setup

#### Download and Verify CCPP Dataset
```matlab
% The UCI CCPP dataset should be automatically included
data_path = fullfile(pwd, 'data', 'processed', 'ccpp_simin_cleaned.mat');
if ~exist(data_path, 'file')
    warning('CCPP dataset not found. Downloading...');
    downloadCCPPDataset(); % Function downloads from UCI repository
end

% Verify data integrity
ccpp_data = load(data_path);
if size(ccpp_data.data, 1) ~= 9568 || size(ccpp_data.data, 2) ~= 5
    error('Invalid CCPP dataset. Expected 9568×5 matrix.');
else
    fprintf('✓ CCPP dataset verified: %d samples\n', size(ccpp_data.data, 1));
end
```

#### Train ML Model (First Time Setup)
```matlab
% Train the Random Forest model (achieves 95.9% accuracy)
fprintf('Training Random Forest model (this may take 2-3 minutes)...\n');
[model, validation_results] = trainCCPPModel();

% Verify model performance
if validation_results.r2_score >= 0.95
    fprintf('✓ Model training successful: R² = %.4f (%.1f%% accuracy)\n', ...
            validation_results.r2_score, validation_results.r2_score * 100);
else
    error('❌ Model training failed to achieve 95%% accuracy');
end

% Save trained model
model_path = fullfile(pwd, 'core', 'models', 'ccpp_random_forest_model.mat');
save(model_path, 'model', 'validation_results');
```

### 3. Simulink Configuration

#### Initialize Enhanced Simulink System
```matlab
% Set up all 4 specialized Simulink blocks
initializeEnhancedSimulink();

% Verify block functionality
blocks = {
    'mlPowerPredictionBlock',
    'environmentalConditionsBlock', 
    'industrialIoTBlock',
    'advancedMPCBlock'
};

for i = 1:length(blocks)
    if exist(fullfile('simulation', 'blocks', [blocks{i} '.m']), 'file')
        fprintf('✓ %s ready\n', blocks{i});
    else
        error('❌ Missing block: %s', blocks{i});
    end
end
```

#### Test Simulink Integration
```matlab
% Test each block individually
test_time = 10; % 10 seconds

% Test environmental conditions
[AT, V, AP, RH] = environmentalConditionsBlock(test_time);
fprintf('Environmental test: T=%.1f°C, V=%.1f cmHg, P=%.1f mbar, H=%.1f%%\n', AT, V, AP, RH);

% Test ML prediction block
[pred_power, confidence, status] = mlPowerPredictionBlock(AT, V, AP, RH);
if status == 2
    fprintf('✓ ML prediction: %.1f MW (confidence: %.1f%%)\n', pred_power, confidence*100);
end
```

### 4. Industrial IoT Setup

#### Configure Data Acquisition
```matlab
% Configure supported industrial protocols
iot_config = struct();
iot_config.protocols = {'Modbus', 'OPC-UA', 'MQTT'};
iot_config.sampling_rate = 10; % Hz
iot_config.data_quality_threshold = 85; % %

% Initialize IoT monitoring
configureEnergiSense(struct('iot_config', iot_config));
```

#### Test IoT Functionality
```matlab
% Simulate sensor inputs
sensor_inputs = struct();
sensor_inputs.temperature_sensors = 25 + 5*rand(5,1); % 5 temperature sensors
sensor_inputs.vibration_sensors = 0.1*rand(3,1);      % 3 vibration sensors  
sensor_inputs.pressure_sensors = 1013 + 10*rand(4,1); % 4 pressure sensors
sensor_inputs.flow_sensors = 100 + 20*rand(2,1);      % 2 flow sensors

% Test IoT block
[health, metrics, alerts] = industrialIoTBlock(sensor_inputs, 85);
fprintf('✓ IoT monitoring: Overall health %.1f%%, Data quality %.1f%%\n', ...
        health.overall_health, metrics.data_quality);
```

## Advanced Configuration

### Custom ML Model Training

If you want to retrain the model with custom data:

```matlab
function customModelTraining()
    % Load your custom dataset
    % Expected format: [AT, V, AP, RH, PE] where PE is target power
    custom_data = load('your_custom_ccpp_data.mat');
    
    % Training configuration
    config = struct();
    config.n_trees = 100;
    config.min_leaf_size = 5;
    config.cross_validation_folds = 5;
    config.test_split = 0.2;
    
    % Train custom model
    [custom_model, results] = trainCCPPModel(config);
    
    % Validate performance threshold
    if results.r2_score < 0.90
        warning('Custom model accuracy below 90%%: R² = %.4f', results.r2_score);
    end
    
    % Save custom model
    save('core/models/custom_ccpp_model.mat', 'custom_model', 'results');
end
```

### Controller Parameter Optimization

```matlab
% Optimize PID controller for your specific plant
target_rmse = 2.0; % MW
max_iterations = 50;

optimization_results = optimizeControllerPerformance(target_rmse, max_iterations);

% Apply optimized parameters
pid_params = optimization_results.optimal_params;
configureEnergiSense(struct('control_parameters', pid_params));

fprintf('✓ Controller optimized: RMSE improved by %.1f%%\n', ...
        optimization_results.improvement_pct);
```

### Database Integration

```matlab
% Configure database connection for data logging
db_config = struct();
db_config.type = 'SQLite'; % or 'MySQL', 'PostgreSQL'
db_config.connection_string = 'energisense_data.db';
db_config.auto_create_tables = true;
db_config.logging_interval = 1; % seconds

% Initialize database
initializeDatabaseLogging(db_config);
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "TreeBagger not found"
```matlab
% Solution: Install Statistics and Machine Learning Toolbox
>> ver % Check installed toolboxes
% If missing, install from MATLAB Add-Ons
```

#### Issue 2: "Model file not found"
```matlab
% Solution: Retrain the model
>> [model, results] = trainCCPPModel();
>> save('core/models/ccpp_random_forest_model.mat', 'model', 'results');
```

#### Issue 3: "Simulink blocks not working"
```matlab
% Solution: Reinitialize Simulink system
>> initializeEnhancedSimulink();
>> rehash toolboxcache;
```

#### Issue 4: "Poor ML model performance"
```matlab
% Solution: Verify data quality and retrain
>> validateCCPPData(); % Check data integrity
>> [model, results] = trainCCPPModel();
>> if results.r2_score < 0.95
       error('Training failed - check data quality');
   end
```

#### Issue 5: "Simulink cannot find blocks"
```matlab
% Solution: Add block directory to path
>> addpath(fullfile(pwd, 'simulation', 'blocks'));
>> rehash toolbox;
```

### Performance Optimization

#### For Large-Scale Simulations:
```matlab
% Enable parallel processing (if Parallel Computing Toolbox available)
if license('test', 'Distrib_Computing_Toolbox')
    parpool('local', 4); % Use 4 cores
    configureEnergiSense(struct('use_parallel', true));
end

% Optimize memory usage
configureEnergiSense(struct('memory_optimization', true));
```

#### For Real-Time Applications:
```matlab
% Configure for minimal latency
rt_config = struct();
rt_config.prediction_cache = true;    % Cache ML predictions
rt_config.update_rate = 100;         % 100 Hz control loop
rt_config.buffer_size = 1000;        % Data buffer size

configureEnergiSense(rt_config);
```

## Validation and Testing

### Run Complete System Validation
```matlab
function validateInstallation()
    fprintf('Running EnergiSense installation validation...\n\n');
    
    % Test 1: ML Model Performance
    fprintf('1. Testing ML Model (95.9%% accuracy target):\n');
    [power, conf, status] = predictPowerEnhanced([20, 45, 1015, 70]);
    if status == 2 && conf >= 0.95
        fprintf('   ✓ ML prediction successful: %.2f MW (%.1f%% confidence)\n', power, conf*100);
    else
        fprintf('   ❌ ML prediction failed\n');
        return;
    end
    
    % Test 2: Simulink Integration
    fprintf('\n2. Testing Simulink Integration:\n');
    try
        [AT, V, AP, RH] = environmentalConditionsBlock(10);
        [pred, conf, stat] = mlPowerPredictionBlock(AT, V, AP, RH);
        fprintf('   ✓ Simulink blocks operational\n');
    catch ME
        fprintf('   ❌ Simulink integration failed: %s\n', ME.message);
        return;
    end
    
    % Test 3: Control Systems
    fprintf('\n3. Testing Control Systems:\n');
    pid_params = struct('Kp', 5.0, 'Ki', 0.088, 'Kd', 0.171);
    try
        [u, ~, ~] = predictivePIDController(450, pred, 445, 0.1, pid_params);
        fprintf('   ✓ Enhanced PID controller functional\n');
    catch ME
        fprintf('   ❌ Control system failed: %s\n', ME.message);
        return;
    end
    
    % Test 4: Complete Simulation
    fprintf('\n4. Running 30-second validation simulation:\n');
    try
        initial_conditions = struct('initial_power', 450, 'initial_temperature', 20);
        results = runEnhancedSimulation(30, 0.1, initial_conditions);
        fprintf('   ✓ Simulation completed: %.1f%% ML accuracy achieved\n', results.ml_accuracy * 100);
    catch ME
        fprintf('   ❌ Simulation failed: %s\n', ME.message);
        return;
    end
    
    fprintf('\n🎉 EnergiSense installation validation PASSED!\n');
    fprintf('    System ready for production use.\n\n');
end

% Run validation
validateInstallation();
```

## Getting Started

Once installation is complete, follow this workflow:

### 1. Basic Usage
```matlab
% Quick prediction
[power, confidence] = predictPowerEnhanced([25, 50, 1013, 60]);

% Launch interactive dashboard
launchInteractiveDashboard();
```

### 2. Run Full Simulation
```matlab
% 5-minute enhanced simulation with all features
initial_conditions = struct('initial_power', 450, 'initial_temperature', 20);
results = runEnhancedSimulation(300, 0.05, initial_conditions);

% Analyze results
analyzeSimulationResults(results);
```

### 3. Explore Documentation
```matlab
% Open comprehensive documentation
open('docs/README.md');           % Main documentation
open('docs/USER_GUIDE.md');       % User workflows  
open('docs/API_REFERENCE.md');    % Function reference
```

## Support and Updates

### Update Check
```matlab
% Check for system updates (implement as needed)
checkEnergiSenseUpdates();
```

### Get Help
- **Documentation**: See `docs/` directory for comprehensive guides
- **Examples**: Run `launchInteractiveDashboard()` for guided examples
- **Issues**: Check validation results and error messages

### Performance Monitoring
```matlab
% Monitor ML model performance over time
model_health = monitorMLModel(recent_predictions, recent_actuals);
if strcmp(model_health.alert, 'RETRAIN_RECOMMENDED')
    fprintf('⚠️ Consider model retraining for optimal performance\n');
end
```

---

**Installation Complete!** 🎉

Your EnergiSense system is now ready with:
- ✅ 95.9% accurate ML power prediction
- ✅ 4 enhanced Simulink blocks  
- ✅ Optimized PID and MPC controllers
- ✅ Industrial IoT monitoring
- ✅ Comprehensive documentation

*Next: Explore the User Guide (`docs/USER_GUIDE.md`) for detailed workflows and examples.*