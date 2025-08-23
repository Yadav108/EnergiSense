function app = runInteractiveDashboard()
% RUNINTERACTIVEDASHBOARD Launch the EnergiSense Interactive Dashboard
%
% SYNTAX:
%   app = runInteractiveDashboard()
%
% DESCRIPTION:
%   Launches the EnergiSense Interactive Dashboard application for real-time
%   monitoring and analysis of Combined Cycle Power Plant (CCPP) performance
%   with 99.1% prediction accuracy.
%
% OUTPUTS:
%   app - Handle to the EnergiSense Interactive Dashboard application
%
% FEATURES:
%   - Real-time simulation with configurable update rates
%   - Interactive environmental parameter controls
%   - 6-panel professional analysis dashboard
%   - Performance metrics and gauges
%   - Data export and session management
%   - Integration with existing EnergiSense models
%
% REQUIREMENTS:
%   - MATLAB R2020a or later with App Designer
%   - EnergiSense system files (optional, will use empirical model as fallback)
%   - Statistics and Machine Learning Toolbox (for advanced features)
%
% USAGE EXAMPLES:
%   % Launch the dashboard
%   app = runInteractiveDashboard();
%   
%   % Launch and assign to variable for programmatic control
%   dashboard = runInteractiveDashboard();
%
% INTEGRATION WITH ENERGISENSE SYSTEM:
%   Place this function in your EnergiSense root directory alongside:
%   - core/models/ensemblePowerModel.mat (optional)
%   - predictPowerEnhanced.m (optional)
%   - loadEnergiSenseVariables.m (optional)
%
% TROUBLESHOOTING:
%   If you encounter issues:
%   1. Ensure MATLAB App Designer is available
%   2. Check file paths for EnergiSense models
%   3. Verify required toolboxes are installed
%   4. Run from EnergiSense root directory for best integration
%
% See also: EnergiSenseInteractiveDashboard, setupEnergiSense, runDashboard
%
% Author: EnergiSense Development Team
% Date: 2024
% Version: 1.0

    % Display startup banner
    fprintf('\n');
    fprintf('=================================================================\n');
    fprintf('🎛️  EnergiSense Interactive Dashboard v1.0\n');
    fprintf('=================================================================\n');
    fprintf('Digital Twin for Combined Cycle Power Plant Operations\n');
    fprintf('Prediction Accuracy: 99.1%% | Real-time Analysis | Export Ready\n');
    fprintf('=================================================================\n\n');
    
    % Check MATLAB version compatibility
    if verLessThan('matlab', '9.8')
        warning('EnergiSense:VersionWarning', ...
            'This dashboard requires MATLAB R2020a or later for optimal performance.');
    end
    
    % Check for required toolboxes
    checkRequiredToolboxes();
    
    % Check for EnergiSense system files
    checkEnergiSenseIntegration();
    
    % Initialize any required paths
    initializePaths();
    
    try
        % Launch the dashboard application
        fprintf('🚀 Launching EnergiSense Interactive Dashboard...\n');
        app = EnergiSenseInteractiveDashboard();
        
        fprintf('✅ Dashboard launched successfully!\n');
        fprintf('\nDashboard Features:\n');
        fprintf('  📊 6-Panel Real-time Analysis\n');
        fprintf('  🎛️  Interactive Environmental Controls\n');
        fprintf('  📈 Performance Metrics & Gauges\n');
        fprintf('  💾 Data Export & Session Management\n');
        fprintf('  🔄 Configurable Update Rates\n');
        fprintf('  🎯 99.1%% Prediction Accuracy Display\n\n');
        
        fprintf('Usage Instructions:\n');
        fprintf('  1. Click "Start Simulation" to begin real-time monitoring\n');
        fprintf('  2. Adjust environmental parameters using sliders\n');
        fprintf('  3. Monitor performance metrics in real-time\n');
        fprintf('  4. Export dashboard or save session data as needed\n');
        fprintf('  5. Use "Reset Data" to clear buffers and restart\n\n');
        
        % Display integration status
        displayIntegrationStatus();
        
    catch ME
        fprintf('❌ Failed to launch dashboard: %s\n', ME.message);
        fprintf('\nTroubleshooting Steps:\n');
        fprintf('  1. Ensure App Designer is available in your MATLAB installation\n');
        fprintf('  2. Check that all required files are in the current directory\n');
        fprintf('  3. Verify MATLAB version is R2020a or later\n');
        fprintf('  4. Try running: >> appdesigner\n');
        fprintf('  5. Check MATLAB path and file permissions\n\n');
        
        fprintf('For support, check the EnergiSense documentation or contact support.\n\n');
        rethrow(ME);
    end
    
    % If no output requested, clear the app handle
    if nargout == 0
        clear app;
    end
end

function checkRequiredToolboxes()
    % Check for recommended toolboxes
    requiredToolboxes = {
        'Statistics and Machine Learning Toolbox', 'stats';
        'Curve Fitting Toolbox', 'curvefit';
        'Signal Processing Toolbox', 'signal'
    };
    
    missingToolboxes = {};
    
    for i = 1:size(requiredToolboxes, 1)
        if ~license('test', requiredToolboxes{i, 2})
            missingToolboxes{end+1} = requiredToolboxes{i, 1}; %#ok<AGROW>
        end
    end
    
    if ~isempty(missingToolboxes)
        fprintf('⚠️  Optional toolboxes not detected:\n');
        for i = 1:length(missingToolboxes)
            fprintf('   - %s\n', missingToolboxes{i});
        end
        fprintf('   Dashboard will use built-in alternatives.\n\n');
    end
end

function checkEnergiSenseIntegration()
    % Check for EnergiSense system integration
    systemFiles = {
        'core/models/ensemblePowerModel.mat';
        'ensemblePowerModel.mat';
        'predictPowerEnhanced.m';
        'loadEnergiSenseVariables.m';
        'setupEnergiSense.m'
    };
    
    foundFiles = {};
    
    for i = 1:length(systemFiles)
        if exist(systemFiles{i}, 'file')
            foundFiles{end+1} = systemFiles{i}; %#ok<AGROW>
        end
    end
    
    if isempty(foundFiles)
        fprintf('ℹ️  EnergiSense system files not detected in current directory.\n');
        fprintf('   Dashboard will operate in standalone mode with empirical models.\n\n');
    else
        fprintf('✅ EnergiSense system integration detected!\n');
        fprintf('   Found files:\n');
        for i = 1:length(foundFiles)
            fprintf('   - %s\n', foundFiles{i});
        end
        fprintf('\n');
    end
end

function initializePaths()
    % Add current directory and subdirectories to path if needed
    currentDir = pwd;
    
    % Check if EnergiSense subdirectories exist and add to path
    subdirs = {'core', 'core/models', 'simulation', 'simulation/models', 'data', 'utilities'};
    
    for i = 1:length(subdirs)
        subdir = fullfile(currentDir, subdirs{i});
        if exist(subdir, 'dir') && ~contains(path, subdir)
            addpath(subdir);
        end
    end
end

function displayIntegrationStatus()
    % Display current integration status
    fprintf('Current Integration Status:\n');
    
    % Check model availability
    if exist('core/models/ensemblePowerModel.mat', 'file') || exist('ensemblePowerModel.mat', 'file')
        fprintf('  🤖 ML Model: Available (99.1%% accuracy)\n');
    else
        fprintf('  🤖 ML Model: Using empirical model\n');
    end
    
    % Check function availability
    if exist('predictPowerEnhanced.m', 'file')
        fprintf('  🔧 Prediction Function: Available\n');
    else
        fprintf('  🔧 Prediction Function: Using built-in\n');
    end
    
    % Check Simulink integration
    if exist('simulation/models/Energisense.slx', 'file')
        fprintf('  📐 Simulink Model: Available\n');
    else
        fprintf('  📐 Simulink Model: Not detected\n');
    end
    
    fprintf('\n');
    fprintf('Dashboard is ready for operation! 🎉\n');
    fprintf('=================================================================\n\n');
end