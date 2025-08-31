function simpleVerification()
%SIMPLEVERIFICATION Simple verification system for EnergiSense
%
% This is a fallback verification system that works with all MATLAB versions

    fprintf('\n🔍 EnergiSense Simple Verification System\n');
    fprintf('==========================================\n\n');
    
    % Test 1: ML Prediction
    fprintf('1. Testing ML Prediction System\n');
    fprintf('--------------------------------\n');
    try
        testInput = [25, 40, 1013, 60]; % [AT, V, AP, RH]
        [power, confidence] = predictPowerEnhanced(testInput);
        
        if power > 400 && power < 500
            fprintf('   ✅ ML Prediction: WORKING\n');
            fprintf('   📊 Result: %.2f MW\n', power);
            fprintf('   🎯 Confidence: %.1f%%\n', confidence * 100);
        else
            fprintf('   ❌ ML Prediction: Invalid range\n');
        end
    catch ME
        fprintf('   ❌ ML Prediction: ERROR - %s\n', ME.message);
    end
    
    % Test 2: Weather Intelligence
    fprintf('\n2. Testing Weather Intelligence\n');
    fprintf('-------------------------------\n');
    try
        weatherData = getWeatherIntelligence();
        fprintf('   ✅ Weather System: WORKING\n');
        fprintf('   🌡️ Temperature: %.1f°C\n', weatherData.temperature);
        fprintf('   💧 Humidity: %.1f%%\n', weatherData.humidity);
        fprintf('   📊 Pressure: %.1f mbar\n', weatherData.pressure);
    catch ME
        fprintf('   ❌ Weather System: ERROR - %s\n', ME.message);
    end
    
    % Test 3: Model Files
    fprintf('\n3. Testing Model Files\n');
    fprintf('----------------------\n');
    modelFiles = {
        'core/models/ensemblePowerModel.mat';
        'core/models/ccpp_random_forest_model.mat';
        'core/models/digitaltwin.mat'
    };
    
    for i = 1:length(modelFiles)
        if exist(modelFiles{i}, 'file')
            fprintf('   ✅ %s: FOUND\n', modelFiles{i});
        else
            fprintf('   ❌ %s: MISSING\n', modelFiles{i});
        end
    end
    
    % Test 4: Core Functions
    fprintf('\n4. Testing Core Functions\n');
    fprintf('-------------------------\n');
    functions = {
        'predictPowerEnhanced';
        'predictPowerML';
        'getWeatherIntelligence';
        'checkModel';
        'setupEnergiSense'
    };
    
    for i = 1:length(functions)
        if exist(functions{i}, 'file')
            fprintf('   ✅ %s: AVAILABLE\n', functions{i});
        else
            fprintf('   ❌ %s: MISSING\n', functions{i});
        end
    end
    
    % Test 5: Multiple Scenarios
    fprintf('\n5. Testing Multiple Scenarios\n');
    fprintf('-----------------------------\n');
    
    scenarios = {
        [15, 35, 1015, 70], 'Cool Conditions';
        [30, 50, 1005, 80], 'Hot Conditions';
        [20, 45, 1020, 50], 'Moderate Conditions';
        [35, 60, 1000, 90], 'Extreme Hot';
        [5, 30, 1025, 40], 'Extreme Cold'
    };
    
    for i = 1:size(scenarios, 1)
        try
            input_vals = scenarios{i,1};
            [pow, conf] = predictPowerEnhanced(input_vals);
            fprintf('   ✅ %s: %.1f MW (%.0f%% conf)\n', scenarios{i,2}, pow, conf*100);
        catch ME
            fprintf('   ❌ %s: ERROR - %s\n', scenarios{i,2}, ME.message);
        end
    end
    
    % Summary
    fprintf('\n=== VERIFICATION COMPLETE ===\n');
    fprintf('All core systems tested.\n');
    fprintf('Check results above for any issues.\n');
    fprintf('Use launchInteractiveDashboard() to start the main system.\n\n');
    
end