%% Test Script for EnergiSense Fixes
% Tests verification system, MAE display, and ML prediction fixes

fprintf('🧪 EnergiSense Fixes Test Suite\n');
fprintf('================================\n\n');

%% Test 1: ML Prediction Input Arguments Fix
fprintf('Test 1: ML Prediction Input Arguments\n');
try
    % Test with correct input format [AT, V, AP, RH]
    testInput = [25, 40, 1013, 60];
    [power, confidence] = predictPowerEnhanced(testInput);
    
    if power > 400 && power < 500 && confidence > 0
        fprintf('✅ ML Prediction: PASSED (%.2f MW, %.0f%% conf)\n', power, confidence*100);
        mlTest = true;
    else
        fprintf('❌ ML Prediction: Invalid result (%.2f MW, %.2f conf)\n', power, confidence);
        mlTest = false;
    end
catch ME
    fprintf('❌ ML Prediction: FAILED - %s\n', ME.message);
    mlTest = false;
end

%% Test 2: Multiple Input Scenarios
fprintf('\nTest 2: Multiple Input Scenarios\n');
testScenarios = [
    15, 35, 1015, 70;  % Cool conditions
    30, 50, 1005, 80;  % Hot conditions
    20, 45, 1020, 50   % Moderate conditions
];

scenarioTest = true;
for i = 1:size(testScenarios, 1)
    try
        [pow, conf] = predictPowerEnhanced(testScenarios(i,:));
        fprintf('   Scenario %d: %.1f MW (%.0f%% conf)\n', i, pow, conf*100);
        if pow < 400 || pow > 500
            scenarioTest = false;
        end
    catch ME
        fprintf('   Scenario %d: FAILED - %s\n', i, ME.message);
        scenarioTest = false;
    end
end

if scenarioTest
    fprintf('✅ Multiple Scenarios: PASSED\n');
else
    fprintf('❌ Multiple Scenarios: FAILED\n');
end

%% Test 3: Weather Intelligence
fprintf('\nTest 3: Weather Intelligence\n');
try
    weatherData = getWeatherIntelligence();
    if isfield(weatherData, 'temperature') && isfield(weatherData, 'humidity')
        fprintf('✅ Weather: PASSED (%.1f°C, %.1f%%)\n', weatherData.temperature, weatherData.humidity);
        weatherTest = true;
    else
        fprintf('❌ Weather: Missing fields\n');
        weatherTest = false;
    end
catch ME
    fprintf('❌ Weather: FAILED - %s\n', ME.message);
    weatherTest = false;
end

%% Test 4: Model Loading
fprintf('\nTest 4: Model File Existence\n');
modelFiles = {
    'core/models/ensemblePowerModel.mat';
    'core/models/ccpp_random_forest_model.mat'
};

fileTest = true;
for i = 1:length(modelFiles)
    if exist(modelFiles{i}, 'file')
        fprintf('✅ %s: Found\n', modelFiles{i});
    else
        fprintf('❌ %s: Missing\n', modelFiles{i});
        fileTest = false;
    end
end

%% Test Summary
fprintf('\n=== TEST SUMMARY ===\n');
totalTests = 4;
passedTests = sum([mlTest, scenarioTest, weatherTest, fileTest]);

fprintf('Passed: %d/%d tests\n', passedTests, totalTests);
fprintf('ML Prediction Fix: %s\n', statusText(mlTest));
fprintf('Multiple Scenarios: %s\n', statusText(scenarioTest));
fprintf('Weather Intelligence: %s\n', statusText(weatherTest));
fprintf('Model Files: %s\n', statusText(fileTest));

if passedTests == totalTests
    fprintf('\n🎉 ALL TESTS PASSED! System is ready.\n');
else
    fprintf('\n⚠️ Some tests failed. Check issues above.\n');
end

function status = statusText(passed)
    if passed
        status = '✅ PASSED';
    else
        status = '❌ FAILED';
    end
end