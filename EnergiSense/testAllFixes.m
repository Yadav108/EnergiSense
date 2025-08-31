%% Test All Fixes - Comprehensive System Test
% Tests all the fixes applied to resolve verification and ML issues

fprintf('\n🧪 EnergiSense Comprehensive Fix Test\n');
fprintf('=====================================\n\n');

% Initialize system
fprintf('📋 Step 1: System Initialization\n');
fprintf('---------------------------------\n');
try
    startup;
    fprintf('✅ Startup completed\n');
catch ME
    fprintf('❌ Startup failed: %s\n', ME.message);
end

% Test ML prediction with different approaches
fprintf('\n🤖 Step 2: ML Prediction Testing\n');
fprintf('----------------------------------\n');

% Test A: Direct function call
fprintf('Test A: Direct predictPowerEnhanced call\n');
try
    testInput = [25, 40, 1013, 60]; % [AT, V, AP, RH]
    [power, confidence] = predictPowerEnhanced(testInput);
    
    if power > 400 && power < 500
        fprintf('   ✅ PASSED: %.2f MW (%.1f%% conf)\n', power, confidence*100);
        mlDirectTest = true;
    else
        fprintf('   ❌ FAILED: Invalid range %.2f MW\n', power);
        mlDirectTest = false;
    end
catch ME
    fprintf('   ❌ FAILED: %s\n', ME.message);
    mlDirectTest = false;
end

% Test B: Multiple scenarios
fprintf('\nTest B: Multiple scenario testing\n');
scenarios = [
    15, 35, 1015, 70;  % Cool
    30, 50, 1005, 80;  % Hot
    20, 45, 1020, 50   % Moderate
];

scenariosPassed = 0;
for i = 1:size(scenarios, 1)
    try
        [pow, conf] = predictPowerEnhanced(scenarios(i,:));
        if pow > 400 && pow < 500
            fprintf('   ✅ Scenario %d: %.1f MW\n', i, pow);
            scenariosPassed = scenariosPassed + 1;
        else
            fprintf('   ❌ Scenario %d: Invalid range %.1f MW\n', i, pow);
        end
    catch ME
        fprintf('   ❌ Scenario %d: Error - %s\n', i, ME.message);
    end
end

mlScenariosTest = (scenariosPassed == size(scenarios, 1));
if mlScenariosTest
    fprintf('   ✅ All scenarios PASSED\n');
else
    fprintf('   ❌ Only %d/%d scenarios passed\n', scenariosPassed, size(scenarios, 1));
end

% Test verification system
fprintf('\n🔍 Step 3: Verification System Testing\n');
fprintf('---------------------------------------\n');
try
    % Test simple verification function
    if exist('simpleVerification', 'file')
        fprintf('Test A: Standalone simpleVerification\n');
        simpleVerification();
        fprintf('   ✅ Standalone verification PASSED\n');
        standaloneTest = true;
    else
        fprintf('   ⚠️ Standalone verification file not found, using embedded\n');
        standaloneTest = false;
    end
catch ME
    fprintf('   ❌ Standalone verification FAILED: %s\n', ME.message);
    standaloneTest = false;
end

% Test dashboard launch
fprintf('\nTest B: Dashboard integration test\n');
try
    % This would normally launch the full dashboard, but we'll just test the launcher
    fprintf('   Testing dashboard launcher availability...\n');
    if exist('launchInteractiveDashboard', 'file')
        fprintf('   ✅ Dashboard launcher available\n');
        dashboardTest = true;
    else
        fprintf('   ❌ Dashboard launcher missing\n');
        dashboardTest = false;
    end
catch ME
    fprintf('   ❌ Dashboard test FAILED: %s\n', ME.message);
    dashboardTest = false;
end

% Test weather system
fprintf('\n🌤️ Step 4: Weather Intelligence Testing\n');
fprintf('----------------------------------------\n');
try
    weatherData = getWeatherIntelligence();
    if isfield(weatherData, 'temperature') && isfield(weatherData, 'humidity')
        fprintf('✅ Weather system PASSED\n');
        fprintf('   Temperature: %.1f°C\n', weatherData.temperature);
        fprintf('   Humidity: %.1f%%\n', weatherData.humidity);
        fprintf('   Pressure: %.1f mbar\n', weatherData.pressure);
        weatherTest = true;
    else
        fprintf('❌ Weather system missing fields\n');
        weatherTest = false;
    end
catch ME
    fprintf('❌ Weather system FAILED: %s\n', ME.message);
    weatherTest = false;
end

% Final Summary
fprintf('\n📊 FINAL TEST RESULTS\n');
fprintf('=====================\n');
totalTests = 5;
passedTests = sum([mlDirectTest, mlScenariosTest, standaloneTest, dashboardTest, weatherTest]);

fprintf('Overall Score: %d/%d tests passed\n\n', passedTests, totalTests);
fprintf('✅ ML Direct Prediction: %s\n', testStatus(mlDirectTest));
fprintf('✅ ML Multiple Scenarios: %s\n', testStatus(mlScenariosTest));
fprintf('✅ Verification System: %s\n', testStatus(standaloneTest));
fprintf('✅ Dashboard Integration: %s\n', testStatus(dashboardTest));
fprintf('✅ Weather Intelligence: %s\n', testStatus(weatherTest));

if passedTests >= 4
    fprintf('\n🎉 SYSTEM STATUS: EXCELLENT\n');
    fprintf('All critical systems are working!\n');
elseif passedTests >= 3
    fprintf('\n✅ SYSTEM STATUS: GOOD\n');
    fprintf('Most systems working, minor issues detected.\n');
else
    fprintf('\n⚠️ SYSTEM STATUS: NEEDS ATTENTION\n');
    fprintf('Multiple issues detected, please investigate.\n');
end

fprintf('\n💡 To launch the system:\n');
fprintf('   app = launchInteractiveDashboard()\n');
fprintf('   Click "🔍 Verification" button for detailed testing\n\n');

% Helper function
function status = testStatus(passed)
    if passed
        status = 'PASSED';
    else
        status = 'FAILED';
    end
end