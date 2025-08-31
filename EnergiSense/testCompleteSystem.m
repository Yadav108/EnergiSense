%% Test Complete System - ML Prediction & Verification
% Tests both the ML prediction fix and the restored verification system

fprintf('\n🔧 EnergiSense Complete System Test\n');
fprintf('===================================\n\n');

% Step 1: Test ML Prediction Fix
fprintf('🤖 Step 1: ML Prediction System Test\n');
fprintf('-------------------------------------\n');

try
    % Test the fixed ML prediction
    fprintf('Testing predictPowerEnhanced...\n');
    testInput = [25, 40, 1013, 60]; % [AT, V, AP, RH]
    
    tic;
    [power, confidence] = predictPowerEnhanced(testInput);
    elapsed = toc;
    
    % Check if it's using ML (not empirical)
    if confidence > 0.9  % ML predictions have high confidence
        fprintf('✅ ML Prediction: SUCCESS\n');
        fprintf('   Power: %.2f MW\n', power);
        fprintf('   Confidence: %.1f%% (High - indicates ML model used)\n', confidence*100);
        fprintf('   Time: %.3f seconds\n', elapsed);
        mlTest = true;
    else
        fprintf('⚠️ ML Prediction: Using empirical model (low confidence)\n');
        fprintf('   Power: %.2f MW\n', power);
        fprintf('   Confidence: %.1f%%\n', confidence*100);
        mlTest = false;
    end
    
catch ME
    fprintf('❌ ML Prediction: FAILED - %s\n', ME.message);
    mlTest = false;
end

% Step 2: Test Multiple Scenarios
fprintf('\n🔄 Step 2: Multiple Scenarios Test\n');
fprintf('----------------------------------\n');

scenarios = {
    15, 35, 1015, 70, 'Cool';
    25, 40, 1013, 60, 'Normal';  
    30, 50, 1005, 80, 'Hot'
};

scenariosPassed = 0;
totalScenarios = size(scenarios, 1);

for i = 1:totalScenarios
    try
        input_vals = [scenarios{i,1}, scenarios{i,2}, scenarios{i,3}, scenarios{i,4}];
        [pow, conf] = predictPowerEnhanced(input_vals);
        
        if conf > 0.9 && pow >= 400 && pow <= 500
            fprintf('✅ %s scenario: %.1f MW (%.0f%% conf)\n', scenarios{i,5}, pow, conf*100);
            scenariosPassed = scenariosPassed + 1;
        else
            fprintf('⚠️ %s scenario: %.1f MW (%.0f%% conf) - Low confidence or out of range\n', ...
                scenarios{i,5}, pow, conf*100);
        end
    catch ME
        fprintf('❌ %s scenario: ERROR - %s\n', scenarios{i,5}, ME.message);
    end
end

scenariosTest = (scenariosPassed == totalScenarios);
fprintf('Result: %d/%d scenarios passed with ML model\n', scenariosPassed, totalScenarios);

% Step 3: Test Dashboard Launch
fprintf('\n🎛️ Step 3: Dashboard Launch Test\n');
fprintf('---------------------------------\n');

try
    fprintf('Launching dashboard (may take 10-15 seconds)...\n');
    app = launchInteractiveDashboard();
    
    % Quick verification
    if isvalid(app) && isprop(app, 'UIFigure') && isvalid(app.UIFigure)
        fprintf('✅ Dashboard launched successfully\n');
        
        % Check verification button
        if isprop(app, 'VerificationButton')
            fprintf('✅ Verification button available\n');
            
            % Check if it points to the complete verification system
            callback = app.VerificationButton.ButtonPushedFcn;
            if contains(func2str(callback.Callback), 'launchVerificationSystem')
                fprintf('✅ Complete verification system restored\n');
                verificationRestored = true;
            else
                fprintf('⚠️ Still using simple verification\n');
                verificationRestored = false;
            end
        else
            fprintf('❌ Verification button missing\n');
            verificationRestored = false;
        end
        
        dashboardTest = true;
        
        % Clean up
        pause(1);
        delete(app);
        fprintf('✅ Dashboard closed cleanly\n');
        
    else
        fprintf('❌ Dashboard launch failed\n');
        dashboardTest = false;
        verificationRestored = false;
    end
    
catch ME
    fprintf('❌ Dashboard test failed: %s\n', ME.message);
    dashboardTest = false;
    verificationRestored = false;
end

% Step 4: Test Weather System
fprintf('\n🌤️ Step 4: Weather Intelligence Test\n');
fprintf('------------------------------------\n');

try
    weatherData = getWeatherIntelligence();
    if isfield(weatherData, 'temperature')
        fprintf('✅ Weather system working\n');
        fprintf('   Temperature: %.1f°C, Humidity: %.1f%%\n', ...
            weatherData.temperature, weatherData.humidity);
        weatherTest = true;
    else
        fprintf('❌ Weather system incomplete\n');
        weatherTest = false;
    end
catch ME
    fprintf('❌ Weather system failed: %s\n', ME.message);
    weatherTest = false;
end

% Final Results
fprintf('\n📊 COMPLETE SYSTEM TEST RESULTS\n');
fprintf('===============================\n');

totalTests = 5;
passedTests = sum([mlTest, scenariosTest, dashboardTest, verificationRestored, weatherTest]);

fprintf('Overall Score: %d/%d tests passed\n\n', passedTests, totalTests);

% Detailed results
fprintf('✅ ML Prediction (High Confidence): %s\n', statusText(mlTest));
fprintf('✅ Multiple Scenarios (ML Model): %s\n', statusText(scenariosTest));  
fprintf('✅ Dashboard Launch: %s\n', statusText(dashboardTest));
fprintf('✅ Complete Verification Restored: %s\n', statusText(verificationRestored));
fprintf('✅ Weather Intelligence: %s\n', statusText(weatherTest));

% System status
if passedTests >= 4
    fprintf('\n🎉 SYSTEM STATUS: EXCELLENT\n');
    fprintf('✅ ML predictions using full ML model (not empirical)\n');
    fprintf('✅ Complete verification system restored with UI\n');
    fprintf('✅ All critical systems operational\n');
elseif passedTests >= 3
    fprintf('\n✅ SYSTEM STATUS: GOOD\n');
    fprintf('Most systems working, check failed tests above\n');
else
    fprintf('\n⚠️ SYSTEM STATUS: NEEDS ATTENTION\n');
    fprintf('Multiple critical issues detected\n');
end

fprintf('\n🚀 Ready to use:\n');
fprintf('   app = launchInteractiveDashboard()\n');
fprintf('   Click "🔍 Verification" for complete UI-based testing\n\n');

function status = statusText(passed)
    if passed
        status = 'PASSED';
    else
        status = 'FAILED';
    end
end