%% Test Dashboard ML Prediction
% Specifically tests ML prediction within the interactive dashboard context

fprintf('🎛️ Testing Dashboard ML Prediction\n');
fprintf('===================================\n\n');

% Step 1: Initialize system
fprintf('Step 1: System Initialization\n');
fprintf('-----------------------------\n');
try
    startup;
    fprintf('✅ System initialized\n');
catch ME
    fprintf('❌ Initialization failed: %s\n', ME.message);
    return;
end

% Step 2: Test ML function directly (as baseline)
fprintf('\nStep 2: Direct ML Function Test\n');
fprintf('-------------------------------\n');
try
    testInput = [25, 40, 1013, 60]; % [AT, V, AP, RH]
    [power, confidence] = predictPowerEnhanced(testInput);
    
    if confidence > 0.9
        fprintf('✅ Direct ML function: WORKING\n');
        fprintf('   Power: %.2f MW, Confidence: %.1f%%\n', power, confidence*100);
        directMLTest = true;
    else
        fprintf('⚠️ Direct ML function: Using empirical (%.1f%% confidence)\n', confidence*100);
        directMLTest = false;
    end
catch ME
    fprintf('❌ Direct ML function: ERROR - %s\n', ME.message);
    directMLTest = false;
end

% Step 3: Launch dashboard and test ML within dashboard
fprintf('\nStep 3: Dashboard ML Test\n');
fprintf('-------------------------\n');

try
    fprintf('Launching dashboard...\n');
    app = launchInteractiveDashboard();
    
    if isvalid(app) && isprop(app, 'ModelType')
        fprintf('✅ Dashboard launched successfully\n');
        fprintf('   Model Type: %s\n', app.ModelType);
        fprintf('   Model Loaded: %s\n', string(app.ModelLoaded));
        fprintf('   Model Accuracy: %.1f%%\n', app.ModelAccuracy);
        
        % Step 4: Test prediction within dashboard context
        fprintf('\nStep 4: Dashboard Prediction Test\n');
        fprintf('---------------------------------\n');
        
        % Create test parameters as the dashboard would
        params = struct();
        params.temperature = 25;
        params.vacuum = 40;
        params.pressure = 1013;
        params.humidity = 60;
        
        try
            % Enable debug mode for detailed output
            app.DebugMode = true;
            
            fprintf('Testing dashboard prediction with debug info:\n');
            [dashPower, dashConfidence] = app.enhancedPredictPower(params);
            
            fprintf('Dashboard Prediction Results:\n');
            fprintf('   Power: %.2f MW\n', dashPower);
            fprintf('   Confidence: %.1f%%\n', dashConfidence * 100);
            
            if dashConfidence > 0.9
                fprintf('✅ Dashboard ML prediction: WORKING (using ML model)\n');
                dashMLTest = true;
            else
                fprintf('⚠️ Dashboard ML prediction: Using empirical fallback\n');
                dashMLTest = false;
            end
            
        catch ME
            fprintf('❌ Dashboard prediction failed: %s\n', ME.message);
            dashMLTest = false;
        end
        
        % Step 5: Test during simulation (if timer exists)
        fprintf('\nStep 5: Simulation Context Test\n');
        fprintf('-------------------------------\n');
        
        try
            if isprop(app, 'SimulationTimer') && ~isempty(app.SimulationTimer)
                fprintf('Testing ML prediction during simulation context...\n');
                
                % Simulate what happens during timer callback
                simulationParams = struct();
                simulationParams.temperature = app.TemperatureSlider.Value;
                simulationParams.vacuum = app.VacuumSlider.Value;
                simulationParams.pressure = app.PressureSlider.Value;
                simulationParams.humidity = app.HumiditySlider.Value;
                
                [simPower, simConfidence] = app.enhancedPredictPower(simulationParams);
                
                if simConfidence > 0.9
                    fprintf('✅ Simulation ML prediction: WORKING\n');
                    simMLTest = true;
                else
                    fprintf('⚠️ Simulation ML prediction: Using empirical\n');
                    simMLTest = false;
                end
            else
                fprintf('ℹ️ No simulation timer found, skipping simulation test\n');
                simMLTest = true; % Not applicable
            end
        catch ME
            fprintf('❌ Simulation prediction failed: %s\n', ME.message);
            simMLTest = false;
        end
        
        dashboardTest = true;
        
        % Cleanup
        pause(1);
        delete(app);
        fprintf('✅ Dashboard closed\n');
        
    else
        fprintf('❌ Dashboard launch failed or invalid\n');
        dashboardTest = false;
        dashMLTest = false;
        simMLTest = false;
    end
    
catch ME
    fprintf('❌ Dashboard test failed: %s\n', ME.message);
    dashboardTest = false;
    dashMLTest = false;
    simMLTest = false;
end

% Results Summary
fprintf('\n📊 DASHBOARD ML TEST RESULTS\n');
fprintf('============================\n');

totalTests = 4;
passedTests = sum([directMLTest, dashboardTest, dashMLTest, simMLTest]);

fprintf('Score: %d/%d tests passed\n\n', passedTests, totalTests);
fprintf('✅ Direct ML Function: %s\n', statusText(directMLTest));
fprintf('✅ Dashboard Launch: %s\n', statusText(dashboardTest));
fprintf('✅ Dashboard ML Prediction: %s\n', statusText(dashMLTest));
fprintf('✅ Simulation ML Context: %s\n', statusText(simMLTest));

if dashMLTest && directMLTest
    fprintf('\n🎉 SUCCESS: ML predictions working in dashboard!\n');
    fprintf('Dashboard is using the full ML model (not empirical)\n');
elseif directMLTest && ~dashMLTest
    fprintf('\n⚠️ PARTIAL SUCCESS: ML works directly but fails in dashboard\n');
    fprintf('Dashboard is falling back to empirical model\n');
    fprintf('Check the debug output above for specific error messages\n');
elseif ~directMLTest
    fprintf('\n❌ PROBLEM: ML function not working at all\n');
    fprintf('Need to fix the basic ML prediction function first\n');
else
    fprintf('\n⚠️ MIXED RESULTS: Check individual test results above\n');
end

fprintf('\n💡 Tips:\n');
fprintf('- If dashboard ML fails, check path issues or model loading\n');
fprintf('- Enable debug mode in dashboard for detailed error info\n');
fprintf('- Verify ModelType is set correctly during dashboard initialization\n\n');

function status = statusText(passed)
    if passed
        status = 'PASSED';
    else
        status = 'FAILED';
    end
end