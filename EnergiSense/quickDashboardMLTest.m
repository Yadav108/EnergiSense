%% Quick Dashboard ML Test
% Fast test to identify ML prediction issues in dashboard

fprintf('⚡ Quick Dashboard ML Test\n');
fprintf('=========================\n');

try
    % Quick initialization
    startup;
    
    % Test direct ML (should work)
    fprintf('Testing direct ML...\n');
    try
        [power, conf] = predictPowerEnhanced([25, 40, 1013, 60]);
        if conf > 0.9
            fprintf('✅ Direct ML: Working (%.1f%% conf)\n', conf*100);
        else
            fprintf('⚠️ Direct ML: Empirical (%.1f%% conf)\n', conf*100);
        end
    catch ME
        fprintf('❌ Direct ML: %s\n', ME.message);
    end
    
    % Launch dashboard with minimal setup
    fprintf('Launching dashboard for ML test...\n');
    app = launchInteractiveDashboard();
    
    % Quick test of dashboard prediction
    fprintf('Testing dashboard ML prediction...\n');
    
    % Enable debug mode
    app.DebugMode = true;
    
    % Create test parameters
    params.temperature = 25;
    params.vacuum = 40; 
    params.pressure = 1013;
    params.humidity = 60;
    
    fprintf('Dashboard ModelType: %s\n', app.ModelType);
    fprintf('Dashboard ModelLoaded: %s\n', string(app.ModelLoaded));
    
    % Test prediction
    [dashPower, dashConf] = app.enhancedPredictPower(params);
    
    if dashConf > 0.9
        fprintf('✅ Dashboard ML: Working (%.1f%% conf, %.1f MW)\n', dashConf*100, dashPower);
    else
        fprintf('⚠️ Dashboard ML: Empirical fallback (%.1f%% conf, %.1f MW)\n', dashConf*100, dashPower);
    end
    
    delete(app);
    
catch ME
    fprintf('❌ Test failed: %s\n', ME.message);
end

fprintf('Test complete.\n');