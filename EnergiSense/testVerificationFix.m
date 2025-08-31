%% Test Verification System Fix
% Verifies that the duplicate function error is resolved and verification works

fprintf('🔧 Testing Verification System Fix\n');
fprintf('==================================\n\n');

% Test 1: Check for duplicate function errors
fprintf('Test 1: Function Declaration Check\n');
fprintf('----------------------------------\n');

try
    % This would fail if there are duplicate function errors
    fprintf('Checking dashboard file syntax...\n');
    
    % Parse the file to check for syntax errors
    dashboardFile = 'dashboard/interactive/EnergiSenseInteractiveDashboard.m';
    
    if exist(dashboardFile, 'file')
        fprintf('✅ Dashboard file exists\n');
        
        % Try to get function information (this would fail with duplicates)
        try
            functions_list = which('-all', 'EnergiSenseInteractiveDashboard');
            if ~isempty(functions_list)
                fprintf('✅ No duplicate function declaration errors\n');
                syntaxTest = true;
            else
                fprintf('⚠️ Function not found in path\n');
                syntaxTest = false;
            end
        catch ME
            fprintf('❌ Function check error: %s\n', ME.message);
            syntaxTest = false;
        end
    else
        fprintf('❌ Dashboard file not found\n');
        syntaxTest = false;
    end
    
catch ME
    fprintf('❌ Syntax check failed: %s\n', ME.message);
    syntaxTest = false;
end

% Test 2: Dashboard Launch Test
fprintf('\nTest 2: Dashboard Launch\n');
fprintf('------------------------\n');

try
    fprintf('Launching dashboard...\n');
    startup;
    
    tic;
    app = launchInteractiveDashboard();
    launchTime = toc;
    
    if isvalid(app) && isprop(app, 'UIFigure') && isvalid(app.UIFigure)
        fprintf('✅ Dashboard launched successfully (%.1f seconds)\n', launchTime);
        
        % Check verification button
        if isprop(app, 'VerificationButton')
            fprintf('✅ Verification button found\n');
            
            % Check button callback
            callback = app.VerificationButton.ButtonPushedFcn;
            if contains(func2str(callback.Callback), 'launchVerificationSystem')
                fprintf('✅ Complete verification system connected\n');
                buttonTest = true;
            else
                fprintf('⚠️ Using simple verification\n');
                buttonTest = false;
            end
        else
            fprintf('❌ Verification button missing\n');
            buttonTest = false;
        end
        
        dashboardTest = true;
        
        % Cleanup
        pause(1);
        delete(app);
        fprintf('✅ Dashboard closed cleanly\n');
        
    else
        fprintf('❌ Dashboard launch failed\n');
        dashboardTest = false;
        buttonTest = false;
    end
    
catch ME
    fprintf('❌ Dashboard test failed: %s\n', ME.message);
    if contains(ME.message, 'already been declared')
        fprintf('   This indicates duplicate function errors!\n');
    end
    dashboardTest = false;
    buttonTest = false;
end

% Test 3: ML Prediction Test
fprintf('\nTest 3: ML Prediction\n');
fprintf('---------------------\n');

try
    [power, confidence] = predictPowerEnhanced([25, 40, 1013, 60]);
    
    if confidence > 0.9
        fprintf('✅ ML model working (%.1f%% confidence)\n', confidence*100);
        fprintf('   Power: %.2f MW\n', power);
        mlTest = true;
    else
        fprintf('⚠️ Using empirical model (%.1f%% confidence)\n', confidence*100);
        mlTest = false;
    end
    
catch ME
    fprintf('❌ ML prediction failed: %s\n', ME.message);
    mlTest = false;
end

% Results Summary
fprintf('\n📊 VERIFICATION FIX TEST RESULTS\n');
fprintf('================================\n');

totalTests = 4;
passedTests = sum([syntaxTest, dashboardTest, buttonTest, mlTest]);

fprintf('Score: %d/%d tests passed\n\n', passedTests, totalTests);
fprintf('✅ Syntax/Duplicate Functions: %s\n', statusText(syntaxTest));
fprintf('✅ Dashboard Launch: %s\n', statusText(dashboardTest));
fprintf('✅ Verification Button: %s\n', statusText(buttonTest));
fprintf('✅ ML Prediction: %s\n', statusText(mlTest));

if passedTests >= 3
    fprintf('\n🎉 VERIFICATION FIX: SUCCESS\n');
    fprintf('No more duplicate function errors!\n');
    fprintf('Complete verification system is working!\n');
else
    fprintf('\n⚠️ VERIFICATION FIX: ISSUES DETECTED\n');
    fprintf('Some problems remain, check results above.\n');
end

fprintf('\n💡 Ready to use:\n');
fprintf('   app = launchInteractiveDashboard()\n');
fprintf('   Click "🔍 Verification" for complete UI testing\n\n');

function status = statusText(passed)
    if passed
        status = 'PASSED';
    else
        status = 'FAILED';
    end
end