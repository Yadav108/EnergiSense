% Diagnose ML Prediction Issues
fprintf('🔍 ML Prediction Diagnostics\n');
fprintf('=============================\n\n');

% Step 1: Path check
fprintf('1. Checking paths...\n');
pathsToCheck = {'core/prediction', 'core/models'};
for i = 1:length(pathsToCheck)
    if exist(pathsToCheck{i}, 'dir')
        fprintf('   ✅ %s: Found\n', pathsToCheck{i});
        addpath(pathsToCheck{i});
    else
        fprintf('   ❌ %s: Missing\n', pathsToCheck{i});
    end
end

% Step 2: Function availability
fprintf('\n2. Checking functions...\n');
funcsToCheck = {'predictPowerEnhanced', 'predictPowerML'};
for i = 1:length(funcsToCheck)
    if exist(funcsToCheck{i}, 'file')
        fprintf('   ✅ %s: Available\n', funcsToCheck{i});
    else
        fprintf('   ❌ %s: Missing\n', funcsToCheck{i});
    end
end

% Step 3: Model files
fprintf('\n3. Checking model files...\n');
modelFiles = {
    'core/models/ccpp_random_forest_model.mat';
    'core/models/ensemblePowerModel.mat'
};
for i = 1:length(modelFiles)
    if exist(modelFiles{i}, 'file')
        fprintf('   ✅ %s: Found\n', modelFiles{i});
        
        % Try to get file info
        try
            info = dir(modelFiles{i});
            fprintf('      Size: %.2f MB\n', info.bytes / 1024 / 1024);
        catch
            fprintf('      Size: Unknown\n');
        end
    else
        fprintf('   ❌ %s: Missing\n', modelFiles{i});
    end
end

% Step 4: Simple prediction test
fprintf('\n4. Testing basic prediction...\n');
try
    % Test input in correct format [AT, V, AP, RH]
    testInput = [25, 40, 1013, 60];
    fprintf('   Input: [AT=%.0f°C, V=%.0f cmHg, AP=%.0f mbar, RH=%.0f%%]\n', testInput);
    
    % Try prediction with timeout protection
    fprintf('   Attempting prediction...\n');
    tic;
    [power, confidence] = predictPowerEnhanced(testInput);
    elapsed = toc;
    
    fprintf('   ✅ Success!\n');
    fprintf('      Power: %.2f MW\n', power);
    fprintf('      Confidence: %.1f%%\n', confidence * 100);
    fprintf('      Time: %.3f seconds\n', elapsed);
    
    % Validate result
    if power > 400 && power < 500
        fprintf('   ✅ Result within expected range\n');
    else
        fprintf('   ⚠️ Result outside expected range (400-500 MW)\n');
    end
    
catch ME
    fprintf('   ❌ Prediction failed: %s\n', ME.message);
    fprintf('   Stack trace:\n');
    for i = 1:min(3, length(ME.stack))
        fprintf('      %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

fprintf('\nDiagnostics complete.\n');