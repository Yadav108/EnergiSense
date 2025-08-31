% Quick ML test without complex loading
fprintf('Quick ML Test\n');
fprintf('=============\n');

% Test 1: Basic prediction function exists
if exist('predictPowerEnhanced', 'file')
    fprintf('✅ predictPowerEnhanced: Found\n');
else
    fprintf('❌ predictPowerEnhanced: Missing\n');
    return;
end

% Test 2: Try simple prediction with fallback
try
    fprintf('Testing basic prediction...\n');
    testInput = [25, 40, 1013, 60];
    
    % Try with minimal overhead
    [power, confidence] = predictPowerEnhanced(testInput);
    
    fprintf('✅ Prediction successful: %.2f MW\n', power);
    fprintf('✅ Confidence: %.1f%%\n', confidence * 100);
    
catch ME
    fprintf('❌ Prediction failed: %s\n', ME.message);
    
    % Try even simpler fallback
    try
        fprintf('Trying empirical fallback...\n');
        AT = testInput(1);
        V = testInput(2); 
        AP = testInput(3);
        RH = testInput(4);
        
        % Simple empirical formula
        power = 454.365 - 1.977 * AT - 0.234 * V + 0.0618 * (AP - 1013) - 0.158 * (RH - 50) / 50;
        power = max(420, min(500, power));
        
        fprintf('✅ Empirical prediction: %.2f MW\n', power);
        
    catch ME2
        fprintf('❌ Even empirical failed: %s\n', ME2.message);
    end
end

fprintf('Test complete.\n');