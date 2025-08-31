%% Quick ML Test
fprintf('🧪 Quick ML Prediction Test\n');
fprintf('===========================\n');

% Setup paths first
startup;

% Test with exact same format as expected by ML model
testInput = [25, 40, 1013, 60]; % [AT, V, AP, RH]

try
    fprintf('Testing predictPowerEnhanced...\n');
    [power, confidence] = predictPowerEnhanced(testInput);
    fprintf('✅ Result: %.2f MW (%.1f%% confidence)\n', power, confidence*100);
catch ME
    fprintf('❌ Error: %s\n', ME.message);
end

fprintf('\nTesting predictPowerML directly...\n');
try
    [power2, conf2] = predictPowerML(testInput);
    fprintf('✅ Result: %.2f MW (%.1f%% confidence)\n', power2, mean(conf2.mean_confidence)*100);
catch ME
    fprintf('❌ Error: %s\n', ME.message);
end

fprintf('\nTest completed.\n');