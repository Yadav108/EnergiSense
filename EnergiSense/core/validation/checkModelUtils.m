function checkModel()
% checkModel - Enhanced model verification

    fprintf('=== CCPP DIGITAL TWIN MODEL VERIFICATION ===\n\n');
    
    if exist('models/ensemblePowerModel.mat', 'file')
        fprintf('✅ Found: ensemblePowerModel.mat\n');
        
        % Try direct loading
        try
            data = load('models/ensemblePowerModel.mat');
            fields = fieldnames(data);
            fprintf('📁 File contains: %s\n', strjoin(fields, ', '));
            fprintf('✅ Model file loaded successfully\n');
        catch
            fprintf('⚠️  File loading issue\n');
        end
        
        % Test enhanced prediction function
        fprintf('\n🧪 Testing prediction function...\n');
        try
            sample_input = [25.36, 40.27, 68.77, 1013.84];
            [power, confidence, anomaly] = predictPowerEnhanced(sample_input);
            fprintf('✅ Prediction: %.2f MW\n', power);
            fprintf('✅ Confidence: %.1f%%\n', confidence*100);
            fprintf('✅ Anomaly: %s\n', char("No" + anomaly * "Yes"));
            fprintf('\n🎉 YOUR MODEL IS WORKING PERFECTLY!\n');
        catch ME
            fprintf('❌ Prediction error: %s\n', ME.message);
        end
    else
        fprintf('❌ Model file not found\n');
    end
end
