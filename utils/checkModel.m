%% Model Verification Script - checkModel.m
% Check if your actual trained model is being used

function checkModel()
    fprintf('=== CCPP DIGITAL TWIN MODEL VERIFICATION ===\n\n');
    
    % Check 1: Model file existence and loading
    checkActualModel();
    
    fprintf('\n' + repmat('=', 1, 50) + '\n\n');
    
    % Check 2: Test with CCPP data
    testWithCCPPData();
    
    fprintf('\n' + repmat('=', 1, 50) + '\n\n');
    
    % Check 3: Simulink integration check
    checkSimulinkModel();
end

%% Local Functions

function checkActualModel()
    fprintf('=== CHECKING YOUR TRAINED MODEL ===\n\n');
    
    % Check if your trained model file exists
    if exist('ensemblePowerModel.mat', 'file')
        fprintf('✅ Found: ensemblePowerModel.mat\n');
        
        % Try to load your actual model
        try
            model = loadLearnerForCoder('ensemblePowerModel');
            fprintf('✅ Successfully loaded your trained model!\n');
            fprintf('   Model type: %s\n', class(model));
            
            % Test with sample CCPP data
            sample_input = [25.36, 40.27, 68.77, 1013.84]; % Sample CCPP values
            prediction = predict(model, sample_input);
            fprintf('✅ Model prediction test: %.2f MW\n', prediction);
            
            fprintf('\n✅ GOOD NEWS: Your actual trained model is working!\n');
            
        catch ME
            fprintf('❌ Error loading your model: %s\n', ME.message);
            fprintf('   This means we are using the fallback linear model\n');
        end
    else
        fprintf('❌ ensemblePowerModel.mat not found in current directory\n');
        fprintf('   Current directory: %s\n', pwd);
        fprintf('   Available .mat files:\n');
        matFiles = dir('*.mat');
        if isempty(matFiles)
            fprintf('   (No .mat files found)\n');
        else
            for i = 1:length(matFiles)
                fprintf('   - %s\n', matFiles(i).name);
            end
        end
    end
end

function testWithCCPPData()
    fprintf('=== TESTING WITH CCPP DATA ===\n\n');
    
    % Typical CCPP ranges from your dataset
    fprintf('Testing with typical CCPP values:\n\n');
    
    test_cases = [
        1.81, 39.42, 88.62, 1013.84;  % Low load
        14.96, 41.76, 80.26, 1010.24; % Medium load  
        25.36, 40.27, 68.77, 1017.00; % High load
        8.34, 40.80, 86.27, 1009.23   % Variable conditions
    ];
    
    labels = {'Low Load', 'Medium Load', 'High Load', 'Variable'};
    
    for i = 1:size(test_cases, 1)
        input_data = test_cases(i, :);
        
        % Test current prediction function
        try
            [power, confidence, anomaly] = predictPowerEnhanced(input_data);
            fprintf('%s: AT=%.1f°C, V=%.1f, RH=%.1f%%, AP=%.1f\n', ...
                labels{i}, input_data(1), input_data(2), input_data(3), input_data(4));
            fprintf('   → Predicted Power: %.2f MW\n', power);
            fprintf('   → Confidence: %.1f%%\n', confidence*100);
            if anomaly
                fprintf('   → Anomaly: YES ⚠️\n\n');
            else
                fprintf('   → Anomaly: NO ✅\n\n');
            end
        catch ME
            fprintf('Error testing %s: %s\n\n', labels{i}, ME.message);
        end
    end
end

function checkSimulinkModel()
    fprintf('=== SIMULINK MODEL STATUS ===\n\n');
    
    fprintf('To verify what model is being used in Simulink:\n\n');
    fprintf('1. Open your Simulink model\n');
    fprintf('2. Double-click "Digital Twin Inference Core"\n');
    fprintf('3. Look for these messages in the MATLAB Function:\n');
    fprintf('   - "Model loaded successfully" = Using your actual model ✅\n');
    fprintf('   - "Using fallback linear model" = Using backup model ⚠️\n\n');
    
    fprintf('MODEL COMPARISON:\n');
    fprintf('┌─────────────────────┬──────────────────────────────────┐\n');
    fprintf('│ Your Actual Model   │ Fallback Linear Model            │\n');
    fprintf('├─────────────────────┼──────────────────────────────────┤\n');
    fprintf('│ ensemblePowerModel  │ Simple linear equations          │\n');
    fprintf('│ High accuracy       │ Basic approximation              │\n');
    fprintf('│ Trained on CCPP     │ Generic power relationship       │\n');
    fprintf('│ Best performance    │ Works but less accurate          │\n');
    fprintf('└─────────────────────┴──────────────────────────────────┘\n\n');
    
    fprintf('RECOMMENDED ACTIONS:\n');
    fprintf('• If using fallback: Copy ensemblePowerModel.mat to current folder\n');
    fprintf('• If using actual model: You are all set! ✅\n');
    fprintf('• Test accuracy: Compare predictions with known CCPP data\n\n');
end