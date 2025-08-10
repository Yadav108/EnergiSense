function setupEnergiSense()
% SETUPENERGISENSE Complete project setup and verification

fprintf('🔧 EnergiSense Project Setup\n');
fprintf('===========================\n\n');

% Step 1: Configure paths
fprintf('Step 1: Configuring paths...\n');
run('startup.m');

% Step 2: Verify key files
fprintf('\nStep 2: Verifying installation...\n');
key_files = {
    'core/models/ensemblePowerModel.mat'
    'core/models/digitaltwin.mat'
    'simulation/models/Energisense.slx'
    'examples/quickstart/demo.m'
};

all_good = true;
for i = 1:length(key_files)
    if exist(key_files{i}, 'file')
        fprintf('   ✅ %s\n', key_files{i});
    else
        fprintf('   ❌ %s (missing)\n', key_files{i});
        all_good = false;
    end
end

% Step 3: Quick model test
if all_good && exist('core/validation/checkModel.m', 'file')
    fprintf('\nStep 3: Testing model...\n');
    try
        checkModel();
    catch ME
        fprintf('   ⚠️ Model test failed: %s\n', ME.message);
    end
end

fprintf('\n🎉 Setup complete!\n');
fprintf('\n📚 Available commands:\n');
fprintf('   demo()                    - Run main demonstration\n');
fprintf('   runDashboard()           - Launch monitoring dashboard\n');
fprintf('   configureEnergiSense()   - Configure control system\n');
fprintf('\n📂 Project structure: docs/user/README.md\n');

end
