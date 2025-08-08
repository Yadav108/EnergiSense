function demo()
% Updated EnergiSense Demo with proper data loading
    fprintf('🏭 EnergiSense Complete Demo\n');
    fprintf('============================\n\n');
    
    % STEP 0: Load Essential Data (CRITICAL!)
    fprintf('Step 0: Loading Digital Twin Data...\n');
    if exist('Digitaltwin.mat', 'file')
        load('Digitaltwin.mat');
        fprintf('✅ Digitaltwin.mat loaded successfully\n');
        fprintf('📁 Workspace now contains CCPP input/output data\n');
        
        % Show what was loaded
        workspace_vars = who;
        fprintf('   Variables loaded: %s\n', strjoin(workspace_vars, ', '));
    else
        fprintf('❌ Digitaltwin.mat not found - this is required!\n');
        fprintf('   Without this file, Simulink model won''t have proper data\n');
        return;
    end
    
    fprintf('\nStep 1: Model Check\n');
    checkModel();
    
    fprintf('\nStep 2: Dashboard Test\n');
    testDashboard();
    
    fprintf('\nStep 3: System Launch Instructions\n');
    fprintf('Now your system has proper data. Next steps:\n');
    fprintf('1. 🔧 Run: open_system(''simulink/Energisense.slx'')\n');
    fprintf('2. ▶️  Click RUN in Simulink (now has real CCPP data!)\n');
    fprintf('3. 📊 Run: runDashboard() (for real-time monitoring)\n');
    fprintf('4. 🎯 Observe: Real predictions vs actual CCPP data\n\n');
    
    fprintf('🎉 Complete system ready with actual CCPP dataset!\n');
    fprintf('💡 Key difference: Simulink now uses real environmental data\n');
    fprintf('   instead of random signals!\n');
end
