%% FILE 2: Save this as "dashboardInstructions.m"

function dashboardInstructions()
    fprintf('\n=== DIGITAL TWIN DASHBOARD SETUP ===\n\n');
    
    fprintf('STEP 1: Update Your Simulink Model\n');
    fprintf('- Add "To Workspace" blocks for:\n');
    fprintf('  * Predicted Power (from Digital Twin Core "y")\n');
    fprintf('  * Control Signal (from PID Controller "u")\n');
    fprintf('  * Actual Power (from PE_ts)\n\n');
    
    fprintf('STEP 2: Configure To Workspace Blocks\n');
    fprintf('- Variable names: predicted_power, control_signal, actual_power\n');
    fprintf('- Save format: Array\n');
    fprintf('- Maximum rows: 1000\n\n');
    
    fprintf('STEP 3: Run Dashboard\n');
    fprintf('- Start Simulink model\n');
    fprintf('- Run: runDashboard()\n');
    fprintf('- Monitor real-time performance\n\n');
    
    fprintf('FEATURES:\n');
    fprintf('✓ Real-time power tracking\n');
    fprintf('✓ Control signal monitoring\n');
    fprintf('✓ Performance metrics\n');
    fprintf('✓ Environmental conditions\n');
    fprintf('✓ System status display\n\n');
    
    fprintf('QUICK TEST:\n');
    fprintf('1. Make sure Simulink model is running\n');
    fprintf('2. Type: runDashboard()\n');
    fprintf('3. You should see a 6-panel dashboard window\n\n');
end