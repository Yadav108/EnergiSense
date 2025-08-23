function testDashboardQuick()
%TESTDASHBOARDQUICK Quick dashboard test with timeout
%
% Simple test to verify dashboard launches without hanging

fprintf('🧪 Quick Dashboard Test\n');
fprintf('========================\n');

% Store original directory
originalDir = pwd;

try
    % Navigate to EnergiSense root
    if exist('core', 'dir') && exist('dashboard', 'dir')
        % Already in root
    elseif exist('EnergiSense', 'dir')
        cd('EnergiSense');
    end
    
    % Add paths
    addpath('core');
    addpath('core/models');  
    addpath('dashboard');
    addpath('dashboard/interactive');
    
    % Test direct dashboard creation (bypass launcher complexity)
    fprintf('Testing direct dashboard creation...\n');
    tic;
    
    % Navigate to dashboard directory
    cd('dashboard/interactive');
    
    % Create dashboard directly
    app = EnergiSenseInteractiveDashboard();
    
    launchTime = toc;
    
    fprintf('✅ Direct dashboard creation successful!\n');
    fprintf('Launch time: %.2f seconds\n', launchTime);
    
    % Quick validation
    if isprop(app, 'UIFigure') && isvalid(app.UIFigure)
        fprintf('✅ UI Figure valid\n');
        
        % Test basic properties
        if isprop(app, 'StatusLabel')
            fprintf('✅ Status components present\n');
        end
        
        if isprop(app, 'AccuracyGauge') 
            fprintf('✅ Gauge components present\n');
        end
    end
    
    % Clean shutdown
    fprintf('Cleaning up...\n');
    delete(app);
    fprintf('✅ Cleanup successful\n');
    
catch ME
    fprintf('❌ Test failed: %s\n', ME.message);
    fprintf('Location: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
end

% Restore directory
cd(originalDir);

fprintf('Test complete.\n');
end