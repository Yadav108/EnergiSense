function systemCheck()
% SYSTEMCHECK Verify EnergiSense installation

fprintf('🔍 EnergiSense System Check\n');
fprintf('=========================\n');

% Check MATLAB version
fprintf('MATLAB Version: %s\n', version('-release'));

% Check directory structure
required_dirs = {'core', 'control', 'dashboard', 'simulation', 'examples'};
fprintf('\nDirectory Structure:\n');
for i = 1:length(required_dirs)
    if exist(required_dirs{i}, 'dir')
        fprintf('   ✅ %s/\n', required_dirs{i});
    else
        fprintf('   ❌ %s/ (missing)\n', required_dirs{i});
    end
end

fprintf('\n🎯 System check complete!\n');
end
