% EnergiSense Startup Script
% Automatically configures paths and environment

fprintf('🚀 Starting EnergiSense environment...\n');

% Add all project paths
addpath(genpath('core'));
addpath(genpath('control'));
addpath(genpath('dashboard'));
addpath(genpath('simulation'));
addpath(genpath('utilities'));
addpath(genpath('validation'));
addpath(genpath('examples'));

fprintf('✅ EnergiSense paths configured\n');
fprintf('💡 Type: setupEnergiSense() for first-time setup\n');
fprintf('🎮 Type: demo() to see the system in action\n');
