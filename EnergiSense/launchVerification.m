function launchVerification()
%LAUNCHVERIFICATION Launch EnergISense verification systems
%
% This function provides access to both verification systems:
% 1. Professional GUI Verification - Modern, elegant interface
% 2. Simple Console Verification - Quick command-line testing
%
% Usage:
%   launchVerification()  - Shows selection dialog
%
% Author: EnergISense Development Team

    fprintf('\n🎯 EnergISense Verification Systems\n');
    fprintf('===================================\n\n');
    
    fprintf('Select verification method:\n');
    fprintf('1. 🖥️  Professional GUI Verification (Recommended)\n');
    fprintf('2. ⚡ Simple Console Verification (Quick)\n');
    fprintf('3. 🚀 Launch Interactive Dashboard\n');
    fprintf('4. ❌ Cancel\n\n');
    
    choice = input('Enter choice (1-4): ');
    
    switch choice
        case 1
            fprintf('\n🚀 Launching Professional GUI Verification...\n');
            try
                % Try compatible GUI first
                compatibleVerificationGUI();
                fprintf('✅ Compatible professional GUI launched successfully\n');
            catch ME
                fprintf('⚠️ Compatible GUI failed: %s\n', ME.message);
                try
                    % Fallback to original GUI
                    professionalVerificationGUI();
                    fprintf('✅ Original professional GUI launched successfully\n');
                catch ME2
                    fprintf('❌ All GUI launches failed: %s\n', ME2.message);
                    fprintf('💡 Try option 2 for console verification\n');
                end
            end
            
        case 2  
            fprintf('\n⚡ Running Simple Console Verification...\n');
            try
                simpleVerification();
            catch ME
                fprintf('❌ Console verification failed: %s\n', ME.message);
            end
            
        case 3
            fprintf('\n🚀 Launching Interactive Dashboard...\n');
            try
                launchInteractiveDashboard();
                fprintf('✅ Dashboard launched successfully\n');
            catch ME
                fprintf('❌ Dashboard launch failed: %s\n', ME.message);
                fprintf('💡 Run verification first to check system status\n');
            end
            
        case 4
            fprintf('Operation cancelled.\n');
            
        otherwise
            fprintf('❌ Invalid choice. Operation cancelled.\n');
    end
    
    fprintf('\n');
end