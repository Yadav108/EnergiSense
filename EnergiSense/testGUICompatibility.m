function testGUICompatibility()
%TESTGUICOMPATIBILITY Test GUI compatibility without full launch
%
% This function tests if the GUI components can be created without
% layout errors, providing a quick validation.

    fprintf('🧪 Testing GUI Compatibility\n');
    fprintf('============================\n\n');
    
    try
        fprintf('1. Testing basic uifigure creation...\n');
        testFig = uifigure('Name', 'Test', 'Position', [100, 100, 400, 300], 'Visible', 'off');
        fprintf('   ✅ uifigure creation successful\n');
        
        fprintf('2. Testing uipanel creation...\n');
        testPanel = uipanel(testFig, 'Position', [10, 10, 380, 280], 'BackgroundColor', [1, 1, 1]);
        fprintf('   ✅ uipanel creation successful\n');
        
        fprintf('3. Testing uilabel creation...\n');
        testLabel = uilabel(testPanel, 'Text', 'Test Label', 'Position', [10, 200, 100, 20]);
        fprintf('   ✅ uilabel creation successful\n');
        
        fprintf('4. Testing uibutton creation...\n');
        testButton = uibutton(testPanel, 'Text', 'Test Button', 'Position', [10, 170, 100, 25]);
        fprintf('   ✅ uibutton creation successful\n');
        
        fprintf('5. Testing uitextarea creation...\n');
        testTextArea = uitextarea(testPanel, 'Position', [10, 80, 200, 80], 'Value', {'Test', 'Text'});
        fprintf('   ✅ uitextarea creation successful\n');
        
        fprintf('6. Testing uilistbox creation...\n');
        testListBox = uilistbox(testPanel, 'Position', [10, 10, 100, 60], 'Items', {'Item 1', 'Item 2'});
        fprintf('   ✅ uilistbox creation successful\n');
        
        fprintf('7. Testing gauge creation...\n');
        try
            testGauge = uisemicirculargauge(testPanel, 'Position', [220, 150, 100, 60]);
            fprintf('   ✅ uisemicirculargauge creation successful\n');
        catch
            fprintf('   ⚠️ uisemicirculargauge not available (older MATLAB version)\n');
        end
        
        fprintf('8. Testing axes creation...\n');
        try
            testAxes = uiaxes(testPanel, 'Position', [220, 20, 150, 100]);
            fprintf('   ✅ uiaxes creation successful\n');
        catch
            % Try regular axes as fallback
            testAxes = axes('Parent', testPanel, 'Position', [0.6, 0.1, 0.35, 0.4]);
            fprintf('   ✅ axes creation successful (fallback)\n');
        end
        
        % Clean up
        close(testFig);
        
        fprintf('\n✅ GUI Compatibility Test PASSED\n');
        fprintf('   All required components are available\n');
        fprintf('   Compatible GUI should work properly\n\n');
        
        fprintf('🚀 You can now safely launch the verification GUI:\n');
        fprintf('   >> compatibleVerificationGUI()\n');
        fprintf('   >> launchVerification()\n\n');
        
        return;
        
    catch ME
        fprintf('\n❌ GUI Compatibility Test FAILED\n');
        fprintf('   Error: %s\n', ME.message);
        fprintf('   Your MATLAB version may not support all GUI components\n');
        fprintf('   Recommendation: Use simpleVerification() instead\n\n');
        
        % Clean up if figure exists
        try
            close(testFig);
        catch
            % Figure may not exist
        end
        
        rethrow(ME);
    end

end