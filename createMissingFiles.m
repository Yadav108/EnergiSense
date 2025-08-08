function createMissingFiles()
% createMissingFiles - Create missing files for EnergiSense repository
% This function creates the LICENSE file and other missing files

    fprintf('Creating missing files for EnergiSense...\n\n');
    
    % Create LICENSE file
    createLicenseFile();
    
    fprintf('Files created successfully!\n\n');
    
    fprintf('Now run these Git commands to upload:\n');
    fprintf('system(''git add .'')\n');
    fprintf('system(''git commit -m "Add LICENSE file"'')\n');
    fprintf('system(''git push origin master'')\n');
end

function createLicenseFile()
    fprintf('Creating LICENSE file...\n');
    
    % Create license content as cell array of strings
    license_lines = {
        'MIT License'
        ''
        'Copyright (c) 2025 EnergiSense Project'
        ''
        'Permission is hereby granted, free of charge, to any person obtaining a copy'
        'of this software and associated documentation files (the "Software"), to deal'
        'in the Software without restriction, including without limitation the rights'
        'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell'
        'copies of the Software, and to permit persons to whom the Software is'
        'furnished to do so, subject to the following conditions:'
        ''
        'The above copyright notice and this permission notice shall be included in all'
        'copies or substantial portions of the Software.'
        ''
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR'
        'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,'
        'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE'
        'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER'
        'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,'
        'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE'
        'SOFTWARE.'
    };
    
    % Write LICENSE file
    fid = fopen('LICENSE', 'w');
    if fid == -1
        error('Could not create LICENSE file');
    end
    
    % Write each line
    for i = 1:length(license_lines)
        fprintf(fid, '%s\n', license_lines{i});
    end
    
    fclose(fid);
    
    fprintf('LICENSE file created successfully!\n');
end