# EnergiSense - Power Plant Optimization System

## Overview
EnergiSense is a comprehensive Combined Cycle Power Plant (CCPP) optimization and control system.

## Features
- 🔮 **99.1% Accuracy Predictions** - Advanced ML ensemble models
- 📊 **Real-time Dashboard** - Professional monitoring interface
- 🎛️ **Simulink Integration** - Complete plant simulation
- 🌤️ **Weather Intelligence** - Environmental data integration
- 🔧 **Control Systems** - PID controllers and automation
- ✅ **Validation Framework** - Model testing and verification

## System Requirements
- MATLAB R2020b or later
- Required Toolboxes:
  - Control System Toolbox 25.1
  - Curve Fitting Toolbox 25.1
  - Deep Learning Toolbox 25.1
  - GPU Coder 25.1
  - Global Optimization Toolbox 25.1
  - Instrument Control Toolbox 25.1
  - MATLAB 25.1
  - MATLAB Coder 25.1
  - MATLAB Compiler 25.1
  - MATLAB Compiler SDK 25.1
  - MATLAB Report Generator 25.1
  - Optimization Toolbox 25.1
  - Parallel Computing Toolbox 25.1
  - Simscape 25.1
  - Simscape Electrical 25.1
  - Simulink 25.1
  - Simulink Control Design 25.1
  - Simulink Report Generator 25.1
  - Statistics and Machine Learning Toolbox 25.1
  - Symbolic Math Toolbox 25.1

## Quick Start
```matlab
% Add EnergiSense to path
addpath(genpath('EnergiSense'));

% Launch interactive dashboard
launchInteractiveDashboard();

% Run prediction
data.AT = 20; data.V = 50; data.AP = 1013; data.RH = 60;
power = predictPowerEnhanced(data);
```

## File Structure
```
EnergiSense/
├── auditEnergiSenseSystem.m
├── fixFromWorkspacePid.m
├── launchInteractiveDashboard.m
├── predictPowerEnhanced.m
├── predictPowerFixed.m
├── setupEnergiSense.m
├── startup.m
├── updateDashboardModels.m
├── predictivePIDController.m
├── configureEnergiSense.m
├── ... (12 more files)
```

## Documentation
- [User Guide](docs/user_guide.md)
- [Technical Documentation](docs/technical_docs.md)
- [API Reference](docs/api_reference.md)

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Contact
For questions and support, please open an issue or contact the development team.
