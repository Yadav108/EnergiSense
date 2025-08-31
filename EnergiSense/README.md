# EnergiSense - Educational Combined Cycle Power Plant Simulation

[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Simulink](https://img.shields.io/badge/Simulink-Supported-blue.svg)](https://www.mathworks.com/products/simulink.html)
[![Educational](https://img.shields.io/badge/Purpose-Educational-blue.svg)](#purpose)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎓 Overview

EnergiSense is an **educational simulation system** for Combined Cycle Power Plant (CCPP) modeling that demonstrates machine learning concepts and control systems. The system includes a Random Forest model trained on the UCI CCPP dataset for educational purposes and learning about industrial process modeling.

## ⚠️ Important Disclaimer

**This is an educational project designed for learning purposes only.** 

- Not intended for production or real industrial use
- Simulated components and simplified models for demonstration
- Performance claims are educational benchmarks, not industrial certifications
- Use for learning ML, control systems, and MATLAB/Simulink concepts

### ✨ Educational Features

- **🤖 Machine Learning Demo**: Random Forest model trained on UCI CCPP dataset (educational implementation)
- **🎛️ Control Systems Learning**: PID controller examples and basic MPC concepts
- **⚙️ Simulink Integration**: Custom blocks demonstrating power plant modeling concepts
- **📊 Data Visualization**: Interactive dashboards for learning data analysis
- **🌡️ Environmental Modeling**: Simulated environmental conditions for demonstration
- **📚 Educational Content**: Comprehensive documentation for learning industrial concepts
- **🔧 Parameter Tuning**: Examples of controller optimization techniques

## 🚀 Quick Start

### Prerequisites

- MATLAB R2021a or later
- Simulink (recommended)
- Statistics and Machine Learning Toolbox
- Control System Toolbox

### Installation

1. **Clone or download** the EnergiSense repository
2. **Navigate** to the EnergiSense directory in MATLAB
3. **Run the setup**:
   ```matlab
   setupEnergiSense()
   ```

### Basic Usage

```matlab
% 1. Configure the enhanced system
configureEnergiSense();

% 2. Run a complete enhanced simulation
runEnhancedSimulation();

% 3. Test ML model predictions
[power, confidence] = predictPowerEnhanced([15, 50, 1013, 65]);
fprintf('Predicted Power: %.1f MW (Confidence: %.1f%%)\n', power, confidence*100);

% 4. Launch interactive dashboard
launchInteractiveDashboard();
```

## 📊 System Architecture

```
EnergiSense/
├── 🧠 core/                    # Core ML and prediction systems
│   ├── training/               # ML model training (95.9% accuracy)
│   ├── prediction/             # Production prediction engines
│   ├── models/                 # Trained models and data
│   └── validation/             # Comprehensive testing framework
│
├── 🎛️ control/                 # Advanced control systems
│   ├── controllers/            # Enhanced PID, MPC controllers
│   ├── advanced/               # Advanced control strategies
│   └── tuning/                 # Auto-optimization and configuration
│
├── ⚙️ simulation/              # Simulink integration
│   ├── blocks/                 # 4 specialized Simulink blocks
│   ├── analysis/               # Results analysis and validation
│   └── initializeEnhancedSimulink.m
│
├── 📡 analytics/               # Industrial analytics
│   └── maintenance/            # Predictive maintenance engine
│
├── 🖥️ dashboard/              # User interfaces
│   ├── interactive/            # Interactive GUI dashboard
│   └── main/                   # Analytics dashboard
│
└── 📚 docs/                   # Comprehensive documentation
```

## 🤖 Machine Learning Model (Educational)

### Model Specifications
- **Algorithm**: Random Forest Regression (100 trees)
- **Training Data**: UCI Combined Cycle Power Plant Dataset (9,568 samples)
- **Purpose**: Educational demonstration of ML regression techniques
- **Performance Metrics** (on training data):
  - R² Score: ~0.96 (typical for this dataset)
  - MAE: ~2.4 MW (Mean Absolute Error)
  - RMSE: ~3.5 MW (Root Mean Square Error)

**Note**: Performance metrics are based on standard ML practices with the UCI dataset. This is an educational implementation for learning purposes, not a production system.

### Input Features
1. **AT** - Ambient Temperature (°C): -6.23 to 37.11
2. **V** - Exhaust Vacuum (cm Hg): 25.36 to 81.56  
3. **AP** - Atmospheric Pressure (mbar): 992.89 to 1033.30
4. **RH** - Relative Humidity (%): 25.56 to 100.16

### Model Training
```matlab
% Train new model (if needed)
[model, validation_results] = trainCCPPModel();

% Validate performance
validateEnhancedSystem();
```

## 🎛️ Control Systems (Educational)

### PID Controller Example
- **Demo Parameters**: Kp=5.0, Ki=0.088, Kd=0.171
- **Purpose**: Demonstrate basic feedback control concepts
- **Features**: Basic PID implementation with simple tuning examples
- **Learning Goals**: Understanding proportional, integral, and derivative control

### Model Predictive Control (Conceptual)
- **Purpose**: Educational introduction to MPC concepts
- **Implementation**: Basic MPC framework for learning
- **Features**: Simple prediction horizon and constraint handling examples
- **Learning Goals**: Understanding predictive control strategies

```matlab
% Configure enhanced controller
configureEnergiSense();

% Optimize controller parameters
optimizeControllerPerformance();
```

## ⚙️ Simulink Integration

### Specialized Simulink Blocks

1. **`mlPowerPredictionBlock`**: Real-time 95.9% ML predictions
2. **`environmentalConditionsBlock`**: Realistic environmental modeling
3. **`industrialIoTBlock`**: IoT monitoring and maintenance alerts
4. **`advancedMPCBlock`**: Model Predictive Control with constraints

### Usage in Simulink
```matlab
% Initialize enhanced Simulink environment
initializeEnhancedSimulink();

% Open Simulink model
open('Energisense.slx');

% Run simulation
simout = sim('Energisense');

% Analyze results
analyzeEnergiSenseResults(simout);
```

## 📡 Simulation Features

### Monitoring System Demo
- **Data Quality Simulation**: Demonstrates data validation concepts
- **System Health Examples**: Shows basic component monitoring principles
- **Alert System Demo**: Examples of warning and alert mechanisms
- **Maintenance Concepts**: Educational predictive maintenance examples

### Environmental Modeling
- **Daily Cycles**: Simulated temperature and humidity patterns for learning
- **Weather Simulation**: Basic environmental condition modeling
- **Parameter Variation**: Demonstrates how environmental factors affect power output
- **Educational Examples**: Helps understand industrial process dependencies

## 📊 Educational Performance & Validation

### Learning Objectives Achieved
- **ML Concepts**: Demonstrates Random Forest regression on real dataset
- **Control Theory**: Shows basic PID and MPC principles
- **System Integration**: Examples of MATLAB/Simulink coordination
- **Data Analysis**: Interactive visualization and result interpretation

### Comprehensive Testing
```matlab
% Run complete system validation
validateEnhancedSystem();

% Performance optimization
optimizeControllerPerformance();

% Complete simulation test
runEnhancedSimulation();
```

## 🖥️ User Interfaces

### Interactive Dashboard
```matlab
% Launch main interactive GUI
launchInteractiveDashboard();
```
- Real-time system monitoring
- Interactive parameter adjustment
- Live performance visualization
- System status and health indicators

### Analytics Dashboard
```matlab
% Launch comprehensive analytics
runDashboard();
```
- Historical data analysis
- Performance trend monitoring
- Predictive maintenance scheduling
- System optimization recommendations

## 📚 Documentation Structure

- **[Installation Guide](docs/INSTALLATION.md)**: Complete setup instructions
- **[API Reference](docs/API_REFERENCE.md)**: Function and class documentation
- **[User Guide](docs/USER_GUIDE.md)**: Step-by-step usage instructions
- **[ML Model Documentation](docs/ML_MODEL.md)**: Detailed model specifications
- **[Control Systems Guide](docs/CONTROL_SYSTEMS.md)**: Controller configuration
- **[Simulink Integration](docs/SIMULINK_INTEGRATION.md)**: Simulink blocks and usage
- **[Industrial Features](docs/INDUSTRIAL_FEATURES.md)**: IoT and maintenance systems

## 🔧 Configuration

### System Configuration
```matlab
% Basic configuration
configureEnergiSense();

% Advanced customization
pid_params.Kp = 5.0;           % Proportional gain
pid_params.Ki = 0.088;         % Integral gain
pid_params.Kd = 0.171;         % Derivative gain
pid_params.prediction_weight = 0.621;  % ML model weight
```

### Environment Variables
- `ENERGISENSE_DATA_PATH`: Custom data directory
- `ENERGISENSE_MODEL_PATH`: Custom model directory
- `ENERGISENSE_CACHE_PATH`: Custom cache directory

## 📈 Educational Benchmarks

| Learning Area | Implementation | Status |
|---------------|----------------|--------|
| ML Regression | Random Forest on UCI data | ✅ Working |
| Control Theory | Basic PID demonstration | ✅ Working |
| Data Analysis | Interactive visualization | ✅ Working |
| System Integration | MATLAB/Simulink examples | ✅ Working |
| Documentation | Comprehensive guides | ✅ Available |

**Note**: This is an educational system designed for learning, not production use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository**: CCPP dataset for educational use
- **MATLAB/Simulink**: Development platform for educational demonstrations
- **Academic Community**: Control systems and machine learning education resources

## 📞 Educational Support

This is an educational project for learning purposes:
- 📖 Documentation: See `docs/` directory for learning materials
- 🎓 Purpose: Understanding industrial ML and control concepts
- 📚 Learning Path: Follow the user guides for step-by-step learning

---

**EnergiSense** - An educational simulation for learning industrial ML and control systems! 🎓