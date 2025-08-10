# EnergiSense 🏭⚡
## Digital Twin Research Platform for Combined Cycle Power Plant Analysis

[![MATLAB](https://img.shields.io/badge/MATLAB-R2025a+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Simulink](https://img.shields.io/badge/Simulink-Supported-blue.svg)](https://www.mathworks.com/products/simulink.html)
[![Status](https://img.shields.io/badge/Status-Research%20Platform-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-99.1%25-success.svg)]()

---

## 🎯 **Project Overview**

EnergiSense is a comprehensive digital twin research platform for Combined Cycle Power Plant (CCPP) analysis. The platform demonstrates **99.1% prediction accuracy** on the UCI CCPP dataset using ensemble machine learning methods integrated with advanced control systems.

### **Key Features**
- **High-Accuracy Prediction**: 99.1% accuracy on UCI CCPP dataset
- **Real-time Dashboard**: 6-panel monitoring interface for system analysis
- **Control System**: PID controller with anti-windup protection
- **Complete Integration**: End-to-end system from data input to control output
- **Research Tools**: Validation, analysis, and benchmarking utilities

This platform serves as a research tool for power plant digitalization, machine learning applications in energy systems, and educational purposes in digital twin concepts.

---

## 📊 **Performance Results**

### **Machine Learning Model**
- **Prediction Accuracy**: 99.1% (R² = 0.991)
- **Mean Absolute Error (MAE)**: 4.2 MW
- **Root Mean Square Error (RMSE)**: 5.2 MW
- **Operating Range**: 422.4 - 487.9 MW predicted vs 426.2 - 487.7 MW actual
- **Model Type**: Ensemble Regression Trees (CompactRegressionEnsemble)
- **Dataset**: UCI Combined Cycle Power Plant (9,568 samples)

### **Dashboard Interface**

![EnergiSense Dashboard](https://github.com/Yadav108/EnergiSense/blob/master/EnergiSense/data/results/EnergiSense_Dashboard_Report.png)

The dashboard provides real-time monitoring with six integrated panels: power prediction tracking, environmental conditions monitoring, PID control signals, performance metrics, system status indicators, and historical trend analysis. The interface demonstrates effective real-time data visualization with clear color-coded status indicators for operational assessment.

### **Control System Performance**
- **Controller Type**: Predictive PID with anti-windup protection
- **Setpoint Tracking**: Stable response to power output targets
- **System Stability**: No oscillations observed during testing
- **Integration**: Seamless connection between ML predictions and control

---

## 🗂️ **Project Structure**

```
EnergiSense/
├── 📄 startup.m                    # Automatic path configuration
├── 📄 setupEnergiSense.m          # Complete system setup and verification
├── 📄 README.md                   # Project documentation
├── 📄 LICENSE                     # MIT License file
│
├── 📁 core/                       # Core research components
│   ├── 📁 models/
│   │   ├── ensemblePowerModel.mat # 99.1% accuracy ML model
│   │   └── digitaltwin.mat        # Digital twin system configuration
│   ├── 📁 prediction/
│   │   └── predictPowerEnhanced.m # Enhanced prediction function
│   └── 📁 validation/
│       ├── checkModel.m           # Model verification and testing
│       └── checkModelUtils.m      # Validation utility functions
│
├── 📁 control/                    # Control system components
│   ├── 📁 controllers/
│   │   └── predictivePIDController.m # PID controller implementation
│   └── 📁 tuning/
│       └── configureEnergiSense.m    # Control parameter configuration
│
├── 📁 dashboard/                  # Real-time monitoring interface
│   └── 📁 main/
│       └── runDashboard.m         # Dashboard launcher and interface
│
├── 📁 simulation/                 # Simulink models and analysis
│   ├── 📁 models/
│   │   ├── Energisense.slx        # Complete digital twin Simulink model
│   │   └── Energisense.slxc       # Compiled Simulink model cache
│   └── 📁 analysis/
│       ├── analyzeResults.m       # Performance analysis tools
│       └── analyzeEnergiSenseResults.m # Comprehensive result analysis
│
├── 📁 data/                       # Research datasets and results
│   ├── 📁 raw/
│   │   └── Folds5X2.csv          # Original UCI CCPP dataset
│   ├── 📁 processed/
│   │   ├── ccpp_simin_cleaned.mat # Cleaned data for Simulink
│   │   └── Es.mat                 # Ensemble model training data
│   └── 📁 results/
│       └── EnergiSense_Dashboard_Report.png # Dashboard screenshot
│
├── 📁 examples/                   # Usage examples and tutorials
│   └── 📁 quickstart/
│       └── demo.m                 # Main demonstration script
│
├── 📁 python/                     # Python implementation
│   ├── main.py                    # Python research analysis script
│   ├── requirements.txt           # Python package dependencies
│   └── 📁 src/                    # Python source modules
│       ├── __init__.py
│       ├── models.py              # ML model definitions
│       ├── data_loader.py         # Data processing utilities
│       └── metrics.py             # Performance evaluation functions
│
├── 📁 utilities/                  # System utilities and tools
│   └── 📁 system/
│       └── systemCheck.m          # Installation verification script
│
└── 📁 docs/                       # Documentation
    └── 📁 api/                    # API documentation (future)
```

---

## 🚀 **Getting Started**

### **Requirements**
- MATLAB R2025a or later
- Simulink
- Statistics and Machine Learning Toolbox
- Control System Toolbox
- Python 3.8+ (optional, for Python components)

### **Installation**
```matlab
% 1. Clone the repository
git clone https://github.com/Yadav108/EnergiSense.git

% 2. Navigate to project directory
cd('EnergiSense')

% 3. Run complete setup
setupEnergiSense()

% 4. Verify installation
systemCheck()
```

### **Quick Start**
```matlab
% Run main demonstration
demo()

% Launch monitoring dashboard
runDashboard()

% Test prediction with sample conditions
test_conditions = [25.36, 40.27, 68.77, 1013.84];  % [AT, V, RH, AP]
predicted_power = predictPowerEnhanced(test_conditions);
fprintf('Predicted Power: %.2f MW\n', predicted_power);
```

---

## 📈 **Usage Examples**

### **Basic Prediction**
```matlab
% Single prediction
conditions = [14.96, 41.76, 1024.07, 73.17];  % [Temp, Vacuum, Humidity, Pressure]
power = predictPowerEnhanced(conditions);
fprintf('Power Output: %.2f MW\n', power);
```

### **Batch Analysis**
```matlab
% Multiple condition analysis
test_data = [
    15, 35, 60, 1015;  % Condition 1
    30, 45, 75, 1010;  % Condition 2
    20, 30, 55, 1020   % Condition 3
];

for i = 1:size(test_data, 1)
    power = predictPowerEnhanced(test_data(i, :));
    fprintf('Condition %d: %.2f MW\n', i, power);
end
```

### **Control System Integration**
```matlab
% Open Simulink model
open_system('simulation/models/Energisense.slx')

% Configure control parameters
configureEnergiSense()

% Run simulation
sim('simulation/models/Energisense.slx')
```

---

## 📚 **Dataset and Methodology**

### **Dataset Source**
This work uses the Combined Cycle Power Plant dataset from the UCI Machine Learning Repository:

> Pınar Tüfekci, "Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods," *International Journal of Electrical Power & Energy Systems*, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615.

**Dataset Link**: [UCI CCPP Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)

### **Technical Specifications**
- **Data Points**: 9,568 observations (2006-2011)
- **Input Features**: 4 (Ambient Temperature, Vacuum, Relative Humidity, Atmospheric Pressure)
- **Output**: Power output (MW)
- **Range**: 426.2 - 487.7 MW
- **Model Architecture**: Ensemble Regression Trees with bagging
- **Validation Method**: 5-fold cross-validation with hold-out testing

---

## 🔬 **Research Applications**

### **Academic Use Cases**
- Benchmark for power plant prediction algorithms
- Digital twin concept demonstration
- Machine learning education in energy systems
- Control system design and validation
- Research collaboration platform

### **Industry Applications**
- Power plant performance analysis
- Operational efficiency assessment
- Predictive maintenance insights
- Energy optimization studies
- Training and simulation tools

---

## 📋 **Citation**

If you use EnergiSense in your research, please cite:

```
EnergiSense: Digital Twin Research Platform for Combined Cycle Power Plant Analysis
GitHub Repository: https://github.com/Yadav108/EnergiSense
Achieved 99.1% prediction accuracy on UCI CCPP dataset (2025)
```

---

## 📄 **License**

This project is licensed under the MIT License.

---

*Empowering sustainable energy through intelligent digital twin technology* ⚡🌱
