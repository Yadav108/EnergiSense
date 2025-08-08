# 🚀 How to Run EnergiSense - Complete Guide

## 📋 Quick Start Checklist

- [ ] MATLAB R2021a+ with Simulink installed
- [ ] Python 3.8+ (for research implementation)
- [ ] Git installed on your system
- [ ] 15 minutes for complete setup

---

## 🎯 Method 1: Complete MATLAB Digital Twin (Recommended)

### **Step 1: Clone and Setup** (2 minutes)
```bash
# Clone the repository
git clone https://github.com/Yadav108/EnergiSense.git
cd EnergiSense
```

**In MATLAB:**
```matlab
% Navigate to project folder
cd('path/to/EnergiSense');  % Replace with your actual path

% Add all folders to MATLAB path
addpath(genpath(pwd));

% Verify setup
fprintf('✅ EnergiSense loaded successfully!\n');
fprintf('📁 Current directory: %s\n', pwd);
```

### **Step 2: Verify Your Model** (1 minute)
```matlab
% Check if your trained model is working
checkModel();
```

**Expected Output:**
```
=== CCPP DIGITAL TWIN MODEL VERIFICATION ===
✅ Found: ensemblePowerModel.mat
✅ Successfully loaded your trained model!
   Model type: classreg.learning.regr.CompactRegressionEnsemble
✅ Model prediction test: 442.22 MW
✅ GOOD NEWS: Your actual trained model is working!
```

### **Step 3: Test the Dashboard** (1 minute)
```matlab
% Test dashboard functionality
testDashboard();
```

**Expected Result:**
- A window with 4 test plots should appear
- Power tracking, control signals, performance metrics
- Sample data showing realistic CCPP behavior

### **Step 4: Launch the Complete Digital Twin** (5 minutes)

**4A. Open Simulink Model:**
```matlab
% Open the main digital twin model
open_system('simulink/Energisense.slx');
```

**4B. Run Simulation:**
```matlab
% Start the simulation
sim('simulink/Energisense.slx');
```

**Expected Simulink Behavior:**
- Model should compile without errors
- Simulation runs for 100 seconds (default)
- Scope shows power predictions vs actual data
- Control signal scope shows PID controller working

**4C. Launch Real-time Dashboard:**
```matlab
% Start real-time monitoring (while simulation is running)
runDashboard();
```

### **Step 5: Expected Complete Results** 📊

**You should see:**

1. **Simulink Model Running:**
   - Blue line: Predicted power (~400-500 MW)
   - Yellow line: Actual power from CCPP dataset
   - Control signal near 0 (PID working correctly)

2. **Real-time Dashboard (6 panels):**
   - **Power Tracking**: Live prediction vs actual vs setpoint
   - **Control Signal**: PID controller output
   - **Performance Metrics**: Efficiency (~85%), Accuracy (~95%), Stability (~100%)
   - **Environmental Conditions**: Temperature, wind, humidity, pressure
   - **System Status**: Current power, error, control values
   - **Control Panel**: PID parameters and setpoint info

3. **Console Output:**
   ```
   Starting Digital Twin Dashboard...
   Dashboard started! Monitoring data from Simulink...
   Power: 442.2 MW
   Confidence: 92.1%
   Anomaly: Normal ✅
   ```

---

## 🐍 Method 2: Python Research Implementation

### **Step 1: Python Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Or manually install
pip install numpy pandas scikit-learn matplotlib seaborn
```

### **Step 2: Run Python Analysis**
```bash
# Run the main research script
python main.py
```

**Expected Python Output:**
- Data loading and preprocessing
- Model training results
- Performance metrics
- Visualization plots saved in img/ folder

---

## 🎯 Complete Workflow Demo (10 minutes)

### **Full System Demonstration:**

```matlab
%% Complete EnergiSense Demo Script
% Run this for a full demonstration

fprintf('🏭 EnergiSense Digital Twin Demonstration\n');
fprintf('==========================================\n\n');

% Step 1: Verify everything is working
fprintf('Step 1: Model Verification\n');
checkModel();

fprintf('\nStep 2: Testing Dashboard\n');
testDashboard();

fprintf('\nStep 3: Sample Predictions\n');
% Test with different CCPP conditions
test_conditions = [
    25.36, 40.27, 68.77, 1013.84;  % High efficiency conditions
    10.82, 39.4, 88.62, 1009.23;   % Low efficiency conditions  
    14.96, 41.76, 80.26, 1010.24;  % Average conditions
];

condition_names = {'High Efficiency', 'Low Efficiency', 'Average'};

for i = 1:size(test_conditions, 1)
    [power, confidence, anomaly] = predictPowerEnhanced(test_conditions(i,:));
    fprintf('%s: %.1f MW (Confidence: %.1f%%, Anomaly: %s)\n', ...
        condition_names{i}, power, confidence*100, char("No" + anomaly*"Yes"));
end

fprintf('\nStep 4: Opening Simulink Model\n');
open_system('simulink/Energisense.slx');

fprintf('\nStep 5: Run simulation and launch dashboard manually:\n');
fprintf('1. Click Run in Simulink\n');
fprintf('2. Execute: runDashboard()\n');
fprintf('3. Observe real-time monitoring\n\n');

fprintf('🎉 Demo complete! Your digital twin is ready!\n');
```

---

## 📊 Performance Benchmarks

### **What You Should See:**

| Metric | Expected Value | Description |
|--------|---------------|-------------|
| Prediction Accuracy | 95%+ | ML model performance on test data |
| Control Error | <5% | PID setpoint tracking accuracy |
| Response Time | <30s | Time to reach 95% of setpoint |
| Dashboard Update | 0.5s | Real-time monitoring refresh rate |
| Anomaly Detection | 3σ threshold | Statistical outlier identification |

### **Sample Test Results:**
```
Input: [25.36°C, 40.27, 68.77%, 1013.84 hPa]
Output: 442.22 MW
Confidence: 92.1%
Anomaly Status: Normal
Control Signal: 0.09
```

---

## 🛠️ Troubleshooting Guide

### **Common Issues & Solutions:**

#### **Issue 1: Model Loading Error**
```
Error: Could not load ensemblePowerModel.mat
```
**Solution:**
```matlab
% Check if file exists
if exist('models/ensemblePowerModel.mat', 'file')
    fprintf('✅ Model file found\n');
else
    fprintf('❌ Copy ensemblePowerModel.mat to models/ folder\n');
end
```

#### **Issue 2: Simulink Compilation Error**
```
Error: Try and catch are not supported for code generation
```
**Solution:**
- Your model is using the enhanced prediction function
- This is normal and expected
- The system falls back to linear model for Simulink compatibility

#### **Issue 3: Dashboard Not Updating**
```
Waiting for Simulink data...
```
**Solution:**
```matlab
% Make sure simulation is running first
sim('simulink/Energisense.slx');

% Then launch dashboard
runDashboard();
```

#### **Issue 4: Git/Path Issues**
```matlab
% Check current directory
pwd

% Make sure you're in the right folder
cd('path/to/EnergiSense');

% Verify files exist
dir('models/*.mat')
dir('dashboard/*.m')
```

---

## 🎥 Expected Visual Results

### **Simulink Model View:**
- Environmental inputs connected to Digital Twin Core
- PID Controller receiving predicted power feedback
- Scopes showing real-time power and control signals
- Clean, professional block diagram layout

### **Dashboard Interface:**
- Dark theme with 6 professional panels
- Live updating charts and gauges
- Color-coded status indicators (green = normal, red = alert)
- Real-time numerical displays

### **Performance Plots:**
- Power tracking: Smooth lines around 400-500 MW range
- Control signals: Small oscillations around zero
- Metrics bars: High efficiency (85%+), accuracy (95%+), stability (100%)

---

## 🚀 Next Steps After Running

### **Once Everything Works:**
1. **Take Screenshots** of your dashboard
2. **Record a Demo Video** (2-3 minutes)
3. **Test Different Setpoints** (try 350 MW, 450 MW, 500 MW)
4. **Experiment with PID Tuning** (modify Kp, Ki, Kd values)
5. **Add Your Own Data** (replace CCPP dataset with your data)

### **Customization Options:**
```matlab
% Change PID parameters in Simulink model
Kp = 2.0;  % More aggressive control
Ki = 0.15; % Faster integral action
Kd = 0.08; % More derivative damping

% Change setpoint
new_setpoint = 450; % MW

% Modify alarm thresholds
high_power_limit = 700; % MW
low_power_limit = 50;   % MW
```

---

## 📞 Support

### **If You Need Help:**
1. **Check Issues**: [GitHub Issues](https://github.com/Yadav108/EnergiSense/issues)
2. **Verify Prerequisites**: MATLAB version, toolboxes installed
3. **Check File Paths**: Ensure all files are in correct folders
4. **Run Diagnostics**: Use `checkModel()` and `testDashboard()`

### **Expected Timeline:**
- **Setup**: 5 minutes
- **First Run**: 10 minutes  
- **Understanding Results**: 15 minutes
- **Customization**: 30+ minutes

**🎯 Goal: Within 15 minutes, you should have a fully operational digital twin with real-time monitoring!**
