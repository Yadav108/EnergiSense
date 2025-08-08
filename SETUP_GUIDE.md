# 🚀 EnergiSense Complete Setup Guide (CORRECTED)

## ⚠️ IMPORTANT UPDATE
**A critical step was missing from previous instructions: loading the CCPP dataset!**

---

## 🎯 Method 1: Complete MATLAB Digital Twin (CORRECTED)

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
```

### **Step 2: 🔑 CRITICAL - Load CCPP Dataset** (30 seconds)
```matlab
% **THIS STEP WAS MISSING - IT'S ESSENTIAL!**
load('Digitaltwin.mat');

% Verify what was loaded
fprintf('📁 CCPP Dataset loaded:\n');
whos
fprintf('✅ Simulink model now has access to real power plant data!\n');
```

**This provides:**
- Real environmental conditions (AT_ts, V_ts, RH_ts, AP_ts)
- Actual power output data (PE_ts) for validation
- Proper time vectors for simulation

### **Step 3: Verify Your Model** (1 minute)
```matlab
% Check if your trained model is working
checkModel();
```

### **Step 4: Test the Dashboard** (1 minute)
```matlab
% Test dashboard functionality
testDashboard();
```

### **Step 5: Launch the Complete Digital Twin** (5 minutes)

**5A. Open Simulink Model:**
```matlab
% Open the main digital twin model
open_system('simulink/Energisense.slx');
```

**5B. Run Simulation with Real Data:**
```matlab
% Start the simulation (now using real CCPP data!)
sim('simulink/Energisense.slx');
```

**Expected Simulink Behavior:**
- Environmental inputs: Real CCPP time series data
- Power predictions: Based on actual plant conditions
- Validation data: Real PE_ts for comparison
- Control signals: Responding to realistic variations

**5C. Launch Real-time Dashboard:**
```matlab
% Start real-time monitoring (while simulation is running)
runDashboard();
```

---

## 🚀 Updated Complete Workflow Demo

### **One-Command Complete Setup:**
```matlab
%% Complete EnergiSense Setup with Real Data
fprintf('🏭 EnergiSense Complete Setup\n');
fprintf('=============================\n\n');

% Navigate to project
cd('path/to/EnergiSense');  % Update this path!
addpath(genpath(pwd));

% CRITICAL: Load CCPP dataset
fprintf('Loading CCPP dataset...\n');
load('Digitaltwin.mat');
fprintf('✅ Real power plant data loaded!\n\n');

% Verify system
checkModel();
testDashboard();

% Launch system
fprintf('\nLaunching complete digital twin...\n');
open_system('simulink/Energisense.slx');
fprintf('✅ Click RUN in Simulink to start with real data\n');
fprintf('📊 Then run: runDashboard()\n');
```

---

## 🔑 Key Takeaway

**The `load('Digitaltwin.mat')` step is ESSENTIAL!**
- Without it: Random/empty data, unrealistic results
- With it: Real CCPP data, meaningful power plant simulation

**Always run this command before starting your Simulink model!**
