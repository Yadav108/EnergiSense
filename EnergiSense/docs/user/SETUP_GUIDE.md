# EnergiSense Complete Setup Guide (CORRECTED)

## IMPORTANT UPDATE
**A critical step was missing from previous instructions: loading the CCPP dataset!**

---

## Complete MATLAB Digital Twin Setup

### Step 1: Clone and Setup (2 minutes)
```bash
git clone https://github.com/Yadav108/EnergiSense.git
cd EnergiSense
```

**In MATLAB:**
```matlab
cd('path/to/EnergiSense');  % Replace with your path
addpath(genpath(pwd));
```

### Step 2: CRITICAL - Load CCPP Dataset (30 seconds)
```matlab
% THIS STEP WAS MISSING - ITS ESSENTIAL!
load('Digitaltwin.mat');
fprintf('CCPP Dataset loaded successfully!\n');
```

**This provides:**
- Real environmental conditions (AT_ts, V_ts, RH_ts, AP_ts)
- Actual power output data (PE_ts) for validation
- Proper time vectors for simulation

### Step 3: Verify Your Model
```matlab
checkModel();
```

### Step 4: Test the Dashboard
```matlab
testDashboard();
```

### Step 5: Launch Complete Digital Twin
```matlab
open_system('simulink/Energisense.slx');
sim('simulink/Energisense.slx');
runDashboard();
```

## Key Takeaway

**The load('Digitaltwin.mat') step is ESSENTIAL!**
- Without it: Random/empty data, unrealistic results
- With it: Real CCPP data, meaningful power plant simulation

**Always run this command before starting your Simulink model!**
