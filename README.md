# ⚡ EnergiSense: AI-Powered Smart Prediction System for Power Plant Efficiency

This project uses a trained **Ensemble Machine Learning model** in Simulink to predict **Power Output (PE)** based on real-time sensor inputs: **Ambient Temperature (AT)**, **Exhaust Vacuum (V)**, **Ambient Pressure (AP)**, and **Relative Humidity (RH)**.

It compares predicted PE with actual values from the workspace, allowing visualization and feedback mechanisms for power plant efficiency analysis.

---

## 🧠 Project Highlights

- ✅ MATLAB trained `ensemblePowerModel` used via `loadLearnerForCoder`
- ✅ Simulink integration using `MATLAB Function` block
- ✅ Inputs (`AT`, `V`, `AP`, `RH`) fed using `From Workspace` blocks
- ✅ Real-time prediction and Scope-based visualization
- ✅ Supports feedback/error signal calculation for further optimization

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `EnergiSense_Model.slx` | Main Simulink model implementing Phases 1–4 |
| `ensemblePowerModel.mat` | Trained ensemble model exported using `saveLearnerForCoder` |
| `predictPower.m` | MATLAB function to load model and perform prediction |
| `README.md` | Project documentation |
| Input files (`AT`, `V`, `AP`, `RH`, `PE_actual`) | Time-series input data used for simulation |

---

## 🔧 Requirements

- MATLAB R2022a or later  
- Simulink  
- Statistics and Machine Learning Toolbox (for training the model)  
- Embedded Coder (optional for code generation)

---

## 📊 Inputs and Outputs

### Inputs from Workspace:
- `AT`: Ambient Temperature
- `V`: Exhaust Vacuum
- `AP`: Ambient Pressure
- `RH`: Relative Humidity
- `PE_actual`: Actual power output (used for comparison)

### Output:
- `PE_predicted`: Predicted using trained ML model

---

## 🔄 System Phases

### ✅ Phase 1: Input Configuration
- Inputs imported from MATLAB Workspace using `From Workspace` blocks
- Must be in format: `[time, data]` with `double`, 2D, no NaN/Inf

### ✅ Phase 2: Model Prediction
- Uses `predictPower.m` inside a `MATLAB Function` block
- Predicts PE using `[AT, V, AP, RH]` as 1x4 input vector

### ✅ Phase 3: Output Comparison
- Actual vs Predicted PE signals connected to Scope

### ✅ Phase 4: Visualization & Feedback
- Scope visualizes real-time predicted vs actual PE
- Feedback (error = PE_actual - PE_predicted) can be used to optimize further

---

## 📈 Sample Scope Output

The following scope shows a good match between predicted and actual PE:

> ![Alt text](C:\Users\Aryan\PycharmProjects\pythonProject\EnergiSense\img\Actual Vs Predicted PE.jpg) 
> > 📉 **Scope Output**: The plot shows a strong alignment between predicted and actual power output, indicating high model accuracy and reliability in real-time prediction scenarios.


---

## 🚀 Getting Started

1. Open MATLAB and load the workspace variables (`AT`, `V`, `AP`, `RH`, `PE_actual`)
2. Open `EnergiSense_Model.slx`
3. Click ▶️ **Run**
4. Observe the **Scope** for prediction accuracy

---

## 🔁 Optional Extensions

- Integrate Digital Twin (Simscape or external plant)
- Online learning or model update logic
- Feedback controllers to optimize system efficiency
- Export C/C++ using Embedded Coder for real-time deployment

---

## 🧠 Authors & Credits

- Developed by: **Aryan Yadav**  
- Tools: MATLAB, Simulink, Machine Learning Toolbox
