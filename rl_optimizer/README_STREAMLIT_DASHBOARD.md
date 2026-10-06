# Streamlit dashboard

Run from this folder:

```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

## What it shows

- All three Indian urban cycles: Chennai/CMDC, Pune, Delhi/DMDC reconstructed trace
- Healthy, fuel-cell failure and ultracapacitor failure scenarios
- Non-optimized deterministic baseline
- PPO-optimized results when `ems_ppo_model.zip` is present
- Speed profile
- Fuel-cell, battery and ultracapacitor power as three separate subplots
- Storage states: battery SOC, UC voltage, backup SOC
- Cycle statistics and optimizer comparison tables/charts
- CSV download for the selected case

## Run order

1. `python traffic_scenario.py --cycle all`
2. `python train_rl.py --timesteps 500000 --cycles cmdc pune delhi`
3. `streamlit run streamlit_dashboard.py`

The dashboard is independent of MATLAB and reads the same cycle/model files used by the Python optimizer.
