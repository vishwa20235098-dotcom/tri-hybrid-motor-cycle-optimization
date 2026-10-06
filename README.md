# Failure-aware tri-hybrid Indian urban driving-cycle project

## Sources
- Fuel cell
- Traction battery
- Ultracapacitor
- Emergency backup battery

## Indian driving cycles
- Chennai / CMDC
- Pune
- Delhi / DMDC (reconstructed from published statistics; not raw author data)

No foreign driving cycle is included in the default optimizer.

## MATLAB
Edit the top of `main_rl.m`:

```matlab
CYCLE = 'cmdc';              % cmdc | pune | delhi
FAILURE_MODE = 'none';       % none | fuel_cell | ultracapacitor
```

The three primary source power plots are placed as three subplots on one page. Backup power, power balance and storage states are shown separately.

## Python RL
From `rl_optimizer/`:

```bash
pip install -r requirements.txt
python traffic_scenario.py --cycle all
python train_rl.py --timesteps 500000 --cycles cmdc pune delhi
```

The PPO environment randomly trains across all registered Indian cycles and across healthy/failure cases. The observation includes source-health flags and backup SOC.

Export a selected case:

```bash
python export_results.py --model ems_ppo_model.zip --cycle cmdc --failure none
python export_results.py --model ems_ppo_model.zip --cycle cmdc --failure fuel_cell
python export_results.py --model ems_ppo_model.zip --cycle cmdc --failure ultracapacitor
python export_results.py --model ems_ppo_model.zip --cycle pune --failure none
python export_results.py --model ems_ppo_model.zip --cycle delhi --failure none
```

Then set the matching `CYCLE` and `FAILURE_MODE` in `main_rl.m` and run it in MATLAB.

## Important data note

The Delhi paper gives the representative DMDC as a plotted speed-time trace and reports its summary statistics, but the downloaded paper does not provide the raw machine-readable trace. The project therefore labels its Delhi CSV as reconstructed. Replace it with the actual trace if you have it.

## Streamlit dashboard

The dashboard is a separate Python application that collects the Indian-cycle results and presents:
- Chennai, Pune and Delhi cycle statistics
- speed-time plots
- three primary-source power plots on one page
- battery / ultracapacitor / backup states
- healthy, fuel-cell-failure and ultracapacitor-failure cases
- with-PPO vs without-optimizer comparisons
- downloadable selected-case CSV results

From `rl_optimizer/` after installing the requirements:

```bash
streamlit run streamlit_dashboard.py
```

The dashboard expects `ems_ppo_model.zip` in the same `rl_optimizer/` directory for the optimized results. Without the model it still shows the non-optimized baseline and cycle statistics.
