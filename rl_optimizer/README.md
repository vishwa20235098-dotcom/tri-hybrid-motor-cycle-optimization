# Python RL optimizer — Indian urban tri-hybrid EMS

## Default cycles

- `cmdc` — Chennai Motorcycle Driving Cycle
- `pune` — Pune urban cycle
- `delhi` — Delhi Motorcycle Driving Cycle, reconstructed from published statistics because the downloaded paper contains a plotted trace rather than a machine-readable CSV.

## Generate traffic-aware versions

```bash
python traffic_scenario.py --cycle all
```

This adds controlled traffic-signal stops and speed-limit zones to each cycle. The same processed speed trace is then used by the RL environment and MATLAB export.

## Train one PPO policy for all Indian cycles and failure cases

```bash
python train_rl.py --timesteps 500000 --cycles cmdc pune delhi
```

The environment randomizes both cycle and source-health scenario:

- healthy
- fuel-cell failure
- ultracapacitor failure

The backup battery is available as an emergency source.

## Export

```bash
python export_results.py --model ems_ppo_model.zip --cycle cmdc --failure none
python export_results.py --model ems_ppo_model.zip --cycle cmdc --failure fuel_cell
python export_results.py --model ems_ppo_model.zip --cycle cmdc --failure ultracapacitor
```

Repeat for `pune` and `delhi` as required.

The generated MATLAB files are named:

`rl_results_<cycle>_<failure>.mat`

## Important

The PPO model must be retrained after changing the environment observation space. The old model from the earlier project version is deliberately not shipped as the new environment includes source-health and backup-SOC observations.
