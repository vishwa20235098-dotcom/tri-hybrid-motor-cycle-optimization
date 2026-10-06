# Indian driving-cycle status

This project is **India-only**. The earlier NYCC cycle has been removed completely.

## Included

1. **Chennai – CMDC**: existing supplied cycle.
2. **Pune**: existing supplied cycle.
3. **Delhi – DMDC**: an additional Indian urban motorcycle cycle.
   - The source reports an urban cycle duration of 847.8 s, average speed 34.36 km/h, running speed 36.61 km/h, and total length 8054.7 m; Delhi riders did not exceed 50 km/h in the study. citeturn6view0turn8view0
   - The article provides a plotted representative DMDC but no machine-readable CSV in the downloaded article. Therefore `rl_optimizer/DELHI_DMDC_reconstructed.csv` is a reproducible reconstruction matching those headline statistics. **It is not the original raw DMDC trace.**

## Other Indian cycles

Indian literature also contains city/region-specific cycles for Bangalore/Bengaluru, Hyderabad and Lucknow, as well as the national IDC/MIDC cycles. A review of Indian drive-cycle work identifies Chennai, Pune and Bengaluru as separate real-world cycles, and later work reports Hyderabad electric-car cycles. citeturn0search0turn2search8turn2search41

The project does **not** fabricate raw traces for those cycles when a machine-readable source trace is unavailable. If the original raw CSVs from your Chennai/Lucknow/Pune papers are available, place them in `rl_optimizer/data/` and register them in `cycles.py`; the PPO code already supports an arbitrary number of cycles.

### Why this matters

A driving cycle is vehicle-class and location dependent. Published work explicitly notes that driving patterns vary by city/region and vehicle class, so one city's cycle should not silently be treated as another city's cycle. citeturn0search1turn0search5
