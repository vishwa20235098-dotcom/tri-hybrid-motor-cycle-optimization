# Indian urban driving-cycle suite

The RL optimizer is intentionally **India-only**. No NYCC, NEDC, WLTC or other foreign cycle is part of the default training set.

## Cycles

| Key | City / cycle | Status |
|---|---|---|
| `cmdc` | Chennai Motorcycle Driving Cycle (CMDC) | Supplied project data |
| `pune` | Pune urban driving cycle | Supplied project data |
| `delhi` | Delhi Motorcycle Driving Cycle (DMDC) | Reconstructed trace from published statistics; **not raw author data** |

The Delhi paper reports a representative urban DMDC duration of about 847.8 s, average speed 34.36 km/h, running speed 36.61 km/h, total distance 8054.7 m, and a maximum speed not exceeding 50 km/h. The project therefore generates a reproducible stop-go trace matching those headline constraints. The paper itself contains the plotted representative trace but does not provide a machine-readable CSV in the downloaded article. citeturn6view0turn8view0

## Important

Do **not** call the Delhi reconstructed CSV the original DMDC dataset in a report. If the raw/official DMDC speed-time data becomes available, replace `DELHI_DMDC_reconstructed.csv` and keep the same two columns: `Time,Speed`.

The literature also documents additional Indian cycles, including Bangalore, Hyderabad, Lucknow and the national IDC/MIDC cycles. Their availability, vehicle class and raw speed-time data differ. The project should only add those as actual traces when their source data are available rather than inventing a trace.
