# Cavitation source synthesis and session-level injection for centrifugal-pump vibration monitoring

Code for synthesizing cavitation and impact signatures, injecting them into recorded pump
vibration sessions, and characterizing how a deployed monitoring pipeline responds. The
injected sessions keep the storage format of the originals, so the same feature-extraction and
detection code reads synthetic and recorded data alike.

## What is in here

```
equipment_analysis/      the vibration-monitoring package that reads the sessions
  npz_io.py, dataio.py     session I/O (npz and CSV), safe path resolution
  features.py              time- and frequency-domain features, rotation-speed detection
  anomaly.py, model.py     robust z-score against a baseline; IsolationForest model
  cache.py, registry.py    per-session feature cache; folder -> equipment configuration
  plotting.py, diagnosis.py, equipment.py, train_model.py
  tests/

impeller_pinn/           synthesis, injection and characterization around that package
  synth.py, modal.py       impact ring-down and modal superposition (damped multi-mode sum)
  beam_modes.py            Euler-Bernoulli cantilever frequencies (vane approximation)
  classical_ritz.py        polynomial Rayleigh-Ritz cantilever eigensolver
  inject.py                writes injected copies of recorded sessions; originals untouched
  detection_sweep.py       detection grid: severity x in-band SNR x repetition rate
  run_detection_grid.py    runs the grid: detection table, ROC, minimum detectable severity
  detection_diagnostics.py leave-one-out false alarms and band-placement sweep
  cavitation_baseline.py   normal baseline of the impeller-side accelerometer and the microphone
  tier1_natural_variation.py  normal-to-normal variation thresholds per channel and band
  thermal_analysis.py      water and pipe temperature logs: drift, cavitation-number sensitivity
  pilot_cavitation.py      cavitation number and channel response of suction-throttle runs
  epoch_inventory.py       inventory of a continuous-monitoring folder: sessions, logs, level steps
  load_hose_design.py      discharge load-hose sizing: hose friction, trim valve, sigma_ISA,
                           differential-pressure flow measurement
  validate.py              end-to-end demonstration on recorded sessions
  tests/

docs/_generated/
  fig_gen_cavitation.py    reduced cavitation source model (Blake nucleation, Wood two-phase
                           attenuation, shot-noise band power) and the figure generator
  data/paper1/load_hose_design.json   output of load_hose_design.py
  figures/paper1/          the source-model figure and three schematics
```

The field-analysis scripts read recorded sessions (seven channels at 51.2 kHz, with pressure and
temperature logs) from a local archive set at the top of each script (`DATA_ROOT`). Those
sessions and the results computed from them are not distributed here. The tests build synthetic
sessions and need no field data.

## Running it

```bash
pip install numpy scipy pandas scikit-learn joblib pyyaml openpyxl matplotlib pytest
pip install torch      # optional: one test builds mode shapes with torch and skips without it

PYTHONPATH=. python -m pytest impeller_pinn/tests equipment_analysis/tests -q

PYTHONPATH=. python -m impeller_pinn.load_hose_design   # rewrites docs/_generated/data/paper1/load_hose_design.json
python docs/_generated/fig_gen_cavitation.py             # source-model figure and schematics
```

`fig_gen_cavitation.py` draws every figure whose input is present and names the inputs it did
not find.

## Environment

| | |
|---|---|
| Python | 3.12.3 |
| NumPy / SciPy | 2.5.0 / 1.18.0 |
| pandas / scikit-learn | 3.0.3 / 1.9.0 |
| Matplotlib | 3.11.0 |
| PyTorch (one test only) | 2.12.0a0 (NGC 26.04) |

## License

See `LICENSE`.
