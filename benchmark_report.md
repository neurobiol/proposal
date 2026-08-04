# SINAPs quantum predictive-memory feasibility benchmark

## Data status

- Source used: **synthetic published-design benchmark**.
- Any file whose name contains `synthetic` is simulated and must not be presented as participant data.
- The 2025 article states that raw data are available from the authors on request; no public participant-level download was located.

## Main numerical results

- In the synthetic history-dependent scenario, adding observed history changed held-out RMSE from **0.0438** to **0.0278** (36.6% improvement).
- In the no-history control, the corresponding change was **0.0363** to **0.0355** (2.2% improvement).
- The known overlapping-futures process produced a memory gap of **1.0210 bits**.
- The orthogonal control produced a gap of **2.2204e-16 bits**; the one-state null produced **0.0000e+00 bits**.
- The empirical synthetic transducer gave **C_mu=1.9765 bits**, **C_q=1.9377 qubits**, gap **0.0388**.
- Participant bootstrap 95% interval for the synthetic empirical gap: **[0.0282, 0.0532]** (median 0.0383).

## Feasibility decision

- Exact quantum-metric recovery tests: **PASS**.
- History-specific predictive utility test: **PASS**.
- Synthetic empirical memory-gap interval above zero: **PASS**.
- Overall recommendation: **PROCEED TO A REAL-DATA PILOT**.

This recommendation means the computational question is testable and the implementation behaves correctly on controlled data. It does **not** establish a quantum advantage in human motor learning. That conclusion requires the actual de-identified trial-level dataset and participant-held-out validation.

## Published-effect calibration check

- online_random: 10.47%
- online_repeated: 13.57%
- offline_APnegative_implicit: 17.19%
- offline_APnegative_explicit: 4.55%
- offline_APpositive_implicit: 9.35%
- offline_APpositive_explicit: 13.71%

## Files

- `sinaps_quantum_feasibility_benchmark.py`: the complete executable benchmark.
- `synthetic_sinaps_history.csv`: synthetic history-dependent dataset.
- `synthetic_sinaps_no_history.csv`: synthetic null-control dataset.
- `fold_metrics.csv`: participant-held-out predictive metrics.
- `known_process_validation.csv`: exact quantum-memory controls.
- `empirical_quantum_memory_bootstrap.csv`: participant bootstrap results.
- `predictive_state_assignments.csv`: fitted predictive states for the selected source data.
- PNG figures summarize learning curves, held-out RMSE, and predictive memory.

## Recommended real-data next step

Request de-identified trial-level RMSE for all 72 participants, group assignments, AP30 threshold status/value, waveform-level performance, and delayed-retention trials. Run this same script with `--data-csv`. The real-data decision should be based on held-out NLL/RMSE and a participant-bootstrap interval for `C_mu-C_q`, not on the synthetic result.
