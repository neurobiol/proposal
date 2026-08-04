# SINAPs quantum predictive-memory benchmark

## What this package does

The script tests a computational pipeline for trial-by-trial motor-learning data:

1. Generate a **clearly labelled synthetic** version of the published SINAPs design when raw data are unavailable.
2. Compare a static model with a history-aware model using participant-held-out cross-validation.
3. Fit a finite predictive-state transducer.
4. Compute classical predictive memory `C_mu` and quantum predictive memory `C_q`.
5. Validate the quantum calculation on known overlapping, orthogonal, and one-state processes.
6. Bootstrap participants to quantify uncertainty in `C_mu - C_q`.

The synthetic data are calibrated to the design and aggregate effects reported by Perrier et al. (2025), but are **not reconstructed participant data**.

## Requirements

Python 3.10 or newer with:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

Install with:

```bash
python -m pip install -r requirements_sinaps_benchmark.txt
```

## Run the synthetic feasibility benchmark

```bash
python sinaps_quantum_feasibility_benchmark.py \
  --output-dir sinaps_benchmark_outputs \
  --seed 20260730 \
  --participants 72 \
  --bootstrap 300
```

## Run later with real data

```bash
python sinaps_quantum_feasibility_benchmark.py \
  --data-csv real_sinaps_trials.csv \
  --output-dir sinaps_real_benchmark_outputs \
  --seed 20260730 \
  --bootstrap 1000
```

Required columns:

```text
participant_id, trial, phase, ap_positive, explicit_instruction, waveform, rmse
```

Allowed values:

- `phase`: `baseline`, `practice`, or `retention`
- `waveform`: `random` or `repeated`
- `ap_positive`: `0` or `1`
- `explicit_instruction`: `0` or `1`

Each participant should have rows ordered by trial within each waveform.

## What counts as a promising result

A real-data pilot is promising only when:

- history improves participant-held-out NLL or RMSE beyond the observed baseline and known covariates;
- the selected classical process predicts held-out participants credibly;
- the quantum construction reproduces the same fitted output probabilities;
- the participant-bootstrap interval for `C_mu - C_q` is above zero;
- the result is stable to state number, output binning, and orthogonal-state controls.

Synthetic results establish only that the question is computationally testable. They do not establish quantum advantage in human learning.
