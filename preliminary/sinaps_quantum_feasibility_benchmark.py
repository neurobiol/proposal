#!/usr/bin/env python3
"""SINAPs quantum predictive-memory feasibility benchmark.

This script has two purposes:

1. It generates a *clearly labelled synthetic* version of the published SINAPs
   motor-learning design when participant-level raw data are not supplied.
2. It tests whether (a) observed trial history improves held-out prediction and
   (b) a finite predictive-state transducer admits quantum memory compression.

The synthetic generator is calibrated to the design and aggregate effects in:
Perrier et al., Brain Sciences 2025, 15, 645,
doi:10.3390/brainsci15060645. It is not a reconstruction of the original data.

Usage
-----
Synthetic benchmark (recommended first run):
    python sinaps_quantum_feasibility_benchmark.py \
        --output-dir sinaps_benchmark_outputs --seed 20260730

Use a real CSV later:
    python sinaps_quantum_feasibility_benchmark.py \
        --data-csv real_sinaps_trials.csv \
        --output-dir sinaps_real_benchmark_outputs

Required real-data columns
--------------------------
participant_id, trial, phase, ap_positive, explicit_instruction, waveform, rmse

Optional columns are preserved. `phase` must be one of baseline, practice,
retention; `waveform` must be random or repeated. The script assumes trial order
within each participant and waveform.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import eigvalsh
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
PHASE_TO_INT = {"baseline": 0, "practice": 1, "retention": 2}
WAVEFORM_TO_INT = {"random": 0, "repeated": 1}


@dataclass
class PredictionMetrics:
    model: str
    scenario: str
    fold: int
    rmse: float
    nll: float
    sigma: float
    n_test: int


def shannon_entropy_bits(probabilities: NDArray[np.float64]) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > EPS]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def gaussian_nll(y: NDArray[np.float64], pred: NDArray[np.float64], sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    residual = np.asarray(y) - np.asarray(pred)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + residual**2 / (2.0 * sigma**2)))


def rmse(y: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def _balanced_conditions(n_participants: int, rng: np.random.Generator) -> pd.DataFrame:
    if n_participants < 8:
        raise ValueError("At least 8 participants are required.")
    conditions = [(ap, instruction) for ap in (0, 1) for instruction in (0, 1)]
    rows: List[Tuple[int, int]] = []
    while len(rows) < n_participants:
        rows.extend(conditions)
    rows = rows[:n_participants]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["ap_positive", "explicit_instruction"])


def generate_synthetic_sinaps(
    n_participants: int,
    seed: int,
    scenario: str,
) -> pd.DataFrame:
    """Generate a synthetic version of the published 72-participant design.

    Parameters
    ----------
    scenario:
        `history`: plausible participant-specific latent state and autocorrelation.
        `no_history`: same aggregate condition effects but independent residuals.
    """
    if scenario not in {"history", "no_history"}:
        raise ValueError("scenario must be 'history' or 'no_history'")

    rng = np.random.default_rng(seed)
    conditions = _balanced_conditions(n_participants, rng)
    records: List[dict] = []

    # Published offline percentage improvements by AP30 x instruction group.
    offline_gain = {
        (0, 0): 0.171,  # AP30-negative, implicit
        (0, 1): 0.036,  # AP30-negative, explicit
        (1, 0): 0.088,  # AP30-positive, implicit
        (1, 1): 0.133,  # AP30-positive, explicit
    }

    # Shared slow latent state; 0=low engagement, 1=intermediate, 2=stable.
    state_effect = np.array([0.018, 0.0, -0.018])

    for participant_idx, cond in conditions.iterrows():
        pid = f"P{participant_idx + 1:03d}"
        ap = int(cond["ap_positive"])
        instruction = int(cond["explicit_instruction"])

        baseline_level = float(rng.normal(1.0, 0.075))
        if scenario == "history":
            learning_ability = float(np.clip(rng.normal(1.0, 0.16), 0.55, 1.45))
            personal_retention = float(rng.normal(0.0, 0.018))
        else:
            # Null control: participant baseline may differ, but there is no
            # participant-specific learning rate, retention tendency, latent state,
            # or residual autocorrelation beyond the published condition effects.
            learning_ability = 1.0
            personal_retention = 0.0
        latent_state = int(rng.choice([0, 1], p=[0.58, 0.42]))
        ar_noise = {"random": 0.0, "repeated": 0.0}

        for trial in range(1, 133):
            if trial <= 24:
                phase = "baseline"
                phase_progress = trial / 24.0
            elif trial <= 120:
                phase = "practice"
                phase_progress = (trial - 24) / 96.0
            else:
                phase = "retention"
                phase_progress = (trial - 120) / 12.0

            if scenario == "history":
                # Slowly changing participant state. Explicit instruction has a small
                # state-dependent influence only after the baseline period.
                if phase == "practice":
                    if latent_state == 0:
                        p_up = np.clip(0.035 * learning_ability + 0.008 * ap - 0.006 * instruction * (1 - ap), 0.004, 0.10)
                        if rng.random() < p_up:
                            latent_state = 1
                    elif latent_state == 1:
                        p_up = np.clip(0.025 * learning_ability + 0.006 * ap, 0.003, 0.08)
                        if rng.random() < p_up:
                            latent_state = 2
                        elif rng.random() < 0.008:
                            latent_state = 0
                    else:
                        if rng.random() < 0.005:
                            latent_state = 1
                elif phase == "retention" and rng.random() < 0.01:
                    latent_state = max(0, latent_state - 1)
            else:
                latent_state = 1

            for waveform in ("random", "repeated"):
                # Published online improvements: repeated 13.5%, random 10.5%.
                endpoint_improvement = 0.107 if waveform == "repeated" else 0.077
                endpoint_improvement *= learning_ability

                if phase == "baseline":
                    # Small familiarization during the first 24 trials.
                    improvement = 0.018 * (1.0 - np.exp(-2.0 * phase_progress)) / (1.0 - np.exp(-2.0))
                    expected = baseline_level * (1.0 - improvement)
                elif phase == "practice":
                    smooth_progress = (1.0 - np.exp(-3.0 * phase_progress)) / (1.0 - np.exp(-3.0))
                    expected = baseline_level * (1.0 - 0.018 - endpoint_improvement * smooth_progress)
                else:
                    final_training = baseline_level * (1.0 - 0.018 - endpoint_improvement)
                    gain = float(np.clip(offline_gain[(ap, instruction)] + personal_retention, -0.02, 0.24))
                    expected = final_training * (1.0 - gain)
                    # Very small within-retention re-familiarization.
                    expected *= 1.0 - 0.006 * phase_progress

                if scenario == "history":
                    expected += state_effect[latent_state]
                    innovation = float(rng.normal(0.0, 0.026))
                    ar_noise[waveform] = 0.62 * ar_noise[waveform] + innovation
                    noise = ar_noise[waveform]
                else:
                    noise = float(rng.normal(0.0, 0.034))

                observed = float(max(0.25, expected + noise))
                records.append(
                    {
                        "participant_id": pid,
                        "trial": trial,
                        "session": 1 if trial <= 120 else 2,
                        "phase": phase,
                        "ap_positive": ap,
                        "explicit_instruction": instruction,
                        "instruction_active": int(instruction and trial > 24 and trial <= 120),
                        "waveform": waveform,
                        "rmse": observed,
                        "synthetic_latent_state": latent_state,
                        "data_source": f"synthetic_{scenario}_published_design",
                    }
                )

    data = pd.DataFrame.from_records(records)
    return data.sort_values(["participant_id", "waveform", "trial"]).reset_index(drop=True)


def validate_input_data(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant_id",
        "trial",
        "phase",
        "ap_positive",
        "explicit_instruction",
        "waveform",
        "rmse",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = data.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    out["phase"] = out["phase"].astype(str).str.lower()
    out["waveform"] = out["waveform"].astype(str).str.lower()
    if not set(out["phase"]).issubset(PHASE_TO_INT):
        raise ValueError(f"phase values must be in {sorted(PHASE_TO_INT)}")
    if not set(out["waveform"]).issubset(WAVEFORM_TO_INT):
        raise ValueError(f"waveform values must be in {sorted(WAVEFORM_TO_INT)}")
    for col in ["trial", "ap_positive", "explicit_instruction", "rmse"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if "session" not in out:
        out["session"] = np.where(out["phase"].eq("retention"), 2, 1)
    if "instruction_active" not in out:
        out["instruction_active"] = (
            out["explicit_instruction"].astype(int)
            & out["phase"].eq("practice")
        ).astype(int)
    if "data_source" not in out:
        out["data_source"] = "user_supplied"
    return out.sort_values(["participant_id", "waveform", "trial"]).reset_index(drop=True)


def _rolling_slope(values: NDArray[np.float64]) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def add_history_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["phase_code"] = out["phase"].map(PHASE_TO_INT).astype(int)
    out["waveform_code"] = out["waveform"].map(WAVEFORM_TO_INT).astype(int)
    max_trial = max(float(out["trial"].max()), 1.0)
    out["trial_norm"] = out["trial"].astype(float) / max_trial
    baseline_lookup = (
        out[out["phase"].eq("baseline")]
        .groupby(["participant_id", "waveform"])["rmse"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "baseline_mean", "std": "baseline_std"})
        .reset_index()
    )
    out = out.merge(baseline_lookup, on=["participant_id", "waveform"], how="left")
    out["baseline_std"] = out["baseline_std"].fillna(0.0)
    out["ap_x_instruction"] = out["ap_positive"].astype(float) * out["explicit_instruction"].astype(float)
    out["instruction_x_active"] = out["explicit_instruction"].astype(float) * out["instruction_active"].astype(float)

    groups = out.groupby(["participant_id", "waveform"], sort=False)["rmse"]
    out["lag1"] = groups.shift(1)
    out["lag2"] = groups.shift(2)
    out["rolling3_mean"] = groups.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out["rolling5_std"] = groups.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    out["rolling5_slope"] = groups.transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).apply(_rolling_slope, raw=True)
    )
    out["prior_mean"] = groups.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    out["prior_min"] = groups.transform(lambda s: s.shift(1).expanding(min_periods=1).min())

    global_median = float(out["rmse"].median())
    for col in ["lag1", "lag2", "rolling3_mean", "rolling5_std", "rolling5_slope", "prior_mean", "prior_min"]:
        if col == "rolling5_std" or col == "rolling5_slope":
            out[col] = out[col].fillna(0.0)
        else:
            out[col] = out[col].fillna(global_median)
    return out


STATIC_FEATURES = [
    "trial_norm",
    "phase_code",
    "ap_positive",
    "explicit_instruction",
    "instruction_active",
    "waveform_code",
    "ap_x_instruction",
    "instruction_x_active",
    "baseline_mean",
    "baseline_std",
]

HISTORY_FEATURES = STATIC_FEATURES + [
    "lag1",
    "lag2",
    "rolling3_mean",
    "rolling5_std",
    "rolling5_slope",
    "prior_mean",
    "prior_min",
]

STATE_FEATURES = [
    "trial_norm",
    "phase_code",
    "ap_positive",
    "explicit_instruction",
    "waveform_code",
    "lag1",
    "rolling3_mean",
    "rolling5_slope",
    "prior_mean",
]


def make_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.20,
        random_state=seed,
    )


def cross_validated_prediction_benchmark(
    data: pd.DataFrame,
    scenario: str,
    seed: int,
    n_splits: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    participant_count = data["participant_id"].nunique()
    n_splits = min(n_splits, participant_count)
    if n_splits < 2:
        raise ValueError("At least two participants are required for held-out validation.")

    # The first 24 trials establish an observed participant baseline. Models are
    # evaluated prospectively from trial 25 onward, avoiding leakage while making
    # the static comparator clinically meaningful.
    data = data[data["trial"] > 24].copy().reset_index(drop=True)
    groups = data["participant_id"].to_numpy()
    y = data["rmse"].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    metrics: List[PredictionMetrics] = []
    predictions: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, y, groups), start=1):
        for model_name, feature_names in [("static", STATIC_FEATURES), ("history", HISTORY_FEATURES)]:
            model = make_regressor(seed + fold * 100 + (1 if model_name == "history" else 0))
            x_train = data.iloc[train_idx][feature_names].to_numpy(dtype=float)
            x_test = data.iloc[test_idx][feature_names].to_numpy(dtype=float)
            y_train = y[train_idx]
            y_test = y[test_idx]
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            sigma = float(np.std(y_train - train_pred, ddof=1))
            metrics.append(
                PredictionMetrics(
                    model=model_name,
                    scenario=scenario,
                    fold=fold,
                    rmse=rmse(y_test, test_pred),
                    nll=gaussian_nll(y_test, test_pred, sigma),
                    sigma=sigma,
                    n_test=len(test_idx),
                )
            )
            pred_frame = data.iloc[test_idx][["participant_id", "trial", "phase", "waveform", "rmse"]].copy()
            pred_frame["prediction"] = test_pred
            pred_frame["model"] = model_name
            pred_frame["scenario"] = scenario
            pred_frame["fold"] = fold
            predictions.append(pred_frame)

    metric_frame = pd.DataFrame([m.__dict__ for m in metrics])
    return metric_frame, pd.concat(predictions, ignore_index=True)


def quantile_edges(values: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    raw = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = raw.copy()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_values(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def finite_bin_centres(values: NDArray[np.float64], bins: NDArray[np.int64], n_bins: int) -> NDArray[np.float64]:
    overall = float(np.mean(values))
    centres = np.full(n_bins, overall, dtype=float)
    for b in range(n_bins):
        mask = bins == b
        if np.any(mask):
            centres[b] = float(np.mean(values[mask]))
    return centres


def transition_count_tensor(
    data: pd.DataFrame,
    states: NDArray[np.int64],
    edges: NDArray[np.float64],
    n_states: int,
    n_inputs: int,
    n_bins: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]:
    frame = data[["participant_id", "waveform", "trial", "phase_code", "rmse"]].copy()
    frame["state"] = states
    frame["output_bin"] = bin_values(frame["rmse"].to_numpy(dtype=float), edges)

    total_counts = np.zeros((n_states, n_inputs, n_bins, n_states), dtype=float)
    occupancy = np.zeros(n_states, dtype=float)
    participant_transitions: Dict[str, NDArray[np.float64]] = {}
    participant_occupancy: Dict[str, NDArray[np.float64]] = {}

    for pid, participant_frame in frame.groupby("participant_id", sort=False):
        p_counts = np.zeros_like(total_counts)
        p_occ = np.zeros_like(occupancy)
        for s in participant_frame["state"].astype(int):
            p_occ[s] += 1.0
        for _, seq in participant_frame.groupby("waveform", sort=False):
            seq = seq.sort_values("trial").reset_index(drop=True)
            for i in range(len(seq) - 1):
                s = int(seq.loc[i, "state"])
                sp = int(seq.loc[i + 1, "state"])
                x = int(seq.loc[i + 1, "phase_code"])
                yb = int(seq.loc[i + 1, "output_bin"])
                p_counts[s, x, yb, sp] += 1.0
        participant_transitions[str(pid)] = p_counts
        participant_occupancy[str(pid)] = p_occ
        total_counts += p_counts
        occupancy += p_occ
    return total_counts, occupancy, participant_transitions, participant_occupancy


def counts_to_transition_probabilities(counts: NDArray[np.float64], alpha: float = 0.25) -> NDArray[np.float64]:
    smoothed = np.asarray(counts, dtype=float) + alpha
    denominator = smoothed.sum(axis=(2, 3), keepdims=True)
    return smoothed / np.maximum(denominator, EPS)


def gram_from_transducer(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Construct the product-input Gram matrix from transition probabilities."""
    n_states, n_inputs, _, _ = T.shape
    gram = np.ones((n_states, n_states), dtype=float)
    for x in range(n_inputs):
        amplitudes = np.sqrt(T[:, x, :, :].reshape(n_states, -1))
        gram_x = amplitudes @ amplitudes.T
        gram *= gram_x
    gram = 0.5 * (gram + gram.T)
    diag = np.sqrt(np.clip(np.diag(gram), EPS, None))
    gram = gram / np.outer(diag, diag)
    np.fill_diagonal(gram, 1.0)
    return gram


def quantum_memory_from_gram(pi: NDArray[np.float64], gram: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
    pi = np.asarray(pi, dtype=float)
    pi = pi / max(pi.sum(), EPS)
    root_pi = np.sqrt(pi)
    ensemble_matrix = root_pi[:, None] * gram * root_pi[None, :]
    eigenvalues = eigvalsh(0.5 * (ensemble_matrix + ensemble_matrix.T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.sum() > 0:
        eigenvalues /= eigenvalues.sum()
    cq = shannon_entropy_bits(eigenvalues)
    return cq, eigenvalues


def memory_from_counts(counts: NDArray[np.float64], occupancy: NDArray[np.float64]) -> dict:
    T = counts_to_transition_probabilities(counts)
    pi = occupancy / max(occupancy.sum(), EPS)
    gram = gram_from_transducer(T)
    cmu = shannon_entropy_bits(pi)
    cq, rho_eigenvalues = quantum_memory_from_gram(pi, gram)
    gram_eigenvalues = eigvalsh(gram)
    return {
        "C_mu_bits": cmu,
        "C_q_qubits": cq,
        "memory_gap": cmu - cq,
        "gram_min_eigenvalue": float(gram_eigenvalues.min()),
        "gram_max_diagonal_error": float(np.max(np.abs(np.diag(gram) - 1.0))),
        "transition_max_normalization_error": float(np.max(np.abs(T.sum(axis=(2, 3)) - 1.0))),
        "rho_rank": int(np.sum(rho_eigenvalues > 1e-10)),
        "gram": gram,
        "T": T,
        "pi": pi,
    }


def fit_empirical_transducer(
    data: pd.DataFrame,
    seed: int,
    n_states: int = 4,
    n_bins: int = 4,
    n_bootstrap: int = 300,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[STATE_FEATURES].to_numpy(dtype=float))
    kmeans = KMeans(n_clusters=n_states, n_init=30, random_state=seed)
    states = kmeans.fit_predict(x_scaled).astype(int)
    edges = quantile_edges(data["rmse"].to_numpy(dtype=float), n_bins=n_bins)

    counts, occupancy, p_counts, p_occ = transition_count_tensor(
        data=data,
        states=states,
        edges=edges,
        n_states=n_states,
        n_inputs=len(PHASE_TO_INT),
        n_bins=n_bins,
    )
    memory = memory_from_counts(counts, occupancy)

    rng = np.random.default_rng(seed + 991)
    participants = np.array(sorted(p_counts), dtype=object)
    bootstrap_rows: List[dict] = []
    for b in range(n_bootstrap):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        b_counts = np.zeros_like(counts)
        b_occ = np.zeros_like(occupancy)
        for pid in sampled:
            b_counts += p_counts[str(pid)]
            b_occ += p_occ[str(pid)]
        b_memory = memory_from_counts(b_counts, b_occ)
        bootstrap_rows.append(
            {
                "bootstrap": b + 1,
                "C_mu_bits": b_memory["C_mu_bits"],
                "C_q_qubits": b_memory["C_q_qubits"],
                "memory_gap": b_memory["memory_gap"],
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    assigned = data[["participant_id", "trial", "phase", "waveform", "rmse"]].copy()
    assigned["predictive_state"] = states
    return memory, bootstrap, assigned


def known_process_validation() -> pd.DataFrame:
    rows: List[dict] = []

    # Similar predictive futures: states 0 and 1 have high quantum overlap.
    overlap = np.zeros((3, 1, 3, 3), dtype=float)
    overlap[0, 0, 0, 0] = 0.55
    overlap[0, 0, 1, 1] = 0.35
    overlap[0, 0, 2, 2] = 0.10
    overlap[1, 0, 0, 0] = 0.50
    overlap[1, 0, 1, 1] = 0.40
    overlap[1, 0, 2, 2] = 0.10
    overlap[2, 0, 0, 0] = 0.05
    overlap[2, 0, 1, 1] = 0.15
    overlap[2, 0, 2, 2] = 0.80
    processes = [("overlapping_futures", overlap, np.array([0.35, 0.35, 0.30]))]

    # Perfectly distinguishable futures: orthogonal quantum states, no compression.
    orthogonal = np.zeros((3, 1, 3, 3), dtype=float)
    for s in range(3):
        orthogonal[s, 0, s, s] = 1.0
    processes.append(("orthogonal_futures", orthogonal, np.array([0.35, 0.35, 0.30])))

    # Single predictive state: zero classical and quantum memory.
    one_state = np.ones((1, 1, 1, 1), dtype=float)
    processes.append(("one_state_null", one_state, np.array([1.0])))

    for name, T, pi in processes:
        gram = gram_from_transducer(T)
        cmu = shannon_entropy_bits(pi)
        cq, _ = quantum_memory_from_gram(pi, gram)
        rows.append(
            {
                "process": name,
                "C_mu_bits": cmu,
                "C_q_qubits": cq,
                "memory_gap": cmu - cq,
                "gram_min_eigenvalue": float(eigvalsh(gram).min()),
                "max_transition_error": float(np.max(np.abs(T.sum(axis=(2, 3)) - 1.0))),
            }
        )
    return pd.DataFrame(rows)


def published_effect_check(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    # Online improvement: mean first 24 versus final 12 practice trials.
    session1 = data[data["trial"] <= 120]
    baseline = session1[session1["trial"] <= 24].groupby(["participant_id", "waveform"])["rmse"].mean()
    late = session1[(session1["trial"] >= 109) & (session1["trial"] <= 120)].groupby(["participant_id", "waveform"])["rmse"].mean()
    online = (baseline - late) / baseline * 100.0
    online = online.rename("percent_improvement").reset_index()
    for waveform, group in online.groupby("waveform"):
        rows.append({"effect": f"online_{waveform}", "estimate_percent": float(group["percent_improvement"].mean())})

    last_practice = data[(data["trial"] >= 109) & (data["trial"] <= 120)].groupby(
        ["participant_id", "ap_positive", "explicit_instruction", "waveform"]
    )["rmse"].mean()
    retention = data[data["phase"].eq("retention")].groupby(
        ["participant_id", "ap_positive", "explicit_instruction", "waveform"]
    )["rmse"].mean()
    offline = ((last_practice - retention) / last_practice * 100.0).rename("percent_improvement").reset_index()
    for (ap, instruction), group in offline.groupby(["ap_positive", "explicit_instruction"]):
        rows.append(
            {
                "effect": f"offline_AP{'positive' if ap else 'negative'}_{'explicit' if instruction else 'implicit'}",
                "estimate_percent": float(group["percent_improvement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    synthetic_history: pd.DataFrame,
    process_validation: pd.DataFrame,
    empirical_memory: Mapping[str, float],
) -> None:
    # Figure 1: learning curves.
    curve = (
        synthetic_history.groupby(["trial", "ap_positive", "explicit_instruction"], as_index=False)["rmse"].mean()
    )
    plt.figure(figsize=(8.2, 5.0))
    for (ap, instruction), group in curve.groupby(["ap_positive", "explicit_instruction"]):
        label = f"AP{'+' if ap else '-'} / {'explicit' if instruction else 'implicit'}"
        plt.plot(group["trial"], group["rmse"], label=label)
    plt.axvline(24, linestyle="--", linewidth=1)
    plt.axvline(120, linestyle="--", linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Synthetic RMSE (lower is better)")
    plt.title("Synthetic SINAPs-like learning and delayed retention")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_learning_curves.png", dpi=220)
    plt.close()

    # Figure 2: held-out RMSE comparison.
    summary = fold_metrics.groupby(["scenario", "model"], as_index=False)["rmse"].mean()
    order = [("history", "static"), ("history", "history"), ("no_history", "static"), ("no_history", "history")]
    summary["order"] = summary.apply(lambda r: order.index((r["scenario"], r["model"])), axis=1)
    summary = summary.sort_values("order")
    label_map = {
        ("history", "static"): "Dynamic scenario\nstatic model",
        ("history", "history"): "Dynamic scenario\nhistory model",
        ("no_history", "static"): "Null scenario\nstatic model",
        ("no_history", "history"): "Null scenario\nhistory model",
    }
    labels = [label_map[(r.scenario, r.model)] for r in summary.itertuples()]
    plt.figure(figsize=(7.8, 4.8))
    plt.bar(labels, summary["rmse"].to_numpy())
    plt.ylabel("Participant-held-out RMSE")
    plt.title("Does observed trial history improve prediction?")
    plt.tight_layout()
    plt.savefig(output_dir / "heldout_rmse_comparison.png", dpi=220)
    plt.close()

    # Figure 3: exact process validation and empirical synthetic estimate.
    pretty_process = {
        "overlapping_futures": "Overlapping\nfutures",
        "orthogonal_futures": "Orthogonal\nfutures",
        "one_state_null": "One-state\nnull",
    }
    labels = [pretty_process[x] for x in process_validation["process"].tolist()] + ["Synthetic\nempirical"]
    cmu = process_validation["C_mu_bits"].tolist() + [float(empirical_memory["C_mu_bits"])]
    cq = process_validation["C_q_qubits"].tolist() + [float(empirical_memory["C_q_qubits"])]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x - width / 2, cmu, width, label="Classical Cμ")
    plt.bar(x + width / 2, cq, width, label="Quantum Cq")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Predictive memory")
    plt.title("Classical and quantum predictive-memory benchmark")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "predictive_memory_comparison.png", dpi=220)
    plt.close()


def build_report(
    output_dir: Path,
    source_description: str,
    fold_metrics: pd.DataFrame,
    process_validation: pd.DataFrame,
    memory: Mapping[str, object],
    bootstrap: pd.DataFrame,
    effect_check: pd.DataFrame,
) -> str:
    mean_metrics = fold_metrics.groupby(["scenario", "model"])[["rmse", "nll"]].mean().reset_index()

    def metric_value(scenario: str, model: str, column: str) -> float:
        row = mean_metrics[(mean_metrics["scenario"] == scenario) & (mean_metrics["model"] == model)]
        return float(row.iloc[0][column])

    hist_static = metric_value("history", "static", "rmse")
    hist_history = metric_value("history", "history", "rmse")
    null_static = metric_value("no_history", "static", "rmse")
    null_history = metric_value("no_history", "history", "rmse")
    history_improvement = 100.0 * (hist_static - hist_history) / hist_static
    null_improvement = 100.0 * (null_static - null_history) / null_static

    gap_ci = np.quantile(bootstrap["memory_gap"], [0.025, 0.5, 0.975])
    overlap_gap = float(
        process_validation.loc[process_validation["process"] == "overlapping_futures", "memory_gap"].iloc[0]
    )
    orthogonal_gap = float(
        process_validation.loc[process_validation["process"] == "orthogonal_futures", "memory_gap"].iloc[0]
    )
    null_gap = float(process_validation.loc[process_validation["process"] == "one_state_null", "memory_gap"].iloc[0])

    recovery_pass = overlap_gap > 0.05 and abs(orthogonal_gap) < 1e-8 and abs(null_gap) < 1e-8
    history_pass = history_improvement > 5.0 and (history_improvement - null_improvement) > 10.0
    empirical_gap_pass = gap_ci[0] > 0.0
    proceed = recovery_pass and history_pass

    lines = [
        "# SINAPs quantum predictive-memory feasibility benchmark",
        "",
        "## Data status",
        "",
        f"- Source used: **{source_description}**.",
        "- Any file whose name contains `synthetic` is simulated and must not be presented as participant data.",
        "- The 2025 article states that raw data are available from the authors on request; no public participant-level download was located.",
        "",
        "## Main numerical results",
        "",
        f"- In the synthetic history-dependent scenario, adding observed history changed held-out RMSE from **{hist_static:.4f}** to **{hist_history:.4f}** ({history_improvement:.1f}% improvement).",
        f"- In the no-history control, the corresponding change was **{null_static:.4f}** to **{null_history:.4f}** ({null_improvement:.1f}% improvement).",
        f"- The known overlapping-futures process produced a memory gap of **{overlap_gap:.4f} bits**.",
        f"- The orthogonal control produced a gap of **{orthogonal_gap:.4e} bits**; the one-state null produced **{null_gap:.4e} bits**.",
        f"- The empirical synthetic transducer gave **C_mu={float(memory['C_mu_bits']):.4f} bits**, **C_q={float(memory['C_q_qubits']):.4f} qubits**, gap **{float(memory['memory_gap']):.4f}**.",
        f"- Participant bootstrap 95% interval for the synthetic empirical gap: **[{gap_ci[0]:.4f}, {gap_ci[2]:.4f}]** (median {gap_ci[1]:.4f}).",
        "",
        "## Feasibility decision",
        "",
        f"- Exact quantum-metric recovery tests: **{'PASS' if recovery_pass else 'FAIL'}**.",
        f"- History-specific predictive utility test: **{'PASS' if history_pass else 'FAIL'}**.",
        f"- Synthetic empirical memory-gap interval above zero: **{'PASS' if empirical_gap_pass else 'FAIL'}**.",
        f"- Overall recommendation: **{'PROCEED TO A REAL-DATA PILOT' if proceed else 'REVISE THE PIPELINE BEFORE REQUESTING DATA'}**.",
        "",
        "This recommendation means the computational question is testable and the implementation behaves correctly on controlled data. It does **not** establish a quantum advantage in human motor learning. That conclusion requires the actual de-identified trial-level dataset and participant-held-out validation.",
        "",
        "## Published-effect calibration check",
        "",
    ]
    for row in effect_check.itertuples(index=False):
        lines.append(f"- {row.effect}: {row.estimate_percent:.2f}%")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `sinaps_quantum_feasibility_benchmark.py`: the complete executable benchmark.",
            "- `synthetic_sinaps_history.csv`: synthetic history-dependent dataset.",
            "- `synthetic_sinaps_no_history.csv`: synthetic null-control dataset.",
            "- `fold_metrics.csv`: participant-held-out predictive metrics.",
            "- `known_process_validation.csv`: exact quantum-memory controls.",
            "- `empirical_quantum_memory_bootstrap.csv`: participant bootstrap results.",
            "- `predictive_state_assignments.csv`: fitted predictive states for the selected source data.",
            "- PNG figures summarize learning curves, held-out RMSE, and predictive memory.",
            "",
            "## Recommended real-data next step",
            "",
            "Request de-identified trial-level RMSE for all 72 participants, group assignments, AP30 threshold status/value, waveform-level performance, and delayed-retention trials. Run this same script with `--data-csv`. The real-data decision should be based on held-out NLL/RMSE and a participant-bootstrap interval for `C_mu-C_q`, not on the synthetic result.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-csv", type=Path, default=None, help="Optional real trial-level CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("sinaps_benchmark_outputs"))
    parser.add_argument("--participants", type=int, default=72, help="Synthetic participant count.")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--states", type=int, default=4, help="Empirical predictive-state count.")
    parser.add_argument("--bins", type=int, default=4, help="Adaptive RMSE output bins.")
    parser.add_argument("--bootstrap", type=int, default=300, help="Participant bootstrap replicates.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Always run two synthetic scenarios to test specificity, even when a real CSV is supplied.
    synthetic_history = add_history_features(
        generate_synthetic_sinaps(args.participants, args.seed, scenario="history")
    )
    synthetic_null = add_history_features(
        generate_synthetic_sinaps(args.participants, args.seed + 1, scenario="no_history")
    )
    synthetic_history.to_csv(args.output_dir / "synthetic_sinaps_history.csv", index=False)
    synthetic_null.to_csv(args.output_dir / "synthetic_sinaps_no_history.csv", index=False)

    history_metrics, history_predictions = cross_validated_prediction_benchmark(
        synthetic_history, scenario="history", seed=args.seed
    )
    null_metrics, null_predictions = cross_validated_prediction_benchmark(
        synthetic_null, scenario="no_history", seed=args.seed + 1
    )
    fold_metrics = pd.concat([history_metrics, null_metrics], ignore_index=True)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat([history_predictions, null_predictions], ignore_index=True).to_csv(
        args.output_dir / "heldout_predictions.csv", index=False
    )

    process_validation = known_process_validation()
    process_validation.to_csv(args.output_dir / "known_process_validation.csv", index=False)

    source_data = synthetic_history
    source_description = "synthetic published-design benchmark"
    if args.data_csv is not None:
        if not args.data_csv.exists():
            raise FileNotFoundError(f"Real-data CSV does not exist: {args.data_csv}")
        source_data = add_history_features(validate_input_data(pd.read_csv(args.data_csv)))
        source_description = f"user-supplied real-data CSV `{args.data_csv.name}`"
        real_metrics, real_predictions = cross_validated_prediction_benchmark(
            source_data, scenario="real_data", seed=args.seed + 10
        )
        real_metrics.to_csv(args.output_dir / "real_data_fold_metrics.csv", index=False)
        real_predictions.to_csv(args.output_dir / "real_data_heldout_predictions.csv", index=False)

    memory, bootstrap, state_assignments = fit_empirical_transducer(
        source_data,
        seed=args.seed,
        n_states=args.states,
        n_bins=args.bins,
        n_bootstrap=args.bootstrap,
    )
    bootstrap.to_csv(args.output_dir / "empirical_quantum_memory_bootstrap.csv", index=False)
    state_assignments.to_csv(args.output_dir / "synthetic_predictive_state_assignments.csv", index=False)

    memory_summary = {
        key: value
        for key, value in memory.items()
        if key not in {"gram", "T", "pi"}
    }
    gap_ci = np.quantile(bootstrap["memory_gap"], [0.025, 0.5, 0.975])
    memory_summary.update(
        {
            "bootstrap_gap_ci_low": float(gap_ci[0]),
            "bootstrap_gap_median": float(gap_ci[1]),
            "bootstrap_gap_ci_high": float(gap_ci[2]),
        }
    )
    with (args.output_dir / "empirical_memory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(memory_summary, handle, indent=2)
    np.savetxt(args.output_dir / "empirical_gram_matrix.csv", memory["gram"], delimiter=",")

    effect_check = published_effect_check(synthetic_history)
    effect_check.to_csv(args.output_dir / "synthetic_published_effect_check.csv", index=False)

    make_figures(
        output_dir=args.output_dir,
        fold_metrics=fold_metrics,
        synthetic_history=synthetic_history,
        process_validation=process_validation,
        empirical_memory=memory,
    )

    report = build_report(
        output_dir=args.output_dir,
        source_description=source_description,
        fold_metrics=fold_metrics,
        process_validation=process_validation,
        memory=memory,
        bootstrap=bootstrap,
        effect_check=effect_check,
    )
    (args.output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")

    run_config = {
        "seed": args.seed,
        "participants": args.participants,
        "states": args.states,
        "bins": args.bins,
        "bootstrap": args.bootstrap,
        "data_csv": str(args.data_csv) if args.data_csv else None,
        "python": sys.version,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
