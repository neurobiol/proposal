# Mathematical and reproducibility guide

## Purpose

This document explains the mathematical and computational framework used in the thesis proposal and records the main implementation and decision rules.

## Memory-Efficient Modeling of Human Sensorimotor Learning: A Quantum-Inspired Approach

### Master of Science Thesis Proposal

**Student:** Yashine H. Goolam Hossen  
**Principal Investigator:** Prof. Travis J. A. Craddock  
**Department:** Biology, University of Waterloo  
**Location:** Waterloo, Ontario, Canada  
**Proposal date:** August 5, 2026

> **Viewing note:** GitHub renders the equations in the normal file view or Preview. The Raw view and some generic text viewers show the LaTeX source instead.

---

## 1. Overview

The biological question is whether AP-sensitive motor-system physiology, instruction, movement-segment type, and earlier practice together provide useful information about later sensorimotor performance and delayed retention.

The central modeling question is:

> How much information about earlier practice must a fitted model retain to preserve its predictions, and can a quantum-inspired representation reproduce the same predictions with less memory?

The project first compares five classical model families on participants who were not used for training. The predictive-state model is used for the memory comparison only if its prediction is competitive and its state structure is stable. A quantum representation is then constructed from that fitted classical process. It is not trained as an independent predictor.

The word **memory** in this document means information stored by a mathematical predictor. It does not mean conscious recall, biological memory capacity, a synaptic memory trace, or evidence of quantum activity in the brain.

The current preliminary analysis and simulation pipeline is stored in the [`preliminary/`](./preliminary/) directory. The real de-identified SINAPs participant data are not included in the public repository.

---

## Quick glossary

- **$AP_{30}$**: a motor-threshold measure obtained using a 30-microsecond anterior-posterior TMS pulse.
- **TMS**: transcranial magnetic stimulation.
- **EMG**: electromyography.
- **RMSE**: root mean square error, the trial-level measure of how closely the participant followed the target.
- **RMSPE**: root mean squared prediction error, the difference between predicted and observed trial-level RMSE.
- **NLL**: negative log-likelihood, a measure of how much probability a model assigns to observed outcomes.
- **Calibration**: agreement between predicted uncertainty and observed outcomes.
- **Hidden state**: an internal model variable that is inferred rather than directly measured.
- **Predictive state**: a group of earlier histories that supports the same future predictions at the tested resolution.
- **Unifilar**: once the current state, input, and observed outcome are known, the next state is determined.
- **Classical predictive memory**: the information needed to distinguish the fitted classical predictive states, measured in bits.
- **Quantum predictive memory**: the corresponding information cost after allowing quantum states to overlap, measured using base-2 von Neumann entropy.
- **Step-wise inefficiency**: a condition under which distinct classical states can sometimes produce the same output and next state for every allowed input, making strict quantum memory reduction possible.
- **Cross-validation**: fitting a model on one set of participants and testing it on different participants.
- **Nested cross-validation**: using inner participant folds to choose model settings and separate outer participant folds for final evaluation.
- **Bootstrap**: repeated resampling of complete participants to estimate uncertainty.

---

## 2. Data, prediction targets, and notation

The SINAPs dataset contains 72 healthy adults. Participants completed 24 baseline trials, 96 practice trials, and 12 delayed-retention trials 24–48 hours later. Each trial contained random and repeated movement segments. AP-sensitive physiology was measured before practice using TMS and EMG and was not measured again after training.

For participant $n$ and trial $t$:

- $n$ labels the participant.
- $t$ labels the trial position.
- $x_{nt}$ contains information known before the outcome. Fixed information includes instruction condition and AP-sensitive physiology. Trial-varying information includes session, block, trial position, movement-segment type, and earlier performance.
- $y_{nt}$ is an observed behavioral outcome. The main trial-level outcome is tracking RMSE.
- $h_{nt}$ is the observed history available before trial $t$.

The history is written as

$$
h_{nt}=(x_{n1:t-1},y_{n1:t-1}).
$$

Here, $x_{n1:t-1}$ and $y_{n1:t-1}$ contain all inputs and outcomes observed before trial $t$.

The main predictive distribution is

$$
p(y_{nt}\mid h_{nt},x_{nt}).
$$

This is the distribution of the next outcome given the participant's earlier history and the current input.

The three prespecified prediction targets are:

1. next-trial tracking RMSE;
2. end-of-practice performance; and
3. delayed-retention performance.

Delayed retention is predicted using only information available before the retention outcome. Information measured after an outcome is not used to predict that earlier outcome.

### Four distinct levels

The analysis keeps four levels separate:

- **Inputs:** AP-sensitive physiology, instruction, experimental conditions, and earlier trial history.
- **Predicted outcomes:** next-trial tracking RMSE, end-of-practice performance, and delayed retention.
- **Evaluation measures:** participant-held-out NLL, RMSPE, and calibration.
- **Memory measures:** $C_\mu$ and $C_q$.

The memory measures are properties of the fitted representations. They are not behavioral outcomes or post-training physiological measurements.

---

## 3. Prediction measures

All models are evaluated on participants who were not used to fit them.

### 3.1 Root mean squared prediction error

$$
\mathrm{RMSPE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}.
$$

Definitions:

- $N$ is the number of held-out observations.
- $y_i$ is the observed value.
- $\hat y_i$ is the predicted value.
- Lower RMSPE means smaller prediction errors.

### 3.2 Negative log-likelihood

$$
\mathrm{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log p(y_i\mid h_i,x_i).
$$

Definitions:

- $p(y_i\mid h_i,x_i)$ is the probability density or probability assigned to the observed outcome.
- The natural logarithm is used unless otherwise stated.
- Lower NLL means that the model assigns more probability to the outcomes that occur.

NLL is the primary model-comparison measure because it evaluates the full predictive distribution, including both the expected outcome and its uncertainty.

### 3.3 Calibration

Calibration asks whether predicted uncertainty agrees with observed outcomes. For example, intervals presented as 90% prediction intervals should contain approximately 90% of comparable held-out observations.

NLL is primary. RMSPE and calibration are secondary.

The published SINAPs ANOVA will be reproduced as a validation step, but it will not be used as evidence of quantum advantage.

---

## 4. Classical model comparison

The project compares five classical model families. Each model uses only information available before the outcome being predicted.

### 4.1 Model C0: mixed-effects regression with history

A representative form is

$$
y_{nt}=\beta_0+b_n+\boldsymbol{\beta}^{\mathsf T}\mathbf z_{nt}+\sum_{\ell=1}^{L}a_{\ell}y_{n,t-\ell}+\epsilon_{nt}.
$$

Definitions:

- $\beta_0$ is the overall intercept.
- $b_n$ is a participant-specific intercept.
- $\mathbf z_{nt}$ contains known predictors such as trial position, instruction, segment type, and AP-sensitive context.
- $L$ is the number of earlier trials included.
- $a_\ell$ is the coefficient for the outcome observed $\ell$ trials earlier.
- $\epsilon_{nt}$ is residual variation.

C0 tests whether earlier performance adds useful prediction beyond current context and stable participant differences.

### 4.2 Model C1: linear state-space model

A state-space model introduces continuous hidden learning variables:

$$
\mathbf s_{n,t+1}=A_{x_{nt}}\mathbf s_{nt}+B_{x_{nt}}\mathbf u_{nt}+\mathbf w_{nt},
$$

$$
y_{nt}=C\mathbf s_{nt}+d_{x_{nt}}+v_{nt}.
$$

Models with one to four hidden dimensions will be compared. The hidden dimension and all other settings are selected using training participants only.

### 4.3 Model C2: input-output hidden Markov model

This model assumes a small set of discrete hidden states with input-dependent transitions:

$$
P(S_{t+1}=j\mid S_t=i,X_t=x)=A_{ij}^{(x)}.
$$

Each hidden state also has an output distribution $P(Y_t\mid S_t,X_t)$.

### 4.4 Model C3: empirical predictive-state model

The exact theory groups two histories together when they give the same probabilities for all allowed future outputs under all allowed future inputs:

$$
h\sim_{\epsilon}h'
\quad\Longleftrightarrow\quad
P(\vec Y\mid \vec X,h)=P(\vec Y\mid \vec X,h').
$$

Finite human data cannot establish exact predictive equivalence for every possible future. C3 is therefore treated as an **empirical predictive-state approximation**, not an exact reconstructed $\epsilon$-transducer.

Within each outer training fold, separate inner participant folds will select:

- how much earlier history to include;
- how the outcomes are represented;
- how similar two predictive distributions must be before their histories are grouped; and
- how many predictive states to retain.

Histories are grouped using their supported predictions for the three prespecified targets: next-trial tracking error, end-of-practice performance, and delayed retention.

The fitted transitions are constrained to be unifilar. The input-conditioned transition probabilities are

$$
T_{ss'}^{y\mid x}=P(Y_t=y,S_{t+1}=s'\mid S_t=s,X_t=x).
$$

Definitions:

- $s$ is the current predictive state.
- $s'$ is the next predictive state.
- $x$ is the current input.
- $y$ is the observed output category or range.

C3 stability will be tested by resampling complete participants and checking whether the following remain similar after state-label alignment:

- the number of states;
- the histories assigned to the states; and
- the fitted transition probabilities.

Only a C3 model that passes the prespecified stability rule will be used to estimate predictive memory and construct the quantum representation.

### 4.5 Model C4: nonlinear tree-ensemble benchmark

C4 contains random-forest and XGBoost variants. It tests whether nonlinear thresholds and interactions improve prediction beyond the more structured models.

Model variants and hyperparameters are selected only in inner participant-level folds. For continuous RMSE, C4 estimates both the mean outcome and its uncertainty. The uncertainty is estimated from prediction errors in inner validation folds. The Gaussian error assumption will be checked, and held-out calibration will be reported. All NLL comparisons will use the same outcome scale.

C4 is a flexible prediction benchmark. It is not used to estimate $C_\mu$ or construct the quantum representation.

### 4.6 Role of the five models

- C0, C1, C2, and C4 are conventional prediction benchmarks.
- C3 is the classical predictive-memory model.
- C3 proceeds to quantum encoding only if it is both stable and predictively competitive.

---

## 5. Classical predictive memory

Let $\pi_t(s)$ be the probability that the fitted process occupies predictive state $s$ at trial $t$. Classical predictive memory is

$$
C_{\mu}(t)=-\sum_s\pi_t(s)\log_2\pi_t(s).
$$

Definitions:

- $C_\mu(t)$ is the classical predictive-memory cost at trial $t$.
- $s$ runs over the fitted predictive states.
- $\pi_t(s)$ is the occupancy probability of state $s$ at trial $t$.
- The unit is bits.

The main classical-memory summary is

$$
\overline{C}_{\mu}=\frac{1}{T}\sum_{t=1}^{T}C_{\mu}(t).
$$

This is a trial-position-dependent, protocol-specific quantity for the fitted SINAPs process. It is not a universal complexity of human learning.

---

## 6. Quantum predictive representation

### 6.1 Quantum memory states

Each classical predictive state $s$ is represented by a normalized quantum state

$$
\lvert\sigma_s\rangle.
$$

The use of quantum states is mathematical. It does not assume that neurons or motor cortex physically implement this calculation.

### 6.2 Quantum-state construction

The quantum representation is constructed from C3 after C3 has been fitted. It is not trained as a separate predictor.

For classical state $s$ and input $x$, define

$$
\lvert\sigma_s^x\rangle = \sum_{y,s'}
\sqrt{T_{ss'}^{y\mid x}}\,\lvert y\rangle\lvert s'\rangle.
$$

All allowed inputs are combined as

$$
\lvert\sigma_s\rangle=\bigotimes_{x\in\mathcal X}\lvert\sigma_s^x\rangle.
$$

The fitted C3 transitions determine the quantum states and their overlaps. No independent overlap parameters are optimized.

### 6.3 Overlap between quantum states

For states $r$ and $s$,

$$
\langle\sigma_r\mid\sigma_s\rangle
=\prod_{x\in\mathcal X}\sum_{y,s'}
\sqrt{T_{rs'}^{y\mid x}T_{ss'}^{y\mid x}}.
$$

The pairwise overlaps form the Gram matrix

$$
G_{rs}=\langle\sigma_r\mid\sigma_s\rangle.
$$

A valid Gram matrix must be Hermitian, positive semidefinite, and have ones on its diagonal.

### 6.4 When strict reduction is possible

A strict memory reduction is possible only when at least two classical states can, for every allowed input, sometimes produce the same output and move to the same next state. This condition is called **step-wise inefficiency**.

If this condition is absent, the quantum states should remain orthogonal and

$$
C_q=C_\mu.
$$

### 6.5 Average quantum state

At trial $t$,

$$
\rho_t=\sum_s\pi_t(s)\lvert\sigma_s\rangle\langle\sigma_s\rvert.
$$

A valid density matrix must have trace one and non-negative eigenvalues.

### 6.6 Quantum predictive memory

Quantum predictive memory is the von Neumann entropy

$$
C_q(t)=-\mathrm{Tr}\left[\rho_t\log_2\rho_t\right].
$$

If $\lambda_k(t)$ are the eigenvalues of $\rho_t$, the same quantity is

$$
C_q(t)=-\sum_k\lambda_k(t)\log_2\lambda_k(t).
$$

The protocol average is

$$
\overline{C}_q=\frac{1}{T}\sum_{t=1}^{T}C_q(t).
$$

$C_\mu(t)$ and $C_q(t)$ are averaged over exactly the same analyzed trials so that the classical and quantum representations are compared under the same task positions and state occupancies.

---

## 7. Continuous outcomes

The finite-state quantum construction requires a finite output representation. Continuous RMSE values will first be grouped into ranges defined using training participants only.

Sensitivity analyses will:

- vary the number of RMSE ranges;
- test alternative training-only range definitions; and
- use a continuous Gaussian-output model when possible.

A held-out participant never influences the outcome ranges or model settings used to predict that participant.

---

## 8. Controls and sensitivity analyses

### 8.1 Orthogonal-state implementation control

All off-diagonal overlaps are forced to zero:

$$
G_{ss'}=0\quad\text{for }s\neq s'.
$$

This control must recover

$$
\overline{C}_q=\overline{C}_{\mu}.
$$

It is an implementation check, not an independent biological hypothesis.

### 8.2 Matched-memory approximation comparison

A secondary analysis will compare classical and quantum approximations given the same predictive-memory limit. Equal parameter count, equal rank, or fewer fitted parameters will not by themselves count as quantum advantage.

### 8.3 Label shuffling

Instruction and AP labels will be randomly reassigned at the participant level while preserving each participant's trial order. This tests whether those variables add predictive information beyond chance.

### 8.4 Simulations with known answers

Synthetic data will be generated from known classical and quantum processes. The full analysis will then be rerun to test whether it recovers the generating process and the correct memory relationship.

### 8.5 Missing-data sensitivity

Missingness patterns and reasons will be reported. The primary analysis will use observed pre-outcome trials. Complete-participant and training-fold-only imputation analyses will test robustness.

### 8.6 Other sensitivity analyses

The analysis will vary:

- history length;
- RMSE representation;
- the number of hidden or predictive states;
- AP-threshold treatment; and
- other model settings selected within training folds.

---

## 9. Decision rules

### 9.1 Classical model comparison

The five classical model families are ranked by participant-held-out NLL. RMSPE and calibration are secondary measures.

C3 proceeds to quantum encoding only if both conditions are met:

1. C3 passes the prespecified state-stability rule.
2. Its mean held-out NLL is within one standard error of the best benchmark among C0–C4.

The standard error is calculated from paired held-out NLL differences for the same participants.

### 9.2 Reproduction of the classical process

The quantum representation must reproduce C3's output probabilities and state changes with a maximum absolute error below

$$
10^{-8}.
$$

This is a numerical implementation requirement, not evidence of quantum advantage.

### 9.3 Primary memory comparison

The main statistic is

$$
\overline{C}_{\mu}-\overline{C}_q.
$$

Evidence for lower quantum memory requires the entire 95% bootstrap interval, obtained by resampling complete participants, to remain above zero.

### 9.4 Secondary analyses

- The matched-memory approximation comparison is secondary.
- Instruction–retention context analyses are secondary and will use intervals adjusted for the number of comparisons.
- Having fewer parameters does not count as quantum advantage.

If the primary criterion is not met, the conclusion is that the available data do not support a quantum memory reduction at the tested resolution.

---

## 10. Validation and leakage prevention

- Six-fold cross-validation keeps every trial from one participant in the same fold.
- Folds are made as similar as possible in AP status and instruction condition, although exact balance may not be possible.
- Inner participant folds select history length, outcome representation, state number, regularization, similarity thresholds, and C4 settings.
- Outer test participants do not influence those choices.
- The 72 participants, rather than the many repeated trials, are treated as the independent sample units.
- Delayed-retention outcomes are not used to construct predictors available before the delay.
- Recognition status, if measured after practice, is used only for outcomes occurring after it was measured.
- Bootstrap resampling is performed by participant.
- State labels are aligned before states and transitions are compared across resampled fits.
- If AP threshold is analyzed numerically, AP-negative thresholds are treated as above the highest tested output, not as zero.

---

## 11. Required implementation checks

Before interpreting real-data results, the software must verify that:

1. every fitted transition probability is non-negative;
2. transition probabilities sum to one for every current state and input;
3. the fitted C3 transition structure is unifilar;
4. every constructed quantum memory state is normalized;
5. the Gram matrix is Hermitian, positive semidefinite, and has ones on its diagonal;
6. the quantum representation reproduces C3's output probabilities and state changes with maximum absolute error below $10^{-8}$;
7. every density matrix has trace one and non-negative eigenvalues;
8. forcing quantum states to be orthogonal recovers $\overline{C}_q=\overline{C}_{\mu}$;
9. known-history, no-history, orthogonal-state, and single-state simulations give the expected results; and
10. no information from a held-out participant enters model selection, range construction, imputation, or fitting.

---

## 12. Interpretation of possible results

### Quantum predictive memory is lower

The same fitted observable process can be represented with less predictive information when states with similar futures are encoded by overlapping quantum states.

### Classical and quantum predictive memory are equal

The fitted histories require separate predictive states at the tested resolution, step-wise inefficiency is absent, or the available data do not resolve useful overlap.

### C3 is not predictively competitive

Quantum encoding is not used for the primary comparison because the predictive-state model is not an adequate representation of the observed process relative to the other benchmarks.

### C3 is unstable

The thesis reports the range of classical-memory values supported by the data without claiming an exact classical–quantum comparison.

### All models predict poorly

The available variables or participant sample do not support reliable individual prediction. This limitation is reported rather than interpreting memory differences.

None of these outcomes demonstrates or rules out microscopic quantum processes in the brain. The comparison concerns mathematical representations of the fitted behavioral process.

---

## 13. Current research progress

The preliminary pipeline has been tested using simulated data based on the published SINAPs design.

- When the simulated outcome depended on recent history, including that history reduced held-out RMSPE by 36.6%.
- When history did not matter, the reduction was 2.2%.
- Lower quantum memory appeared only when different classical states predicted similar futures.
- No reduction appeared when states were fully separate or when only one state was required.

These are software and model-recovery checks. They are not results from the real SINAPs participants. The real data are still required to determine whether the human motor-learning process supports stable predictive-state reconstruction or quantum predictive-memory reduction.

---

## 14. Reproducibility record

The project will record and version-control:

- analysis code;
- model settings and configuration files;
- random seeds;
- data-processing decisions;
- software versions;
- model-recovery tests;
- non-identifying outputs; and
- the exact tagged software release used for the thesis results.

The public repository should not contain identifiable participant data. A tagged release may be archived on Zenodo so that the exact cited software version receives a permanent DOI.

---

## 15. Core references

- Barnett N, Crutchfield JP. Computational Mechanics of Input-Output Processes: Structured Transformations and the epsilon-Transducer. *Journal of Statistical Physics*. 2015;161:404–451. doi:10.1007/s10955-015-1327-5.
- Thompson J, Garner AJP, Vedral V, Gu M. Using Quantum Theory to Simplify Input-Output Processes. *npj Quantum Information*. 2017;3:6. doi:10.1038/s41534-016-0001-3.
- Perrier ML, Graham KR, Vander Vaart JE, Staines WR, Meehan SK. Sensitivity to Instruction Strategies in Motor Learning Is Predicted by Anterior-Posterior TMS Motor Thresholds. *Brain Sciences*. 2025;15:645. doi:10.3390/brainsci15060645.
- Laird NM, Ware JH. Random-Effects Models for Longitudinal Data. *Biometrics*. 1982;38:963–974. doi:10.2307/2529876.
- Smith MA, Ghazizadeh A, Shadmehr R. Interacting Adaptive Processes with Different Timescales Underlie Short-Term Motor Learning. *PLOS Biology*. 2006;4:e179. doi:10.1371/journal.pbio.0040179.
- Tanaka H, Krakauer JW, Sejnowski TJ. Generalization and Multirate Models of Motor Adaptation. *Neural Computation*. 2012;24:939–966. doi:10.1162/NECO_a_00262.
- Bengio Y, Frasconi P. An Input Output HMM Architecture. *Advances in Neural Information Processing Systems*. 1994;7:427–434.
- Breiman L. Random Forests. *Machine Learning*. 2001;45:5–32. doi:10.1023/A:1010933404324.
- Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794. doi:10.1145/2939672.2939785.
