# Mathematical specification: quantum predictive modelling of the SINAPs motor-learning data

## 1. Purpose

This document gives the detailed mathematical definitions and implementation checks used by the proposal **Quantum Predictive Modelling of Human Sensorimotor Learning**. The main proposal keeps equations within the text to preserve the required page format; this companion document records the full model.

## 2. Data as a controlled input-output process

For participant `n` and trial `t`:

- `x_nt` is the controlled input/context: session, block, instruction status, waveform type, and participant AP30 context.
- `y_nt` is the observed output. The primary output is standardized trial-level RMSE. Secondary outputs, if available, are temporal lag, velocity error, and within-epoch variability.
- `h_nt = (x_n1:t-1, y_n1:t-1)` is the observed history before trial `t`.

The model estimates the conditional distribution

$$p(y_{nt}\mid h_{nt},x_{nt}).$$

Delayed retention is predicted only from information available at the end of session one.

## 3. Prediction metrics

For held-out observations `i = 1,...,N`,

$$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i(y_i-\hat y_i)^2},$$

and

$$\mathrm{NLL}=-\frac{1}{N}\sum_i \log p(y_i\mid h_i,x_i).$$

Participant-level cross-validation keeps every trial from one participant in a single fold.

## 4. Classical model hierarchy

### C0: mixed-effects autoregression

A representative Gaussian form is

$$y_{nt}=\beta_0+b_n+\beta^\top z_{nt}+\sum_{\ell=1}^{L}a_\ell y_{n,t-\ell}+\epsilon_{nt},$$

where `b_n` is a participant intercept, `z_nt` contains trial, instruction, waveform, and AP30 variables, and `epsilon_nt` is Gaussian residual noise.

### C1: linear state-space model

For latent state vector `s_nt`,

$$s_{n,t+1}=A_{x_{nt}}s_{nt}+B_{x_{nt}}u_{nt}+w_{nt}, \qquad y_{nt}=Cs_{nt}+d_{x_{nt}}+v_{nt}.$$

One to four latent dimensions are considered. `w_nt` and `v_nt` are process and observation noise.

### C2: input-output hidden Markov model

For finite hidden state `S_t`,

$$P(S_{t+1}=j\mid S_t=i,X_t=x)=A^{(x)}_{ij},$$

with Gaussian or discretized emissions `P(Y_t|S_t,X_t)`.

### C3: classical causal-state transducer

In the exact theory, two histories are predictively equivalent when

$$h\sim_\epsilon h' \quad\Longleftrightarrow\quad P(\vec Y\mid \vec X,h)=P(\vec Y\mid \vec X,h')$$

for every allowed future input sequence `\vec X`. Each equivalence class is a causal state `s`. With finite human data, the empirical reconstruction uses pre-specified future targets: next-trial performance, end-of-block performance, and delayed retention. Minimality is therefore claimed only within those horizons and the supported model class. The input-conditioned edge probabilities are

$$T_{ss'}^{y\mid x}=P(Y_t=y,S_{t+1}=s'\mid S_t=s,X_t=x).$$

Because learning is non-stationary, let `\pi_t(s)` denote predictive-state occupancy at trial `t`. Classical predictive memory is

$$C_\mu(t)=-\sum_s\pi_t(s)\log_2\pi_t(s),$$

and the primary summary is the average across the observed protocol, `\bar C_\mu=T^{-1}\sum_t C_\mu(t)`.

## 5. Quantum transducer

For each classical predictive state `s`, input `x`, output `y`, and successor state `s'`, let

$$T_{ss'}^{y\mid x}=P(Y_t=y,S_{t+1}=s'\mid S_t=s,X_t=x).$$

Following the explicit quantum-transducer construction, define

$$|\sigma_s^x\rangle=\sum_{y,s'}\sqrt{T_{ss'}^{y\mid x}}\,|y\rangle|s'\rangle,$$

and combine the allowed inputs as

$$|\sigma_s\rangle=\bigotimes_{x\in\mathcal X}|\sigma_s^x\rangle.$$

The overlap between two predictive memory states is therefore

$$\langle\sigma_r|\sigma_s\rangle=\prod_{x\in\mathcal X}\sum_{y,s'}\sqrt{T_{rs'}^{y\mid x}T_{ss'}^{y\mid x}}.$$

This construction reproduces the fitted classical input-output statistics while allowing states with similar future behaviour to overlap. The Gram matrix `G_rs=<sigma_r|sigma_s>` must be Hermitian, positive semidefinite, and have unit diagonal.

At trial `t`, the quantum memory state is

$$\rho_t=\sum_s\pi_t(s)|\sigma_s\rangle\langle\sigma_s|,$$

and quantum predictive memory is

$$C_q(t)=-\mathrm{Tr}(\rho_t\log_2\rho_t).$$

The primary summary is `\bar C_q=T^{-1}\sum_t C_q(t)` over the same observed protocol used for `\bar C_\mu`. The supported quantum rank at trial `t` is `D_q(t)=rank(rho_t)`; the maximum and trial-averaged ranks are reported.

## 6. Continuous outputs

The exact transducer uses pre-specified adaptive RMSE bins fit only to training participants. Sensitivity analyses use multiple bin counts and a Gaussian-emission quantum model. Binning is never fitted using held-out participants.

## 7. Controls

1. **Orthogonal control:** set `G_ss' = 0` for `s != s'` while retaining the same transition probabilities. Then `C_q` reduces to the classical state entropy.
2. **Matched rank:** compare low-dimensional classical and quantum approximations with the same supported memory rank.
3. **Label permutation:** permute instruction or AP30 labels at participant level.
4. **Model recovery:** simulate data from each fitted model class at observed sequence lengths and noise levels, then refit all classes.

## 8. Pre-specified advantage criteria

A **quantum memory advantage** requires:

- numerical reproduction of the selected classical process output probabilities; and
- a 95% participant-bootstrap interval for `\bar C_\mu-\bar C_q` entirely above zero.

Held-out RMSE and NLL select the credible classical source process before quantum encoding. A secondary matched-rank comparison may test approximate low-memory classical and quantum models. If the primary memory criterion is not met, the classical representation is preferred.

## 9. Uncertainty and leakage prevention

- Six-fold outer cross-validation is stratified by AP30 status and instruction; all trials from a participant remain in one fold.
- History length, state number, emission family, and regularization are tuned only inside the training fold.
- Bootstrap resampling is performed by participant.
- Delayed retention is never used to construct features available at the end of session one.
- State labels are aligned across bootstrap fits before reporting transition uncertainty.

## 10. Required implementation tests

- Transition probabilities are normalized and non-negative.
- The Gram matrix is Hermitian, positive semidefinite, and has unit diagonal.
- The explicit quantum-transducer simulator reproduces fitted output probabilities to numerical tolerance.
- Every trial-specific quantum average state has unit trace and non-negative eigenvalues.
- Orthogonalization recovers `C_mu` within numerical tolerance.
- Synthetic known-state processes are recovered before fitting the SINAPs data.

## 11. Core references

- Barnett N, Crutchfield JP. Computational Mechanics of Input-Output Processes: Structured Transformations and the epsilon-Transducer. *Journal of Statistical Physics*. 2015;161:404-451. doi:10.1007/s10955-015-1327-5.
- Thompson J, Garner AJP, Vedral V, Gu M. Using Quantum Theory to Simplify Input-Output Processes. *npj Quantum Information*. 2017;3:6. doi:10.1038/s41534-016-0001-3.
- Perrier ML, Graham KR, Vander Vaart JE, Staines WR, Meehan SK. Sensitivity to Instruction Strategies in Motor Learning Is Predicted by Anterior-Posterior TMS Motor Thresholds. *Brain Sciences*. 2025;15:645. doi:10.3390/brainsci15060645.
