# Mathematical guide for the SINAPs quantum predictive modelling proposal

> **Viewing note:** GitHub renders the equations in the normal file view or Preview. The Raw view and some generic text viewers show the LaTeX source instead.

## 1. Purpose of this guide

This document explains the mathematics used in the proposal **Quantum Predictive Modelling of Human Sensorimotor Learning**. 

The central question is:

> What is the smallest amount of information about a participant's past practice that is needed to predict future learning and delayed retention?

The project first finds the strongest supported classical predictor. It then builds a quantum representation of the same observable process and asks whether that representation needs less predictive memory.

The word **memory** in this document means information stored by a mathematical predictor. It does not mean conscious recall or a biological memory trace.

---

## Quick glossary

- **$AP_{30}$**: a motor-threshold measure obtained with a 30-microsecond anterior-posterior TMS pulse.
- **TMS**: transcranial magnetic stimulation.
- **RMSE**: root mean square error, a measure of the size of prediction errors.
- **NLL**: negative log-likelihood, a measure of how much probability a model assigns to observed outcomes.
- **Hidden state**: an internal model variable that is inferred rather than directly measured.
- **Predictive state**: a summary of past observations used to predict future observations.
- **Classical predictive memory**: information stored in fully distinguishable classical states, measured in bits.
- **Quantum predictive memory**: information stored in possibly overlapping quantum states, measured in qubits.
- **Cross-validation**: fitting a model on one set of participants and testing it on different participants.
- **Bootstrap**: repeated resampling used to estimate uncertainty.

---

## 2. Data and notation

The dataset contains repeated trials from several participants.

For participant $n$ and trial $t$:

- $n$ labels the participant.
- $t$ labels the trial number.
- $x_{nt}$ is the input or context known before the trial. It may include session, block, instruction condition, waveform type, and AP-sensitive motor-cortical response category.
- $y_{nt}$ is the observed outcome. The main outcome is standardized root mean square error, or RMSE.
- $h_{nt}$ is the complete observed history before trial $t$.

The history is written as

$$
h_{nt}=(x_{n1:t-1},y_{n1:t-1}).
$$

Here:

- $x_{n1:t-1}$ means all inputs for participant $n$ from trial 1 through trial $t-1$.
- $y_{n1:t-1}$ means all observed outcomes from trial 1 through trial $t-1$.
- The notation $1:t-1$ means the sequence of indices from 1 to $t-1$.

The main predictive quantity is

$$
p(y_{nt}\mid h_{nt},x_{nt}).
$$

This is the probability distribution for the next outcome $y_{nt}$, given the participant's earlier history $h_{nt}$ and the current input $x_{nt}$. The vertical bar means "given" or "conditional on."

Delayed retention is predicted using only information available at the end of the first session. Data from the later retention test are never used to construct earlier predictors.

---

## 3. Prediction measures

All models are evaluated on participants who were not used to fit the model.

### 3.1 Root mean square error

Prediction error is measured by

$$
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}.
$$

Definitions:

- $N$ is the number of held-out observations.
- $i$ labels one held-out observation.
- $y_i$ is the observed value.
- $\hat y_i$ is the model's predicted value.
- $y_i-\hat y_i$ is the prediction error.
- Squaring makes positive and negative errors contribute equally.
- The square root returns the result to the same units as the outcome.

Lower RMSE means more accurate numerical predictions.

### 3.2 Negative log-likelihood

The full predictive distribution is evaluated by

$$
\mathrm{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log p(y_i\mid h_i,x_i).
$$

Definitions:

- $p(y_i\mid h_i,x_i)$ is the probability density or probability assigned to the observed outcome $y_i$.
- $h_i$ is the history available before observation $i$.
- $x_i$ is the current input or context.
- $\log$ is the natural logarithm unless otherwise stated.
- The minus sign makes better predictions produce smaller values.

Lower NLL means that the model assigns more probability to outcomes that actually occur. NLL is important because it evaluates both the predicted mean and the predicted uncertainty.

### 3.3 Calibration

Calibration asks whether predicted probabilities match observed frequencies. For example, among outcomes assigned a probability near 0.7, an event should occur about 70 percent of the time. Calibration will be reported as a secondary measure.

---

## 4. Classical model hierarchy

The project compares four classical models. Each model uses only information that would be available before the outcome being predicted.

### 4.1 Model C0: mixed-effects regression with history

A representative form is

$$
y_{nt}=\beta_0+b_n+\boldsymbol{\beta}^{\mathsf T}\mathbf z_{nt}+\sum_{\ell=1}^{L}a_{\ell}y_{n,t-\ell}+\epsilon_{nt}.
$$

Definitions:

- $y_{nt}$ is the observed outcome for participant $n$ on trial $t$.
- $\beta_0$ is the overall intercept, or average starting level.
- $b_n$ is a participant-specific intercept that captures stable differences between participants.
- $\mathbf z_{nt}$ is a vector containing known predictors, such as trial number, instruction, waveform type, and AP context.
- $\boldsymbol{\beta}$ is the vector of coefficients associated with $\mathbf z_{nt}$.
- The superscript $\mathsf T$ means transpose, so $\boldsymbol{\beta}^{\mathsf T}\mathbf z_{nt}$ is the weighted sum of the predictors.
- $L$ is the number of earlier trials included.
- $\ell$ is the lag, meaning how many trials back the model looks.
- $a_{\ell}$ is the coefficient for the outcome observed $\ell$ trials earlier.
- $y_{n,t-\ell}$ is the earlier outcome at that lag.
- $\epsilon_{nt}$ is unexplained residual noise, usually assumed to follow a normal distribution with mean zero.

This model tests whether recent trial history improves prediction beyond current context and stable participant differences.

### 4.2 Model C1: linear state-space model

A state-space model introduces hidden variables that summarize learning processes:

$$
\mathbf s_{n,t+1}=A_{x_{nt}}\mathbf s_{nt}+B_{x_{nt}}\mathbf u_{nt}+\mathbf w_{nt},
$$

$$
y_{nt}=C\mathbf s_{nt}+d_{x_{nt}}+v_{nt}.
$$

Definitions for the first equation:

- $\mathbf s_{nt}$ is a vector of hidden learning variables for participant $n$ at trial $t$.
- $\mathbf s_{n,t+1}$ is the hidden state on the next trial.
- $A_{x_{nt}}$ is the state-transition matrix under input $x_{nt}$. It determines how much of the previous hidden state is retained and how its components interact.
- $B_{x_{nt}}$ maps the known trial input $\mathbf u_{nt}$ into changes in the hidden state.
- $\mathbf u_{nt}$ is a vector of known trial-level inputs.
- $\mathbf w_{nt}$ is process noise, which represents unobserved variation in how the hidden state changes.

Definitions for the second equation:

- $C$ maps the hidden state to the observed outcome.
- $d_{x_{nt}}$ is an input-dependent shift in the observed outcome.
- $v_{nt}$ is observation noise.

Models with one to four hidden dimensions will be tested. The number of hidden variables is selected using training participants only.

### 4.3 Model C2: input-output hidden Markov model

This model assumes a small set of distinct hidden conditions. Its transition rule is

$$
P(S_{t+1}=j\mid S_t=i,X_t=x)=A_{ij}^{(x)}.
$$

Definitions:

- $S_t$ is the hidden state at trial $t$.
- $i$ is the current hidden-state label.
- $j$ is the next hidden-state label.
- $X_t=x$ means that the input at trial $t$ has value $x$.
- $A_{ij}^{(x)}$ is the probability of moving from state $i$ to state $j$ under input $x$.
- $P(\cdot)$ denotes probability.

Each state also has an output distribution, written as $P(Y_t\mid S_t,X_t)$. This distribution describes the RMSE values expected from each hidden state under each input.

### 4.4 Model C3: classical predictive-state model

The exact theory groups two histories together when they give the same probabilities for all allowed future outputs under all allowed future inputs:

$$
h\sim_{\epsilon}h'
\quad\Longleftrightarrow\quad
P(\vec Y\mid \vec X,h)=P(\vec Y\mid \vec X,h').
$$

Definitions:

- $h$ and $h'$ are two different observed histories.
- $\sim_{\epsilon}$ means "predictively equivalent."
- $\Longleftrightarrow$ means that the statement on either side is true exactly when the other is true.
- $\vec X$ is a possible future sequence of inputs.
- $\vec Y$ is a possible future sequence of outputs.
- $P(\vec Y\mid \vec X,h)$ is the probability distribution over future outputs given future inputs and history $h$.

All histories in one equivalence class form one **causal state**. In finite human data, exact equality over an unlimited future cannot be established. The empirical model therefore uses pre-specified outcomes: next-trial RMSE, end-of-block performance, and delayed retention. Any minimality claim is limited to these outcomes, the observed protocol, and the supported model class.

The input-conditioned transition probabilities are

$$
T_{ss'}^{y\mid x}=P(Y_t=y,S_{t+1}=s'\mid S_t=s,X_t=x).
$$

Definitions:

- $s$ is the current predictive state.
- $s'$ is the next predictive state.
- $x$ is the current input.
- $y$ is the observed output category.
- $T_{ss'}^{y\mid x}$ is the probability of observing output $y$ and moving to state $s'$, given current state $s$ and input $x$.

---

## 5. Classical predictive memory

Let $\pi_t(s)$ be the probability that the process occupies predictive state $s$ at trial $t$. Classical predictive memory is

$$
C_{\mu}(t)=-\sum_s\pi_t(s)\log_2\pi_t(s).
$$

Definitions:

- $C_{\mu}(t)$ is the classical predictive memory at trial $t$.
- The subscript $\mu$ is conventional notation for classical statistical complexity.
- $s$ runs over all classical predictive states.
- $\pi_t(s)$ is the probability of state $s$ at trial $t$.
- $\log_2$ is the logarithm with base 2.
- The unit is bits.

The quantity is larger when the model must keep track of more distinct states and when those states are used with similar frequency.

Because learning changes across the experiment, memory is calculated at each trial. The main summary is

$$
\overline{C}_{\mu}=\frac{1}{T}\sum_{t=1}^{T}C_{\mu}(t).
$$

Definitions:

- $\overline{C}_{\mu}$ is the classical memory averaged across the protocol.
- $T$ is the total number of trial positions included.
- The overline means an average.

---

## 6. Quantum predictive representation

### 6.1 Quantum memory states

Each classical predictive state $s$ is represented by a normalized quantum state

$$\lvert\sigma_s\rangle.$$

Definitions:

- $\lvert\sigma_s\rangle$ is read as "ket sigma s."
- It is a vector in a mathematical state space.
- Normalized means that its total probability is one.
- The label $s$ links it to one classical predictive state.

The use of quantum states is a mathematical representation. It does not assume that neurons or motor cortex perform the proposed calculation physically.

### 6.2 State construction for each input

For classical state $s$ and input $x$, define

$$\lvert\sigma_s^x\rangle = \sum_{y,s'}\sqrt{T_{ss'}^{y\mid x}}\,\lvert y\rangle\lvert s'\rangle.$$

Definitions:

- $\lvert\sigma_s^x\rangle$ is the quantum state associated with classical state $s$ under input $x$.
- The sum runs over every possible output $y$ and successor state $s'$.
- $T_{ss'}^{y\mid x}$ is the corresponding classical transition probability.
- The square root converts a probability into a quantum amplitude.
- $\lvert y\rangle$ is a basis vector representing output $y$.
- $\lvert s'\rangle$ is a basis vector representing next state $s'$.
- Writing two kets together denotes a tensor product, which combines the output and next-state labels in one state space.

All allowed inputs are combined as

$$\lvert\sigma_s\rangle=\bigotimes_{x\in\mathcal X}\lvert\sigma_s^x\rangle.$$

Definitions:

- $\mathcal X$ is the set of allowed inputs.
- $x\in\mathcal X$ means that $x$ is one member of that set.
- $\bigotimes$ means the tensor product over all inputs.
- The result is one quantum memory state that contains the predictive response to every allowed input.

### 6.3 Overlap between quantum states

The overlap between states $r$ and $s$ is

$$\langle\sigma_r\mid\sigma_s\rangle=\prod_{x\in\mathcal X}\sum_{y,s'}\sqrt{T_{rs'}^{y\mid x}T_{ss'}^{y\mid x}}.$$

Definitions:

- $\langle\sigma_r\mid\sigma_s\rangle$ is the inner product between two quantum states.
- $r$ and $s$ label two different classical predictive states.
- $\prod$ means multiply the input-specific overlaps across all inputs.
- A value near 1 means the states have very similar predicted futures.
- A value near 0 means the states are nearly fully distinguishable.

The matrix containing all pairwise overlaps is the **Gram matrix**:

$$G_{rs}=\langle\sigma_r\mid\sigma_s\rangle.$$

Definitions:

- $G$ is the Gram matrix.
- $G_{rs}$ is the entry in row $r$, column $s$.
- The matrix must be Hermitian, meaning $G_{rs}=G_{sr}^{*}$, where $*$ denotes complex conjugation.
- It must be positive semidefinite, meaning it cannot assign a negative squared length to any vector.
- Its diagonal entries must equal one because every state is normalized.

### 6.4 Average quantum state

At trial $t$, the average quantum memory state is

$$\rho_t=\sum_s\pi_t(s)\lvert\sigma_s\rangle\langle\sigma_s\rvert.$$

Definitions:

- $\rho_t$ is a density matrix. It represents the quantum memory before trial $t$ when the exact predictive state is uncertain.
- $\pi_t(s)$ is the probability of classical predictive state $s$ at trial $t$.
- $\lvert\sigma_s\rangle\langle\sigma_s\rvert$ is the matrix associated with quantum state $s$.
- The sum mixes the state matrices according to their occupancy probabilities.

A valid density matrix must have trace one and non-negative eigenvalues. The trace is the sum of its diagonal entries. An eigenvalue is a weight associated with one independent direction in the state space.

### 6.5 Quantum predictive memory

Quantum memory is the von Neumann entropy

$$C_q(t)=-\mathrm{Tr}\!\left[\rho_t\log_2\rho_t\right].$$

Definitions:

* $C_q(t)$ is quantum predictive memory at trial $t$.
* $q$ denotes the quantum representation.
* $\mathrm{Tr}$ is the matrix trace.
* $\log_2\rho_t$ is the base-2 matrix logarithm.
* The unit is qubits.

In practice, if $\lambda_k(t)$ are the eigenvalues of $\rho_t$, the same quantity is

$$C_q(t)=-\sum_k\lambda_k(t)\log_2\lambda_k(t).$$

Definitions:

- $k$ labels the eigenvalues.
- $\lambda_k(t)$ is the $k$-th eigenvalue at trial $t$.
- Eigenvalues equal to zero contribute zero by continuity.

The average quantum memory is

$$\overline{C}_q=\frac{1}{T}\sum_{t=1}^{T}C_q(t).$$

It is averaged over the same trial positions as $\overline{C}_{\mu}$.

---

## 7. Continuous outcomes

The exact transducer construction requires a finite set of output categories. Trial-level RMSE will therefore be divided into pre-specified adaptive bins. Bin boundaries are fitted using training participants only.

Sensitivity analyses will:

- vary the number of RMSE bins;
- use alternative binning rules;
- fit Gaussian output distributions when possible;
- confirm that conclusions are not produced by one arbitrary output resolution.

A held-out participant never influences the bin boundaries or model settings used to predict that participant.

---

## 8. Controls

### 8.1 Fully distinct-state control

All off-diagonal overlaps are set to zero:

$$G_{ss'}=0\quad\text{for }s\neq s'.$$

Definitions:

- $s\neq s'$ means the two state labels are different.
- Off-diagonal entries compare different states.

This control keeps the fitted transition probabilities but removes quantum overlap. It must recover the classical predictive memory.

### 8.2 Matched-memory or matched-rank comparison

Low-dimensional classical and quantum approximations will be compared under the same supported memory size. Rank means the number of independent directions needed to represent a state mixture. This analysis is secondary because equal rank does not guarantee equal model flexibility.

### 8.3 Label permutation

Instruction or AP labels will be randomly reassigned at the participant level while keeping each participant's trial order intact. This tests whether those labels add predictive information beyond chance.

### 8.4 Model recovery

Synthetic data will be generated from each candidate model class. The full analysis will then be rerun to test whether it identifies the process that generated the data.

---

## 9. Decision rules

A primary quantum memory advantage requires both conditions below:

1. The quantum representation reproduces the selected classical process's output probabilities within numerical tolerance.
2. The 95 percent participant-bootstrap confidence interval for

$$\overline{C}_{\mu}-\overline{C}_q$$

lies entirely above zero.

Definitions:

- $\overline{C}_{\mu}-\overline{C}_q$ is the average reduction in predictive memory.
- A positive value means the quantum representation uses less memory.
- A participant bootstrap repeatedly resamples whole participants, not individual trials.
- A 95 percent confidence interval summarizes uncertainty across these resamples.

Held-out NLL and RMSE are used first to identify a credible classical source process. A memory reduction in a poorly predicting model is not treated as an advantage.

If the primary criterion is not met, the conclusion is that the available data do not support a quantum memory reduction at the tested resolution.

---

## 10. Validation and leakage prevention

- Every trial from one participant remains in the same cross-validation fold.
- History length, state number, output bins, and regularization are chosen using training participants only.
- Delayed-retention outcomes are not used to create predictors available before the delay.
- Bootstrap resampling is performed by participant.
- State labels are aligned across resampled fits before transitions are compared.
- AP-negative thresholds are treated as right-censored above the maximum measured value, not as zero.

Right-censored means that the true threshold is known only to exceed the largest value that could be measured.

---

## 11. Required implementation checks

Before interpreting real-data results, the software must pass the following checks:

1. Every transition probability is non-negative.
2. Transition probabilities sum to one for every current state and input.
3. The Gram matrix is Hermitian, positive semidefinite, and has ones on its diagonal.
4. The quantum simulator reproduces the fitted classical output probabilities within numerical tolerance.
5. Every density matrix has trace one and non-negative eigenvalues.
6. Forcing the quantum states to be fully distinct recovers classical predictive memory.
7. Synthetic processes with known history dependence and known state overlap are recovered correctly.
8. No-history, fully distinct-state, and single-state controls produce no artificial advantage.

---

## 12. Interpretation of possible results

### Quantum memory is lower

The same observable learning process can be represented using less predictive information when similar futures are encoded by overlapping states.

### Classical and quantum memory are equal

The measured learning histories require fully distinguishable predictive states, or the available data do not resolve useful overlap.

### The classical model predicts better

The quantum approximation is not justified for these data and the classical representation is preferred.

### Both models predict poorly

The available variables or sample size are insufficient for reliable participant-level prediction. The project should then report this limitation rather than interpret memory differences.

---

## 13. Core references

- Barnett N, Crutchfield JP. Computational Mechanics of Input-Output Processes: Structured Transformations and the epsilon-Transducer. *Journal of Statistical Physics*. 2015;161:404-451. doi:10.1007/s10955-015-1327-5.
- Thompson J, Garner AJP, Vedral V, Gu M. Using Quantum Theory to Simplify Input-Output Processes. *npj Quantum Information*. 2017;3:6. doi:10.1038/s41534-016-0001-3.
- Perrier ML, Graham KR, Vander Vaart JE, Staines WR, Meehan SK. Sensitivity to Instruction Strategies in Motor Learning Is Predicted by Anterior-Posterior TMS Motor Thresholds. *Brain Sciences*. 2025;15:645. doi:10.3390/brainsci15060645.
