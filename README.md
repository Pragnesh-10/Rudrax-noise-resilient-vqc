# QFF 2026 Phase 1 — PS-Q3: Noise-Resilient Classifier


**Team Name:** RudraX  
**Team Members:**
- Y Naga Pragnesh
- M Sirisha
- Abiramasree K
- Y Siva Shanmukh
**Institution:** SRM University, AP  
**Hackathon:** Qiskit Fall Fest 2026  
**Problem Statement:** PS-Q3 — Noise-Resilient Classification & Error Mitigation

---

## 1. Project Overview

This project implements a four-qubit Variational Quantum Classifier (VQC) for a synthetic binary-classification task. It evaluates the trained classifier under ideal simulation and synthetic device noise, comparing Processor A and Processor B.

Two error-mitigation approaches are investigated:

- **Analytic readout-error correction**
- **Zero-noise extrapolation (ZNE)**

The experiment compares classification accuracy, Brier score, compiled circuit resources, mitigation overhead, and repeated-run variability.

**Scope:** This is a simulation-based study. It does not demonstrate performance on physical quantum hardware or on a real-world dataset.

## 2. Problem Statement and Objectives

The selected challenge is **PS-Q3: Noise-Resilient Classification & Error Mitigation**.

The main objectives are to:

1. Train a four-qubit VQC for binary classification.
2. Compare ideal evaluation with noisy simulation on Processor A and Processor B.
3. Evaluate the effect of readout correction and ZNE.
4. Compare classification quality and compiled circuit resource requirements.
5. Record execution costs, repeated-run variability, and experimental limitations.

## 3. Dataset and Training

The project uses a locally generated synthetic classification dataset.

| Property | Configuration |
|---|---|
| Total samples | 200 |
| Features | 4 |
| Classes | 2 |
| Random seed | 42 |
| Training split | 80% (160 samples) |
| Held-out test split | 20% (40 samples) |
| Classifier | Four-qubit VQC |
| Optimizer | COBYLA |
| Maximum objective evaluations | 100 |
| Transpiler optimization level | 1 |
| Transpiler seed | 42 |
| Baseline simulation shots | 4,096 per sample |

The dataset and preprocessing workflow are implemented in `src/problem.py`. The training and benchmark workflow is orchestrated by `main.py`.

### Training status

The recorded training run reached its 100-evaluation limit and reported `success=false`.

| Training metric | Value |
|---|---:|
| Initial objective | 0.2597988579 |
| Final objective | 0.1996649351 |
| Absolute objective reduction | 0.0601339228 |
| Relative objective reduction | Approximately 23.1% |
| Objective evaluations | 100 |

A diagnostic rerun reproduced the final objective under the same seed and evaluation budget. This supports reproducibility of the recorded optimization result, but **convergence was not confirmed**. The objective reduction does not establish that a local or global optimum was reached.

The held-out test set contains only 40 examples. One additional correct prediction changes accuracy by 2.5 percentage points. Results should therefore be interpreted as preliminary rather than evidence of statistical significance or broad generalization.

## 4. Processor Configurations

The benchmark uses the configurations defined in the repository.

| Property | Processor A | Processor B |
|---|---|---|
| Configured qubits | 5 | 9 |
| Selected topology | Linear chain | Participant-defined graph |
| Configuration file | `processors/processor_A.json` | `processors/processor_B.json` |

The coupling graphs and noise settings are defined in the processor configuration files and validated by `src/processors.py`. The selected topologies are used for the current Processor A/B benchmark.

The results describe these particular configurations and compilation settings. They do not establish that one architecture is universally superior to the other.

A Processor D configuration file is present in the repository, but **the current benchmark results cover Processor A and Processor B only**. The presence of a configuration file does not mean that Processor D was benchmarked.

### Synthetic noise settings

| Noise component | Configured probability |
|---|---:|
| Single-qubit depolarizing noise | 0.001 |
| CX depolarizing noise | 0.005 |
| Symmetric readout bit-flip noise | 0.01 |

These values represent synthetic simulation settings, not measured physical-hardware calibration data.

## 5. Error Mitigation Methods

### 5.1 Analytic readout-error correction

The readout-correction implementation uses the configured symmetric readout-error probability to correct measured probabilities analytically.

- It does not estimate a calibration matrix from separate calibration circuits.
- It uses the same baseline circuit-execution count in this implementation.
- Corrected probabilities are constrained to valid probability bounds.

This method should not be described as measured calibration-matrix mitigation.

### 5.2 Zero-noise extrapolation (ZNE)

The ZNE implementation scales simulator noise parameters at factors 1, 2, and 3, then extrapolates toward the zero-noise limit.

- The implementation scales simulation noise parameters.
- It does not use gate-folding-based noise amplification.
- The reported evaluation uses three times the baseline circuit executions.

The results are specific to this simulation-based noise-scaling procedure and should not be interpreted as evidence of equivalent improvements on physical hardware.

### 5.3 Ideal reference

The ideal reference uses statevector evaluation. It is not a shot-matched noisy baseline, which is an important distinction when interpreting ideal-versus-noisy differences.

## 6. Latest Recorded Classification Results

The following results are from the saved benchmark outputs. Accuracy is measured on 40 held-out examples.

| Processor / condition | Accuracy | Brier score |
|---|---:|---:|
| Ideal reference | 75.0% | 0.207742 |
| Processor A — noisy | 75.0% | 0.210657 |
| Processor A — readout corrected | 75.0% | 0.210028 |
| Processor A — ZNE | 77.5% | 0.208720 |
| Processor A — repeated-run mean | 74.17% | 0.211164 |
| Processor B — noisy | 72.5% | 0.209538 |
| Processor B — readout corrected | 72.5% | 0.208888 |
| Processor B — ZNE | 72.5% | 0.207499 |
| Processor B — repeated-run mean | 74.17% | 0.210203 |

A lower Brier score indicates better probabilistic prediction quality.

### Observations

- ZNE lowered the Brier score relative to the noisy baseline on both processors.
- Processor A's observed accuracy increased from 75.0% to 77.5% after ZNE.
- Processor B's observed accuracy remained at 72.5% after ZNE.
- Readout correction lowered the Brier score on both processors without changing observed accuracy.
- Repeated-run summaries use three runs per processor.

The accuracy increase for Processor A represents one additional correct prediction out of 40. These results do not establish statistical significance or guarantee similar improvements on other datasets or hardware.

## 7. Circuit Resources and Execution Cost

The saved benchmark reports the following compiled circuit metrics.

| Resource metric | Processor A | Processor B |
|---|---:|---:|
| Mean circuit depth | 23 | 24 |
| Mean CX count | 8 | 7 |
| Mean explicit SWAP count | 0 | 0 |
| Mean operation count | 64 | 57 |
| Compiled circuit width | 5 | 9 |

Processor A has slightly lower compiled depth in this benchmark. Processor B has fewer CX gates and fewer total operations, while using a wider compiled circuit.

These values describe the selected circuits and topologies. They should not be interpreted as a universal ranking of processor architectures. An explicit SWAP count of zero alone does not establish that routing has no overhead, because routing operations may be decomposed into other gates.

### Mitigation execution overhead

| Evaluation condition | Circuit executions | Shots |
|---|---:|---:|
| Noisy baseline | 40 | 163,840 |
| Analytic readout correction | 40 | 163,840 |
| ZNE | 120 | 491,520 |

Readout correction is analytic and does not add separate calibration-circuit executions in this implementation. ZNE evaluates three noise-scaling factors, giving three times the baseline executions and shots.

These figures describe the implemented simulation workflow. They are not measurements of physical-hardware runtime or monetary cost.

## 8. Setup and Execution

### Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- `pytest` for the test suite

### macOS / Linux

```bash
git clone https://github.com/Pragnesh-10/Rudrax-NISQ.git
cd Rudrax-NISQ

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest

python main.py
```

### Windows PowerShell

```powershell
git clone https://github.com/Pragnesh-10/Rudrax-NISQ.git
cd Rudrax-NISQ

py -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest

python main.py
```

### Run tests

```bash
python -m pytest -q
```

The development environment previously passed all 21 tests. Run the test suite again after installing dependencies or making changes.

### Notebook

Open `main.ipynb` in Jupyter from the repository root and run its cells. The notebook invokes `main.py`; it does not implement a separate training pipeline.

### Reproducibility notes

The experiment uses fixed seeds for dataset generation, training initialization, and transpilation. Reproducibility can also depend on installed dependency versions and simulator behavior. The optimizer's evaluation limit and unconfirmed convergence should be considered when interpreting results.

## 9. Repository Structure

```text
Rudrax-NISQ/
├── main.py
├── main.ipynb
├── requirements.txt
├── README.md
├── report.md
├── processors/
│   ├── processor_A.json
│   ├── processor_B.json
│   └── processor_D.json
├── src/
│   ├── problem.py
│   ├── processors.py
│   ├── routing.py
│   ├── metrics.py
│   └── visualization.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── training_summary.json
└── tests/
    └── test_core.py
```

### Main files

- `main.py` — training, evaluation, benchmarking, and result generation.
- `main.ipynb` — notebook entry point.
- `src/problem.py` — dataset and classifier workflow.
- `src/processors.py` — processor configuration validation and noise models.
- `src/routing.py` — transpilation and circuit resource metrics.
- `src/metrics.py` — evaluation metrics and readout correction.
- `src/visualization.py` — benchmark visualizations.
- `processors/` — synthetic processor configurations.
- `results/tables/` — benchmark tables, predictions, and repeated-run summaries.
- `results/figures/` — generated figures.
- `results/training_summary.json` — recorded optimizer outcome.
- `report.md` — detailed methodology, results, resource comparison, and limitations.
- `tests/` — automated tests.

## 10. Limitations and Future Work

### Current limitations

1. The optimizer reached its evaluation limit without confirmed convergence.
2. The held-out test set is small and provides coarse-grained accuracy estimates.
3. Noise parameters are synthetic rather than hardware-calibrated.
4. Readout correction uses a configured probability rather than a measured calibration matrix.
5. ZNE scales simulator noise parameters rather than folding gates.
6. The ideal reference uses statevector evaluation rather than a shot-matched baseline.
7. Three repeated runs provide only a preliminary estimate of variability.
8. The current benchmark compares Processor A and Processor B; Processor D is not included in the recorded results.

### Future work

- Increase the training evaluation budget and record objective history.
- Evaluate performance across multiple independent dataset seeds.
- Run more repetitions and use stronger statistical analysis.
- Evaluate measured readout calibration and additional mitigation techniques.
- Benchmark additional processor configurations if required by the applicable phase instructions.
- Evaluate the workflow on real hardware if access and scope permit.

## 11. Conclusion

This project presents a simulation-based comparison of a four-qubit VQC under ideal and synthetic noisy conditions on Processor A and Processor B. It evaluates analytic readout correction and ZNE using classification metrics, compiled circuit resources, execution counts, and repeated-run summaries.

ZNE improved the Brier score on both processors in the recorded benchmark, while the observed accuracy response differed between Processor A and Processor B. These findings are preliminary and must be interpreted alongside the small test set, synthetic noise assumptions, and unconfirmed optimizer convergence.

The project provides an experimental study of noise mitigation and processor architecture trade-offs for QFF 2026 Phase 1 PS-Q3.
