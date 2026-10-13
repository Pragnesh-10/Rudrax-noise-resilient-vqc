# QFF 2026 Phase 1 — PS-Q3: Noise-Resilient Classifier

## Project overview

This project implements a four-qubit variational quantum classifier (VQC) for
a synthetic binary-classification task. It compares ideal evaluation with noisy
simulation on Processor A and Processor B and evaluates two mitigation
approaches: analytic readout-error correction and zero-noise extrapolation
(ZNE).

The experiment is a simulation-based comparison. It does not demonstrate
performance on physical quantum hardware or on a real-world dataset.

## Dataset and experiment

- Synthetic dataset: 200 samples, 4 features, random seed 42.
- Train/test split: 80% training and 20% held-out test data.
- Held-out test set: 40 examples.
- Optimizer: COBYLA, limited to 100 objective evaluations.
- Transpilation: optimization level 1, transpiler seed 42.
- Simulation shots: 4,096 per sample for each baseline evaluation.
- Noise settings: single-qubit depolarizing probability 0.001, CX depolarizing
  probability 0.005, and symmetric readout bit-flip probability 0.01.

The latest training run reached the 100-evaluation limit and reported
`success=false`. Optimizer convergence has therefore **not been confirmed**.

## Processor configurations

The benchmark uses the topology and noise settings in:

- `processors/processor_A.json` — 5-qubit linear chain.
- `processors/processor_B.json` — 9-qubit graph.

The exact coupling edges are validated in `src/processors.py`. The Phase 1
Master Guide permits participant-defined coupling maps. Our selected graphs
and their limitations are documented in `audit/specification_conflicts.md`.

Processor D has a configuration file, but the current benchmark workflow
compares Processor A and Processor B. Do not interpret the existence of the
Processor D file as evidence that Processor D was benchmarked.

## Mitigation methods and caveats

- **Readout correction:** uses the configured symmetric readout-error
  probability analytically. It does not estimate a calibration matrix from
  calibration circuits.
- **Zero-noise extrapolation (ZNE):** scales simulator noise parameters and
  extrapolates across noise factors 1, 2, and 3. It is not gate-folding-based
  noise amplification. The reported ZNE evaluation uses three times the
  baseline circuit executions.
- **Ideal reference:** uses ideal statevector evaluation and is not a
  shot-matched noisy baseline.

These implementation details should be disclosed when presenting results.

## Latest recorded results

Results below are from the latest saved benchmark run. Accuracy is measured
on 40 held-out examples, so each individual prediction changes accuracy by
2.5 percentage points.

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

In this run, ZNE lowered the Brier score on both processors. It raised
Processor A's observed accuracy from 75.0% to 77.5%; Processor B's observed
accuracy remained at 72.5%. These results are preliminary, not evidence of
statistical significance or broad generalization. Repeated-run summaries use
three runs.

## Setup and execution

Python 3.10 or newer is recommended. Use a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Run the test suite:

```bash
python -m pytest -q
```

The current development environment passed 21 tests. Re-run the tests after
installing dependencies or making changes.

## Notebook

Open `main.ipynb` in Jupyter from the repository root and run its cells. The
notebook invokes `main.py`; it does not implement a separate training pipeline.

## Repository map

- `main.py` — training, evaluation, and result generation.
- `main.ipynb` — notebook entry point.
- `src/problem.py` — dataset and classifier workflow.
- `src/processors.py` — processor configuration validation and noise models.
- `src/routing.py` — transpilation and circuit resource metrics.
- `src/metrics.py` — evaluation and readout correction.
- `src/visualization.py` — benchmark visualization.
- `processors/` — processor JSON configurations.
- `results/tables/` — benchmark tables, predictions, and repeated-run summaries.
- `results/figures/` — generated figures.
- `results/training_summary.json` — recorded optimizer outcome.
- `report.md` — methodology, results, resource comparison, and limitations.
- `audit/specification_conflicts.md` — topology design rationale and benchmark limitations.

## Before submission

- [ ] Verify the selected coupling maps, noise settings, and design rationale against the Phase 1 Master Guide.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python main.py` and verify that the output files are refreshed.
- [ ] Inspect `report.md`, the CSV results, and the generated figure.
- [ ] Confirm the required submission format and repository visibility.
- [ ] Keep the optimizer non-convergence, test-set size, mitigation methods,
      and ZNE execution overhead explicit in the presentation.
