# QFF 2026 Phase 1 — PS-Q3: Noise-Resilient Classifier

## 1. Objective

This project investigates how a four-qubit Variational Quantum Classifier (VQC)
behaves under simulated quantum noise and how error-mitigation techniques affect
classification quality. Two participant-defined processor topologies, Processor A
and Processor B, are compared using the same synthetic binary classification task.

All reported results are from simulation, not physical quantum hardware.

## 2. Dataset and training

- Dataset: synthetic binary classification data generated with scikit-learn.
- Samples: 200.
- Features: 4.
- Dataset generation seed: 42.
- Split: 80% training and 20% held-out test data.
- Training subset: first 80 samples of the training split.
- Model: four-qubit variational quantum classifier.
- Optimizer: COBYLA.
- Maximum objective evaluations: 100.
- Initial training objective: 0.2597988579.
- Final training objective: 0.1996649351.
- Absolute objective reduction: 0.0601339228 (approximately 23.1%).
- Training initialization seed: 42.

A read-only diagnostic run with the same seed reproduced the final objective
and the same 100-evaluation termination status. This supports reproducibility
and shows objective improvement during optimization, but the objective
history was not recorded, so the detailed optimization trajectory cannot be
assessed. The optimizer stopped because it reached its objective-evaluation
limit. Successful convergence was not established. The final objective value
must not be interpreted as proof of a global or local optimum.

The held-out test set contains 40 samples. Because it is small, each test
prediction changes accuracy by 2.5 percentage points. Results should therefore
be interpreted as preliminary rather than evidence of general performance.

## 3. Processor configurations and simulation

The project compares two configured coupling maps:

- Processor A: 5 physical qubits.
- Processor B: 9 physical qubits.

Both configurations use the same configured synthetic noise probabilities:
single-qubit depolarizing probability 0.001, CX depolarizing probability 0.005,
and symmetric readout bit-flip probability 0.01.

Transpilation uses optimization level 1 and seed 42. The noisy baseline uses
4096 shots per test sample. The exact topology and gate configuration are
defined in the processor configuration files; consult those files when
reproducing the experiment.

These processor labels describe simulated configurations, not real quantum
devices or measured hardware fidelities.

## 4. Error mitigation methods

### 4.1 Analytic readout correction

The readout-mitigation method analytically corrects measured probabilities using
the configured symmetric readout error rate. It does not estimate a readout
calibration matrix from separately executed calibration circuits. Its results
depend on the configured error rate representing the simulated readout noise.

### 4.2 Zero-Noise Extrapolation (ZNE)

ZNE evaluates the circuit at noise-scaling factors 1, 2, and 3, then uses
Richardson-style extrapolation. In this implementation, the single-qubit and
CX depolarizing probabilities are scaled while readout error is held fixed.
This is simulator noise-parameter scaling; it is not physical gate folding.

The reported circuit-execution overhead is 3x compared with the noisy baseline.
The CSV records the implementation's ZNE stability diagnostic and the fraction
of raw extrapolated values that required clipping. These diagnostics do not,
by themselves, prove that the mitigation estimate is physically accurate.

## 5. Latest benchmark results

The following values were read from `results/tables/q3_comparison.csv`.

| Processor | Condition | Accuracy | Brier score |
|---|---|---:|---:|
| Ideal reference | Ideal | 0.750 | 0.207742 |
| Processor A | Noisy | 0.750 | 0.210657 |
| Processor A | Readout mitigated | 0.750 | 0.210028 |
| Processor A | ZNE | 0.775 | 0.208720 |
| Processor A | Repeated-run mean | 0.742 | 0.211164 |
| Processor B | Noisy | 0.725 | 0.209538 |
| Processor B | Readout mitigated | 0.725 | 0.208888 |
| Processor B | ZNE | 0.725 | 0.207499 |
| Processor B | Repeated-run mean | 0.742 | 0.210203 |

Lower Brier scores indicate better probability predictions under this metric.
ZNE lowered the Brier score on both processors. It raised Processor A's observed
accuracy from 75.0% to 77.5%, while Processor B remained at 72.5%. Readout
correction lowered the Brier score slightly without changing accuracy.

The repeated-run results are based on three runs. Processor A has accuracy
standard deviation 0.014434 and Brier-score standard deviation
0.000897. Processor B has accuracy standard deviation
0.014434 and Brier-score standard deviation
0.001937. Three runs provide only a preliminary estimate of variability.

## 6. Circuit resource comparison

| Metric | Processor A | Processor B |
|---|---:|---:|
| Mean compiled depth | 23 | 24 |
| Mean CX count | 8 | 7 |
| Explicit SWAP count | 0 | 0 |
| Mean operation count | 64 | 57 |
| Compiled circuit width | 5 | 9 |
| Baseline circuit executions | 40 | 40 |
| Baseline shots | 163,840 | 163,840 |
| ZNE circuit executions | 120 | 120 |
| ZNE shots | 491,520 | 491,520 |

For this circuit and transpilation configuration, Processor A has slightly
lower compiled depth (23 versus 24), while Processor B has fewer CX gates
(7 versus 8) and fewer total operations (57 versus 64). These results describe
the current configuration and do not establish that either processor will
perform better for every circuit.

A zero explicit SWAP count does not establish that routing is free. Interpret
SWAP counts alongside depth, CX count, operation count, and the transpiled
circuit itself.

## 7. Interpretation

1. Processor A achieved 75% noisy test accuracy; Processor B achieved 72.5%.
2. ZNE raised Processor A's observed test accuracy to 77.5%; Processor B
   remained at 72.5%. Readout correction did not change accuracy on either
   processor.
3. Readout correction and ZNE lowered the Brier score relative to the noisy
   baseline on both processors.
4. ZNE used three times as many circuit executions as the noisy baseline.
5. Processor A had slightly lower compiled depth, while Processor B had
   fewer CX gates and fewer total operations in this experiment.
6. The ideal reference is a statevector result, so it is not a shot-matched
   baseline for the noisy simulations.
7. Accuracy is measured on only 40 held-out examples, and repeated-run
   summaries use three runs. These results do not establish statistical
   significance or generalization beyond this dataset and configuration.

## 8. Limitations and reproducibility

- COBYLA reached its 100-evaluation limit (`MAXFUN`) and reported `success=false`; convergence was not confirmed.
- Only 40 held-out samples were evaluated.
- Repeated-run analysis uses three runs only.
- The dataset is synthetic and does not establish performance on a real-world
  application.
- Readout correction uses the configured error rate rather than a measured
  calibration matrix.
- ZNE scales simulator depolarizing-noise parameters; it does not use physical
  gate folding.
- Simulated noise parameters are assumptions, not measured hardware properties.
- The ideal reference does not use the same shot-based evaluation as the noisy
  conditions.
- Results may depend on Qiskit versions, transpilation, random seeds, and the
  supplied processor configurations.

To reproduce the run, activate the project environment, install the dependencies
listed in `requirements.txt`, and run:

    python -m pytest -q
    python main.py

Check the generated CSV and JSON outputs after the run. The recorded benchmark
should not be considered current if the code or configuration changes without
regenerating the results.

## 9. Software environment

Versions recorded for this run:

- Python: 3.14.2
- Qiskit: 2.5.2
- Qiskit Aer: 0.17.2
- NumPy: 2.5.4
- SciPy: 1.18.1
- scikit-learn: 1.9.1

## 10. Conclusion

This experiment provides a reproducible comparison of a four-qubit VQC under
two simulated processor topologies. In the latest run, ZNE improved the Brier
score on both processors and raised Processor A's observed accuracy from 75% to
77.5%; Processor B's accuracy remained at 72.5%. Processor B used fewer CX
gates and total operations, while Processor A had slightly lower compiled
depth. The 3x ZNE execution overhead,
40-example test set, three repeated runs, and optimizer termination limit the
strength of the conclusions. These limitations should be included when
presenting the results.
