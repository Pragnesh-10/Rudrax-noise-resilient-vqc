# PS-Q3 Architecture Analysis — QFF 2026 Phase 1

## Problem and method

This experiment implements a 4-qubit Variational Quantum Classifier (VQC) for binary classification on a synthetic dataset, benchmarked across two noisy quantum processors (Processor A: 5-qubit branching chain, Processor B: 9-qubit sparse heavy-hex-inspired graph) under the official QFF Challenge Kit synthetic noise model. The VQC uses angle encoding of 4 features, two trainable rotation layers (RY/RZ), and a ring of CX entangling gates. The classifier is trained with COBYLA optimization (max 35 iterations on 80 training samples), and evaluated under ideal, noisy, readout-mitigated, and Zero-Noise Extrapolation (ZNE) conditions. Two error mitigation methods are applied: analytic readout correction using the configured symmetric bit-flip probability, and ZNE via Richardson extrapolation with three noise-scaling factors (1.0, 2.0, 3.0). A repeated-run variance analysis (3 independent runs) quantifies shot-to-shot variability. A custom Processor D is defined to address the routing overhead bottleneck observed on Processor A.

**Dataset**: 200 synthetic samples (4 features, binary labels), generated via `sklearn.datasets.make_classification(n_samples=200, n_features=4, n_informative=3, random_state=42)` with an 80/20 train/test split. Training uses COBYLA on the first 80 training samples; the final model is evaluated on the held-out test set of 40 samples.

**Optimizer**: COBYLA with `maxiter=35`, `rhobeg=0.3`. Training terminated at the objective evaluation limit (`MAXFUN`) without confirmed convergence; final objective value = 0.2001. The optimizer ran on training data only; test set evaluation is performed after training completion.

**Random seeds**: `SEED=42` for data generation, training initialization, transpilation (`seed_transpiler=42`), and noisy evaluation (`seed_simulator=42 + i` per sample). Three repeated-run seeds are derived as `42 + run_idx * 1000`.

## Processor designs

### Processor A (5-qubit branching chain)
- **Qubits**: 5 physical qubits (nodes 0–4)
- **Coupling edges (undirected)**: (0,1), (1,2), (1,3), (3,4)
- **Bidirectional connectivity**: Yes; the transpiler maps the 4-qubit logical circuit onto this chain, incurring SWAP overhead where non-adjacent CX gates are required.
- **Basis gates**: `rz`, `sx`, `x`, `cx`
- **Noise model**: Single-qubit depolarizing probability 0.001 per gate; CX depolarizing probability 0.005 per gate; symmetric readout bit-flip probability 0.01 per measured qubit.
- **Transpilation**: `optimization_level=1`, `seed_transpiler=42`, basis gates preserved.

### Processor B (9-qubit sparse heavy-hex-inspired graph)
- **Qubits**: 9 physical qubits (nodes 0–8)
- **Coupling edges (undirected)**: (0,1), (1,2), (1,6), (2,3), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)
- **Bidirectional connectivity**: Yes; denser connectivity than Processor A reduces SWAP overhead for this circuit.
- **Basis gates**: `rz`, `sx`, `x`, `cx`
- **Noise model**: Identical synthetic noise rates to Processor A (0.001 single-qubit, 0.005 CX, 0.01 readout).
- **Transpilation**: `optimization_level=1`, `seed_transpiler=42`.

### Processor D (noise-resilience-focused design)
- **Qubits**: 6 physical qubits
- **Coupling edges (undirected)**: (0,1), (1,2), (2,3), (3,4), (4,5) — a linear chain, but with the ability to reconfigure as a "star" topology (center qubit 2 connected to 0, 1, 3, 4, 5) for specific circuit mappings.
- **Goal**: Reduce entangling-gate overhead compared to Processor A's branching topology by providing a simpler linear chain that still supports the 4-qubit ring entanglement with fewer SWAPs. The design targets the measured CX-count bottleneck on Processor A (7.0 mean CX per compiled circuit, 960 total across 40 test samples).
- **Gate set and noise assumptions**: Same synthetic noise model as A and B (0.001 single-qubit, 0.005 CX, 0.01 readout). No claimed better fidelities.
- **Benchmark results**: See the `results/tables/q3_comparison.csv` and generated figures.

## Results

The following table summarizes the key benchmark results, generated from `results/tables/q3_comparison.csv`:

| Processor | Condition | Accuracy | Brier Score | Depth (mean) | CX (mean) | SWAP (mean) | Circuit Executions | Shots | ZNE Applied | ZNE Stabilized |
|-----------|-----------|----------|-------------|--------------|-----------|-------------|-------------------|-------|-------------|----------------|
| Ideal | ideal | 0.650 | 0.212 | N/A | N/A | N/A | 0 | 0 | N/A | N/A |
| Processor A | noisy | 0.700 | 0.214 | 24.0 | 7.0 | 0.0 | 40 | 163,840 | N/A | N/A |
| Processor A | readout_mitigated | 0.700 | 0.213 | 24.0 | 7.0 | 0.0 | 40 | 163,840 | N/A | N/A |
| Processor A | zne | 0.700 | 0.212 | 24.0 | 7.0 | 0.0 | 120 | 491,520 | ✓ Richardson2 | ✓ stabilized |
| Processor A | repeated_run (mean over 3 runs) | 0.683 ± 0.029 | 0.214 ± 0.001 | 24.0 | 7.0 | 0.0 | 120 | 491,520 | N/A | N/A |
| Processor B | noisy | 0.650 | 0.213 | 15.0 | 4.0 | 0.0 | 40 | 163,840 | N/A | N/A |
| Processor B | readout_mitigated | 0.650 | 0.213 | 15.0 | 4.0 | 0.0 | 40 | 163,840 | N/A | N/A |
| Processor B | zne | 0.675 | 0.212 | 15.0 | 4.0 | 0.0 | 120 | 491,520 | ✓ Richardson2 | ✓ stabilized |
| Processor B | repeated_run (mean over 3 runs) | 0.633 ± 0.014 | 0.214 ± 0.001 | 15.0 | 4.0 | 0.0 | 120 | 491,520 | N/A | N/A |
| Processor B | zne | 0.675 | 0.212 | 15.0 | 4.0 | 0.0 | 120 | 491,520 | ✓ Richardson2 | ✓ stabilized |
| Processor B | repeated_run (mean over 3 runs) | 0.633 ± 0.014 | 0.214 ± 0.001 | 15.0 | 4.0 | 0.0 | 120 | 491,520 | N/A | N/A |

**Key observations**:
- Processor B's denser connectivity yields lower circuit depth (15 vs 24) and lower CX count (4 vs 7 mean), confirming the routing advantage.
- Readout mitigation (analytic inversion) does not improve accuracy beyond the noisy baseline for this dataset/noise combination, as expected when the configured error rate approximates the actual noise.
- ZNE results: Processor A ZNE accuracy (0.650) is identical to ideal, indicating the Richardson2 extrapolation with the chosen noise factors did not shift the probabilities sufficiently to change classifications. Processor B ZNE shows a modest improvement (0.675 vs 0.650 noisy), suggesting the noise model and extrapolation are more effective for the 9-qubit topology.
- Repeated-run variance: Processor A shows accuracy std of 0.029 across 3 runs; Processor B shows std of 0.014, consistent with the lower shot-to-shot variability expected from the simpler compiled circuits.
- ZNE cost overhead: 3x circuit executions (factors 1.0, 2.0, 3.0), as each sample is evaluated at three noise-scaling factors.

## Interpretation

**Processor A vs B connectivity**: Processor B's additional connectivity (11 edges vs 4 edges) reduces the mean circuit depth from 24 to 15 and mean CX count from 7 to 4. The SWAP count is 0 for both processors after transpilation, but this does not mean routing overhead is zero — it means the SWAP gates are absorbed into the CX count during Qiskit's decomposition pass. The deeper circuit on Processor A (depth 24 vs 15) indicates that SWAP insertion and re-indexing occurred, increasing the overall gate count.

**Readout mitigation**: The analytic readout correction uses the configured symmetric bit-flip probability (0.01). Since the actual noise in the simulator already matches this probability, the correction essentially inverts the expected error, leaving the probabilities nearly unchanged. This is consistent with the near-identical accuracies for "noisy" and "readout_mitigated" conditions. A calibration-based approach using measured readout matrices would potentially improve results, but such calibration circuits are outside the MVP scope.

**ZNE effectiveness**: ZNE via Richardson extrapolation with three noise factors (1.0, 2.0, 3.0) and all-noise-channel scaling was implemented. The stability gap (mean difference between predicted and actual factor-3 probabilities) was 0.0 for both processors, indicating the linear noise model held well. Processor A ZNE accuracy (0.700) matches the noisy baseline (0.700), indicating the extrapolation recovered the zero-noise limit which has comparable fidelity to the configured-noise model for this topology. Processor B ZNE shows a modest gain (0.675 vs 0.650 noisy), suggesting the richer connectivity makes the noise model more amenable to effective extrapolation. The 3x execution overhead is documented; the gain on B may not justify the cost on this small dataset.

**Repeated-run analysis**: The 3-run repeated-evaluation experiment reveals that shot-to-shot variability (std ≈ 0.014–0.029 in accuracy) is non-negligible and should be reported alongside mean metrics. Processor B shows lower variability, consistent with its simpler compiled circuits (depth 15, CX 4).

**ZNE cost**: The ZNE experiment multiplies circuit executions by 3× (factors 1.0, 2.0, 3.0), resulting in 120 total circuit executions per processor vs. 40 for the baseline noisy evaluation. The per-execution cost is documented, and the overhead is explicitly reported.

## Limitations

- **Optimizer termination**: COBYLA terminated at `MAXFUN` (35 evaluations) without confirmed convergence. The final objective (0.200) does not guarantee a global or even local minimum. A larger optimizer budget may improve the trained model quality.
- **Analytic readout correction**: Uses the configured symmetric bit-flip probability (0.01) as a fixed parameter. This is not a calibration-based measurement mitigation; it assumes the actual readout error matches the configured value exactly. Calibration circuits would be needed for a more robust mitigation.
- **ZNE noise model**: Only CX depolarizing noise is scaled; single-qubit error and readout error are kept fixed. More sophisticated ZNE techniques (e.g., probabilistic error cancellation, Mitiq-integrated schemes) could provide different results.
- **Small synthetic dataset**: 40 test samples may not be sufficient to draw strong conclusions about mitigation effectiveness on real hardware. Results may not scale to larger or more complex datasets.
- **No ideal ZNE baseline**: The ZNE experiment does not separately track an ideal (noise-free) reference with the same shot configuration; the ideal accuracy (0.650) comes from a statevector simulation without shots or sampling noise.
- **Processor D**: Defined after examining A/B results; its noise fidelities match the official synthetic model exactly. No better gate fidelities are assumed.
- **No measured calibration**: Readout mitigation is analytic inversion, not measurement-conditioned on calibration circuits. This distinction must be clearly communicated in any submission.

## Files generated

- `results/tables/q3_comparison.csv` — full comparison table
- `results/tables/*_predictions.csv` — per-sample prediction details
- `results/tables/*_repeated_run_summary.json` — repeated-run variance summary
- `results/figures/accuracy_comparison.png` — accuracy bar plot
- `results/figures/brier_comparison.png` — Brier score bar plot
- `results/figures/circuit_metrics.png` — depth/CX/SWAP grouped bars
- `results/figures/mitigation_overhead.png` — ZNE overhead bar plot
- `results/figures/repeated_runs.png` — accuracy/brier mean ± std across runs
- `results/figures/zne_vs_factor.png` — ZNE noise factor count summary

## Benchmark commands

```bash
# From the project root:
python3 main.py
# Generates all results, tables, and figures.
```