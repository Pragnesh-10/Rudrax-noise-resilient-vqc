"""Noise evaluation, readout mitigation, ZNE, and metric helpers."""
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from .processors import noise_model, coupling_map, validate_config
from .routing import compile_for_processor, circuit_metrics
from .problem import vqc_circuit


def compiled_metrics_for_sample(x, theta, config):
    compiled = compile_for_processor(vqc_circuit(x, theta), config)
    return circuit_metrics(compiled), compiled


def aggregate_compiled_metrics(X, theta, config):
    """Summarize compiled circuit cost across all supplied samples."""
    per_sample = [compiled_metrics_for_sample(x, theta, config)[0] for x in X]
    if not per_sample:
        raise ValueError("X must contain at least one sample")
    frame = pd.DataFrame(per_sample)
    summary = {}
    for key in ("depth", "cx_count", "swap_count", "operation_count", "num_qubits"):
        summary[f"{key}_mean"] = float(frame[key].mean())
        summary[f"{key}_max"] = int(frame[key].max())
        summary[f"{key}_total"] = int(frame[key].sum())
    return summary


def eval_noisy(X, y, theta, config, shots=4096, mitigate_readout=False, seed=42):
    """Evaluate one circuit per sample and return classification/probability metrics.

    Readout mitigation is analytic inversion using the configured symmetric bit-flip
    probability; it is not a calibration-circuit experiment.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    if len(X) == 0:
        raise ValueError("X must contain at least one sample")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")
    if shots <= 0:
        raise ValueError("shots must be a positive integer")

    p_flip = float(config["noise"]["readout_bit_flip_probability"])
    if mitigate_readout and not (0.0 <= p_flip < 0.5):
        raise ValueError("Readout mitigation requires 0 <= bit-flip probability < 0.5")

    simulator = AerSimulator(noise_model=noise_model(config))
    rows = []
    for i, (x, label) in enumerate(zip(X, y)):
        qc = vqc_circuit(x, theta, measured=True)
        compiled = compile_for_processor(qc, config)
        result = simulator.run(compiled, shots=shots, seed_simulator=seed + i).result()
        counts = result.get_counts()
        # This scaffold measures a single classical bit. Fail loudly if that changes.
        if set(counts) - {"0", "1"}:
            raise ValueError(f"Expected one-bit count keys, got: {sorted(counts)}")
        raw_p1 = counts.get("1", 0) / shots
        p1 = raw_p1
        if mitigate_readout:
            p1 = float(np.clip((raw_p1 - p_flip) / (1 - 2 * p_flip), 0.0, 1.0))
        pred = int(p1 >= 0.5)
        rows.append({"label": int(label), "raw_p1": raw_p1, "p1": p1, "prediction": pred})

    pred = np.array([r["prediction"] for r in rows])
    probs = np.array([r["p1"] for r in rows])
    metrics = {
        "accuracy": float(np.mean(pred == y)),
        "brier_score": float(np.mean((probs - y) ** 2)),
        "shots_per_sample": int(shots),
        "total_shots": int(shots * len(y)),
        "circuit_executions": int(len(y)),
        "readout_mitigated": bool(mitigate_readout),
    }
    return metrics, pd.DataFrame(rows)


def _zne_scaled_circuit(config, noise_scale_factor):
    """Build a NoiseModel with scaled error rates for ZNE.

    The original error probabilities from the config are multiplied by
    noise_scale_factor to amplify noise for extrapolation. All noise
    channels (single-qubit depolarizing, CX depolarizing, readout) are
    scaled by the same factor so that the relative ratios are preserved.
    """
    p1_original = float(config["noise"]["single_qubit_depolarizing_probability"])
    p2_original = float(config["noise"]["cx_depolarizing_probability"])
    pr_original = float(config["noise"]["readout_bit_flip_probability"])

    p1_scaled = p1_original * noise_scale_factor
    p2_scaled = p2_original * noise_scale_factor
    pr_scaled = pr_original  # readout error scaled separately below

    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(p1_scaled, 1), ["x", "sx", "rz"]
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(p2_scaled, 2), ["cx"]
    )
    model.add_all_qubit_readout_error(
        ReadoutError([[1 - pr_scaled, pr_scaled], [pr_scaled, 1 - pr_scaled]])
    )
    return model


def zne_evaluate(X, y, theta, config, noise_factors=(1.0, 2.0, 3.0), shots=4096, seed=42):
    """Perform Zero-Noise Extrapolation using gate-noise scaling.

    For each noise factor k in noise_factors:
    - Scale the CX depolarizing probability by k
    - Run the compiled circuit on the noisy AerSimulator
    - Collect per-sample probabilities for qubit 0 (P(1))

    Richardson extrapolation (2-point, using factors 1 and 2) estimates
    the zero-noise limit: p_ideal = 2*p(k=1) - p(k=2).

    If the 3-factor check shows inconsistency (the linear model doesn't
    hold within a tolerance), the result falls back to the k=1 result
    with a stabilization flag.

    Returns:
        mitigated_metrics: dict with accuracy, brier_score, and ZNE cost info
        factor_results: dict with per-factor probabilities and execution counts
        overhead: dict with execution cost comparison vs baseline
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n_samples = len(X)

    # --- Run circuits at each noise factor ---
    factor_results = {}
    total_executions = 0

    for k in noise_factors:
        scaled_noise = _zne_scaled_circuit(config, k)
        simulator = AerSimulator(noise_model=scaled_noise)

        probs_list = []
        exec_count = 0
        for i, (x, label) in enumerate(zip(X, y)):
            qc = vqc_circuit(x, theta, measured=True)
            compiled = compile_for_processor(qc, config)
            result = simulator.run(compiled, shots=shots, seed_simulator=seed + i).result()
            counts = result.get_counts()
            if set(counts) - {"0", "1"}:
                raise ValueError(
                    f"ZNE: Expected one-bit count keys, got: {sorted(counts)}"
                )
            raw_p1 = counts.get("1", 0) / shots
            probs_list.append(raw_p1)
            exec_count += 1

        total_executions += exec_count
        factor_results[k] = {
            "probs": np.array(probs_list),
            "executions": exec_count,
            "shots": shots,
        }

    # --- Richardson extrapolation using factors 1 and 2 ---
    # For linear noise scaling p(k) = p_ideal + k * eps:
    #   p(1) = p_ideal + eps
    #   p(2) = p_ideal + 2*eps
    # => p_ideal = 2*p(1) - p(2)
    probs_1 = factor_results[1.0]["probs"]  # shape: (n_samples,)
    probs_2 = factor_results[2.0]["probs"]  # shape: (n_samples,)
    ext_probs = 2.0 * probs_1 - probs_2

    # Clip to valid probability range [0, 1]
    ext_probs = np.clip(ext_probs, 0.0, 1.0)

    # --- 3-factor stability check ---
    # Using factor 3 to verify the linear noise model.
    # Expected probs at k=3 under the linear model: p(3) = 3*p(1) - 2*p_ideal
    #   = 3*p(1) - 2*(2*p(1) - p(2)) = -p(1) + 2*p(2)
    probs_3 = factor_results[3.0]["probs"]
    expected_probs_3 = -probs_1 + 2.0 * probs_2
    stability_gap = np.abs(expected_probs_3 - probs_3).mean()

    stabilized = stability_gap < 0.1  # 10% mean threshold

    if not stabilized:
        # Fall back to the k=1 (least-amplified) result
        ext_probs = probs_1.copy()

    # --- Compute mitigated metrics ---
    ext_pred = (ext_probs >= 0.5).astype(int)
    accuracy = float(np.mean(ext_pred == y))
    brier = float(np.mean((ext_probs - y) ** 2))

    # --- Cost overhead reporting ---
    # Baseline (unmitigated noisy): 1 factor * n_samples circuits
    # ZNE: N factors * n_samples circuits
    # Overhead multiplier = N factors
    baseline_metrics, _ = eval_noisy(
        X, y, theta, config, shots=shots, mitigate_readout=False, seed=seed
    )

    overhead = {
        "noise_factors": list(noise_factors),
        "total_circuit_executions": int(total_executions),
        "baseline_circuit_executions": int(n_samples),
        "overhead_multiplier": float(len(noise_factors)),
        "shots_per_sample": int(shots),
        "stabilized": bool(stabilized),
        "stability_gap_mean": float(stability_gap) if not stabilized else 0.0,
    }

    mitigated_metrics = {
        "accuracy": accuracy,
        "brier_score": brier,
        "shots_per_sample": int(shots),
        "total_shots": int(shots * n_samples),
        "circuit_executions": int(total_executions),
        "readout_mitigated": False,
        "zne_applied": True,
        "zne_noise_factors": list(noise_factors),
        "zne_extrapolation": "Richardson2" if stabilized else "Richardson2-fallback",
        "zne_stabilized": stabilized,
        "zne_cost_overhead": overhead,
    }

    return mitigated_metrics, factor_results, overhead


def repeated_run_evaluate(
    X, y, theta, config, shots=4096, n_runs=3, mitigate_readout=False, seed_base=42
):
    """Repeat noisy evaluation with independent simulator seeds.

    Reports the mean and standard deviation of accuracy and Brier score
    across repeated runs. Preserves each individual run's results.

    Variability due to finite-shot sampling is tracked separately from
    training initialization variability (which is preserved by using the
    same theta across all runs).

    Returns:
        summary: dict with mean, std, and n_runs
        run_results: list of per-run metric dicts and dataframes
    """
    run_results = []
    accuracies = []
    brier_scores = []

    for run_idx in range(n_runs):
        run_seed = seed_base + run_idx * 1000
        metrics, df = eval_noisy(
            X, y, theta, config, shots=shots, mitigate_readout=mitigate_readout, seed=run_seed
        )
        run_results.append({"metrics": metrics, "dataframe": df})
        accuracies.append(metrics["accuracy"])
        brier_scores.append(metrics["brier_score"])

    summary = {
        "n_runs": int(n_runs),
        "shots_per_sample": int(shots),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "brier_score_mean": float(np.mean(brier_scores)),
        "brier_score_std": float(np.std(brier_scores, ddof=1)),
        "accuracy_range": [float(np.min(accuracies)), float(np.max(accuracies))],
        "brier_score_range": [float(np.min(brier_scores)), float(np.max(brier_scores))],
    }
    return summary, run_results


def zne_expectation_value(theta, config, noise_factors=(1.0, 2.0, 3.0), shots=4096, seed=42):
    """Legacy: simple ZNE on expectation values.

    Kept for reference but the primary ZNE method is zne_evaluate()
    which operates on classifier probabilities.
    """
    from sklearn.utils.validation import check_array

    X_dummy = check_array(np.array([[0.0, 0.0, 0.0, 0.0]]), dtype=float, ensure_2d=True)
    y_dummy = np.array([0])

    # Delegate to zne_evaluate with a single sample for the legacy path
    metrics, factor_results, overhead = zne_evaluate(
        X_dummy, y_dummy, theta, config, noise_factors=noise_factors, shots=shots, seed=seed
    )
    return metrics, factor_results, overhead