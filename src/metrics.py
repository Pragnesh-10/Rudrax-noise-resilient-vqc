"""Noise evaluation, readout mitigation, ZNE, and metric helpers."""

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from .processors import noise_model
from .routing import compile_for_processor, circuit_metrics
from .problem import vqc_circuit


def _validate_binary_labels(y):
    """Validate binary labels without silently truncating non-integer values."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if labels.size == 0:
        raise ValueError("y must contain at least one label")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("y must contain only binary labels: 0 and 1")
    return labels.astype(int)


def _validate_shots(shots):
    """Require a positive integer shot count (excluding booleans)."""
    if (
        isinstance(shots, (bool, np.bool_))
        or not isinstance(shots, (int, np.integer))
        or shots <= 0
    ):
        raise ValueError("shots must be a positive integer")
    return int(shots)


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


def eval_noisy(
    X, y, theta, config, shots=4096, mitigate_readout=False, seed=42
):
    """Evaluate one circuit per sample and return classification/probability metrics.

    Readout mitigation is analytic inversion using the configured symmetric
    bit-flip probability; it is not a calibration-circuit experiment.
    """
    X = np.asarray(X)
    y = _validate_binary_labels(y)
    shots = _validate_shots(shots)

    if X.ndim == 0 or len(X) == 0:
        raise ValueError("X must contain at least one sample")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")

    p_flip = float(config["noise"]["readout_bit_flip_probability"])
    if mitigate_readout and not (0.0 <= p_flip < 0.5):
        raise ValueError(
            "Readout mitigation requires 0 <= bit-flip probability < 0.5"
        )

    simulator = AerSimulator(noise_model=noise_model(config))
    rows = []
    for i, (x, label) in enumerate(zip(X, y)):
        qc = vqc_circuit(x, theta, measured=True)
        compiled = compile_for_processor(qc, config)
        result = simulator.run(
            compiled, shots=shots, seed_simulator=seed + i
        ).result()
        counts = result.get_counts()

        # This scaffold measures a single classical bit. Fail loudly if that changes.
        if set(counts) - {"0", "1"}:
            raise ValueError(f"Expected one-bit count keys, got: {sorted(counts)}")

        raw_p1 = counts.get("1", 0) / shots
        p1 = raw_p1
        if mitigate_readout:
            p1 = float(np.clip((raw_p1 - p_flip) / (1 - 2 * p_flip), 0.0, 1.0))

        pred = int(p1 >= 0.5)
        rows.append(
            {
                "label": int(label),
                "raw_p1": raw_p1,
                "p1": p1,
                "prediction": pred,
            }
        )

    pred = np.array([row["prediction"] for row in rows])
    probs = np.array([row["p1"] for row in rows])
    metrics = {
        "accuracy": float(np.mean(pred == y)),
        "brier_score": float(np.mean((probs - y) ** 2)),
        "shots_per_sample": shots,
        "total_shots": int(shots * len(y)),
        "circuit_executions": int(len(y)),
        "readout_mitigated": bool(mitigate_readout),
    }
    return metrics, pd.DataFrame(rows)


def _zne_scaled_circuit(config, noise_scale_factor):
    """Build a noise model with scaled gate-depolarizing probabilities.

    Single-qubit and CX depolarizing probabilities are multiplied by the
    noise scale factor. Readout bit-flip probability remains fixed.

    This is simulator noise-parameter scaling, not gate folding.
    """
    factor = float(noise_scale_factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("noise_scale_factor must be finite and positive")

    p1 = float(config["noise"]["single_qubit_depolarizing_probability"])
    p2 = float(config["noise"]["cx_depolarizing_probability"])
    pr = float(config["noise"]["readout_bit_flip_probability"])

    p1_scaled = p1 * factor
    p2_scaled = p2 * factor

    if not (0 <= p1_scaled <= 1):
        raise ValueError(
            "Scaled single-qubit depolarizing probability must be in [0, 1]"
        )
    if not (0 <= p2_scaled <= 1):
        raise ValueError("Scaled CX depolarizing probability must be in [0, 1]")
    if not (0 <= pr <= 1):
        raise ValueError("Readout bit-flip probability must be in [0, 1]")

    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(p1_scaled, 1), ["x", "sx", "rz"]
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(p2_scaled, 2), ["cx"]
    )
    model.add_all_qubit_readout_error(
        ReadoutError([[1 - pr, pr], [pr, 1 - pr]])
    )
    return model


def zne_evaluate(
    X, y, theta, config, noise_factors=(1.0, 2.0, 3.0), shots=4096, seed=42
):
    """Evaluate ZNE using simulator gate-noise scaling.

    For each noise factor, scale single-qubit and CX depolarizing
    probabilities while keeping readout error fixed. Execute the compiled
    circuit and estimate P(1) for each sample.

    Two-point Richardson extrapolation uses factors 1 and 2:
        p_extrapolated = 2*p(1) - p(2).

    Factor 3 provides a consistency check against the assumed linear
    noise-scaling model. If the mean absolute gap exceeds 0.1, the method
    falls back to the factor-1 probabilities.

    This is simulator noise-parameter scaling, not gate folding.
    """
    X = np.asarray(X)
    y = _validate_binary_labels(y)
    shots = _validate_shots(shots)

    if X.ndim == 0 or len(X) == 0:
        raise ValueError("X must contain at least one sample")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")

    try:
        noise_factors = tuple(float(k) for k in noise_factors)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "noise_factors must be (1.0, 2.0, 3.0)"
        ) from exc

    if noise_factors != (1.0, 2.0, 3.0):
        raise ValueError(
            "This implementation requires noise_factors=(1.0, 2.0, 3.0)"
        )

    n_samples = len(X)
    factor_results = {}
    total_executions = 0

    # Run the same samples at each supported noise scale.
    for factor in noise_factors:
        scaled_noise = _zne_scaled_circuit(config, factor)
        simulator = AerSimulator(noise_model=scaled_noise)
        probs_list = []

        for i, _label in enumerate(y):
            qc = vqc_circuit(X[i], theta, measured=True)
            compiled = compile_for_processor(qc, config)
            result = simulator.run(
                compiled,
                shots=shots,
                seed_simulator=seed + i,
            ).result()
            counts = result.get_counts()

            if set(counts) - {"0", "1"}:
                raise ValueError(
                    f"ZNE: Expected one-bit count keys, got: {sorted(counts)}"
                )

            probs_list.append(counts.get("1", 0) / shots)

        executions = len(probs_list)
        total_executions += executions
        factor_results[factor] = {
            "probs": np.asarray(probs_list, dtype=float),
            "executions": int(executions),
            "shots": shots,
        }

    # Two-point Richardson extrapolation from noise factors 1 and 2.
    probs_1 = factor_results[1.0]["probs"]
    probs_2 = factor_results[2.0]["probs"]
    raw_ext_probs = 2.0 * probs_1 - probs_2

    # Track clipping rather than silently hiding out-of-range extrapolations.
    clipped_mask = (raw_ext_probs < 0.0) | (raw_ext_probs > 1.0)
    clipped_fraction = float(np.mean(clipped_mask))
    ext_probs = np.clip(raw_ext_probs, 0.0, 1.0)

    # Third-factor consistency check under the assumed linear model.
    probs_3 = factor_results[3.0]["probs"]
    expected_probs_3 = -probs_1 + 2.0 * probs_2
    stability_gap = float(np.mean(np.abs(expected_probs_3 - probs_3)))
    stabilized = stability_gap < 0.1

    if not stabilized:
        # Fall back to the least-amplified (factor-1) result.
        ext_probs = probs_1.copy()

    predictions = (ext_probs >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == y))
    brier = float(np.mean((ext_probs - y) ** 2))

    overhead = {
        "noise_factors": list(noise_factors),
        "total_circuit_executions": int(total_executions),
        "baseline_circuit_executions": int(n_samples),
        "overhead_multiplier": float(total_executions / n_samples),
        "shots_per_sample": shots,
        "stabilized": bool(stabilized),
        "stability_gap_mean": stability_gap,
        "stability_threshold": 0.1,
        "raw_extrapolation_clipped_fraction": clipped_fraction,
    }

    mitigated_metrics = {
        "accuracy": accuracy,
        "brier_score": brier,
        "shots_per_sample": shots,
        "total_shots": int(shots * n_samples * len(noise_factors)),
        "circuit_executions": int(total_executions),
        "readout_mitigated": False,
        "zne_applied": True,
        "zne_noise_factors": list(noise_factors),
        "zne_extrapolation": (
            "Richardson2" if stabilized else "Richardson2-fallback"
        ),
        "zne_stabilized": bool(stabilized),
        "zne_cost_overhead": overhead,
    }

    return mitigated_metrics, factor_results, overhead


def repeated_run_evaluate(
    X, y, theta, config, shots=4096, n_runs=3,
    mitigate_readout=False, seed_base=42
):
    """Repeat noisy evaluation with independent simulator seeds.

    Reports mean and standard deviation of accuracy and Brier score across
    repeated runs. The same theta is used for all runs, so this measures
    finite-shot/simulator-seed variability, not training-initialization
    variability.

    Returns:
        summary: dict with mean, standard deviation, and number of runs
        run_results: list of per-run metric dictionaries and dataframes
    """
    shots = _validate_shots(shots)
    if isinstance(n_runs, (bool, np.bool_)) or not isinstance(
        n_runs, (int, np.integer)
    ) or n_runs < 2:
        raise ValueError("n_runs must be an integer >= 2 to estimate sample std")

    run_results = []
    accuracies = []
    brier_scores = []

    for run_idx in range(n_runs):
        run_seed = seed_base + run_idx * 1000
        metrics, dataframe = eval_noisy(
            X,
            y,
            theta,
            config,
            shots=shots,
            mitigate_readout=mitigate_readout,
            seed=run_seed,
        )
        run_results.append({"metrics": metrics, "dataframe": dataframe})
        accuracies.append(metrics["accuracy"])
        brier_scores.append(metrics["brier_score"])

    summary = {
        "n_runs": int(n_runs),
        "shots_per_sample": shots,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "brier_score_mean": float(np.mean(brier_scores)),
        "brier_score_std": float(np.std(brier_scores, ddof=1)),
        "accuracy_range": [
            float(np.min(accuracies)),
            float(np.max(accuracies)),
        ],
        "brier_score_range": [
            float(np.min(brier_scores)),
            float(np.max(brier_scores)),
        ],
    }
    return summary, run_results


def zne_expectation_value(
    theta, config, noise_factors=(1.0, 2.0, 3.0), shots=4096, seed=42
):
    """Legacy ZNE helper delegating to the primary probability-based method."""
    X_dummy = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=float)
    y_dummy = np.array([0], dtype=int)

    return zne_evaluate(
        X_dummy,
        y_dummy,
        theta,
        config,
        noise_factors=noise_factors,
        shots=shots,
        seed=seed,
    )
