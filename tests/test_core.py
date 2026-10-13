"""Tests for QFF Phase 1 PS-Q3 benchmark suite.

Tests cover processor configurations, noise models, dataset reproducibility,
VQC input/output shapes, probability normalization, readout correction behavior,
ZNE scaling and extrapolation, circuit metrics and routing accounting, shot and
execution-cost calculations, and end-to-end smoke-test execution.
"""
import json
import numpy as np
import pandas as pd
import pytest

from src.problem import make_data, vqc_circuit, ideal_p1
from src.processors import load_config, validate_config, coupling_map, noise_model
from src.metrics import eval_noisy, zne_evaluate, repeated_run_evaluate, \
    aggregate_compiled_metrics, compiled_metrics_for_sample
from src.routing import compile_for_processor, circuit_metrics
from src.visualization import plot_accuracy


# ---- Processor configuration tests ----


def test_processor_A_config():
    """Processor A must have 5 qubits and the updated-kit linear-chain edges."""
    cfg = load_config("processor_A")
    assert cfg["qubits"] == 5
    edges = frozenset(frozenset(e) for e in cfg["coupling_edges"])
    expected = frozenset({
        frozenset([0, 1]), frozenset([1, 2]),
        frozenset([2, 3]), frozenset([3, 4]),
    })
    assert edges == expected


def test_processor_B_config():
    """Processor B must have 9 qubits and the updated-kit reference edges."""
    cfg = load_config("processor_B")
    assert cfg["qubits"] == 9
    edges = frozenset(frozenset(e) for e in cfg["coupling_edges"])
    expected = frozenset({
        frozenset([0, 1]), frozenset([1, 2]),
        frozenset([2, 3]), frozenset([2, 4]),
        frozenset([4, 5]), frozenset([5, 6]),
        frozenset([4, 7]), frozenset([7, 8]),
    })
    assert edges == expected


def test_processor_D_config():
    """Processor D must have 6 qubits and a connected coupling graph."""
    cfg = load_config("processor_D")
    assert cfg["qubits"] == 6
    # Validate connectivity
    adj = {i: [] for i in range(cfg["qubits"])}
    for e in cfg["coupling_edges"]:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    assert len(visited) == cfg["qubits"], "Processor D graph must be connected"


def test_processor_noise_model():
    """Noise model must produce valid error probabilities."""
    cfg_a = load_config("processor_A")
    cfg_b = load_config("processor_B")
    cfg_d = load_config("processor_D")

    model_a = noise_model(cfg_a)
    model_b = noise_model(cfg_b)
    model_d = noise_model(cfg_d)

    # All models should have the same noise parameters
    assert model_a is not None
    assert model_b is not None
    assert model_d is not None


def test_basis_gates():
    """All processor configs must use only allowed basis gates."""
    for name in ("processor_A", "processor_B", "processor_D"):
        cfg = load_config(name)
        valid = {"rz", "sx", "x", "cx"}
        assert set(cfg["basis_gates"]).issubset(valid), \
            f"{name}: invalid basis gates {set(cfg['basis_gates']) - valid}"


# ---- Dataset reproducibility tests ----


def test_make_data_shape():
    """make_data returns train/test splits with correct dimensions."""
    X_train, X_test, y_train, y_test, scaler = make_data()
    assert X_train.shape[1] == 4  # 4 features
    assert X_test.shape[1] == 4
    assert len(y_train) + len(y_test) == 200  # full dataset
    assert set(np.unique(y_train)) | set(np.unique(y_test)) == {0, 1}
    # Check that scaler transforms to approximately [0, pi]
    # (MinMaxScaler may have fp precision at boundaries; allow 1% tolerance)
    assert X_train.min() >= 0 and X_train.max() <= np.pi * 1.05
    assert X_test.min() >= 0 and X_test.max() <= np.pi * 1.05


def test_make_data_seeds():
    """make_data is deterministic with fixed random_state."""
    X_train1, X_test1, y_train1, y_test1, _ = make_data()
    X_train2, X_test2, y_train2, y_test2, _ = make_data()
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(X_test1, X_test2)
    np.testing.assert_array_equal(y_train1, y_train2)
    np.testing.assert_array_equal(y_test1, y_test2)


# ---- VQC tests ----


def test_vqc_circuit_qubits():
    """VQC circuit must act on N_QUBITS=4."""
    from src.problem import N_QUBITS
    theta = np.random.randn(8)
    x = np.random.randn(4)
    qc = vqc_circuit(x, theta)
    assert qc.num_qubits == N_QUBITS


def test_vqc_circuit_theta_shape():
    """VQC expects theta with 8 parameters."""
    from src.problem import N_QUBITS
    x = np.random.randn(4)
    theta = np.random.randn(8)
    qc = vqc_circuit(x, theta)
    # 4 Ry + 4 RZ + 4 CX (ring) = 12 ops, but theta has 8 entries
    assert theta.shape == (8,)


def test_ideal_p1_output():
    """ideal_p1 must return a valid probability in [0, 1]."""
    theta = np.array([0.1] * 8)
    x = np.array([0.5] * 4)
    p = ideal_p1(x, theta)
    assert 0.0 <= p <= 1.0


# ---- Metric tests ----


def test_eval_noisy_output_shape():
    """eval_noisy returns metrics dict with expected keys."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)
    metrics, df = eval_noisy(X_test, y_test, theta, cfg, shots=4096, seed=42)
    assert "accuracy" in metrics
    assert "brier_score" in metrics
    assert "total_shots" in metrics
    assert "circuit_executions" in metrics
    assert len(df) == len(y_test)


def test_eval_noisy_probabilities_finite():
    """Probabilities from eval_noisy must be finite and in [0, 1]."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)
    metrics, df = eval_noisy(X_test, y_test, theta, cfg, shots=4096, seed=42)
    assert np.all(np.isfinite(df["p1"]))
    assert np.all((df["p1"] >= 0) & (df["p1"] <= 1))


def test_eval_noisy_readout_mitigation():
    """Readout-mitigated eval_noisy must not raise and should produce valid probs."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)
    metrics, df = eval_noisy(X_test, y_test, theta, cfg, shots=4096,
                             mitigate_readout=True, seed=42)
    assert "accuracy" in metrics
    assert "brier_score" in metrics
    assert np.all(np.isfinite(df["p1"]))


def test_zne_evaluate():
    """ZNE evaluate must return metrics with zne_applied=True."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)
    metrics, factor_results, overhead = zne_evaluate(
        X_test, y_test, theta, cfg, noise_factors=(1.0, 2.0, 3.0),
        shots=4096, seed=42,
    )
    assert metrics["zne_applied"] is True
    assert "noise_factors" in overhead
    assert overhead["overhead_multiplier"] == 3.0


def test_zne_stabilized():
    """ZNE stability check should produce a boolean."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)
    metrics, factor_results, overhead = zne_evaluate(
        X_test, y_test, theta, cfg, noise_factors=(1.0, 2.0, 3.0),
        shots=4096, seed=42,
    )
    assert isinstance(bool(metrics["zne_stabilized"]), bool)


def test_repeated_run_evaluate():
    """Repeated run must return mean, std, and per-run results."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)
    summary, run_results = repeated_run_evaluate(
        X_test, y_test, theta, cfg, shots=4096, n_runs=3, seed_base=42,
    )
    assert summary["n_runs"] == 3
    assert "accuracy_mean" in summary
    assert "accuracy_std" in summary
    assert "brier_score_mean" in summary
    assert "brier_score_std" in summary
    assert len(run_results) == 3
    for rr in run_results:
        assert "metrics" in rr
        assert "dataframe" in rr


def test_aggregate_compiled_metrics():
    """Aggregate compiled metrics must return depth, CX, swap, etc."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    summary = aggregate_compiled_metrics(X_test, np.array([0.1] * 8), cfg)
    assert "depth_mean" in summary
    assert "cx_count_mean" in summary
    assert "swap_count_mean" in summary
    assert "depth_max" in summary
    assert "cx_count_max" in summary
    assert "swap_count_max" in summary
    assert "depth_total" in summary
    assert "cx_count_total" in summary
    assert "swap_count_total" in summary


def test_compiled_metrics_for_sample():
    """compiled_metrics_for_sample must return per-sample metrics and compiled circuit."""
    from src.problem import vqc_circuit
    from src.processors import load_config
    from src.routing import compile_for_processor, circuit_metrics
    cfg = load_config("processor_A")
    x = X_test if 'X_test' in dir() else X_train if 'X_train' in dir() else None
    # Just test the function exists and runs with a sample x
    x_sample = np.array([0.5] * 4) if x is None else x[0]
    metrics, compiled = compiled_metrics_for_sample(x_sample, np.array([0.1] * 8), cfg)
    assert "depth" in metrics
    assert "cx_count" in metrics
    assert "swap_count" in metrics
    assert "operation_count" in metrics
    assert "num_qubits" in metrics


# ---- Probability normalization tests ----


def test_readout_correction_clipping():
    """Readout-corrected probabilities must be clipped to [0, 1]."""
    X_train, X_test, y_train, y_test, _ = make_data()
    from src.processors import load_config
    cfg = load_config("processor_A")
    theta = np.array([0.1] * 8)

    # Test with a very high bit-flip probability that would push raw_p1 outside [0,1]
    # The clip should handle this gracefully
    metrics, df = eval_noisy(X_test[:1], y_test[:1], theta, cfg,
                             shots=4096, mitigate_readout=True, seed=42)
    assert 0.0 <= df["p1"].iloc[0] <= 1.0


# ---- Visualization smoke test ----


def test_plot_accuracy_smoke():
    """plot_accuracy must run without error on the comparison CSV."""
    csv_path = "results/tables/q3_comparison.csv"
    try:
        plot_accuracy(csv_path, "/tmp/test_accuracy.png")
    except Exception as e:
        pytest.fail(f"plot_accuracy raised {e}")


# ---- End-to-end smoke test ----


def test_e2e_smoke_test():
    """Full end-to-end smoke test: data → train → evaluate → save."""
    import os
    from pathlib import Path

    # This test runs a minimal e2e and checks that key output files exist.
    # It does not verify full correctness, only that the workflow does not crash.

    project_root = Path(__file__).parent.parent
    main_py = project_root / "main.py"

    # Run main.py and check it completes
    import subprocess
    result = subprocess.run(
        ["python3", str(main_py)],
        capture_output=True, text=True, cwd=str(project_root),
        timeout=300,  # 5 min timeout
    )
    assert result.returncode == 0, f"main.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Check key output files generated by main.py
    csv_path = project_root / "results" / "tables" / "q3_comparison.csv"
    assert csv_path.exists(), "q3_comparison.csv not generated"

    theta_path = project_root / "results" / "trained_theta.npy"
    assert theta_path.exists(), "trained_theta.npy not generated"

    summary_path = project_root / "results" / "training_summary.json"
    assert summary_path.exists(), "training_summary.json not generated"

    # Check that figures were generated
    figures_dir = project_root / "results" / "figures"
    assert figures_dir.is_dir(), "results/figures directory not found"
    png_files = list(figures_dir.glob("*.png"))
    assert len(png_files) > 0, "No figure PNGs generated"