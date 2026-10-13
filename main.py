"""End-to-end QFF Phase 1 PS-Q3 experiment.

Run from the project root: python main.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.problem import make_data, ideal_p1
from src.processors import load_config
from src.metrics import eval_noisy, zne_evaluate, repeated_run_evaluate, aggregate_compiled_metrics
from src.visualization import plot_accuracy

ROOT = Path(__file__).resolve().parent
SHOTS = 4096
MAXITER = 35
SEED = 42


def objective(theta, X, y):
    """Deterministic statevector training objective: mean squared probability error."""
    probs = np.array([ideal_p1(x, theta) for x in X])
    return float(np.mean((probs - y) ** 2))


def main():
    for directory in (
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "results" / "tables",
        ROOT / "results" / "figures",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, _ = make_data()
    np.savez(ROOT / "data" / "processed" / "synthetic_qml.npz",
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    rng = np.random.default_rng(SEED)
    initial = rng.uniform(-0.3, 0.3, size=8)
    train_idx = np.arange(min(80, len(X_train)))
    opt = minimize(objective, initial, args=(X_train[train_idx], y_train[train_idx]),
                   method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.3})
    theta = opt.x
    np.save(ROOT / "results" / "trained_theta.npy", theta)
    with open(ROOT / "results" / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"optimizer": "COBYLA", "success": bool(opt.success),
                   "message": str(opt.message), "iterations": int(opt.nfev),
                   "train_subset_size": int(len(train_idx)), "seed": SEED,
                   "final_objective": float(opt.fun)}, f, indent=2)

    ideal_probs = np.array([ideal_p1(x, theta) for x in X_test])
    rows = [{"processor": "Ideal", "condition": "ideal",
             "accuracy": float(np.mean((ideal_probs >= 0.5) == y_test)),
             "brier_score": float(np.mean((ideal_probs - y_test) ** 2)),
             "total_shots": 0, "circuit_executions": 0,
             "depth_mean": None, "depth_max": None, "depth_total": None,
             "cx_count_mean": None, "cx_count_max": None, "cx_count_total": None,
             "swap_count_mean": None, "swap_count_max": None, "swap_count_total": None,
             "zne_applied": False, "zne_stabilized": None,
             "n_repeated_runs": None}]

    for name in ("processor_A", "processor_B"):
        config = load_config(name)
        circuit_summary = aggregate_compiled_metrics(X_test, theta, config)

        # --- Ideal / reference already in first row ---

        # --- Noisy evaluation (readout mitigation False) ---
        metrics, predictions = eval_noisy(X_test, y_test, theta, config,
                                          shots=SHOTS, mitigate_readout=False,
                                          seed=SEED)
        predictions.to_csv(ROOT / "results" / "tables" /
                           f"{name}_{'mitigated' if False else 'noisy'}_predictions.csv",
                           index=False)
        rows.append({"processor": config["name"],
                     "condition": "noisy",
                     "accuracy": metrics["accuracy"], "brier_score": metrics["brier_score"],
                     "total_shots": metrics["total_shots"],
                     "circuit_executions": metrics["circuit_executions"],
                     **circuit_summary,
                     "zne_applied": False, "zne_stabilized": None,
                     "n_repeated_runs": None})

        # --- Readout-mitigated evaluation ---
        metrics_rm, predictions_rm = eval_noisy(X_test, y_test, theta, config,
                                                shots=SHOTS, mitigate_readout=True,
                                                seed=SEED)
        predictions_rm.to_csv(ROOT / "results" / "tables" /
                              f"{name}_readout_mitigated_predictions.csv",
                              index=False)
        rows.append({"processor": config["name"],
                     "condition": "readout_mitigated",
                     "accuracy": metrics_rm["accuracy"], "brier_score": metrics_rm["brier_score"],
                     "total_shots": metrics_rm["total_shots"],
                     "circuit_executions": metrics_rm["circuit_executions"],
                     **circuit_summary,
                     "zne_applied": False, "zne_stabilized": None,
                     "n_repeated_runs": None})

        # --- ZNE evaluation ---
        metrics_zne, factor_results, overhead = zne_evaluate(X_test, y_test, theta, config,
                                                             noise_factors=(1.0, 2.0, 3.0),
                                                             shots=SHOTS, seed=SEED)

        rows.append({"processor": config["name"],
                     "condition": "zne",
                     "accuracy": metrics_zne["accuracy"], "brier_score": metrics_zne["brier_score"],
                     "total_shots": metrics_zne["total_shots"],
                     "circuit_executions": metrics_zne["circuit_executions"],
                     **circuit_summary,
                     "zne_applied": True,
                     "zne_stabilized": metrics_zne["zne_stabilized"],
                     "zne_extrapolation": metrics_zne["zne_extrapolation"],
                     "zne_noise_factors": ";".join(str(f) for f in metrics_zne["zne_noise_factors"]),
                     "zne_cost_overhead_multiplier": metrics_zne["zne_cost_overhead"]["overhead_multiplier"],
                     "n_repeated_runs": None})

        # --- Repeated-run evaluation (variance analysis) ---
        summary, run_results = repeated_run_evaluate(X_test, y_test, theta, config,
                                                     shots=SHOTS, n_runs=3, seed_base=SEED)
        # Save per-run results
        for rr_idx, rr in enumerate(run_results):
            rr["dataframe"].to_csv(ROOT / "results" / "tables" /
                                  f"{name}_run{rr_idx+1}_predictions.csv",
                                  index=False)
        # Save aggregated run summary
        run_summary_path = ROOT / "results" / "tables" / f"{name}_repeated_run_summary.json"
        with open(run_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        rows.append({"processor": config["name"],
                     "condition": "repeated_run",
                     "accuracy": summary["accuracy_mean"], "brier_score": summary["brier_score_mean"],
                     "total_shots": summary["n_runs"] * SHOTS * len(y_test),
                     "circuit_executions": summary["n_runs"] * len(y_test),
                     **{k: v for k, v in circuit_summary.items()},
                     "zne_applied": False, "zne_stabilized": None,
                     "n_repeated_runs": summary["n_runs"],
                     "accuracy_std": summary["accuracy_std"],
                     "brier_score_std": summary["brier_score_std"],
                     "accuracy_range_min": summary["accuracy_range"][0],
                     "accuracy_range_max": summary["accuracy_range"][1],
                     "brier_score_range_min": summary["brier_score_range"][0],
                     "brier_score_range_max": summary["brier_score_range"][1],
                     "zne_cost_overhead_multiplier": None})

    df = pd.DataFrame(rows)
    output_csv = ROOT / "results" / "tables" / "q3_comparison.csv"
    df.to_csv(output_csv, index=False)
    plot_accuracy(output_csv, ROOT / "results" / "figures" / "accuracy_comparison.png")
    print(df.to_string(index=False))
    print(f"\nSaved results to {output_csv}")
    print("Note: readout mitigation is analytic inversion using the configured error rate, not measured calibration.")
    print(f"\nZNE cost overhead: {overhead['overhead_multiplier']}x circuit executions "
          f"(factors {overhead['noise_factors']}, stability gap: {overhead['stability_gap_mean']:.4f})")


if __name__ == "__main__":
    main()