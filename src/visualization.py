"""Create comparison plots for QFF Phase 1 PS-Q3 results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_accuracy(csv_path, output_path):
    """Bar plot of test accuracy by processor and condition."""
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="processor", columns="condition", values="accuracy")
    ax = pivot.plot(kind="bar", ylim=(0, 1), ylabel="Test accuracy",
                    title="Q3: accuracy by processor and mitigation method")
    ax.figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def plot_brier_score(csv_path, output_path):
    """Bar plot of Brier score by processor and condition."""
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="processor", columns="condition", values="brier_score")
    ax = pivot.plot(kind="bar", ylim=(0, 1), ylabel="Brier score",
                    title="Q3: Brier score by processor and mitigation method")
    ax.figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def plot_circuit_metrics(csv_path, output_path):
    """Grouped bar plot of depth, CX count, and SWAP count by processor."""
    df = pd.read_csv(csv_path)
    # Filter out ideal (no metrics) and repeated_run rows for this plot
    plot_df = df[df["condition"].isin(["noisy", "readout_mitigated", "zne"])].copy()

    metrics = ["depth_mean", "cx_count_mean", "swap_count_mean"]
    labels = ["Depth (mean)", "CX count (mean)", "SWAP count (mean)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        pivot = plot_df.pivot(index="processor", columns="condition", values=metric)
        pivot.plot(kind="bar", ax=axes[i], legend=False)
        axes[i].set_ylabel(label)
        axes[i].set_xlabel("Processor")
        axes[i].set_title(f"{label} by processor and condition")
        axes[i].legend(pivot.columns, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mitigation_overhead(csv_path, output_path):
    """Plot mitigation execution cost overhead."""
    df = pd.read_csv(csv_path)
    zne_rows = df[df["condition"] == "zne"]

    overhead_data = []
    for _, row in zne_rows.iterrows():
        nfactors = row.get("zne_noise_factors", "")
        nfactors_list = [float(x.strip()) for x in nfactors.split(";") if x.strip()]
        overhead_data.append({
            "processor": row["processor"],
            "noise_factors": nfactors,
            "n_factors": len(nfactors_list),
            "overhead_multiplier": row.get("zne_cost_overhead_multiplier", len(nfactors_list)),
            "circuit_executions": row["circuit_executions"],
        })

    overhead_df = pd.DataFrame(overhead_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of overhead multiplier
    ax = axes[0]
    overhead_df.plot(x="processor", y="overhead_multiplier", kind="bar", ax=ax,
                     legend=False, title="ZNE overhead multiplier (circuit executions)")
    ax.set_ylabel("Multiplier")
    ax.set_xlabel("Processor")

    # Bar chart of noise factors
    ax2 = axes[1]
    overhead_df.plot(x="processor", y="n_factors", kind="bar", ax=ax2,
                     legend=False, title="Number of ZNE noise factors used")
    ax2.set_ylabel("Number of factors")
    ax2.set_xlabel("Processor")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_repeated_runs(csv_path, output_path):
    """Plot mean and std across repeated runs."""
    df = pd.read_csv(csv_path)
    run_rows = df[df["condition"].str.contains("repeated_run")]

    if run_rows.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No repeated-run data available",
                ha="center", va="center")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    # Build plot data from run rows
    plot_data = []
    for _, row in run_rows.iterrows():
        processor = row["processor"]
        acc_mean = row["accuracy"]
        acc_std = row.get("accuracy_std", 0.0)
        brier_mean = row.get("brier_score_mean", row["brier_score"])
        brier_std = row.get("brier_score_std", 0.0)
        plot_data.append({
            "processor": processor,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "brier_mean": brier_mean,
            "brier_std": brier_std,
        })

    plot_df = pd.DataFrame(plot_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy mean ± std
    ax = axes[0]
    xs = range(len(plot_df))
    ax.errorbar(xs, plot_df["accuracy_mean"].values, yerr=plot_df["accuracy_std"].values,
                fmt="o", capsize=3, label="Accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_df["processor"].values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy mean ± std across repeated runs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Brier score mean ± std
    ax2 = axes[1]
    ax2.errorbar(xs, plot_df["brier_mean"].values, yerr=plot_df["brier_std"].values,
                 fmt="o", capsize=3, label="Brier score")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(plot_df["processor"].values)
    ax2.set_ylabel("Brier score")
    ax2.set_title("Brier score mean ± std across repeated runs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_zne_vs_factor(csv_path, output_path):
    """Plot ZNE results versus noise-scaling factor count."""
    df = pd.read_csv(csv_path)
    zne_rows = df[df["condition"] == "zne"]

    if zne_rows.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ZNE data available",
                ha="center", va="center")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    # Show the number of noise factors used per processor
    fig, ax = plt.subplots()

    for _, row in zne_rows.iterrows():
        nfactors = row.get("zne_noise_factors", "")
        nfactors_list = [float(x.strip()) for x in nfactors.split(";") if x.strip()]
        ax.bar(row["processor"], len(nfactors_list),
               label=f"{row['processor']} (factors={len(nfactors_list)})")

    ax.set_ylabel("Number of noise factors")
    ax.set_title("ZNE noise factor count per processor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)