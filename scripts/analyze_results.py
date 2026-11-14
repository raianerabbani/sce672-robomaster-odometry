"""Statistical Analysis, Visualization & Plot Generation Script.

SCE-672: Advanced Robotics, KFUPM
Analyzes trial CSVs and generates publication-grade figures:
1. Trajectory comparison (Pure Odometry vs Vision vs Kalman Filter vs Ground Truth)
2. Covariance convergence P(t) over time
3. Multi-trial error comparison bar chart (KF vs Pure Odometry)
4. Error distribution and trial consistency analysis
Outputs figures in PNG and PDF formats for LaTeX report compilation.
"""

import os
import sys
import glob
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration for IEEE/academic publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def generate_plots(results_dir: str = "results", output_dir: str = "results/plots"):
    """Generate all figures and plots from trial result CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    summary_json_path = os.path.join(results_dir, "experiment_summary.json")

    if not os.path.exists(summary_json_path):
        print(f"[Analyze Warning] Summary file {summary_json_path} not found.")
        return

    with open(summary_json_path, "r") as f:
        data = json.load(f)

    stats = data["statistics"]
    trials = data["trials"]
    print(f"[Analyze] Loaded {len(trials)} trials from {summary_json_path}.")

    # -------------------------------------------------------------
    # FIGURE 1: Multi-Trial Error Scatterplot (KF vs Pure Odometry)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=300)

    trial_ids = [t["trial_id"] for t in trials]
    kf_errors_cm = [t["kf_error"] * 100 for t in trials]
    odo_errors_cm = [t["odo_error"] * 100 for t in trials]

    ax.scatter(
        trial_ids, kf_errors_cm, s=75, marker="o",
        color="#1b7837", edgecolor="#00441b", zorder=4,
        label="Kalman Filter (Prediction + Correction)"
    )
    ax.scatter(
        trial_ids, odo_errors_cm, s=75, marker="s",
        color="#b2182b", edgecolor="#67001f", alpha=0.85, zorder=3,
        label="Pure Odometry (Prediction Only)"
    )

    # Zero-error reference and MAE band
    ax.axhline(0, color="black", linestyle="-", linewidth=0.9)
    kf_mae_cm = stats["kf_mae"] * 100
    odo_mae_cm = stats["odo_mae"] * 100
    ax.axhline(kf_mae_cm, color="#1b7837", linestyle=":", linewidth=1.4, label=f"KF MAE = {kf_mae_cm:.2f} cm")
    ax.axhline(odo_mae_cm, color="#b2182b", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Pure Odo MAE = {odo_mae_cm:.2f} cm")

    ax.set_xlabel("Trial Number", fontweight="bold")
    ax.set_ylabel("Distance Estimation Error (cm)\n(Estimated − Measured)", fontweight="bold")
    ax.set_title("Final Distance Estimation Error Across Repeat Trials", fontweight="bold")
    ax.set_xticks(trial_ids)
    ax.set_xticklabels([f"{i}" for i in trial_ids])
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True)
    plt.tight_layout()

    fig1_path_png = os.path.join(output_dir, "trial_error_comparison.png")
    fig1_path_pdf = os.path.join(output_dir, "trial_error_comparison.pdf")
    fig.savefig(fig1_path_png)
    fig.savefig(fig1_path_pdf)
    plt.close(fig)
    print(f"  [Figure] Saved: {fig1_path_png}")

    # -------------------------------------------------------------
    # FIGURE 2: Time-Series Trajectory Analysis (Trial 1 or Best)
    # -------------------------------------------------------------
    sample_csv = trials[0]["time_series_path"]
    if not os.path.isabs(sample_csv):
        sample_csv = os.path.join(results_dir, "..", sample_csv)
    if not os.path.exists(sample_csv):
        # Search locally in results_dir
        sample_csv = os.path.join(results_dir, os.path.basename(sample_csv))

    if os.path.exists(sample_csv):
        time_vals = []
        true_dist = []
        pure_odo = []
        vis_dist = []
        kf_dist = []
        cov_pos = []

        with open(sample_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_vals.append(float(row["elapsed_time_s"]))
                pure_odo.append(float(row["pure_odo_dist_m"]))
                kf_dist.append(float(row["kf_dist_m"]))
                cov_pos.append(float(row["cov_pos_m2"]))
                true_dist.append(float(row["true_distance_m"]) if row["true_distance_m"] else np.nan)
                vis_dist.append(float(row["visual_dist_m"]) if row["visual_dist_m"] else np.nan)

        time_arr = np.array(time_vals)
        kf_arr = np.array(kf_dist)
        odo_arr = np.array(pure_odo)
        vis_arr = np.array(vis_dist)
        cov_arr = np.array(cov_pos)
        true_arr = np.array(true_dist)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, dpi=300)

        # Subplot 1: Distance Profiles
        ax1.plot(time_arr, odo_arr, label="Pure Odometry Prediction (Dead Reckoning)", color="#d95f02", linestyle="--", alpha=0.9)
        valid_vis = ~np.isnan(vis_arr)
        if np.any(valid_vis):
            ax1.scatter(time_arr[valid_vis], vis_arr[valid_vis], label="Visual Tag Sensor Measurements $z_k$", color="#7570b3", s=18, alpha=0.7)
        ax1.plot(time_arr, kf_arr, label="Kalman Filter Estimate $\\hat{d}$ (Prediction + Correction)", color="#1b9e77", linewidth=2.5)

        if not np.all(np.isnan(true_arr)):
            ax1.plot(time_arr, true_arr, label="True Distance $d_{\\mathrm{true}}$", color="black", linestyle=":", linewidth=1.8)

        ax1.axhline(stats["target_stop_distance"], color="red", linestyle="-.", label=f"Stop Target ({stats['target_stop_distance']:.2f} m)")
        ax1.set_ylabel("Distance to Tag (m)", fontweight="bold")
        ax1.set_title("Single-Trial Real-Time Trajectory and Odometry Estimation", fontweight="bold")
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.grid(True)

        # Subplot 2: Instantaneous Estimation Error
        ref_dist = true_arr if not np.all(np.isnan(true_arr)) else kf_arr[-1]
        kf_err_ts = (kf_arr - ref_dist) * 100
        odo_err_ts = (odo_arr - ref_dist) * 100

        ax2.plot(time_arr, odo_err_ts, label="Pure Odometry Error (Drift from Wheel Slip)", color="#d95f02", linestyle="--")
        ax2.plot(time_arr, kf_err_ts, label="Kalman Filter Error", color="#1b9e77", linewidth=2.0)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax2.set_ylabel("Error vs Ground Truth (cm)", fontweight="bold")
        ax2.legend(loc="upper left", framealpha=0.9)
        ax2.grid(True)

        # Subplot 3: Kalman Covariance Convergence P(t)
        pos_std_cm = np.sqrt(cov_arr) * 100
        ax3.plot(time_arr, pos_std_cm, color="#2b83ba", linewidth=2.0, label="Estimated State Uncertainty $\\sigma_d = \\sqrt{P_{11}}$")
        ax3.set_xlabel("Elapsed Time (seconds)", fontweight="bold")
        ax3.set_ylabel("Uncertainty $\\sigma$ (cm)", fontweight="bold")
        ax3.legend(loc="upper right", framealpha=0.9)
        ax3.grid(True)

        plt.tight_layout()
        fig2_path_png = os.path.join(output_dir, "trial_trajectory_analysis.png")
        fig2_path_pdf = os.path.join(output_dir, "trial_trajectory_analysis.pdf")
        fig.savefig(fig2_path_png)
        fig.savefig(fig2_path_pdf)
        plt.close(fig)
        print(f"  [Figure] Saved: {fig2_path_png}")

    # -------------------------------------------------------------
    # FIGURE 3: Error Distribution & Consistency Box Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    data_box = [kf_errors_cm, odo_errors_cm]
    labels_box = ["Kalman Filter\n(Prediction + Correction)", "Pure Odometry\n(Dead Reckoning)"]

    bp = ax.boxplot(data_box, patch_artist=True, labels=labels_box, widths=0.45)
    colors = ["#a6dba0", "#f4a582"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Estimation Error (cm)", fontweight="bold")
    ax.set_title("Estimation Error Consistency across 10 Trials", fontweight="bold")
    ax.grid(True)
    plt.tight_layout()

    fig3_path_png = os.path.join(output_dir, "trial_consistency_boxplot.png")
    fig3_path_pdf = os.path.join(output_dir, "trial_consistency_boxplot.pdf")
    fig.savefig(fig3_path_png)
    fig.savefig(fig3_path_pdf)
    plt.close(fig)
    print(f"  [Figure] Saved: {fig3_path_png}")

    print("[Analyze] All figures successfully generated in 'results/plots/'.")


if __name__ == "__main__":
    generate_plots()
