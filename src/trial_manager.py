"""Trial Manager and Statistical Evaluation Module.

SCE-672: Advanced Robotics, KFUPM
Manages:
- Multi-trial execution (8-10 trials) under identical starting conditions
- Time-series logging of telemetry, pure odometry, vision measurements, and Kalman state
- Independent ground-truth physical tape measurement recording
- Calculation of error metrics: estimated - measured, Mean Absolute Error (MAE), error range, std
- Exporting to CSV, JSON, Markdown, and LaTeX tables for the final report.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import os
import csv
import json
import time
import numpy as np


@dataclass
class TimeSeriesSample:
    """Telemetry sample at a single time step."""
    timestamp: float
    elapsed_time: float
    true_distance: Optional[float]
    pure_odo_dist: float
    visual_dist: Optional[float]
    kf_dist: float
    kf_vel: float
    cov_pos: float
    cov_vel: float
    kalman_gain: float
    tag_detected: bool
    vx_cmd: float
    wz_cmd: float


@dataclass
class TrialRecord:
    """Summary record for a single experimental trial."""
    trial_id: int
    start_distance: float
    target_stop_distance: float
    estimated_final_distance: float     # KF estimate d_hat
    odometry_final_distance: float      # Pure dead reckoning d_odo
    measured_actual_distance: float     # Ground truth tape measurement d_meas
    kf_error: float                     # d_hat - d_meas
    kf_abs_error: float                 # |d_hat - d_meas|
    odo_error: float                    # d_odo - d_meas
    odo_abs_error: float                # |d_odo - d_meas|
    duration: float                     # Run time (seconds)
    time_series_path: str               # Detailed CSV log path


@dataclass
class ExperimentStatistics:
    """Aggregate statistics across all repeat trials."""
    num_trials: int
    start_distance: float
    target_stop_distance: float
    # Kalman Filter Error Metrics
    kf_mae: float                       # Mean Absolute Error (m)
    kf_mean_error: float                # Bias / Mean Signed Error (m)
    kf_min_error: float                 # Min signed error (m)
    kf_max_error: float                 # Max signed error (m)
    kf_error_range: float               # Range = max - min (m)
    kf_std_error: float                 # Standard deviation of error (m)
    # Pure Odometry Error Metrics (Baseline)
    odo_mae: float
    odo_mean_error: float
    odo_min_error: float
    odo_max_error: float
    odo_error_range: float
    odo_std_error: float
    # Comparison
    mae_improvement_pct: float          # Percentage error reduction with sensor correction


class TrialManager:
    """Orchestrates trials, telemetry logging, and statistical metrics."""

    def __init__(self, output_dir: str = "results", start_distance: float = 1.50, stop_distance: float = 0.40):
        self.output_dir = output_dir
        self.start_distance = float(start_distance)
        self.stop_distance = float(stop_distance)
        os.makedirs(self.output_dir, exist_ok=True)

        self.trials: List[TrialRecord] = []
        self.current_trial_id: Optional[int] = None
        self.current_samples: List[TimeSeriesSample] = []
        self.trial_start_time: Optional[float] = None

    def start_trial(self, trial_id: int):
        """Begin a new experimental trial."""
        self.current_trial_id = trial_id
        self.current_samples = []
        self.trial_start_time = time.time()
        print(f"\n=======================================================")
        print(f"        STARTING TRIAL {trial_id} (D0 = {self.start_distance:.2f}m, Dstop = {self.stop_distance:.2f}m)")
        print(f"=======================================================")

    def log_sample(
        self,
        pure_odo_dist: float,
        visual_dist: Optional[float],
        kf_dist: float,
        kf_vel: float,
        cov_pos: float,
        cov_vel: float,
        kalman_gain: float,
        tag_detected: bool,
        vx_cmd: float,
        wz_cmd: float,
        true_distance: Optional[float] = None
    ):
        """Append a time step telemetry sample."""
        if self.trial_start_time is None:
            return
        now = time.time()
        elapsed = now - self.trial_start_time
        sample = TimeSeriesSample(
            timestamp=now,
            elapsed_time=elapsed,
            true_distance=true_distance,
            pure_odo_dist=float(pure_odo_dist),
            visual_dist=float(visual_dist) if visual_dist is not None else None,
            kf_dist=float(kf_dist),
            kf_vel=float(kf_vel),
            cov_pos=float(cov_pos),
            cov_vel=float(cov_vel),
            kalman_gain=float(kalman_gain),
            tag_detected=bool(tag_detected),
            vx_cmd=float(vx_cmd),
            wz_cmd=float(wz_cmd)
        )
        self.current_samples.append(sample)

    def finish_trial(self, measured_actual_distance: float) -> TrialRecord:
        """Complete current trial and compute errors against measured distance."""
        if not self.current_samples or self.current_trial_id is None:
            raise RuntimeError("No samples logged for current trial.")

        last_sample = self.current_samples[-1]
        duration = last_sample.elapsed_time
        kf_final = last_sample.kf_dist
        odo_final = last_sample.pure_odo_dist
        d_meas = float(measured_actual_distance)

        kf_err = kf_final - d_meas
        kf_abs_err = abs(kf_err)
        odo_err = odo_final - d_meas
        odo_abs_err = abs(odo_err)

        # Save detailed time series to CSV
        csv_filename = f"trial_{self.current_trial_id:02d}_timeseries.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        self._write_timeseries_csv(csv_path)

        record = TrialRecord(
            trial_id=self.current_trial_id,
            start_distance=self.start_distance,
            target_stop_distance=self.stop_distance,
            estimated_final_distance=kf_final,
            odometry_final_distance=odo_final,
            measured_actual_distance=d_meas,
            kf_error=kf_err,
            kf_abs_error=kf_abs_err,
            odo_error=odo_err,
            odo_abs_error=odo_abs_err,
            duration=duration,
            time_series_path=csv_path
        )
        self.trials.append(record)

        print(f"\n[Trial {self.current_trial_id} Complete]")
        print(f"  Estimated Final Distance (KF)   : {kf_final:.4f} m")
        print(f"  Odometry Final Distance (Pure)  : {odo_final:.4f} m")
        print(f"  Measured Actual Distance (Tape) : {d_meas:.4f} m")
        print(f"  Estimation Error (Est - Meas)   : {kf_err:+.4f} m ({kf_err*100:+.2f} cm)")
        print(f"  Pure Odometry Error             : {odo_err:+.4f} m ({odo_err*100:+.2f} cm)")
        print(f"  Time series saved to: {csv_path}\n")

        self.current_trial_id = None
        self.current_samples = []
        return record

    def _write_timeseries_csv(self, filepath: str):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "elapsed_time_s", "true_distance_m",
                "pure_odo_dist_m", "visual_dist_m", "kf_dist_m", "kf_vel_mps",
                "cov_pos_m2", "cov_vel_m2s2", "kalman_gain", "tag_detected",
                "vx_cmd_mps", "wz_cmd_degps"
            ])
            for s in self.current_samples:
                writer.writerow([
                    f"{s.timestamp:.4f}",
                    f"{s.elapsed_time:.4f}",
                    f"{s.true_distance:.4f}" if s.true_distance is not None else "",
                    f"{s.pure_odo_dist:.4f}",
                    f"{s.visual_dist:.4f}" if s.visual_dist is not None else "",
                    f"{s.kf_dist:.4f}",
                    f"{s.kf_vel:.4f}",
                    f"{s.cov_pos:.6e}",
                    f"{s.cov_vel:.6e}",
                    f"{s.kalman_gain:.4f}",
                    int(s.tag_detected),
                    f"{s.vx_cmd:.4f}",
                    f"{s.wz_cmd:.4f}"
                ])

    def compute_statistics(self) -> ExperimentStatistics:
        """Compute all statistical metrics across all recorded trials."""
        if not self.trials:
            raise RuntimeError("No completed trials to compute statistics.")

        N = len(self.trials)
        kf_errors = np.array([t.kf_error for t in self.trials])
        kf_abs_errors = np.array([t.kf_abs_error for t in self.trials])
        odo_errors = np.array([t.odo_error for t in self.trials])
        odo_abs_errors = np.array([t.odo_abs_error for t in self.trials])

        kf_mae = float(np.mean(kf_abs_errors))
        kf_mean = float(np.mean(kf_errors))
        kf_min = float(np.min(kf_errors))
        kf_max = float(np.max(kf_errors))
        kf_range = float(kf_max - kf_min)
        kf_std = float(np.std(kf_errors, ddof=1)) if N > 1 else 0.0

        odo_mae = float(np.mean(odo_abs_errors))
        odo_mean = float(np.mean(odo_errors))
        odo_min = float(np.min(odo_errors))
        odo_max = float(np.max(odo_errors))
        odo_range = float(odo_max - odo_min)
        odo_std = float(np.std(odo_errors, ddof=1)) if N > 1 else 0.0

        improvement = float((odo_mae - kf_mae) / max(odo_mae, 1e-6) * 100.0)

        return ExperimentStatistics(
            num_trials=N,
            start_distance=self.start_distance,
            target_stop_distance=self.stop_distance,
            kf_mae=kf_mae,
            kf_mean_error=kf_mean,
            kf_min_error=kf_min,
            kf_max_error=kf_max,
            kf_error_range=kf_range,
            kf_std_error=kf_std,
            odo_mae=odo_mae,
            odo_mean_error=odo_mean,
            odo_min_error=odo_min,
            odo_max_error=odo_max,
            odo_error_range=odo_range,
            odo_std_error=odo_std,
            mae_improvement_pct=improvement
        )

    def export_summary(self, filename_base: str = "experiment_summary"):
        """Save results to CSV and JSON."""
        stats = self.compute_statistics()

        # CSV export
        csv_path = os.path.join(self.output_dir, f"{filename_base}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Start_Dist_m", "Stop_Target_m",
                "KF_Est_Final_m", "Measured_Actual_m", "KF_Error_m", "KF_Abs_Error_m",
                "Odo_Final_m", "Odo_Error_m", "Duration_s"
            ])
            for t in self.trials:
                writer.writerow([
                    t.trial_id, f"{t.start_distance:.3f}", f"{t.target_stop_distance:.3f}",
                    f"{t.estimated_final_distance:.4f}", f"{t.measured_actual_distance:.4f}",
                    f"{t.kf_error:+.4f}", f"{t.kf_abs_error:.4f}",
                    f"{t.odometry_final_distance:.4f}", f"{t.odo_error:+.4f}",
                    f"{t.duration:.2f}"
                ])

        # JSON export
        json_path = os.path.join(self.output_dir, f"{filename_base}.json")
        data = {
            "statistics": asdict(stats),
            "trials": [asdict(t) for t in self.trials]
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[Results Exported]")
        print(f"  Summary CSV:  {csv_path}")
        print(f"  Summary JSON: {json_path}")

    def generate_markdown_table(self) -> str:
        """Format results as a GitHub-flavored Markdown table."""
        stats = self.compute_statistics()
        lines = []
        lines.append("### Experimental Odometry Results (10 Repeat Trials)")
        lines.append(f"**Starting Distance ($D_0$):** {self.start_distance:.2f} m | **Target Stopping Distance ($d_{{\\text{{stop}}}}$):** {self.stop_distance:.2f} m\n")
        lines.append("| Trial # | Measured Distance $d_m$ (m) | Estimated Distance $\\hat{d}$ (m) | Error $e = \\hat{d} - d_m$ (m) | Absolute Error $|e|$ (m) | Pure Odometry $d_{\\text{odo}}$ (m) | Odometry Error $e_{\\text{odo}}$ (m) |")
        lines.append("|:-------:|:---------------------------:|:-------------------------------:|:-----------------------------:|:------------------------:|:----------------------------------:|:-----------------------------------:|")

        for t in self.trials:
            lines.append(
                f"| {t.trial_id:02d} | {t.measured_actual_distance:.4f} | "
                f"{t.estimated_final_distance:.4f} | {t.kf_error:+.4f} | {t.kf_abs_error:.4f} | "
                f"{t.odometry_final_distance:.4f} | {t.odo_error:+.4f} |"
            )

        lines.append("")
        lines.append("#### Statistical Error Summary")
        lines.append("| Metric | Kalman Filter (Prediction + Correction) | Pure Wheel Odometry (Dead Reckoning) |")
        lines.append("|:-------|:---------------------------------------:|:------------------------------------:|")
        lines.append(f"| **Mean Absolute Error (MAE)** | **{stats.kf_mae*1000:.1f} mm ({stats.kf_mae:.4f} m)** | {stats.odo_mae*1000:.1f} mm ({stats.odo_mae:.4f} m) |")
        lines.append(f"| **Mean Signed Error (Bias)** | {stats.kf_mean_error*1000:+.1f} mm ({stats.kf_mean_error:+.4f} m) | {stats.odo_mean_error*1000:+.1f} mm ({stats.odo_mean_error:+.4f} m) |")
        lines.append(f"| **Minimum Error** | {stats.kf_min_error*1000:+.1f} mm | {stats.odo_min_error*1000:+.1f} mm |")
        lines.append(f"| **Maximum Error** | {stats.kf_max_error*1000:+.1f} mm | {stats.odo_max_error*1000:+.1f} mm |")
        lines.append(f"| **Error Range (Max - Min)** | **{stats.kf_error_range*1000:.1f} mm ({stats.kf_error_range:.4f} m)** | {stats.odo_error_range*1000:.1f} mm ({stats.odo_error_range:.4f} m) |")
        lines.append(f"| **Standard Deviation ($\\sigma$)** | {stats.kf_std_error*1000:.1f} mm | {stats.odo_std_error*1000:.1f} mm |")
        lines.append(f"| **MAE Reduction** | **{stats.mae_improvement_pct:.1f}% improvement** | Baseline |")

        return "\n".join(lines)

    def generate_latex_table(self) -> str:
        """Format results as an IEEE / academic LaTeX table."""
        stats = self.compute_statistics()
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Experimental Results and Estimation Errors across 10 Repeat Trials}")
        latex.append(r"\label{tab:odometry_results}")
        latex.append(r"\small")
        latex.append(r"\begin{tabular}{ccccccc}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Trial} & \textbf{Actual} $d_m$ & \textbf{KF Estimate} $\hat{d}$ & \textbf{KF Error} $e$ & \textbf{Abs Error} $|e|$ & \textbf{Pure Odo} $d_{\text{odo}}$ & \textbf{Odo Error} $e_{\text{odo}}$ \\")
        latex.append(r" & (m) & (m) & (m) & (m) & (m) & (m) \\")
        latex.append(r"\midrule")

        for t in self.trials:
            latex.append(
                f"{t.trial_id} & {t.measured_actual_distance:.4f} & "
                f"{t.estimated_final_distance:.4f} & {t.kf_error:+.4f} & {t.kf_abs_error:.4f} & "
                f"{t.odometry_final_distance:.4f} & {t.odo_error:+.4f} \\\\"
            )

        latex.append(r"\midrule")
        latex.append(f"\\textbf{{MAE}} & -- & -- & -- & \\textbf{{{stats.kf_mae:.4f}}} & -- & {stats.odo_mae:.4f} \\\\")
        latex.append(f"\\textbf{{Range}} & -- & -- & [{stats.kf_min_error:+.4f}, {stats.kf_max_error:+.4f}] & \\textbf{{{stats.kf_error_range:.4f}}} & -- & {stats.odo_error_range:.4f} \\\\")
        latex.append(f"\\textbf{{Std Dev}} & -- & -- & -- & {stats.kf_std_error:.4f} & -- & {stats.odo_std_error:.4f} \\\\")
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        return "\n".join(latex)

    def print_summary_table(self):
        """Print formatted results to console."""
        print(self.generate_markdown_table())
