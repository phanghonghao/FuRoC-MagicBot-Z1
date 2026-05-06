#!/usr/bin/env python3
"""Generate a single-page A4 PDF report from Z1 training plots + analysis.

Reads 4 plot PNGs and best_models.json, generates a .tex file,
compiles to PDF, and cleans intermediate files.

Usage:
    python gen_report_pdf.py --alias s4_full
    python gen_report_pdf.py --alias phase_p1 --plots_dir ../plots --output_dir ../reports
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Run aliases (same as plot_learning_curves.py) ─────────────────────────── #
RUN_ALIASES = {
    "2026-04-30_04-53-17_s1_flat": "s1_flat",
    "2026-04-30_14-55-05_s1_stable": "s1_stable",
    "2026-05-01_01-21-35_s1_highspeed": "s1_highspeed",
    "2026-05-01_01-31-15_s3_rough_fail": "s3_rough_fail",
    "2026-05-01_04-44-07_s1_flat_retry": "s1_flat_retry",
    "2026-05-01_04-50-05_s2_gentle": "s2_gentle",
    "2026-05-01_07-04-35_s3_rough_l2": "s3_rough_l2",
    "2026-05-04_11-19-50_s3_rough_l1": "s3_rough_l1",
    "2026-05-04_12-30-56_s3_rough_l1_mgpu": "s3_rough_l1_mgpu",
    "2026-05-04_12-34-00_s3_rough_l1_mgpu_4gpu": "s3_rough_l1_mgpu_4gpu",
    "2026-05-04_12-40-26_s3_rough_l1_4gpu": "s3_rough_l1_4gpu",
    "2026-05-04_16-56-05_s4_full_terrain": "s4_full",
    "2026-05-05_04-47-06_s4_flat_deploy": "s4_flat_deploy",
    "2026-05-05_13-57-30_s5_explicit_pd": "s5_explicit_pd",
    "2026-05-06_04-55-58_phase_p1": "phase_p1",
    "2026-05-06_15-47-12_p1_coarse": "p1_coarse",
    "2026-05-06_17-40-13_p1_fine": "p1_fine",
}

PLOT_FILES = [
    "1_reward_trend.png",
    "2_reward_decomposition.png",
    "3_termination.png",
    "4_efficiency.png",
]


def load_best_models(json_path: str) -> dict:
    """Load best_models.json and find the model entry for the given alias."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def fetch_remote_metrics(alias: str) -> dict | None:
    """Fetch live training metrics from RTX server via SSH.

    Parses the training log for the latest iteration snapshot.
    """
    import subprocess as sp

    # Find the run directory on the server
    result = sp.run(
        ["ssh", "-o", "ConnectTimeout=10", "phh@192.168.120.155",
         f"ls -td ~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/*{alias}* 2>/dev/null | head -1"],
        capture_output=True, text=True, timeout=15,
    )
    run_dir = result.stdout.strip()
    if not run_dir:
        return None
    run_dir_name = run_dir.split("/")[-1]

    # Try to find the training log (use larger tail to find metrics past shutdown noise)
    log_text = ""
    for log_name in [f"train_{alias}", f"train_{alias.replace('-', '_')}", "z1_mgpu_" + alias]:
        for tail_n in ["80", "200"]:
            result = sp.run(
                ["ssh", "-o", "ConnectTimeout=10", "phh@192.168.120.155",
                 f"tail -{tail_n} ~/magiclab_rl_lab/logs/{log_name}.log 2>/dev/null || tail -{tail_n} /tmp/{log_name}.log 2>/dev/null"],
                capture_output=True, text=True, timeout=15,
            )
            log_text = result.stdout.strip()
            if log_text and "Mean reward" in log_text:
                break
        if log_text and "Mean reward" in log_text:
            break

    if not log_text:
        return None

    # Parse metrics from log
    metrics = {
        "status": "TRAINING",
        "overfitting_reason": None,
    }

    for line in log_text.split("\n"):
        line = line.strip()
        if "Mean reward" in line:
            try:
                metrics["latest_reward"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Mean episode length" in line:
            try:
                metrics["latest_episode_length"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Episode_Reward/action_rate" in line:
            try:
                metrics["latest_action_rate"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Episode_Termination/time_out" in line:
            try:
                metrics["latest_time_out"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Episode_Termination/bad_orientation" in line:
            try:
                metrics["latest_bad_orientation"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "error_vel_xy" in line:
            try:
                metrics["latest_vel_error"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Time elapsed" in line:
            # Format: "Time elapsed: HH:MM:SS"
            parts = line.split("Time elapsed")
            if len(parts) > 1:
                val = parts[1].lstrip(": ").strip()
                if val:
                    metrics["time_elapsed"] = val
        if "ETA" in line and "Time elapsed" not in line:
            # Format: "ETA: HH:MM:SS"
            parts = line.split("ETA")
            if len(parts) > 1:
                val = parts[1].lstrip(": ").strip()
                if val:
                    metrics["eta"] = val

    # Find latest model checkpoint
    result = sp.run(
        ["ssh", "-o", "ConnectTimeout=10", "phh@192.168.120.155",
         f"ls -t {run_dir}/model_*.pt 2>/dev/null | head -1"],
        capture_output=True, text=True, timeout=15,
    )
    model_path = result.stdout.strip()
    if model_path:
        model_file = os.path.basename(model_path)
        iter_str = model_file.replace("model_", "").replace(".pt", "")
        try:
            metrics["best_model_iteration"] = int(iter_str)
            metrics["latest_iteration"] = int(iter_str)
        except ValueError:
            pass

    # Set peak/latest to same as latest_reward for ongoing runs
    if "latest_reward" in metrics:
        metrics["peak_reward"] = metrics["latest_reward"]
        metrics["peak_reward_iter"] = metrics.get("latest_iteration", 0)
        metrics["best_model_reward"] = metrics["latest_reward"]

    return metrics


def find_model_entry(data: dict, alias: str) -> dict | None:
    """Find the model entry matching the alias.

    Tries, in order:
    1. Exact RUN_ALIASES match (alias → full_dir → run_dir)
    2. Partial match on model["version"] field
    3. Partial match on model["run_dir"] field
    """
    models = data.get("models", [])

    # Strategy 1: exact RUN_ALIASES match
    for model in models:
        run_dir = model.get("run_dir", "")
        for full_name, short_name in RUN_ALIASES.items():
            if short_name == alias and run_dir == full_name:
                return model

    # Strategy 2: partial match on "version" field
    for model in models:
        version = model.get("version", "")
        if version and version == alias:
            return model

    # Strategy 3: partial match on "run_dir" containing alias
    for model in models:
        run_dir = model.get("run_dir", "")
        if alias in run_dir:
            return model

    return None


def generate_analysis(m: dict) -> str:
    """Generate analysis text from model data."""
    lines = []

    status = m.get("status", "UNKNOWN")
    peak = m.get("peak_reward", 0)
    peak_iter = m.get("peak_reward_iter", 0)
    best_iter = m.get("best_model_iteration", 0)
    best_reward = m.get("best_model_reward", 0)
    latest_iter = m.get("latest_iteration", 0)
    latest_reward = m.get("latest_reward", 0)

    # Overfitting analysis
    if status == "OVERFITTING" and peak is not None and latest_reward is not None:
        decline_pct = ((peak - latest_reward) / abs(peak) * 100) if peak != 0 else 0
        reason = m.get("overfitting_reason") or ""
        lines.append(
            f"\\textbf{{Overfitting detected.}} Reward declined {decline_pct:.1f}\\% from peak "
            f"({latest_reward:.2f} vs peak {peak:.2f} @ iter {peak_iter:,})."
        )
        lines.append(
            f"Best checkpoint at iter {best_iter:,} (reward {best_reward:.2f}), "
            f"saved before significant degradation began."
        )
    elif status == "HEALTHY" and peak is not None:
        gap = peak - best_reward
        lines.append(
            f"\\textbf{{Healthy training.}} Policy converged steadily with peak reward "
            f"{peak:.2f} @ iter {peak_iter:,}."
        )
        lines.append(
            f"Best model reward {best_reward:.2f} (gap from peak: {gap:.2f})."
        )
    elif status == "TRAINING":
        lines.append(
            f"\\textbf{{Training in progress.}} Current reward {latest_reward:.2f} "
            f"(iter {latest_iter:,})."
        )
        if "time_elapsed" in m:
            lines.append(f"Elapsed: {m['time_elapsed']}.")
        if "eta" in m:
            lines.append(f"ETA: {m['eta']}.")
    elif status == "UNKNOWN":
        lines.append("No detailed metrics available in best\\_models.json.")
        lines.append("Run /gpu-train health check to update model tracking data.")
    else:
        lines.append(f"Status: {status}.")

    # Termination analysis
    time_out = m.get("latest_time_out")
    bad_orient = m.get("latest_bad_orientation")
    ep_len = m.get("latest_episode_length")

    if time_out is not None and bad_orient is not None:
        if bad_orient > 0.5:
            lines.append(
                f"High fall rate: bad\\_orientation = {bad_orient:.2f}, "
                f"only {time_out:.0%} episodes complete successfully."
            )
        elif time_out and time_out > 0.8:
            lines.append(
                f"Stable locomotion: {time_out:.0%} episodes timeout (complete), "
                f"fall rate only {bad_orient:.2f}."
            )

    if ep_len is not None:
        max_ep = 1000
        pct = ep_len / max_ep * 100
        lines.append(
            f"Mean episode length: {ep_len:.0f}/{max_ep} steps ({pct:.0f}% of max)."
        )

    # Velocity tracking
    vel_err = m.get("latest_vel_error")
    if vel_err is not None:
        if vel_err < 0.4:
            lines.append(f"Good velocity tracking: error = {vel_err:.3f}.")
        elif vel_err < 1.0:
            lines.append(f"Moderate velocity tracking: error = {vel_err:.3f}.")
        else:
            lines.append(f"Poor velocity tracking: error = {vel_err:.3f}.")

    # Action smoothness
    action_rate = m.get("latest_action_rate")
    if action_rate is not None:
        if action_rate < -1.0:
            lines.append(
                f"High action rate penalty ({action_rate:.2f}) suggests erratic policy output."
            )

    # Recommendations
    lines.append("")  # blank line
    lines.append("\\textbf{Recommendations:}")

    if status == "OVERFITTING" and peak is not None and latest_reward is not None:
        decline_pct_val = ((peak - latest_reward) / abs(peak) * 100) if peak != 0 else 0
        if decline_pct_val > 100:
            lines.append(
                "Severe collapse --- consider early stopping, reduced LR, or increased regularization."
            )
        elif decline_pct_val > 50:
            lines.append(
                "Moderate overfitting --- try entropy bonus increase or shorter training schedule."
            )
        else:
            lines.append(
                "Mild overfitting --- current best checkpoint is usable, monitor future runs."
            )
        if bad_orient and bad_orient > 0.3:
            lines.append(
                "High fall rate suggests policy instability; consider reducing action scale or increasing PD gains."
            )
    elif status == "HEALTHY":
        lines.append("Continue monitoring; consider deploying best checkpoint for sim2sim validation.")

    return "\n".join(lines)


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    s = s.replace("\\", "\\\\")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def generate_md(alias: str, model: dict) -> str:
    """Generate a Markdown summary of the training run."""
    status = model.get("status", "UNKNOWN")
    peak = model.get("peak_reward") or 0
    peak_iter = model.get("peak_reward_iter") or 0
    best_iter = model.get("best_model_iteration") or 0
    best_reward = model.get("best_model_reward") or 0
    latest_iter = model.get("latest_iteration") or 0
    latest_reward = model.get("latest_reward") or 0
    ep_len = model.get("latest_episode_length") or 0
    time_out = model.get("latest_time_out")
    bad_ori = model.get("latest_bad_orientation")
    action_rate = model.get("latest_action_rate")
    vel_err = model.get("latest_vel_error")
    time_elapsed = model.get("time_elapsed", "N/A")
    eta = model.get("eta", "N/A")
    reason = model.get("overfitting_reason") or ""

    lines = [
        f"# Z1 12DOF Training Report — {alias}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Status:** {status}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Peak Reward | {peak:.2f} @ iter {peak_iter:,} |",
        f"| Best Checkpoint | model_{best_iter}.pt (reward: {best_reward:.2f}) |",
        f"| Latest | {latest_reward:.2f} @ iter {latest_iter:,} |",
        f"| Episode Length | {ep_len:.0f} / 1000 ({ep_len/10:.0f}%) |",
    ]

    if time_out is not None:
        lines.append(f"| Time-out Rate | {time_out:.2%} |")
    if bad_ori is not None:
        lines.append(f"| Bad Orientation | {bad_ori:.2%} |")
    if action_rate is not None:
        lines.append(f"| Action Rate | {action_rate:.3f} |")
    if vel_err is not None:
        lines.append(f"| Velocity Error | {vel_err:.3f} |")
    if time_elapsed != "N/A":
        lines.append(f"| Elapsed | {time_elapsed} |")
    if eta != "N/A":
        lines.append(f"| ETA | {eta} |")

    lines.append("")
    if reason:
        lines.append(f"**Overfitting reason:** {reason}")
        lines.append("")

    # Plots
    lines.append("## Plots")
    lines.append("")
    for fname in PLOT_FILES:
        name = fname.replace(".png", "").replace("_", " ").title()
        lines.append(f"![{name}]({fname})")
        lines.append("")

    return "\n".join(lines)


def generate_tex(
    alias: str,
    plots: dict[str, str],
    model: dict,
    output_path: str,
) -> str:
    """Generate the .tex file content."""

    status = model.get("status") or "UNKNOWN"
    peak = model.get("peak_reward") or 0
    peak_iter = model.get("peak_reward_iter") or 0
    best_iter = model.get("best_model_iteration") or 0
    best_reward = model.get("best_model_reward") or 0
    latest_iter = model.get("latest_iteration") or 0
    latest_reward = model.get("latest_reward") or 0
    overfitting_reason = model.get("overfitting_reason") or ""

    # Status color
    if status == "HEALTHY":
        status_color = "green!60!black"
    elif status == "OVERFITTING":
        status_color = "red!70!black"
    elif status == "TRAINING":
        status_color = "blue!70!black"
    else:
        status_color = "black"

    analysis = generate_analysis(model)

    # Convert Windows backslash paths to forward slash for LaTeX
    def tex_path(p: str) -> str:
        return p.replace("\\", "/")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    tex = rf"""\documentclass[a4paper,10pt]{{article}}
\usepackage[top=8mm, bottom=8mm, left=8mm, right=8mm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{parskip}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{2pt}}

\begin{{document}}

% ── Header ────────────────────────────────────────────────────────────── %
\begin{{minipage}}{{0.65\textwidth}}
  {{\LARGE\bfseries Z1 12DOF Training Report}}\\[2pt]
  {{\large Run: \texttt{{{escape_latex(alias)}}} \quad|\quad {now}}}
\end{{minipage}}%
\hfill
\begin{{minipage}}{{0.3\textwidth}}
  \raggedleft
  {{\large\bfseries\textcolor{{{status_color}}}{{{escape_latex(status)}}}}}
\end{{minipage}}

\vspace{{3pt}}
\hrule
\vspace{{4pt}}

% ── Summary Table ─────────────────────────────────────────────────────── %
\footnotesize
\begin{{tabularx}}{{\textwidth}}{{|l|l|l|l|l|l|l|}}
\hline
\textbf{{Peak Reward}} & \textbf{{@ Iter}} & \textbf{{Best Ckpt}} & \textbf{{Best Reward}}
  & \textbf{{Latest}} & \textbf{{@ Iter}} & \textbf{{Episode Len}} \\
\hline
{peak:.2f} & {peak_iter:,} & model\_{best_iter}.pt & {best_reward:.2f}
  & {latest_reward:.2f} & {latest_iter:,} & {(model.get("latest_episode_length") or 0):.0f} \\
\hline
\end{{tabularx}}

\normalsize
\vspace{{3pt}}

% ── Plots 2x2 Grid ───────────────────────────────────────────────────── %
% Row 1: Reward Trend + Reward Decomposition
\begin{{minipage}}[t]{{0.49\textwidth}}
  \centering
  \includegraphics[width=\textwidth,height=0.38\textheight,keepaspectratio]{{{tex_path(plots["1_reward_trend.png"])}}}
\end{{minipage}}%
\hfill
\begin{{minipage}}[t]{{0.49\textwidth}}
  \centering
  \includegraphics[width=\textwidth,height=0.38\textheight,keepaspectratio]{{{tex_path(plots["2_reward_decomposition.png"])}}}
\end{{minipage}}

\vspace{{1pt}}

% Row 2: Termination + Efficiency
\begin{{minipage}}[t]{{0.49\textwidth}}
  \centering
  \includegraphics[width=\textwidth,height=0.38\textheight,keepaspectratio]{{{tex_path(plots["3_termination.png"])}}}
\end{{minipage}}%
\hfill
\begin{{minipage}}[t]{{0.49\textwidth}}
  \centering
  \includegraphics[width=\textwidth,height=0.38\textheight,keepaspectratio]{{{tex_path(plots["4_efficiency.png"])}}}
\end{{minipage}}

\vspace{{2pt}}
\hrule
\vspace{{2pt}}

% ── Analysis ──────────────────────────────────────────────────────────── %
\footnotesize
\textbf{{Analysis}} \\[1pt]
{analysis}

\end{{document}}
"""
    return tex


def compile_pdf(tex_path: str) -> str:
    """Compile .tex to .pdf using pdflatex, clean intermediates."""
    tex_dir = os.path.dirname(tex_path)
    tex_name = os.path.splitext(os.path.basename(tex_path))[0]

    # Run pdflatex twice for references
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name + ".tex"],
            cwd=tex_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"pdflatex error:\n{result.stdout[-2000:]}\n{result.stderr[-1000:]}")
            sys.exit(1)

    pdf_path = os.path.join(tex_dir, tex_name + ".pdf")

    # Clean intermediate files
    exts = [".aux", ".log", ".synctex.gz", ".out", ".toc"]
    for ext in exts:
        f = os.path.join(tex_dir, tex_name + ext)
        if os.path.exists(f):
            os.remove(f)

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Generate A4 PDF report from Z1 training plots")
    parser.add_argument("--alias", required=True, help="Run alias (e.g. s4_full, phase_p1)")
    parser.add_argument(
        "--plots_dir",
        default=None,
        help="Root plots directory (default: auto-detect relative to script)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for PDF (default: same as plots_dir/<alias>)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    # Resolve plots directory
    if args.plots_dir:
        plots_root = Path(args.plots_dir).resolve()
    else:
        plots_root = project_dir / "plots"

    run_plots_dir = plots_root / args.alias

    # Check all 4 plots exist
    plot_paths = {}
    for fname in PLOT_FILES:
        fpath = run_plots_dir / fname
        if not fpath.exists():
            print(f"ERROR: Missing plot {fpath}")
            print(f"Run '/plot-train-Z1 --focus {args.alias}' first to generate plots.")
            sys.exit(1)
        plot_paths[fname] = str(fpath.resolve())

    # Load best_models.json
    json_path = project_dir / "best_models.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)

    data = load_best_models(str(json_path))
    model = find_model_entry(data, args.alias)
    if model is None:
        # Try fetching live metrics from RTX server
        print(f"INFO: '{args.alias}' not in best_models.json — fetching live metrics from RTX...")
        model = fetch_remote_metrics(args.alias)
        if model is None:
            print(f"WARNING: Could not fetch live metrics — generating with placeholder data.")
            model = {
                "status": "UNKNOWN",
                "peak_reward": None,
                "peak_reward_iter": None,
                "best_model_iteration": None,
                "best_model_reward": None,
                "latest_iteration": None,
                "latest_reward": None,
                "latest_episode_length": None,
                "latest_time_out": None,
                "latest_bad_orientation": None,
                "latest_vel_error": None,
                "latest_action_rate": None,
                "overfitting_reason": "No data available — run /gpu-train health check to update.",
            }

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = run_plots_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate .tex
    tex_filename = f"report_{args.alias}.tex"
    tex_path = output_dir / tex_filename

    tex_content = generate_tex(args.alias, plot_paths, model, str(tex_path))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Generated: {tex_path}")

    # Compile to PDF
    pdf_path = compile_pdf(str(tex_path))
    print(f"PDF report: {pdf_path}")

    # Generate .MD summary
    md_path = output_dir / f"report_{args.alias}.md"
    md_content = generate_md(args.alias, model)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"MD summary: {md_path}")

    # Keep .tex and .md files
    print(f"TeX source: {tex_path}")

    return pdf_path


if __name__ == "__main__":
    main()
