"""
Generate publication-quality plots from ablation results.

Produces:
  1. Sample efficiency curve (success rate vs demo count, mean ± std)
  2. Capacity comparison bar chart (small vs baseline)
  3. Combined Week 6 summary figure

All plots saved as PNG + PDF in figures/.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


# Consistent styling for publication-quality figures
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})


def load_robust_eval(path):
    with open(path) as f:
        return json.load(f)
    

def load_dagger_results(path):
    """Parse DAgger robust eval JSON into {n_demos: [(seed, success), ...]}"""
    from collections import defaultdict
    with open(path) as f:
        results = json.load(f)
    
    by_demo_count = defaultdict(list)
    for r in results:
        # dagger_init100_seed1 → n_demos=100 (initial), seed=1
        name = Path(r["checkpoint"]).parent.name
        # Expected format: dagger_init<N>_seed<S>
        parts = name.split("_")
        n_demos = int(parts[1].replace("init", ""))
        seed = int(parts[2].replace("seed", ""))
        by_demo_count[n_demos].append((seed, r["success_rate"]))
    
    return by_demo_count


def parse_results(robust_results):
    """
    Organize robust eval results by config type.
    Returns dict: { 'demo_count': {n: [(seed, success), ...]}, 'capacity': {...} }
    """
    by_demo_count = defaultdict(list)
    by_capacity = defaultdict(list)
    baseline_runs = []  # bc_seed42/1/7 — no demos suffix, no cap suffix

    for r in robust_results:
        name = Path(r["checkpoint"]).parent.name
        success = r["success_rate"]

        if "demos" in name:
            # bc_demos250_seed1 → n_demos=250, seed=1
            parts = name.split("_")
            n_demos = int(parts[1].replace("demos", ""))
            seed = int(parts[2].replace("seed", ""))
            by_demo_count[n_demos].append((seed, success))
        elif "cap_" in name:
            # bc_cap_small_seed1 → capacity=small, seed=1
            parts = name.split("_")
            capacity = parts[2]
            seed = int(parts[3].replace("seed", ""))
            by_capacity[capacity].append((seed, success))
        else:
            # bc_seed42 → baseline (full 800 demos, baseline capacity)
            seed = int(name.replace("bc_seed", ""))
            baseline_runs.append((seed, success))

    # Treat baseline runs as the 800-demos point (full train set)
    if baseline_runs:
        by_demo_count[800] = baseline_runs

    return by_demo_count, by_capacity, baseline_runs


def compute_stats(runs):
    """runs = list of (seed, success_rate). Return mean, std, min, max."""
    vals = [s for _, s in runs]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
        "values": vals,
    }


def plot_sample_efficiency(by_demo_count, output_path):
    """Plot success rate vs demo count, mean ± std across seeds."""
    counts = sorted(by_demo_count.keys())
    stats = [compute_stats(by_demo_count[c]) for c in counts]
    means = [s["mean"] * 100 for s in stats]
    stds = [s["std"] * 100 for s in stats]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(counts, means, yerr=stds,
                marker="o", markersize=8, linewidth=2,
                color="#2E86AB", ecolor="#A8DADC", capsize=5,
                label="BC (mean ± std, 3 seeds)")

    # Annotate each point
    for c, m, s in zip(counts, means, stds):
        ax.annotate(f"{m:.0f}%",
                    xy=(c, m), xytext=(8, -12),
                    textcoords="offset points",
                    fontsize=9, color="#2E86AB", fontweight="bold")

    # Reference horizontal lines
    ax.axhline(y=15, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="Random baseline (~15%)")
    ax.axhline(y=100, color="green", linestyle=":", linewidth=1, alpha=0.6,
               label="Expert ceiling (100%)")

    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of demonstration episodes (log scale)")
    ax.set_ylabel("Success rate (%, 100 eval episodes)")
    ax.set_title("BC Sample Efficiency — FetchPickAndPlace")
    ax.set_ylim(-5, 110)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}.png + .pdf")


def plot_capacity(by_capacity, baseline_runs, output_path):
    """Bar chart: small capacity vs baseline."""
    # Map capacity → (mean, std, params, label)
    bars = []
    if "small" in by_capacity:
        s = compute_stats(by_capacity["small"])
        bars.append(("small\n[128, 128]\n~21k params", s["mean"] * 100,
                     s["std"] * 100, "#E63946"))
    if baseline_runs:
        s = compute_stats(baseline_runs)
        bars.append(("baseline\n[256, 256, 256]\n~140k params", s["mean"] * 100,
                     s["std"] * 100, "#2E86AB"))

    labels = [b[0] for b in bars]
    means = [b[1] for b in bars]
    stds = [b[2] for b in bars]
    colors = [b[3] for b in bars]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x_positions = np.arange(len(labels))
    ax.bar(x_positions, means, yerr=stds, color=colors,
           capsize=8, edgecolor="black", linewidth=1, width=0.6)

    for x, m, s in zip(x_positions, means, stds):
        ax.annotate(f"{m:.1f}% ± {s:.1f}%",
                    xy=(x, m), xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=100, color="green", linestyle=":", linewidth=1, alpha=0.6,
               label="Expert ceiling")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%, 100 eval episodes)")
    ax.set_title("BC Network Capacity Comparison")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}.png + .pdf")


def plot_seed_variance(by_demo_count, output_path):
    """Per-seed bars grouped by demo count — show within-config variance."""
    counts = sorted(by_demo_count.keys())
    seeds_unique = sorted({s for c in counts for s, _ in by_demo_count[c]})

    n_groups = len(counts)
    n_bars = len(seeds_unique)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2E86AB", "#E63946", "#F4A261"]
    for i, seed in enumerate(seeds_unique):
        vals = []
        for c in counts:
            seed_results = dict(by_demo_count[c])
            vals.append(seed_results.get(seed, 0) * 100)
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width,
               label=f"seed {seed}", color=colors[i % len(colors)],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of demonstration episodes")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Per-Seed Variance Across Demo Counts")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}.png + .pdf")


def plot_bc_vs_dagger(bc_by_demo_count, dagger_by_demo_count, output_path):
    """BC and DAgger sample efficiency on the same axes."""
    bc_counts = sorted(bc_by_demo_count.keys())
    bc_stats = [compute_stats(bc_by_demo_count[c]) for c in bc_counts]
    bc_means = [s["mean"] * 100 for s in bc_stats]
    bc_stds = [s["std"] * 100 for s in bc_stats]

    dagger_counts = sorted(dagger_by_demo_count.keys())
    dagger_stats = [compute_stats(dagger_by_demo_count[c]) for c in dagger_counts]
    dagger_means = [s["mean"] * 100 for s in dagger_stats]
    dagger_stds = [s["std"] * 100 for s in dagger_stats]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # BC curve (Week 6 result)
    ax.errorbar(bc_counts, bc_means, yerr=bc_stds,
                marker="o", markersize=8, linewidth=2,
                color="#2E86AB", ecolor="#A8DADC", capsize=5,
                label="BC (3 seeds, 100-ep robust eval)")

    # DAgger overlay
    ax.errorbar(dagger_counts, dagger_means, yerr=dagger_stds,
                marker="s", markersize=8, linewidth=2,
                color="#E63946", ecolor="#F4A8B0", capsize=5,
                label="DAgger (3 seeds, 100-ep robust eval)")

    # Annotate
    for c, m in zip(bc_counts, bc_means):
        ax.annotate(f"{m:.0f}%", xy=(c, m), xytext=(8, -14),
                    textcoords="offset points",
                    fontsize=9, color="#2E86AB", fontweight="bold")
    for c, m in zip(dagger_counts, dagger_means):
        ax.annotate(f"{m:.0f}%", xy=(c, m), xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9, color="#E63946", fontweight="bold")

    # Reference lines
    ax.axhline(y=15, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="Random baseline (~15%)")
    ax.axhline(y=100, color="green", linestyle=":", linewidth=1, alpha=0.6,
               label="Expert ceiling (100%)")

    ax.set_xscale("log")
    all_counts = sorted(set(bc_counts) | set(dagger_counts))
    ax.set_xticks(all_counts)
    ax.set_xticklabels([str(c) for c in all_counts])
    ax.set_xlabel("Initial demonstration episodes (log scale)")
    ax.set_ylabel("Success rate (%, 100 eval episodes)")
    ax.set_title("BC vs DAgger Sample Efficiency — FetchPickAndPlace")
    ax.set_ylim(-5, 115)
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}.png + .pdf")


def load_beta_results(linear_path, beta_path):
    """
    Combine linear schedule (from Week 7 dagger_robust_eval_all.json,
    init100 entries) with other schedules from Week 8 beta_robust_eval.json.
    Returns dict: {schedule_name: [(seed, success), ...]}
    """
    from collections import defaultdict
    by_schedule = defaultdict(list)

    # Linear: pull from Week 7 dagger results, only init100 entries
    with open(linear_path) as f:
        week7 = json.load(f)
    for r in week7:
        name = Path(r["checkpoint"]).parent.name
        if "init100" not in name:
            continue
        seed = int(name.split("seed")[1])
        by_schedule["linear"].append((seed, r["success_rate"]))

    # Other schedules from Week 8
    with open(beta_path) as f:
        week8 = json.load(f)
    for r in week8:
        name = Path(r["checkpoint"]).parent.name
        # dagger_beta_<schedule>_seed<S>
        parts = name.split("_")
        schedule = parts[2]
        seed = int(parts[3].replace("seed", ""))
        by_schedule[schedule].append((seed, r["success_rate"]))

    return by_schedule


def plot_beta_schedule_comparison(by_schedule, output_path):
    """Bar chart: success rate by β schedule, mean ± std."""
    # Order: linear, constant, exponential, threshold
    order = ["linear", "constant", "exponential", "threshold"]
    schedules = [s for s in order if s in by_schedule]

    means = []
    stds = []
    labels = []
    for s in schedules:
        stats = compute_stats(by_schedule[s])
        means.append(stats["mean"] * 100)
        stds.append(stats["std"] * 100)
        n = stats["n"]
        labels.append(f"{s}\n(n={n} seed{'s' if n > 1 else ''})")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2E86AB", "#F4A261", "#E76F51", "#6A4C93"]
    x = np.arange(len(schedules))
    ax.bar(x, means, yerr=stds, color=colors[:len(schedules)],
           capsize=8, edgecolor="black", linewidth=1, width=0.6)

    for xi, m, s in zip(x, means, stds):
        if s > 0:
            label = f"{m:.0f}% ± {s:.0f}%"
        else:
            label = f"{m:.0f}%"
        ax.annotate(label, xy=(xi, m), xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=15, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="Random baseline (~15%)")
    ax.axhline(y=100, color="green", linestyle=":", linewidth=1, alpha=0.6,
               label="Expert ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%, 100 eval episodes)")
    ax.set_title("DAgger β-Schedule Ablation (init 100 demos)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}.png + .pdf")


def print_summary_table(by_demo_count, by_capacity, baseline_runs):
    """Markdown table for README."""
    print("\n## Summary table (markdown) — copy to README\n")
    print("### Sample efficiency (A1)\n")
    print("| Demo count | Mean success | Std | Min | Max |")
    print("|---|---|---|---|---|")
    for c in sorted(by_demo_count.keys()):
        s = compute_stats(by_demo_count[c])
        print(f"| {c} | {s['mean']:.1%} | {s['std']:.1%} | "
              f"{s['min']:.0%} | {s['max']:.0%} |")

    print("\n### Network capacity (A2)\n")
    print("| Capacity | Hidden | Params | Mean success | Std |")
    print("|---|---|---|---|---|")
    if "small" in by_capacity:
        s = compute_stats(by_capacity["small"])
        print(f"| small | [128, 128] | ~21k | {s['mean']:.1%} | {s['std']:.1%} |")
    if baseline_runs:
        s = compute_stats(baseline_runs)
        print(f"| baseline | [256, 256, 256] | ~140k | "
              f"{s['mean']:.1%} | {s['std']:.1%} |")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="data/checkpoints/bc/robust_eval_all.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_robust_eval(args.input)
    by_demo_count, by_capacity, baseline_runs = parse_results(results)

    plot_sample_efficiency(by_demo_count, output_dir / "sample_efficiency")
    plot_capacity(by_capacity, baseline_runs, output_dir / "capacity_comparison")
    plot_seed_variance(by_demo_count, output_dir / "seed_variance")

    # Week 7: BC vs DAgger overlay
    dagger_results_path = "data/checkpoints/dagger/dagger_robust_eval_all.json"
    if Path(dagger_results_path).exists():
        dagger_by_demo_count = load_dagger_results(dagger_results_path)

        # Only show demo counts that exist in both BC and DAgger
        bc_subset = {c: by_demo_count[c]
                     for c in dagger_by_demo_count.keys()
                     if c in by_demo_count}
        plot_bc_vs_dagger(bc_subset, dagger_by_demo_count,
                          output_dir / "bc_vs_dagger")
        
        # Week 8: β schedule comparison
        linear_path = "data/checkpoints/dagger/dagger_robust_eval_all.json"
        beta_path = "data/checkpoints/dagger/beta_robust_eval.json"
        if Path(linear_path).exists() and Path(beta_path).exists():
            by_schedule = load_beta_results(linear_path, beta_path)
            plot_beta_schedule_comparison(by_schedule, output_dir / "beta_schedule")

            print("\n### β-schedule comparison (robust eval, DAgger init100)\n")
            print("| Schedule | Mean | Std | n |")
            print("|---|---|---|---|")
            for s in ["linear", "constant", "exponential", "threshold"]:
                if s not in by_schedule:
                    continue
                stats = compute_stats(by_schedule[s])
                print(f"| {s} | {stats['mean']:.1%} | {stats['std']:.1%} | {stats['n']} |")

        # Also print DAgger table
        print("\n### BC vs DAgger comparison (robust eval, mean ± std)\n")
        print("| Demos | BC | DAgger | Δ |")
        print("|---|---|---|---|")
        for c in sorted(dagger_by_demo_count.keys()):
            if c not in bc_subset:
                continue
            bc_s = compute_stats(bc_subset[c])
            d_s = compute_stats(dagger_by_demo_count[c])
            delta = (d_s["mean"] - bc_s["mean"]) * 100
            sign = "+" if delta >= 0 else ""
            print(f"| {c} | {bc_s['mean']:.1%} ± {bc_s['std']:.1%} | "
                  f"{d_s['mean']:.1%} ± {d_s['std']:.1%} | {sign}{delta:.1f} pp |")

    print_summary_table(by_demo_count, by_capacity, baseline_runs)


if __name__ == "__main__":
    main()