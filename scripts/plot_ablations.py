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

    print_summary_table(by_demo_count, by_capacity, baseline_runs)


if __name__ == "__main__":
    main()