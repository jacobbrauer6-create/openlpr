"""
scripts/visualise_results.py
-----------------------------
Reads all completed training run logs from models/checkpoints/
and produces:

  1. A formatted summary table printed to terminal
  2. A multi-panel matplotlib figure saved to evaluation/training_results.pdf
     and evaluation/training_results.png

All labels, tick values, and annotations use LaTeX formatting.

Usage:
    python scripts/visualise_results.py
    python scripts/visualise_results.py --checkpoint-dir models/checkpoints
    python scripts/visualise_results.py --output evaluation/my_results.pdf
    python scripts/visualise_results.py --no-latex   # if LaTeX not installed

LaTeX requirement:
    Full LaTeX rendering needs a system LaTeX installation.
    Windows : install MiKTeX  https://miktex.org/
    macOS   : brew install --cask mactex
    Linux   : sudo apt install texlive-full
    If LaTeX is unavailable, pass --no-latex to use Matplotlib's built-in
    mathtext renderer (no installation required, near-identical output).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe


# ---------------------------------------------------------------------------
# Colour scheme — one colour per architectural family
# ---------------------------------------------------------------------------

FAMILY_COLORS = {
    "ResNet":       "#4C72B0",   # classic blue
    "EfficientNet": "#DD8452",   # warm orange
    "MobileNet":    "#55A868",   # green
    "SqueezeNet":   "#C44E52",   # red
    "ShuffleNet":   "#8172B2",   # purple
    "RegNet":       "#937860",   # brown
    "DenseNet":     "#DA8BC3",   # pink
    "ConvNeXt":     "#8C8C8C",   # grey
    "ViT":          "#CCB974",   # gold
}

FAMILY_MARKERS = {
    "ResNet":       "o",
    "EfficientNet": "s",
    "MobileNet":    "^",
    "SqueezeNet":   "D",
    "ShuffleNet":   "v",
    "RegNet":       "P",
    "DenseNet":     "X",
    "ConvNeXt":     "*",
    "ViT":          "h",
}

# Architecture family → brief description for legend
FAMILY_NOTES = {
    "ResNet":       r"Residual connections (He et al.\ 2015)",
    "EfficientNet": "Compound scaling (Tan \\& Le 2019)",
    "MobileNet":    r"Depthwise separable conv (Howard et al.\ 2019)",
    "SqueezeNet":   r"Fire modules, 1.2M params (Iandola et al.\ 2016)",
    "ShuffleNet":   r"Channel shuffle, ARM-optimised (Ma et al.\ 2018)",
    "RegNet":       r"Grid-search optimal (Radosavovic et al.\ 2020)",
    "DenseNet":     r"Dense connections (Huang et al.\ 2017)",
    "ConvNeXt":     r"Modernised CNN (Liu et al.\ 2022)",
    "ViT":          r"Pure self-attention (Dosovitskiy et al.\ 2020)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(checkpoint_dir: Path) -> list[dict]:
    """
    Walk checkpoint_dir/<backbone>/run_log.json and aggregate results.
    Returns a list of dicts, one per trained backbone.
    """
    results = []

    # Try reading backbone family from train.py's BACKBONE_CONFIGS at runtime
    backbone_meta = _load_backbone_meta()

    for backbone_dir in sorted(checkpoint_dir.iterdir()):
        if not backbone_dir.is_dir():
            continue

        run_log_path = backbone_dir / "run_log.json"
        if not run_log_path.exists():
            continue

        try:
            with open(run_log_path) as f:
                run_log = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not read {run_log_path}: {e}")
            continue

        if not run_log:
            continue

        backbone = backbone_dir.name
        meta     = backbone_meta.get(backbone, {})

        last     = run_log[-1]
        best_iou = max(e["val_iou"] for e in run_log)
        best_epoch = next(i+1 for i, e in enumerate(run_log)
                          if e["val_iou"] == best_iou)

        results.append({
            "backbone":       backbone,
            "family":         meta.get("family", "Unknown"),
            "params_M":       meta.get("params_M", 0.0),
            "size_mb":        meta.get("size_mb", 0.0),
            "lat_p50_ms":     meta.get("lat_ms", 0.0),
            "best_iou":       best_iou,
            "best_epoch":     best_epoch,
            "final_char_acc": last["val_char_acc"],
            "final_train_loss": last["train_loss"],
            "final_val_loss":   last["val_loss"],
            "total_energy_kwh": last["cumulative_energy_kwh"],
            "total_co2_g":      last["cumulative_co2_kg"] * 1000,
            "total_epochs":     len(run_log),
            "avg_epoch_s":      np.mean([e["epoch_time_s"] for e in run_log]),
            "run_log":          run_log,  # full curve for training plots
        })

    results.sort(key=lambda x: x["best_iou"], reverse=True)
    return results


def _load_backbone_meta() -> dict:
    """
    Try to import BACKBONE_CONFIGS from scripts/train.py.
    Falls back to a hardcoded lookup if the import fails.
    """
    try:
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "train",
            os.path.join(os.path.dirname(__file__), "train.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {
            k: {
                "family":   v.get("family", "Unknown"),
                "params_M": v.get("params_M", 0.0),
                "size_mb":  0.0,    # not in BACKBONE_CONFIGS, set later
                "lat_ms":   0.0,
            }
            for k, v in mod.BACKBONE_CONFIGS.items()
        }
    except Exception:
        pass

    # Fallback hardcoded
    return {
        "resnet18":           {"family": "ResNet",       "params_M": 11.2,  "size_mb": 43.1,  "lat_ms": 4.2},
        "resnet34":           {"family": "ResNet",       "params_M": 21.3,  "size_mb": 81.4,  "lat_ms": 6.1},
        "resnet50":           {"family": "ResNet",       "params_M": 25.6,  "size_mb": 98.0,  "lat_ms": 9.3},
        "resnet101":          {"family": "ResNet",       "params_M": 44.5,  "size_mb": 171.4, "lat_ms": 16.8},
        "efficientnet_b0":    {"family": "EfficientNet", "params_M": 5.3,   "size_mb": 21.4,  "lat_ms": 7.1},
        "efficientnet_b2":    {"family": "EfficientNet", "params_M": 7.7,   "size_mb": 31.2,  "lat_ms": 9.8},
        "mobilenet_v3_small": {"family": "MobileNet",    "params_M": 2.5,   "size_mb": 10.2,  "lat_ms": 3.1},
        "mobilenet_v3_large": {"family": "MobileNet",    "params_M": 5.5,   "size_mb": 22.1,  "lat_ms": 5.4},
        "squeezenet1_1":      {"family": "SqueezeNet",   "params_M": 1.2,   "size_mb": 4.9,   "lat_ms": 2.3},
        "shufflenet_v2":      {"family": "ShuffleNet",   "params_M": 2.3,   "size_mb": 9.4,   "lat_ms": 2.9},
        "regnet_y_400mf":     {"family": "RegNet",       "params_M": 4.3,   "size_mb": 17.6,  "lat_ms": 5.8},
        "densenet121":        {"family": "DenseNet",     "params_M": 8.0,   "size_mb": 32.3,  "lat_ms": 11.2},
        "convnext_tiny":      {"family": "ConvNeXt",     "params_M": 28.6,  "size_mb": 114.8, "lat_ms": 12.4},
        "vit_b_16":           {"family": "ViT",          "params_M": 86.6,  "size_mb": 346.2, "lat_ms": 28.6},
    }


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_terminal_table(results: list[dict]):
    """Print a rich summary table to stdout."""
    if not results:
        print("No results found.")
        return

    col_w = [22, 12, 8, 8, 8, 8, 8, 8, 9]
    header = (
        f"{'Backbone':<{col_w[0]}} {'Family':<{col_w[1]}} "
        f"{'IoU':>{col_w[2]}} {'CharAcc':>{col_w[3]}} "
        f"{'p50ms':>{col_w[4]}} {'Params':>{col_w[5]}} "
        f"{'SizeMB':>{col_w[6]}} {'CO₂(g)':>{col_w[7]}} {'kWh':>{col_w[8]}}"
    )
    sep = "─" * sum(col_w) + "─" * (len(col_w) - 1)

    print(f"\n{'OpenLPR — Training Results':^{len(sep)}}")
    print(sep)
    print(header)
    print(sep)

    best_iou = max(r["best_iou"] for r in results)

    for r in results:
        star  = " ★" if r["best_iou"] == best_iou else "  "
        print(
            f"{r['backbone']:<{col_w[0]}}{star[:1]}"
            f"{r['family']:<{col_w[1]}} "
            f"{r['best_iou']:>{col_w[2]}.4f} "
            f"{r['final_char_acc']:>{col_w[3]}.4f} "
            f"{r['lat_p50_ms']:>{col_w[4]}.1f} "
            f"{r['params_M']:>{col_w[5]}.1f}M "
            f"{r['size_mb']:>{col_w[6]}.1f} "
            f"{r['total_co2_g']:>{col_w[7]}.1f} "
            f"{r['total_energy_kwh']:>{col_w[8]}.4f}"
        )

    print(sep)
    best = results[0]
    print(f"  ★ Best IoU: {best['backbone']} ({best['best_iou']:.4f})")
    total_co2 = sum(r["total_co2_g"] for r in results)
    total_kwh = sum(r["total_energy_kwh"] for r in results)
    print(f"  Total CO₂ across all runs : {total_co2:.1f} g")
    print(f"  Total energy              : {total_kwh:.4f} kWh")
    print()


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def setup_matplotlib(use_latex: bool):
    """Configure matplotlib with LaTeX or mathtext rendering."""
    if use_latex:
        try:
            matplotlib.rcParams.update({
                "text.usetex":          True,
                "font.family":          "serif",
                "font.serif":           ["Computer Modern Roman"],
                "axes.formatter.use_mathtext": True,
            })
            # Quick test render
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_title(r"$\alpha$")
            fig.canvas.draw()
            plt.close(fig)
            print("  LaTeX rendering: enabled (system LaTeX found)")
            return True
        except Exception as e:
            print(f"  LaTeX rendering: disabled ({e})")
            print("  Falling back to mathtext — install MiKTeX for full LaTeX")
            matplotlib.rcParams["text.usetex"] = False

    # Mathtext fallback — no system LaTeX needed
    matplotlib.rcParams.update({
        "text.usetex":          False,
        "font.family":          "DejaVu Serif",
        "mathtext.fontset":     "cm",
        "axes.formatter.use_mathtext": True,
    })
    print("  LaTeX rendering: mathtext mode (no system LaTeX required)")
    return False


def apply_plot_style(ax, use_latex: bool):
    """Apply consistent style to an axis."""
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8, length=3)


def tex(s: str, use_latex: bool) -> str:
    """Return LaTeX string if latex enabled, else plain text fallback."""
    if use_latex:
        return s
    # Strip LaTeX commands for plain text
    import re
    s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = s.replace(r"\ ", " ").replace("\\&", "&")
    return s


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------

def plot_accuracy_bars(ax, results, use_latex):
    """Grouped bar chart: IoU + character accuracy per backbone."""
    n     = len(results)
    x     = np.arange(n)
    w     = 0.38
    names = [r["backbone"].replace("_", r"\_" if use_latex else "_")
             for r in results]

    colors_iou  = [FAMILY_COLORS.get(r["family"], "#666") for r in results]
    colors_char = [c + "99" for c in colors_iou]  # semi-transparent variant

    bars_iou = ax.bar(x - w/2, [r["best_iou"]       for r in results],
                      width=w, color=colors_iou, edgecolor="white",
                      linewidth=0.5, label=tex(r"Val $\mathrm{IoU}$", use_latex),
                      zorder=3)
    bars_char = ax.bar(x + w/2, [r["final_char_acc"] for r in results],
                       width=w, color=colors_char, edgecolor="white",
                       linewidth=0.5, label=tex(r"Char Accuracy", use_latex),
                       zorder=3)

    # Value labels on bars
    for bar in list(bars_iou) + list(bars_char):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=7.5)
    ax.set_ylabel(tex(r"Score", use_latex), fontsize=9)
    ax.set_ylim(0.82, 1.01)
    ax.set_title(tex(r"\textbf{Accuracy} -- IoU \& Character Accuracy by Backbone",
                     use_latex), fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    apply_plot_style(ax, use_latex)

    # Colour family legend patches
    seen_families = []
    for r in results:
        if r["family"] not in seen_families:
            seen_families.append(r["family"])
    from matplotlib.patches import Patch
    family_handles = [
        Patch(color=FAMILY_COLORS.get(f, "#666"),
              label=tex(f.replace("&", r"\&"), use_latex))
        for f in seen_families
    ]
    ax.legend(handles=family_handles, fontsize=7, title=tex(r"\textit{Family}",
              use_latex), title_fontsize=7.5, loc="lower left",
              framealpha=0.9, ncol=2)


def plot_accuracy_vs_latency(ax, results, use_latex):
    """Scatter plot: latency on x-axis, IoU on y-axis, point size = params."""
    for r in results:
        color  = FAMILY_COLORS.get(r["family"], "#666")
        marker = FAMILY_MARKERS.get(r["family"], "o")
        size   = max(30, r["params_M"] * 3.5)

        ax.scatter(r["lat_p50_ms"], r["best_iou"],
                   s=size, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8,
                   zorder=4, alpha=0.92)

        # Label each point
        x_off = 0.3
        ax.annotate(
            r["backbone"].replace("_", r"\_" if use_latex else "_"),
            xy=(r["lat_p50_ms"], r["best_iou"]),
            xytext=(r["lat_p50_ms"] + x_off, r["best_iou"] + 0.0005),
            fontsize=6, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                            lw=0.5, shrinkA=0, shrinkB=2),
        )

    ax.set_xlabel(tex(r"Inference latency $p_{50}$ (ms)", use_latex), fontsize=9)
    ax.set_ylabel(tex(r"Best validation $\mathrm{IoU}$", use_latex), fontsize=9)
    ax.set_title(tex(r"\textbf{Accuracy vs.\ Latency} (point area $\propto$ parameters)",
                     use_latex), fontsize=10, pad=8)

    # Efficiency frontier annotation
    ax.axvline(x=10, color="#cc4444", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(10.2, ax.get_ylim()[0] + 0.002,
            tex(r"$10\,\mathrm{ms}$ threshold", use_latex),
            color="#cc4444", fontsize=7, alpha=0.8)

    apply_plot_style(ax, use_latex)


def plot_training_curves(ax, results, use_latex):
    """Line plot of val IoU over epochs for every backbone."""
    for r in results:
        color  = FAMILY_COLORS.get(r["family"], "#666")
        epochs = [e["epoch"]   for e in r["run_log"]]
        ious   = [e["val_iou"] for e in r["run_log"]]
        ax.plot(epochs, ious, color=color, linewidth=1.2,
                alpha=0.85, label=r["backbone"])

    ax.set_xlabel(tex(r"Epoch", use_latex), fontsize=9)
    ax.set_ylabel(tex(r"Validation $\mathrm{IoU}$", use_latex), fontsize=9)
    ax.set_title(tex(r"\textbf{Training Curves} -- Validation IoU over Epochs",
                     use_latex), fontsize=10, pad=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    apply_plot_style(ax, use_latex)


def plot_energy_co2(ax, results, use_latex):
    """Horizontal bar: CO₂ per run, coloured by family."""
    # Sort by CO₂ ascending for readability
    sorted_r = sorted(results, key=lambda x: x["total_co2_g"])
    names    = [r["backbone"].replace("_", r"\_" if use_latex else "_")
                for r in sorted_r]
    co2vals  = [r["total_co2_g"] for r in sorted_r]
    colors   = [FAMILY_COLORS.get(r["family"], "#666") for r in sorted_r]

    bars = ax.barh(names, co2vals, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=3)

    for bar, val in zip(bars, co2vals):
        ax.text(val + max(co2vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                tex(f"${val:.1f}$" + r"\,g", use_latex),
                va="center", ha="left", fontsize=7, color="#333333")

    ax.set_xlabel(tex(r"$\mathrm{CO}_2$ emitted (g)", use_latex), fontsize=9)
    ax.set_title(tex(r"\textbf{Energy Cost} -- $\mathrm{CO}_2$ per Training Run",
                     use_latex), fontsize=10, pad=8)
    ax.set_xlim(0, max(co2vals) * 1.22)
    apply_plot_style(ax, use_latex)


def plot_params_vs_accuracy(ax, results, use_latex):
    """
    Scatter: model size (params) vs IoU.
    Annotates the Pareto-efficient frontier.
    """
    for r in results:
        color  = FAMILY_COLORS.get(r["family"], "#666")
        marker = FAMILY_MARKERS.get(r["family"], "o")
        ax.scatter(r["params_M"], r["best_iou"],
                   s=70, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8,
                   zorder=4, alpha=0.9)
        ax.annotate(
            r["backbone"].replace("_", r"\_" if use_latex else "_"),
            xy=(r["params_M"], r["best_iou"]),
            xytext=(r["params_M"] + 0.5, r["best_iou"] + 0.0004),
            fontsize=6, color="#333333",
        )

    # Pareto frontier
    sorted_by_params = sorted(results, key=lambda x: x["params_M"])
    pareto_x, pareto_y, best_so_far = [], [], -1
    for r in sorted_by_params:
        if r["best_iou"] > best_so_far:
            pareto_x.append(r["params_M"])
            pareto_y.append(r["best_iou"])
            best_so_far = r["best_iou"]
    ax.plot(pareto_x, pareto_y, "--", color="#cc4444",
            linewidth=1.2, alpha=0.7, zorder=3,
            label=tex(r"Pareto frontier", use_latex))

    ax.set_xlabel(tex(r"Parameters ($\times 10^6$)", use_latex), fontsize=9)
    ax.set_ylabel(tex(r"Best validation $\mathrm{IoU}$", use_latex), fontsize=9)
    ax.set_title(tex(r"\textbf{Model Size vs.\ Accuracy} with Pareto Frontier",
                     use_latex), fontsize=10, pad=8)
    ax.legend(fontsize=7, framealpha=0.9)
    apply_plot_style(ax, use_latex)


def plot_summary_table(ax, results, use_latex):
    """Render a formatted data table as a matplotlib axis."""
    ax.axis("off")

    columns = [
        tex(r"\textbf{Backbone}", use_latex),
        tex(r"\textbf{Family}", use_latex),
        tex(r"\textbf{IoU}", use_latex),
        tex(r"\textbf{CharAcc}", use_latex),
        tex(r"\textbf{$p_{50}$ ms}", use_latex),
        tex(r"\textbf{Params}", use_latex),
        tex(r"\textbf{CO$_2$ (g)}", use_latex),
    ]

    rows = []
    for r in results:
        bb = r["backbone"].replace("_", r"\_" if use_latex else "_")
        rows.append([
            bb,
            r["family"],
            f"{r['best_iou']:.4f}",
            f"{r['final_char_acc']:.4f}",
            f"{r['lat_p50_ms']:.1f}",
            f"{r['params_M']:.1f}M",
            f"{r['total_co2_g']:.1f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.55)

    # Style header row
    for col_idx in range(len(columns)):
        cell = table[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Style data rows — alternate shading, highlight best IoU row
    best_iou = max(r["best_iou"] for r in results)
    for row_idx, r in enumerate(results, start=1):
        row_color = "#f0f4f8" if row_idx % 2 == 0 else "#ffffff"
        if r["best_iou"] == best_iou:
            row_color = "#d4edda"   # green highlight for best

        for col_idx in range(len(columns)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(row_color)
            if col_idx == 0:
                cell.set_text_props(fontweight="bold")

    ax.set_title(tex(r"\textbf{Summary Table} -- All Trained Backbones",
                     use_latex), fontsize=10, pad=12, y=0.98)


# ---------------------------------------------------------------------------
# Master figure builder
# ---------------------------------------------------------------------------

def build_figure(results: list[dict], use_latex: bool,
                 output_path: Path = None) -> plt.Figure:
    """
    Compose a 3×2 grid figure with all charts and the summary table.
    """
    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        nrows=4, ncols=2,
        figure=fig,
        hspace=0.52,
        wspace=0.32,
        top=0.94,
        bottom=0.04,
        left=0.07,
        right=0.97,
    )

    # Row 0: main title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.7,
        tex(r"\textbf{OpenLPR} --- Multi-Backbone Training Results",
            use_latex),
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="#2c3e50",
    )
    ax_title.text(
        0.5, 0.25,
        tex(r"Dataset: \texttt{v1.0} synthetic $\cdot$ "
            + f"{len(results)}" +
            r" backbones trained $\cdot$ "
            r"Metric: validation IoU $+$ character accuracy",
            use_latex),
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=10, color="#555555",
    )

    # Row 1: accuracy bars | accuracy vs latency scatter
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    plot_accuracy_bars(ax1, results, use_latex)
    plot_accuracy_vs_latency(ax2, results, use_latex)

    # Row 2: training curves | params vs accuracy
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    plot_training_curves(ax3, results, use_latex)
    plot_params_vs_accuracy(ax4, results, use_latex)

    # Row 3: CO₂ bars | summary table
    ax5 = fig.add_subplot(gs[3, 0])
    ax6 = fig.add_subplot(gs[3, 1])
    plot_energy_co2(ax5, results, use_latex)
    plot_summary_table(ax6, results, use_latex)

    # Family colour legend at the bottom of the figure
    from matplotlib.patches import Patch
    seen_families = list(dict.fromkeys(r["family"] for r in results))
    legend_handles = [
        Patch(color=FAMILY_COLORS.get(f, "#666"),
              label=tex(f + r" --- " + FAMILY_NOTES.get(f, ""), use_latex))
        for f in seen_families
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        title=tex(r"\textit{Architectural Families \& Key Papers}",
                  use_latex),
        title_fontsize=8.5,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.005),
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise OpenLPR multi-backbone training results"
    )
    p.add_argument("--checkpoint-dir", default="models/checkpoints",
                   help="Directory containing backbone subdirectories with run_log.json")
    p.add_argument("--output", default="evaluation/training_results",
                   help="Output path without extension (saves .pdf and .png)")
    p.add_argument("--no-latex", action="store_true",
                   help="Disable LaTeX rendering (use mathtext fallback)")
    p.add_argument("--dpi", type=int, default=150,
                   help="PNG output DPI (default: 150)")
    p.add_argument("--show", action="store_true",
                   help="Open the figure in a window after saving")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"[error] Checkpoint directory not found: {checkpoint_dir}")
        print("  Run training first:")
        print("    python scripts/train_parallel.py --epochs 50")
        sys.exit(1)

    print(f"\nOpenLPR Results Visualiser")
    print(f"  Checkpoint dir : {checkpoint_dir}")

    # LaTeX setup
    use_latex = setup_matplotlib(not args.no_latex)

    # Load results
    print(f"\nLoading run logs...")
    results = load_all_results(checkpoint_dir)

    if not results:
        print(f"[error] No run_log.json files found under {checkpoint_dir}")
        print("  Each backbone subdirectory needs a run_log.json from train.py")
        sys.exit(1)

    print(f"  Found {len(results)} completed training runs")

    # Print terminal table
    print_terminal_table(results)

    # Build figure
    print("Building figure...")
    fig = build_figure(results, use_latex=use_latex)

    # Save outputs
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=args.dpi)
    print(f"  Saved PDF : {pdf_path}")

    fig.savefig(png_path, bbox_inches="tight", dpi=args.dpi)
    print(f"  Saved PNG : {png_path}")

    if args.show:
        plt.show()

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
