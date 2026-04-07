"""
viz.py — Advanced Visualizations for Aircraft Concept Comparison
================================================================

Reads comparison_results.csv (produced by compare_all_planes.py) and generates:
  1. Radar / Spider chart   — all aircraft on 1 normalized plot
  2. Weighted scoring heatmap — color-coded matrix with total score
  3. Scatter plots           — W/S vs W/P, hybridization vs fuel, CL vs CD,
                               specific range vs MTOW
  4. Stacked energy bar + field-performance waterfall
  5. Geometry comparison + fuel-breakdown stacked bar
  6. Payload-Range bubble chart

Usage:
    python viz.py                       # interactive (plt.show)
    python viz.py --save                # save PNGs only
    python viz.py --csv other_file.csv  # use a different CSV
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ARCH_COLORS = {
    "Conv.":    "#607D8B",
    "Wingtip":  "#2196F3",
    "Qtr-Span": "#00BCD4",
    "BLI":      "#4CAF50",
    "BLI+Mid":  "#8BC34A",
    "Series":   "#FF9800",
}


def _arch_color(config_name: str) -> str:
    """Return a color based on the architecture tag in parentheses."""
    for key, color in ARCH_COLORS.items():
        if key in config_name:
            return color
    return "#9E9E9E"


def _short_label(config: str) -> str:
    """Shorten 'BLI+WT (BLI)' -> 'BLI+WT'."""
    return config.split("(")[0].strip() if "(" in config else config


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load the comparison CSV, clean column names."""
    # Try multiple encodings — compare_all_planes may write cp1252 / latin-1
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise RuntimeError(f"Could not decode {csv_path} with any known encoding")
    # Normalise column names: strip whitespace, replace Unicode middle-dot
    df.columns = [c.strip().replace("\ufeff", "")
                    .replace("\u00b7", "\u00b7")   # keep proper Unicode ·
                    .replace("\xb7", "\u00b7")      # fix latin-1 middle dot
                  for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 1. Radar / Spider Chart
# ---------------------------------------------------------------------------
def plot_radar(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Radar chart comparing all aircraft on key normalised metrics.

    Axes (selected by user):
      - PREE            (higher = better -> normalise directly)
      - Fuel/pax/nmi    (lower = better  -> invert)
      - MTOW            (lower = better  -> invert)
      - L/D             (higher = better)
      - BFL             (lower = better  -> invert)
      - CO2/pax/nmi     (lower = better  -> invert)
    """
    # Metrics: (csv_column, display_label, higher_is_better)
    metrics = [
        ("PREE (lb·nmi/kWh)",    "PREE",          True),
        ("Fuel/pax/nmi (lb)",    "Fuel Eff.",      False),
        ("MTOW (lb)",            "MTOW",           False),
        ("L/D",                  "L/D",            True),
        ("BFL (ft)",             "BFL",            False),
        ("CO2/pax/nmi (kg)",     "CO\u2082/pax\u00b7nmi",   False),
    ]

    # Fall back: find matching columns with fuzzy match
    cols_present = []
    for col, label, hib in metrics:
        matches = [c for c in df.columns if col.split("(")[0].strip() in c]
        if col in df.columns:
            cols_present.append((col, label, hib))
        elif matches:
            cols_present.append((matches[0], label, hib))
    metrics = cols_present
    if len(metrics) < 3:
        print("  WARNING: fewer than 3 radar metrics found -- skipping radar.")
        return

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Normalise each metric to [0.05, 1.0]
    norm_data = {}
    for col, label, higher_is_better in metrics:
        vals = df[col].astype(float).values
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            norm = np.ones_like(vals) * 0.5
        elif higher_is_better:
            norm = (vals - vmin) / (vmax - vmin)
        else:
            norm = (vmax - vals) / (vmax - vmin)
        norm = norm * 0.95 + 0.05  # shift to [0.05, 1.0]
        norm_data[label] = norm

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")

    for i, row in df.iterrows():
        name = row["Configuration"]
        color = _arch_color(name)
        values = [norm_data[m[1]][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=_short_label(name), color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8,
              framealpha=0.9)
    ax.set_title("Aircraft Concept Radar Comparison\n(outer = better)",
                 fontsize=14, fontweight="bold", y=1.08)

    plt.tight_layout()
    if save:
        fig.savefig(str(out_dir / "viz_radar.png"), dpi=180, bbox_inches="tight")
        print(f"  Saved: viz_radar.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Weighted Scoring Heatmap
# ---------------------------------------------------------------------------
def plot_scoring_heatmap(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Normalised scoring matrix.  Each metric is scaled 0-10
    (10 = best).  A Total column sums the scores.
    """
    # (csv_column, short label, higher_is_better, weight)
    score_metrics = [
        ("PREE (lb·nmi/kWh)",    "PREE",        True,  2.0),
        ("Fuel/pax/nmi (lb)",    "Fuel/pax/nmi", False, 2.0),
        ("MTOW (lb)",            "MTOW",         False, 1.5),
        ("L/D",                  "L/D",          True,  1.0),
        ("BFL (ft)",             "BFL",          False, 1.0),
        ("CO2/pax/nmi (kg)",     "CO\u2082/pax\u00b7nmi", False, 1.5),
        ("Energy/seat\u00b7nmi (kWh)", "E/seat\u00b7nmi",   False, 1.0),
        ("T/W (TO)",             "T/W",          True,  0.5),
        ("Climb Grad OEI (%)",   "OEI Climb",    True,  0.5),
    ]

    present = []
    for col, label, hib, wt in score_metrics:
        if col in df.columns:
            present.append((col, label, hib, wt))
        else:
            # Try fuzzy match on the label portion
            matches = [c for c in df.columns if label.replace("\u2082", "2").replace("\u00b7", ".").lower() in c.lower()]
            if matches:
                present.append((matches[0], label, hib, wt))
    score_metrics = present

    configs = df["Configuration"].tolist()
    short = [_short_label(c) for c in configs]

    n_metrics = len(score_metrics)
    scores = np.zeros((len(configs), n_metrics))

    for j, (col, label, hib, wt) in enumerate(score_metrics):
        vals = df[col].astype(float).values
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            normed = np.ones_like(vals) * 5.0
        elif hib:
            normed = (vals - vmin) / (vmax - vmin) * 10
        else:
            normed = (vmax - vals) / (vmax - vmin) * 10
        scores[:, j] = normed * wt

    # Append weighted total column
    totals = scores.sum(axis=1)
    col_labels = [m[1] for m in score_metrics] + ["TOTAL"]
    weights = [m[3] for m in score_metrics]
    max_possible = sum(w * 10 for w in weights)

    display = np.column_stack([scores, totals])

    fig, ax = plt.subplots(figsize=(max(12, n_metrics * 1.3), len(configs) * 0.75 + 2))

    # Build a properly scaled colour map
    im = ax.imshow(display, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=max_possible * 0.15)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display[i, j]
            # Choose text colour for readability
            if j == display.shape[1] - 1:
                cell_frac = val / max_possible
            else:
                cell_frac = val / (score_metrics[j][3] * 10) if score_metrics[j][3] > 0 else 0.5
            txt_color = "white" if cell_frac < 0.35 else "black"

            if j == display.shape[1] - 1:
                # Total column: show as percentage of max
                ax.text(j, i, f"{val:.1f}\n({val / max_possible * 100:.0f}%)",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color=txt_color)
            else:
                w = score_metrics[j][3]
                ax.text(j, i, f"{val:.1f}\n(/{w * 10:.0f})",
                        ha="center", va="center", fontsize=8, color=txt_color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold", rotation=30, ha="right")
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short, fontsize=10)
    ax.set_title("Weighted Scoring Matrix  (green = better, weight \u00d7 10 max per cell)",
                 fontsize=13, fontweight="bold", pad=15)

    # Highlight best row
    best_idx = int(np.argmax(totals))
    ax.add_patch(plt.Rectangle((-0.5, best_idx - 0.5), display.shape[1], 1,
                               fill=False, edgecolor="gold", linewidth=3))
    ax.text(display.shape[1] + 0.1, best_idx, "\u2605 BEST",
            va="center", fontsize=11, fontweight="bold", color="goldenrod")

    plt.colorbar(im, ax=ax, shrink=0.5, label="Score")
    plt.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_heatmap.png"), dpi=180, bbox_inches="tight")
        print(f"  Saved: viz_heatmap.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Scatter Plots (2x2)
# ---------------------------------------------------------------------------
def plot_scatters(df: pd.DataFrame, out_dir: Path, save: bool):
    """Four scatter plots revealing cross-metric relationships."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Design-Space Scatter Plots", fontsize=15, fontweight="bold")

    configs = df["Configuration"].tolist()
    colors = [_arch_color(c) for c in configs]
    short = [_short_label(c) for c in configs]

    def _scatter(ax, xcol, ycol, xlabel, ylabel, title,
                 invert_x=False, invert_y=False):
        xv = df[xcol].astype(float).values
        yv = df[ycol].astype(float).values
        for xi, yi, c, s in zip(xv, yv, colors, short):
            ax.scatter(xi, yi, c=c, s=120, edgecolors="black",
                       linewidths=0.5, zorder=5)
            ax.annotate(s, (xi, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=c, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    # 3a. Wing Loading vs Power Loading
    _scatter(axes[0, 0],
             "Wing Loading (psf)", "Power Loading (lb/hp)",
             "Wing Loading (lb/ft\u00b2)", "Power Loading (lb/hp)",
             "Wing Loading vs. Power Loading")

    # 3b. Hybridization vs Fuel per pax-nmi
    _scatter(axes[0, 1],
             "Hybridization", "Fuel/pax/nmi (lb)",
             "Hybridization Factor", "Fuel / pax / nmi (lb)",
             "Hybridization vs. Fuel Economy")

    # 3c. CL vs CD (drag polar)
    _scatter(axes[1, 0],
             "CD cruise", "CL cruise",
             "CD (cruise)", "CL (cruise)",
             "Cruise Drag Polar")

    # 3d. MTOW vs Specific Range
    _scatter(axes[1, 1],
             "MTOW (lb)", "Specific Range (nmi/lb)",
             "MTOW (lb)", "Specific Range (nmi / lb fuel)",
             "MTOW vs. Specific Range")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(str(out_dir / "viz_scatters.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: viz_scatters.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Stacked Energy Bar + Field-Performance Waterfall
# ---------------------------------------------------------------------------
def plot_energy_and_field(df: pd.DataFrame, out_dir: Path, save: bool):
    """Left: stacked fuel/battery energy.  Right: field perf grouped bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Energy Breakdown & Field Performance",
                 fontsize=15, fontweight="bold")

    configs = df["Configuration"].tolist()
    short = [_short_label(c) for c in configs]
    x = np.arange(len(configs))

    # --- 4a. Stacked Energy ---
    fuel_e = df["Fuel Energy (kWh)"].astype(float).values
    batt_e = df["Battery Energy (kWh)"].astype(float).values

    ax1.bar(x, fuel_e, label="Fuel Energy", color="#FF9800",
            edgecolor="black", linewidth=0.4)
    ax1.bar(x, batt_e, bottom=fuel_e, label="Battery Energy", color="#4CAF50",
            edgecolor="black", linewidth=0.4)
    for i, (fe, be) in enumerate(zip(fuel_e, batt_e)):
        ax1.text(i, fe + be + 30, f"{fe + be:,.0f}",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Total On-Board Energy Breakdown")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- 4b. Field Performance grouped bars ---
    w = 0.25
    to_roll = df["TO Ground Roll (ft)"].astype(float).values
    bfl     = df["BFL (ft)"].astype(float).values
    landing = df["Landing Dist (ft)"].astype(float).values

    ax2.bar(x - w, to_roll, w, label="TO Ground Roll",
            color="#1565C0", alpha=0.85)
    ax2.bar(x,     bfl,     w, label="BFL",
            color="#795548", alpha=0.85)
    ax2.bar(x + w, landing, w, label="Landing Dist",
            color="#E53935", alpha=0.85)

    # Constraint line
    ax2.axhline(y=3500, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.text(len(configs) - 0.5, 3550, "3,500 ft limit", fontsize=8,
             color="red", ha="right", va="bottom")

    ax2.set_ylabel("Distance (ft)")
    ax2.set_title("Takeoff & Landing Distances")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(str(out_dir / "viz_energy_field.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Saved: viz_energy_field.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Geometry Comparison + Fuel Breakdown
# ---------------------------------------------------------------------------
def plot_geometry_and_fuel(df: pd.DataFrame, out_dir: Path, save: bool):
    """Left: geometry grouped bars.  Right: fuel breakdown stacked bar."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Geometry & Fuel Breakdown", fontsize=15, fontweight="bold")

    configs = df["Configuration"].tolist()
    short = [_short_label(c) for c in configs]
    x = np.arange(len(configs))

    # --- 5a. Geometry: Wing Span, AR, Propeller Diameter ---
    w = 0.25
    span = df["Wing Span (m)"].astype(float).values
    ar   = df["AR"].astype(float).values
    prop = df["Propeller Diam (m)"].astype(float).values

    ax1.bar(x - w, span, w, label="Wing Span (m)", color="#0288D1")
    ax1.bar(x,     ar,   w, label="Aspect Ratio",  color="#7B1FA2")
    ax1.bar(x + w, prop, w, label="Prop Diameter (m)", color="#FF7043")

    ax1.set_ylabel("Value")
    ax1.set_title("Geometry Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax1.legend(fontsize=9)

    # --- 5b. Fuel Breakdown: Climb + Reserves + Cruise (residual) ---
    fuel_total = df["Fuel (lb)"].astype(float).values
    fuel_climb = df["Fuel Climb (kg)"].astype(float).values * 2.20462   # kg -> lb
    fuel_res   = df["Fuel Reserves (kg)"].astype(float).values * 2.20462
    fuel_cruise = np.maximum(fuel_total - fuel_climb - fuel_res, 0)

    ax2.bar(x, fuel_climb, label="Climb Fuel", color="#FFA726",
            edgecolor="black", linewidth=0.4)
    ax2.bar(x, fuel_cruise, bottom=fuel_climb, label="Cruise Fuel",
            color="#42A5F5", edgecolor="black", linewidth=0.4)
    ax2.bar(x, fuel_res, bottom=fuel_climb + fuel_cruise,
            label="Reserves (45 min)", color="#EF5350",
            edgecolor="black", linewidth=0.4)

    for i, ft in enumerate(fuel_total):
        ax2.text(i, ft + 10, f"{ft:,.0f}", ha="center", va="bottom", fontsize=7)

    ax2.set_ylabel("Fuel Weight (lb)")
    ax2.set_title("Fuel Breakdown by Mission Phase")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(str(out_dir / "viz_geometry_fuel.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Saved: viz_geometry_fuel.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Payload-Range Bubble Chart
# ---------------------------------------------------------------------------
def plot_payload_range_bubble(df: pd.DataFrame, out_dir: Path, save: bool):
    """Bubble: x=range, y=payload, size=total energy, color=architecture."""
    fig, ax = plt.subplots(figsize=(11, 8))

    configs = df["Configuration"].tolist()
    colors = [_arch_color(c) for c in configs]
    short = [_short_label(c) for c in configs]

    rng    = df["Max Range (nmi)"].astype(float).values
    pay    = df["Payload (lb)"].astype(float).values
    energy = (df["Fuel Energy (kWh)"].astype(float).values +
              df["Battery Energy (kWh)"].astype(float).values)

    # Scale bubble area
    e_range = energy.max() - energy.min() + 1
    e_norm = (energy - energy.min()) / e_range * 1500 + 200

    for i in range(len(configs)):
        ax.scatter(rng[i], pay[i], s=e_norm[i], c=colors[i],
                   edgecolors="black", linewidths=0.8, alpha=0.75, zorder=5)
        ax.annotate(short[i], (rng[i], pay[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, fontweight="bold",
                    color=colors[i])

    ax.set_xlabel("Max Range (nmi)", fontsize=11)
    ax.set_ylabel("Payload (lb)", fontsize=11)
    ax.set_title("Payload\u2013Range Bubble Chart\n"
                 "(bubble size \u221d total energy)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Architecture legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=10, label=k) for k, c in ARCH_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              title="Architecture")

    plt.tight_layout()
    if save:
        fig.savefig(str(out_dir / "viz_payload_range.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Saved: viz_payload_range.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Parallel Coordinates Plot
# ---------------------------------------------------------------------------
def plot_parallel_coordinates(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Parallel coordinates: each vertical axis is one metric, each aircraft
    is a polyline connecting its normalised values.  Instantly shows
    trade-offs and dominant designs.
    """
    # Pick 8 axes that cover the design space
    axes_spec = [
        ("MTOW (lb)",            "MTOW\n(lb)",       False),
        ("L/D",                  "L/D",              True),
        ("Fuel/pax/nmi (lb)",    "Fuel/pax\n/nmi",   False),
        ("BFL (ft)",             "BFL\n(ft)",         False),
        ("Hybridization",        "Hybrid.",           True),
        ("Wing Loading (psf)",   "W/S\n(psf)",        None),  # None = neutral
        ("Power Loading (lb/hp)","W/P\n(lb/hp)",      None),
        ("CO2/pax/nmi (kg)",     "CO\u2082/pax\n/nmi", False),
    ]
    # Filter to columns that actually exist
    axes_spec = [(c, l, d) for c, l, d in axes_spec if c in df.columns]
    if len(axes_spec) < 4:
        print("  WARNING: too few columns for parallel coordinates — skipping.")
        return

    n_axes = len(axes_spec)
    fig, host = plt.subplots(figsize=(16, 7))

    # Normalise each column to [0, 1].  For "lower-is-better" columns,
    # invert so that *up* always means *better* (when direction is known).
    norm_vals = np.zeros((len(df), n_axes))
    for j, (col, label, higher_is_better) in enumerate(axes_spec):
        vals = df[col].astype(float).values
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            norm_vals[:, j] = 0.5
        else:
            n = (vals - vmin) / (vmax - vmin)
            if higher_is_better is False:
                n = 1.0 - n
            norm_vals[:, j] = n

    configs = df["Configuration"].tolist()
    colors = [_arch_color(c) for c in configs]
    short = [_short_label(c) for c in configs]

    xs = np.arange(n_axes)
    for i in range(len(df)):
        host.plot(xs, norm_vals[i], color=colors[i], linewidth=2.2,
                  alpha=0.75, label=short[i], marker='o', markersize=5,
                  markeredgecolor='black', markeredgewidth=0.4)

    host.set_xticks(xs)
    host.set_xticklabels([a[1] for a in axes_spec], fontsize=9,
                         fontweight="bold")
    host.set_ylabel("Normalised value  (up = better where direction known)",
                    fontsize=9)
    host.set_ylim(-0.05, 1.15)
    host.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    host.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8,
                         color="grey")
    host.grid(True, axis='x', alpha=0.3, linestyle='--')
    host.set_title("Parallel Coordinates — Multi-Metric Trade-Off View",
                   fontsize=14, fontweight="bold")

    # Add raw-value annotations at top/bottom of each axis
    for j, (col, label, hib) in enumerate(axes_spec):
        vals = df[col].astype(float).values
        lo_raw, hi_raw = vals.min(), vals.max()
        if hib is False:    # inverted axis
            host.text(j, -0.07, f"{hi_raw:.1f}", ha="center", fontsize=6,
                      color="grey")
            host.text(j, 1.07, f"{lo_raw:.1f}", ha="center", fontsize=6,
                      color="grey")
        else:
            host.text(j, -0.07, f"{lo_raw:.1f}", ha="center", fontsize=6,
                      color="grey")
            host.text(j, 1.07, f"{hi_raw:.1f}", ha="center", fontsize=6,
                      color="grey")

    host.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                ncol=5, fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_parallel.png"), dpi=180,
                    bbox_inches="tight")
        print(f"  Saved: viz_parallel.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Pareto Front — two competing objectives
# ---------------------------------------------------------------------------
def plot_pareto(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Scatter with Pareto frontier highlighted.
    Objective 1 (x): MTOW — minimise
    Objective 2 (y): Fuel/pax/nmi — minimise
    Points on the Pareto front are connected and starred.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Pareto Fronts — Non-Dominated Designs",
                 fontsize=15, fontweight="bold")

    configs = df["Configuration"].tolist()
    colors = [_arch_color(c) for c in configs]
    short = [_short_label(c) for c in configs]

    def _draw_pareto(ax, xcol, ycol, xlabel, ylabel, title,
                     minimize_x=True, minimize_y=True):
        xv = df[xcol].astype(float).values
        yv = df[ycol].astype(float).values

        # Find Pareto-optimal indices (both objectives minimised)
        sx = xv if minimize_x else -xv
        sy = yv if minimize_y else -yv
        pareto_mask = np.ones(len(xv), dtype=bool)
        for i in range(len(xv)):
            for j in range(len(xv)):
                if i == j:
                    continue
                if sx[j] <= sx[i] and sy[j] <= sy[i] and \
                   (sx[j] < sx[i] or sy[j] < sy[i]):
                    pareto_mask[i] = False
                    break

        # Plot all points
        for i in range(len(xv)):
            ms = 180 if pareto_mask[i] else 100
            mk = '*' if pareto_mask[i] else 'o'
            ew = 1.5 if pareto_mask[i] else 0.5
            ax.scatter(xv[i], yv[i], c=colors[i], s=ms, marker=mk,
                       edgecolors="black", linewidths=ew, zorder=5)
            offset = (8, 8) if not pareto_mask[i] else (8, -12)
            ax.annotate(short[i], (xv[i], yv[i]), textcoords="offset points",
                        xytext=offset, fontsize=7, fontweight="bold",
                        color=colors[i])

        # Connect Pareto front
        pareto_idx = np.where(pareto_mask)[0]
        has_front_line = False
        if len(pareto_idx) >= 2:
            order = pareto_idx[np.argsort(xv[pareto_idx])]
            ax.plot(xv[order], yv[order], 'k--', linewidth=1.5, alpha=0.5,
                    label="Pareto front", zorder=3)
            has_front_line = True

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if has_front_line:
            ax.legend(fontsize=9, loc="best")

    # Pareto 1: MTOW vs Fuel/pax/nmi (both minimise)
    _draw_pareto(axes[0],
                 "MTOW (lb)", "Fuel/pax/nmi (lb)",
                 "MTOW (lb)", "Fuel / pax / nmi (lb)",
                 "MTOW vs. Fuel Economy\n(\u2605 = Pareto-optimal)")

    # Pareto 2: BFL vs CO2/pax/nmi (both minimise)
    _draw_pareto(axes[1],
                 "BFL (ft)", "CO2/pax/nmi (kg)",
                 "Balanced Field Length (ft)",
                 "CO\u2082 / pax / nmi (kg)",
                 "BFL vs. CO\u2082 Intensity\n(\u2605 = Pareto-optimal)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(str(out_dir / "viz_pareto.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Saved: viz_pareto.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Correlation Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Pearson correlation between key numeric metrics.
    Reveals which design choices are coupled.
    """
    metric_cols = [
        "MTOW (lb)", "L/D", "Fuel/pax/nmi (lb)", "BFL (ft)",
        "Hybridization", "Wing Loading (psf)", "Power Loading (lb/hp)",
        "CO2/pax/nmi (kg)", "T/W (TO)", "AR",
        "Climb Grad OEI (%)", "Cruise Throttle (%)",
        "Thermal Eff", "EW Frac", "Payload Frac",
    ]
    present = [c for c in metric_cols if c in df.columns]
    if len(present) < 5:
        print("  WARNING: too few numeric columns — skipping correlation.")
        return

    sub = df[present].astype(float)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    n = len(present)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [c.split("(")[0].strip() if len(c) > 18 else c for c in present]
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            txt_color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=txt_color, fontweight="bold")

    ax.set_title("Metric Correlation Matrix (Pearson)\n"
                 "Red = positive, Blue = negative",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.75, label="Correlation coefficient")
    plt.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_correlation.png"), dpi=180,
                    bbox_inches="tight")
        print(f"  Saved: viz_correlation.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Normalised Delta from Baseline (Lollipop Chart)
# ---------------------------------------------------------------------------
def plot_delta_from_baseline(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Lollipop chart showing % change of each aircraft vs the SkyCourier
    (conventional) baseline across key metrics.
    Positive = better, negative = worse.
    """
    # (column, short label, higher_is_better)
    delta_metrics = [
        ("Fuel/pax/nmi (lb)",    "Fuel/pax/nmi",  False),
        ("MTOW (lb)",            "MTOW",           False),
        ("L/D",                  "L/D",            True),
        ("BFL (ft)",             "BFL",            False),
        ("CO2/pax/nmi (kg)",     "CO\u2082/pax",   False),
        ("EW Frac",              "EW Frac",        False),
        ("Fuel Burn (kg/hr)",    "Fuel Burn",      False),
    ]
    present = [(c, l, h) for c, l, h in delta_metrics if c in df.columns]
    if len(present) < 3:
        print("  WARNING: too few columns for delta chart — skipping.")
        return

    # Baseline = row 0 (SkyCourier)
    configs = df["Configuration"].tolist()
    baseline_idx = 0
    for i, c in enumerate(configs):
        if "Conv" in c or "SkyCourier" in c:
            baseline_idx = i
            break
    baseline_name = _short_label(configs[baseline_idx])

    # Compute % delta for non-baseline rows
    others = [i for i in range(len(df)) if i != baseline_idx]
    short_others = [_short_label(configs[i]) for i in others]
    n_others = len(others)
    n_metrics = len(present)

    deltas = np.zeros((n_others, n_metrics))
    for j, (col, label, hib) in enumerate(present):
        base_val = df[col].astype(float).iloc[baseline_idx]
        for k, oi in enumerate(others):
            v = df[col].astype(float).iloc[oi]
            if base_val != 0:
                pct = (v - base_val) / abs(base_val) * 100
            else:
                pct = 0
            if not hib:   # invert so positive = improvement
                pct = -pct
            deltas[k, j] = pct

    fig, ax = plt.subplots(figsize=(14, max(6, n_others * 0.9)))

    y_positions = np.arange(n_others)
    bar_h = 0.8 / n_metrics

    cmap = plt.colormaps.get_cmap("tab10")
    for j, (col, label, hib) in enumerate(present):
        yy = y_positions + (j - n_metrics / 2 + 0.5) * bar_h
        vals = deltas[:, j]
        color = cmap(j / max(n_metrics - 1, 1))
        ax.barh(yy, vals, height=bar_h * 0.85, label=label, color=color,
                edgecolor="black", linewidth=0.3, alpha=0.85)
        for k, v in enumerate(vals):
            if abs(v) > 1:
                side = "left" if v > 0 else "right"
                off = 0.5 if v > 0 else -0.5
                ax.text(v + off, yy[k], f"{v:+.1f}%", va="center",
                        ha=side, fontsize=6, fontweight="bold")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(short_others, fontsize=9)
    ax.set_xlabel("% Change from Baseline  (\u2192 positive = better)",
                  fontsize=10)
    ax.set_title(f"Performance Delta vs. Baseline ({baseline_name})\n"
                 f"Positive = improvement over conventional",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_delta_baseline.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Saved: viz_delta_baseline.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Pairwise Trade-Off Matrix (small multiples)
# ---------------------------------------------------------------------------
def plot_pairwise_matrix(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Lower-triangle scatter matrix of 5 key metrics.
    Diagonal shows a simple strip-plot / histogram.
    """
    pair_cols = [
        ("MTOW (lb)",          "MTOW"),
        ("L/D",                "L/D"),
        ("Fuel/pax/nmi (lb)",  "Fuel/pax"),
        ("BFL (ft)",           "BFL"),
        ("Hybridization",      "Hybrid."),
    ]
    pair_cols = [(c, l) for c, l in pair_cols if c in df.columns]
    np_ = len(pair_cols)
    if np_ < 3:
        print("  WARNING: too few columns for pairwise matrix — skipping.")
        return

    configs = df["Configuration"].tolist()
    colors = [_arch_color(c) for c in configs]

    fig, axes = plt.subplots(np_, np_, figsize=(np_ * 2.8, np_ * 2.8))
    fig.suptitle("Pairwise Trade-Off Matrix", fontsize=14, fontweight="bold")

    for i in range(np_):
        for j in range(np_):
            ax = axes[i][j]
            if i == j:
                # Diagonal: strip plot
                vals = df[pair_cols[i][0]].astype(float).values
                for k, v in enumerate(vals):
                    ax.scatter(v, 0.5, c=colors[k], s=60, edgecolors="black",
                               linewidths=0.3, zorder=5)
                ax.set_yticks([])
                ax.set_title(pair_cols[i][1], fontsize=9, fontweight="bold")
                ax.grid(True, axis='x', alpha=0.3)
            elif i > j:
                # Lower triangle: scatter
                xv = df[pair_cols[j][0]].astype(float).values
                yv = df[pair_cols[i][0]].astype(float).values
                for k in range(len(xv)):
                    ax.scatter(xv[k], yv[k], c=colors[k], s=50,
                               edgecolors="black", linewidths=0.3, zorder=5)
                ax.grid(True, alpha=0.2)
            else:
                # Upper triangle: correlation value
                xv = df[pair_cols[j][0]].astype(float).values
                yv = df[pair_cols[i][0]].astype(float).values
                corr_val = np.corrcoef(xv, yv)[0, 1]
                ax.text(0.5, 0.5, f"r={corr_val:.2f}", ha="center",
                        va="center", fontsize=11, fontweight="bold",
                        transform=ax.transAxes,
                        color="red" if corr_val > 0 else "blue")
                ax.set_xticks([])
                ax.set_yticks([])

            # Labels on edges only
            if i < np_ - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(pair_cols[j][1], fontsize=8)
                ax.tick_params(axis='x', labelsize=6)
            if j > 0 and i != j:
                ax.set_yticklabels([])
            elif i > j:
                ax.set_ylabel(pair_cols[i][1], fontsize=8)
                ax.tick_params(axis='y', labelsize=6)

    # Legend below
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=8, label=k)
               for k, c in ARCH_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save:
        fig.savefig(str(out_dir / "viz_pairwise.png"), dpi=150,
                    bbox_inches="tight")
        print(f"  Saved: viz_pairwise.png")
    else:
        plt.show()
    plt.close(fig)


# ===========================================================================
#  BLI+WT ADVOCACY CHARTS  (12-16)
# ===========================================================================
_HERO = "BLI+WT (BLI)"          # the concept we are championing
_HERO_TAG = "BLI+WT"


def _df19(df: pd.DataFrame) -> pd.DataFrame:
    """Return only 19-pax concepts for apples-to-apples comparison."""
    return df[df["Passengers"] == 19].copy()


# ---------------------------------------------------------------------------
# 12.  "Why BLI+WT wins despite lower L/D" — waterfall decomposition
# ---------------------------------------------------------------------------
def plot_efficiency_waterfall(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Waterfall showing how BLI+WT achieves the best fuel/pax/nmi despite
    having lower L/D.  Breaks fuel burn into:
        Fuel_per_seat = f(MTOW, L/D, Thermal Eff, Hybridization, ...)
    Shows each factor relative to conventional baseline (SkyCourier).
    """
    d19 = _df19(df)
    if d19.empty:
        print("  [skip] no 19-pax rows")
        return

    baseline_name = "SkyCourier (Conv.)"
    concepts = [c for c in d19["Configuration"] if c != baseline_name]
    bl_row = d19[d19["Configuration"] == baseline_name].iloc[0]

    fig, axes = plt.subplots(1, len(concepts), figsize=(4 * len(concepts), 6),
                             sharey=True)
    if len(concepts) == 1:
        axes = [axes]

    metric_keys = ["MTOW (lb)", "L/D", "Thermal Eff", "Fuel/pax/nmi (lb)"]
    metric_labels = ["MTOW", "L/D", "Thermal Eff", "Net Fuel/seat"]

    for ax, cname in zip(axes, concepts):
        row = d19[d19["Configuration"] == cname].iloc[0]
        color = _arch_color(cname)

        # Compute % change from baseline for each factor
        pcts = []
        for k in metric_keys:
            bl_val, cval = bl_row[k], row[k]
            if bl_val and bl_val != 0:
                pcts.append((cval - bl_val) / abs(bl_val) * 100)
            else:
                pcts.append(0)

        bar_colors = ["#4CAF50" if p < 0 else "#F44336" for p in pcts]
        # Override: for L/D and Thermal Eff, higher is better → flip colour
        for i, k in enumerate(metric_keys):
            if k in ("L/D", "Thermal Eff"):
                bar_colors[i] = "#4CAF50" if pcts[i] > 0 else "#F44336"

        ax.barh(metric_labels, pcts, color=bar_colors, edgecolor="k",
                linewidth=0.5, height=0.6)
        ax.axvline(0, color="k", linewidth=0.8)
        for i, p in enumerate(pcts):
            ax.text(p + (1 if p >= 0 else -1), i,
                    f"{p:+.1f}%", va="center",
                    ha="left" if p >= 0 else "right", fontsize=9)
        ax.set_title(_short_label(cname), fontsize=11, fontweight="bold",
                     color=color)
        ax.set_xlim(-35, 25)
        ax.set_xlabel("% change vs SkyCourier")

    fig.suptitle("Why BLI+WT Wins Despite Lower L/D\n"
                 "(green = favourable direction)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(str(out_dir / "viz_advocacy_waterfall.png"), dpi=150,
                    bbox_inches="tight")
        print("  Saved: viz_advocacy_waterfall.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 13.  Technology-risk / hybridization comparison
# ---------------------------------------------------------------------------
def plot_tech_risk(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Grouped bar chart:  hybridization level + battery capacity for each
    19-pax concept.  Lower hybridization → less battery-technology risk.
    Also overlays Fuel/pax/nmi to show the efficiency payoff.
    """
    d19 = _df19(df)
    if d19.empty:
        print("  [skip] no 19-pax rows")
        return

    labels = [_short_label(c) for c in d19["Configuration"]]
    hybrid = d19["Hybridization"].values * 100
    batt   = d19["Battery Cap (kWh)"].values
    fuel_eff = d19["Fuel/pax/nmi (lb)"].values

    hero_idx = [i for i, c in enumerate(d19["Configuration"])
                if _HERO in c]
    hero_i = hero_idx[0] if hero_idx else None

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    bars1 = ax1.bar(x - width / 2, hybrid, width, label="Hybridization (%)",
                    color="#42A5F5", edgecolor="k", linewidth=0.5)
    ax1.set_ylabel("Hybridization (%)", color="#42A5F5")
    ax1.set_ylim(0, max(hybrid) * 1.35)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, batt, width, label="Battery Cap (kWh)",
                    color="#FFA726", edgecolor="k", linewidth=0.5)
    ax2.set_ylabel("Battery Capacity (kWh)", color="#FFA726")

    # Fuel efficiency line on top
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    line = ax3.plot(x, fuel_eff, "ko-", markersize=7, linewidth=2,
                    label="Fuel/pax/nmi")
    ax3.set_ylabel("Fuel / pax / nmi (lb)")

    # Highlight BLI+WT
    if hero_i is not None:
        for ax_ in (ax1,):
            ax_.axvspan(hero_i - 0.5, hero_i + 0.5,
                        color="#4CAF50", alpha=0.10)
        ax3.annotate("Best fuel economy\nwith LOWEST hybrid %",
                     xy=(hero_i, fuel_eff[hero_i]),
                     xytext=(hero_i + 1.3, fuel_eff[hero_i] - 0.012),
                     fontsize=9, fontweight="bold", color="#2E7D32",
                     arrowprops=dict(arrowstyle="->", color="#2E7D32"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Technology Risk: Hybridization & Battery Size vs Fuel Efficiency",
                  fontsize=12, fontweight="bold")

    # Combined legend
    handles = list(bars1) [:1] + list(bars2)[:1] + line
    lbls = ["Hybridization (%)", "Battery Cap (kWh)", "Fuel/pax/nmi (lb)"]
    ax1.legend(handles, lbls, loc="upper left", fontsize=8)

    fig.tight_layout()
    if save:
        fig.savefig(str(out_dir / "viz_advocacy_tech_risk.png"), dpi=150,
                    bbox_inches="tight")
        print("  Saved: viz_advocacy_tech_risk.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 14.  Direct Operating Cost (DOC) estimate
# ---------------------------------------------------------------------------
def plot_doc_estimate(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Stacked bar chart estimating DOC per flight-hour across 19-pax concepts.
    Components:
      - Fuel cost  (Fuel_burn × $ per kg Jet-A)
      - Battery depreciation  (Battery_kWh × $/kWh / cycle_life / flt_time)
      - Maintenance  (engine h × rate + airframe rate × MTOW)
      - Crew  (constant across configs)

    These are simplified parametric estimates, not full DAPCA IV, but they
    highlight the economic advantage of low fuel burn + low battery dependency.
    """
    d19 = _df19(df)
    if d19.empty:
        print("  [skip] no 19-pax rows")
        return

    # ---------- assumptions ----------
    fuel_price_per_kg = 1.10           # $/kg Jet-A (≈$3.50/gal)
    batt_cost_per_kwh = 200.0          # $/kWh pack-level (2030 target)
    batt_cycle_life   = 2000           # cycles
    flt_hours_per_cycle = 1.5          # hours per flight
    maint_rate_engine_hr = 80.0        # $/engine/hr (turboprop)
    maint_rate_airframe  = 0.005       # $/lb MTOW/hr
    crew_cost_hr = 500.0               # $/hr (2-crew)

    labels     = [_short_label(c) for c in d19["Configuration"]]
    fuel_burn  = d19["Fuel Burn (kg/hr)"].values
    batt_kwh   = d19["Battery Cap (kWh)"].values
    mtow       = d19["MTOW (lb)"].values

    n_eng_col = "Engine Power ea (hp)"
    eng_power  = d19[n_eng_col].values if n_eng_col in d19.columns else np.ones(len(d19))
    # Assume 2 engines for all concepts (turboprop twin)
    n_engines = 2

    c_fuel  = fuel_burn * fuel_price_per_kg                       # $/hr
    c_batt  = batt_kwh * batt_cost_per_kwh / batt_cycle_life / flt_hours_per_cycle  # $/hr
    c_maint = n_engines * maint_rate_engine_hr + mtow * maint_rate_airframe   # $/hr
    c_crew  = np.full(len(d19), crew_cost_hr)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))

    bot = np.zeros(len(labels))
    for vals, lab, col in [
        (c_fuel,  "Fuel",        "#FF7043"),
        (c_batt,  "Battery dep", "#FFA726"),
        (c_maint, "Maintenance", "#78909C"),
        (c_crew,  "Crew",        "#90A4AE"),
    ]:
        ax.bar(x, vals, bottom=bot, label=lab, color=col, edgecolor="k",
               linewidth=0.5)
        bot += vals

    totals = c_fuel + c_batt + c_maint + c_crew
    for i, t in enumerate(totals):
        ax.text(i, t + 5, f"${t:,.0f}/hr", ha="center", fontsize=9,
                fontweight="bold")

    # Star the hero
    hero_idx = [i for i, c in enumerate(d19["Configuration"]) if _HERO in c]
    if hero_idx:
        hi = hero_idx[0]
        ax.annotate("★ Lowest DOC",
                    xy=(hi, totals[hi] + 15), xytext=(hi, totals[hi] + 55),
                    fontsize=11, fontweight="bold", color="#2E7D32",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="#2E7D32"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Estimated DOC ($/flight-hour)")
    ax.set_title("Direct Operating Cost Estimate — 19-Pax Concepts",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_advocacy_doc.png"), dpi=150,
                    bbox_inches="tight")
        print("  Saved: viz_advocacy_doc.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 15.  Battery-sensitivity / robustness analysis
# ---------------------------------------------------------------------------
def plot_battery_sensitivity(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    Line chart showing how each concept's total-energy metric changes
    when battery specific-energy (Wh/kg) varies from 60% to 140% of
    the baseline assumption.

    Concepts with lower hybridization (BLI+WT = 20%) are less affected
    than concepts with high hybridization (HE-19 = 46%).
    """
    d19 = _df19(df)
    if d19.empty:
        print("  [skip] no 19-pax rows")
        return

    # The sensitivity model:
    # If battery Wh/kg improves by factor k, then for the same energy,
    # battery mass scales by 1/k → MTOW changes → fuel fraction changes.
    # Simplified first-order: fuel_pax_nmi ≈ baseline * (1 + hybridisation * (1/k - 1) * sensitivity)
    # where sensitivity ≈ 0.4  (Breguet partial derivative).
    sensitivity = 0.40
    k_range = np.linspace(0.6, 1.4, 40)       # battery Wh/kg multiplier

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in d19.iterrows():
        hybrid = row["Hybridization"]
        base_fuel = row["Fuel/pax/nmi (lb)"]
        label = _short_label(row["Configuration"])
        color = _arch_color(row["Configuration"])

        fuel_curve = base_fuel * (1 + hybrid * (1.0 / k_range - 1) * sensitivity)
        lw = 3 if _HERO_TAG in row["Configuration"] else 1.3
        ls = "-" if _HERO_TAG in row["Configuration"] else "--"

        ax.plot(k_range * 100, fuel_curve, color=color, linewidth=lw,
                linestyle=ls, label=label)

    ax.axvline(100, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Battery Specific Energy (% of baseline Wh/kg)")
    ax.set_ylabel("Fuel / pax / nmi (lb)")
    ax.set_title("Battery Technology Sensitivity — "
                 "BLI+WT (20% hybrid) Is Most Robust",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_advocacy_battery_sens.png"), dpi=150,
                    bbox_inches="tight")
        print("  Saved: viz_advocacy_battery_sens.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 16.  Head-to-head comparison — BLI+WT vs each competitor
# ---------------------------------------------------------------------------
def plot_head_to_head(df: pd.DataFrame, out_dir: Path, save: bool):
    """
    For each competitor, a horizontal grouped bar chart showing the
    percent difference in 8 key metrics relative to BLI+WT.
    Green bars = BLI+WT advantage, red = competitor advantage.
    """
    d19 = _df19(df)
    if d19.empty or _HERO not in d19["Configuration"].values:
        print("  [skip] missing BLI+WT or 19-pax rows")
        return

    hero_row = d19[d19["Configuration"] == _HERO].iloc[0]
    competitors = [c for c in d19["Configuration"] if c != _HERO]

    metrics = [
        ("Fuel/pax/nmi (lb)", "Fuel / pax / nmi",     "lower"),
        ("CO2/pax/nmi (kg)",  "CO₂ / pax / nmi",      "lower"),
        ("MTOW (lb)",         "MTOW",                  "lower"),
        ("Fuel Burn (kg/hr)", "Fuel burn rate",        "lower"),
        ("L/D",               "Cruise L/D",            "higher"),
        ("Thermal Eff",       "Thermal efficiency",    "higher"),
        ("BFL (ft)",          "Balanced field length", "lower"),
        ("Hybridization",     "Hybridization %",       "lower"),
    ]

    n_comp = len(competitors)
    fig, axes = plt.subplots(1, n_comp, figsize=(4.2 * n_comp, 6), sharey=True)
    if n_comp == 1:
        axes = [axes]

    for ax, comp_name in zip(axes, competitors):
        comp_row = d19[d19["Configuration"] == comp_name].iloc[0]
        color = _arch_color(comp_name)

        labels = []
        pcts = []
        bar_colors = []
        for col, nice_name, direction in metrics:
            hv = hero_row[col]
            cv = comp_row[col]
            if cv and cv != 0:
                pct = (hv - cv) / abs(cv) * 100
            elif hv and hv != 0:
                pct = 100.0   # competitor is zero, hero has a value
            else:
                pct = 0.0
            # Determine if BLI+WT is better
            if direction == "lower":
                bli_wins = hv < cv
            else:
                bli_wins = hv > cv
            bar_colors.append("#4CAF50" if bli_wins else "#EF5350")
            labels.append(nice_name)
            pcts.append(pct)

        y = np.arange(len(labels))
        ax.barh(y, pcts, color=bar_colors, edgecolor="k", linewidth=0.5,
                height=0.6)
        ax.axvline(0, color="k", linewidth=0.8)
        for i, p in enumerate(pcts):
            offset = 1.0 if p >= 0 else -1.0
            ax.text(p + offset, i, f"{p:+.1f}%", va="center",
                    ha="left" if p >= 0 else "right", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("% diff (BLI+WT − competitor) / |competitor|")
        ax.set_title(f"vs {_short_label(comp_name)}", fontsize=11,
                     fontweight="bold", color=color)

    fig.suptitle("BLI+WT Head-to-Head — Green = BLI+WT Advantage",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(str(out_dir / "viz_advocacy_h2h.png"), dpi=150,
                    bbox_inches="tight")
        print("  Saved: viz_advocacy_h2h.png")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Advanced aircraft comparison visualizations")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to comparison_results.csv")
    parser.add_argument("--save", action="store_true",
                        help="Save PNGs instead of showing interactive plots")
    args = parser.parse_args()

    work_dir = Path(__file__).parent.resolve()
    csv_path = Path(args.csv) if args.csv else work_dir / "comparison_results.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        print("       Run compare_all_planes.py first.")
        sys.exit(1)

    df = _load_csv(csv_path)
    print(f"Loaded {len(df)} aircraft from {csv_path.name}")
    print(f"Columns: {len(df.columns)}")

    if args.save:
        matplotlib.use("Agg")

    out_dir = work_dir

    print("\n1/16  Radar chart ...")
    plot_radar(df, out_dir, args.save)

    print("2/16  Scoring heatmap ...")
    plot_scoring_heatmap(df, out_dir, args.save)

    print("3/16  Scatter plots ...")
    plot_scatters(df, out_dir, args.save)

    print("4/16  Energy breakdown & field perf ...")
    plot_energy_and_field(df, out_dir, args.save)

    print("5/16  Geometry & fuel breakdown ...")
    plot_geometry_and_fuel(df, out_dir, args.save)

    print("6/16  Payload-range bubble ...")
    plot_payload_range_bubble(df, out_dir, args.save)

    print("7/16  Parallel coordinates ...")
    plot_parallel_coordinates(df, out_dir, args.save)

    print("8/16  Pareto fronts ...")
    plot_pareto(df, out_dir, args.save)

    print("9/16  Correlation matrix ...")
    plot_correlation_matrix(df, out_dir, args.save)

    print("10/16 Delta from baseline ...")
    plot_delta_from_baseline(df, out_dir, args.save)

    print("11/16 Pairwise trade-off matrix ...")
    plot_pairwise_matrix(df, out_dir, args.save)

    # -- BLI+WT Advocacy charts --
    print("12/16 Efficiency waterfall (BLI+WT advocacy) ...")
    plot_efficiency_waterfall(df, out_dir, args.save)

    print("13/16 Technology risk (BLI+WT advocacy) ...")
    plot_tech_risk(df, out_dir, args.save)

    print("14/16 DOC estimate (BLI+WT advocacy) ...")
    plot_doc_estimate(df, out_dir, args.save)

    print("15/16 Battery sensitivity (BLI+WT advocacy) ...")
    plot_battery_sensitivity(df, out_dir, args.save)

    print("16/16 Head-to-head (BLI+WT advocacy) ...")
    plot_head_to_head(df, out_dir, args.save)

    print("\nDone.")


if __name__ == "__main__":
    main()
