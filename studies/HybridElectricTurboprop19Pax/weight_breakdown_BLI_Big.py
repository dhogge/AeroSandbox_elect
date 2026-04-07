"""
Weight Breakdown Visualization for BLI_Big.py
===============================================

Runs BLI_Big.py, parses the weight breakdown table from its output,
and produces detailed pie charts and a horizontal bar chart of the
component-level weight breakdown.

Usage:
    python weight_breakdown_BLI_Big.py
"""

import subprocess
import sys
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


LBM = 0.453592  # kg per lb


def run_and_parse():
    """Run BLI_Big.py and parse the weight breakdown from stdout."""
    work_dir = Path(__file__).parent.resolve()
    script = work_dir / "BLI_Big.py"

    print("Running BLI_Big.py — this may take a few minutes …")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(work_dir),
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    if result.returncode != 0:
        print("BLI_Big.py failed:")
        for line in result.stderr.strip().splitlines()[-30:]:
            print(f"  STDERR: {line}")
        sys.exit(1)

    raw = result.stdout

    # ------------------------------------------------------------------
    # Parse top-level values
    # ------------------------------------------------------------------
    def _f(pattern, text=raw):
        m = re.search(pattern, text)
        return float(m.group(1).replace(",", "")) if m else 0.0

    mtow_kg = _f(r"MTOW:\s+([\d,.]+)\s+kg")
    empty_kg = _f(r"Empty Weight:\s+([\d,.]+)\s+kg")
    fuel_kg = _f(r"Fuel Weight:\s+([\d,.]+)\s+kg")
    battery_kg = _f(r"Battery Weight:\s+([\d,.]+)\s+kg")
    payload_kg = _f(r"Payload:\s+([\d,.]+)\s+kg")

    # Fallbacks from breakdown table
    if mtow_kg == 0:
        mtow_kg = _f(r"MTOW\s+([\d,.]+)\s+([\d,.]+)")
    if empty_kg == 0:
        m = re.search(r"EMPTY WEIGHT\s+([\d,.]+)", raw)
        if m:
            empty_kg = float(m.group(1).replace(",", ""))

    # ------------------------------------------------------------------
    # Parse component-level breakdown table
    # Each line looks like:  Wing                          123.4      272.0    5.6%
    # ------------------------------------------------------------------
    components = []
    in_breakdown = False
    past_header = False
    for line in raw.splitlines():
        if "Weight Breakdown" in line and not in_breakdown:
            in_breakdown = True
            past_header = False
            continue
        if in_breakdown:
            # Skip the column header row ("Component ... % MTOW") and
            # separator lines (all dashes).  We know real data starts
            # once we've seen the separator after the header.
            stripped = line.strip()
            if not past_header:
                # Still in headers — look for the dash separator
                if stripped.startswith("-"):
                    past_header = True
                continue
            # Stop at the summary separator / EMPTY WEIGHT line
            if stripped.startswith("EMPTY WEIGHT") or stripped.startswith("MTOW"):
                break
            if stripped.startswith("-"):
                break
            # Match: name  mass_kg  mass_lb  pct%
            m = re.match(
                r"^\s{2}(.+?)\s{2,}([\d,.]+)\s+([\d,.]+)\s+([\d.]+)%",
                line,
            )
            if m:
                name = m.group(1).strip()
                mass_kg = float(m.group(2).replace(",", ""))
                mass_lb = float(m.group(3).replace(",", ""))
                pct = float(m.group(4))
                if name:
                    components.append((name, mass_kg, mass_lb, pct))

    # Also extract Payload, Fuel, Battery from the summary rows
    top_level = []
    for line in raw.splitlines():
        for label in ("Payload", "Fuel", "Battery"):
            m = re.match(
                rf"^\s{{2}}{label}\s{{2,}}([\d,.]+)\s+([\d,.]+)\s+([\d.]+)%",
                line,
            )
            if m:
                top_level.append((
                    label,
                    float(m.group(1).replace(",", "")),
                    float(m.group(2).replace(",", "")),
                    float(m.group(3)),
                ))

    return {
        "mtow_kg": mtow_kg,
        "empty_kg": empty_kg,
        "fuel_kg": fuel_kg,
        "battery_kg": battery_kg,
        "payload_kg": payload_kg,
        "components": components,
        "top_level": top_level,
    }


def plot_weight_breakdown(data):
    out_dir = Path(__file__).parent

    components = data["components"]
    top_level = data["top_level"]
    mtow_kg = data["mtow_kg"]

    if not components:
        print("  WARNING: Could not parse component breakdown — skipping plots.")
        return

    # ------------------------------------------------------------------
    # Group components into categories for the nested pie chart
    # ------------------------------------------------------------------
    structure_names = {
        "Wing", "H-Stab", "V-Stab", "Fuselage",
        "Main Landing Gear", "Nose Landing Gear", "Nacelles (wingtip)",
    }
    wingtip_prop_names = {
        "Turboshaft Engines", "Wingtip Electric Motors", "Wingtip ESCs",
        "Wingtip Propellers", "Gearboxes", "Fuel System",
    }
    bli_prop_names = {
        "BLI Motor", "BLI ESC", "BLI Propeller", "BLI Nacelle",
    }
    systems_names = {
        "Instruments", "Electrical System", "Furnishings",
        "Air Conditioning", "Anti-Ice", "Flight Controls",
        "Seats", "Lavatories",
    }

    def group_mass(names):
        return sum(kg for n, kg, _, _ in components if n in names)

    struct_kg = group_mass(structure_names)
    wt_prop_kg = group_mass(wingtip_prop_names)
    bli_prop_kg = group_mass(bli_prop_names)
    systems_kg = group_mass(systems_names)

    fuel_kg = data["fuel_kg"]
    battery_kg = data["battery_kg"]
    payload_kg = data["payload_kg"]

    # ==================================================================
    # FIGURE 1 — High-level pie (4 slices) + detailed bar chart
    # ==================================================================
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(28, 12),
                                          gridspec_kw={"width_ratios": [1, 1.2]})
    fig.suptitle("BLI + Wingtip — Weight Breakdown",
                 fontsize=24, fontweight="bold")

    # --- Pie chart: top-level categories ---
    cat_legend_labels = [
        f"Structure — {struct_kg:.0f} kg ({struct_kg/LBM:,.0f} lb)",
        f"Wingtip Propulsion — {wt_prop_kg:.0f} kg ({wt_prop_kg/LBM:,.0f} lb)",
        f"BLI Propulsion — {bli_prop_kg:.0f} kg ({bli_prop_kg/LBM:,.0f} lb)",
        f"Systems & Cabin — {systems_kg:.0f} kg ({systems_kg/LBM:,.0f} lb)",
        f"Fuel — {fuel_kg:.0f} kg ({fuel_kg/LBM:,.0f} lb)",
        f"Battery — {battery_kg:.0f} kg ({battery_kg/LBM:,.0f} lb)",
        f"Payload — {payload_kg:.0f} kg ({payload_kg/LBM:,.0f} lb)",
    ]
    cat_sizes = [struct_kg, wt_prop_kg, bli_prop_kg, systems_kg,
                 fuel_kg, battery_kg, payload_kg]
    cat_colors = ["#78909C", "#1565C0", "#00897B", "#AB47BC",
                  "#FF9800", "#4CAF50", "#E91E63"]

    # Filter zero slices
    filt = [(l, s, c) for l, s, c in zip(cat_legend_labels, cat_sizes, cat_colors) if s > 0]
    cat_legend_f = [x[0] for x in filt]
    cat_sizes_f = [x[1] for x in filt]
    cat_colors_f = [x[2] for x in filt]

    wedges, texts = ax_pie.pie(
        cat_sizes_f,
        colors=cat_colors_f,
        startangle=90,
        textprops={"fontsize": 20},
    )

    # Place percentage labels: large slices inside, small slices outside with leader lines
    total = sum(cat_sizes_f)
    for i, (wedge, size) in enumerate(zip(wedges, cat_sizes_f)):
        pct = 100.0 * size / total
        if pct < 1.5:
            continue  # skip very tiny slices
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        ang_rad = np.deg2rad(ang)
        if pct >= 10:
            # Large slice: label inside
            x = 0.70 * np.cos(ang_rad)
            y = 0.70 * np.sin(ang_rad)
            ax_pie.text(x, y, f"{pct:.1f}%", ha="center", va="center",
                        fontsize=20, fontweight="bold")
        else:
            # Small slice: label outside with a leader line
            x_text = 1.25 * np.cos(ang_rad)
            y_text = 1.25 * np.sin(ang_rad)
            x_wedge = 0.95 * np.cos(ang_rad)
            y_wedge = 0.95 * np.sin(ang_rad)
            ha = "left" if x_text >= 0 else "right"
            ax_pie.annotate(
                f"{pct:.1f}%",
                xy=(x_wedge, y_wedge),
                xytext=(x_text, y_text),
                ha=ha, va="center",
                fontsize=18, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=1.2),
            )

    ax_pie.legend(wedges, cat_legend_f, fontsize=20, loc="center left",
                  bbox_to_anchor=(-0.35, 0.5), frameon=False)
    ax_pie.set_title(f"MTOW = {mtow_kg:,.0f} kg  ({mtow_kg/LBM:,.0f} lb)",
                     fontsize=22, fontweight="bold")

    # --- Horizontal bar chart: every component ---
    names = [n for n, _, _, _ in components]
    masses_lb = [lb for _, _, lb, _ in components]

    # Sort by weight (heaviest at top)
    order = sorted(range(len(masses_lb)), key=lambda i: masses_lb[i])
    names = [names[i] for i in order]
    masses_lb = [masses_lb[i] for i in order]

    # Color-code by category
    def bar_color(name):
        if name in structure_names:
            return "#78909C"
        if name in wingtip_prop_names:
            return "#1565C0"
        if name in bli_prop_names:
            return "#00897B"
        return "#AB47BC"

    colors = [bar_color(n) for n in names]
    y_pos = range(len(names))

    ax_bar.barh(y_pos, masses_lb, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=20)
    ax_bar.set_xlabel("Mass (lb)", fontsize=22)
    ax_bar.set_title("Empty Weight Components", fontsize=22, fontweight="bold")
    ax_bar.tick_params(axis="x", labelsize=20)
    ax_bar.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Value labels on bars
    for i, v in enumerate(masses_lb):
        ax_bar.text(v + max(masses_lb) * 0.01, i, f"{v:,.0f}",
                    va="center", fontsize=20, fontweight="bold")

    # Legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#78909C", edgecolor="black", label="Structure"),
        Patch(facecolor="#1565C0", edgecolor="black", label="Wingtip Propulsion"),
        Patch(facecolor="#00897B", edgecolor="black", label="BLI Propulsion"),
        Patch(facecolor="#AB47BC", edgecolor="black", label="Systems & Cabin"),
    ]
    ax_bar.legend(handles=legend_elements, fontsize=20, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "weight_breakdown_BLI_Big.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\n  Saved weight breakdown chart to: {out}")
    plt.close(fig)

    # ==================================================================
    # FIGURE 2 — Detailed pie: all individual components + fuel/batt/payload
    # ==================================================================
    fig2, ax2 = plt.subplots(figsize=(18, 14))
    fig2.suptitle("BLI + Wingtip — Full MTOW Breakdown (All Components)",
                  fontsize=24, fontweight="bold")

    all_names = [n for n, _, _, _ in components] + ["Fuel", "Battery", "Payload"]
    all_kg = [kg for _, kg, _, _ in components] + [fuel_kg, battery_kg, payload_kg]

    # Color each slice
    def slice_color(name):
        if name in structure_names:
            return "#78909C"
        if name in wingtip_prop_names:
            return "#1565C0"
        if name in bli_prop_names:
            return "#00897B"
        if name == "Fuel":
            return "#FF9800"
        if name == "Battery":
            return "#4CAF50"
        if name == "Payload":
            return "#E91E63"
        return "#AB47BC"

    all_colors = [slice_color(n) for n in all_names]
    all_legend_labels = [
        f"{n} — {kg:.0f} kg ({kg/LBM:,.0f} lb)"
        for n, kg in zip(all_names, all_kg)
    ]

    # Filter zeros
    filt2 = [(l, s, c) for l, s, c in zip(all_legend_labels, all_kg, all_colors) if s > 0]
    all_legend_f = [x[0] for x in filt2]
    all_kg_f = [x[1] for x in filt2]
    all_colors_f = [x[2] for x in filt2]

    wedges2, texts2, autotexts2 = ax2.pie(
        all_kg_f,
        colors=all_colors_f,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        startangle=90,
        pctdistance=0.80,
        textprops={"fontsize": 20},
    )
    for at in autotexts2:
        at.set_fontsize(20)
    ax2.legend(wedges2, all_legend_f, fontsize=20, loc="center left",
               bbox_to_anchor=(-0.45, 0.5), frameon=False)
    ax2.set_title(f"MTOW = {mtow_kg:,.0f} kg  ({mtow_kg/LBM:,.0f} lb)",
                  fontsize=22, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out2 = out_dir / "weight_breakdown_BLI_Big_detailed.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    print(f"  Saved detailed breakdown chart to: {out2}")
    plt.close(fig2)

    # ==================================================================
    # FIGURE 3 — Standalone category-level pie chart
    # ==================================================================
    fig3, ax3 = plt.subplots(figsize=(16, 12))

    wedges3, _ = ax3.pie(
        cat_sizes_f,
        colors=cat_colors_f,
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )

    # Smart label placement: large slices inside, small slices outside
    total3 = sum(cat_sizes_f)
    # Collect outside annotations for overlap adjustment
    outside_annotations = []
    for i, (wedge, size) in enumerate(zip(wedges3, cat_sizes_f)):
        pct = 100.0 * size / total3
        if pct < 0.5:
            continue
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        ang_rad = np.deg2rad(ang)
        if pct >= 12:
            x = 0.65 * np.cos(ang_rad)
            y = 0.65 * np.sin(ang_rad)
            ax3.text(x, y, f"{pct:.1f}%", ha="center", va="center",
                     fontsize=22, fontweight="bold", color="white")
        else:
            x_wedge = 0.97 * np.cos(ang_rad)
            y_wedge = 0.97 * np.sin(ang_rad)
            x_text = 1.35 * np.cos(ang_rad)
            y_text = 1.35 * np.sin(ang_rad)
            outside_annotations.append((x_text, y_text, x_wedge, y_wedge, pct, ang))

    # Resolve overlapping outside labels by enforcing minimum vertical spacing
    if outside_annotations:
        outside_annotations.sort(key=lambda t: t[1])  # sort by y_text
        min_gap = 0.12
        ys = [a[1] for a in outside_annotations]
        # Push labels apart if too close
        for j in range(1, len(ys)):
            if ys[j] - ys[j - 1] < min_gap:
                ys[j] = ys[j - 1] + min_gap
        # Re-center the adjusted positions
        for j, (x_text, y_orig, x_wedge, y_wedge, pct, ang) in enumerate(outside_annotations):
            y_adj = ys[j]
            ha = "left" if x_text >= 0 else "right"
            ax3.annotate(
                f"{pct:.1f}%",
                xy=(x_wedge, y_wedge),
                xytext=(x_text, y_adj),
                ha=ha, va="center",
                fontsize=20, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=1.2),
            )

    ax3.legend(wedges3, cat_legend_f, fontsize=18, loc="center left",
               bbox_to_anchor=(-0.42, 0.5), frameon=False)
    ax3.set_title(f"MTOW = {mtow_kg:,.0f} kg  ({mtow_kg/LBM:,.0f} lb)",
                  fontsize=24, fontweight="bold", pad=20)

    plt.tight_layout()
    out3 = out_dir / "weight_pie_BLI_Big.png"
    fig3.savefig(str(out3), dpi=150, bbox_inches="tight")
    print(f"  Saved standalone pie chart to: {out3}")
    plt.close(fig3)


def main():
    data = run_and_parse()

    if data["mtow_kg"] == 0:
        print("ERROR: Could not parse MTOW from BLI_Big.py output.")
        sys.exit(1)

    # Print a summary table to console
    print("\n" + "=" * 70)
    print("   BLI + Wingtip — WEIGHT BREAKDOWN SUMMARY")
    print("=" * 70)
    print(f"\n  MTOW:         {data['mtow_kg']:>8.0f} kg   ({data['mtow_kg']/LBM:>8,.0f} lb)")
    print(f"  Empty Weight: {data['empty_kg']:>8.0f} kg   ({data['empty_kg']/LBM:>8,.0f} lb)")
    print(f"  Fuel:         {data['fuel_kg']:>8.0f} kg   ({data['fuel_kg']/LBM:>8,.0f} lb)")
    print(f"  Battery:      {data['battery_kg']:>8.0f} kg   ({data['battery_kg']/LBM:>8,.0f} lb)")
    print(f"  Payload:      {data['payload_kg']:>8.0f} kg   ({data['payload_kg']/LBM:>8,.0f} lb)")

    if data["components"]:
        print(f"\n  {'Component':<28} {'kg':>8} {'lb':>10} {'%MTOW':>7}")
        print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*7}")
        for name, kg, lb, pct in data["components"]:
            print(f"  {name:<28} {kg:8.1f} {lb:10.1f} {pct:6.1f}%")

    print("=" * 70)

    plot_weight_breakdown(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
