"""
aero_visual.py — Aerodynamic Characteristics Visualization for BLI_Big
======================================================================

Runs BLI_Big.py, parses aerodynamic outputs, and generates
presentation-ready figures:

  1. Drag Waterfall  (CD buildup → BLI reduction)
  2. L/D Comparison  (BLI_Big vs. reference turboprops)
  3. Key Aero Summary Card (CL, CD, α, wing loading, L/D)

Usage:
    python aero_visual.py           # saves PNGs
    python aero_visual.py --show    # also opens matplotlib window
"""

import subprocess, sys, re, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colours ──────────────────────────────────────────────────────────
C_BLUE   = "#2196F3"
C_GREEN  = "#4CAF50"
C_RED    = "#F44336"
C_ORANGE = "#FF9800"
C_GREY   = "#9E9E9E"
C_DARK   = "#37474F"
C_TEAL   = "#009688"
C_BG     = "#FAFAFA"


# =====================================================================
# 1. Run BLI_Big.py and capture stdout
# =====================================================================

def run_bli_big() -> str:
    work_dir = Path(__file__).parent.resolve()
    script = work_dir / "BLI_Big.py"
    print("Running BLI_Big.py — this may take a few minutes …")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=600,
        cwd=str(work_dir),
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if result.returncode != 0:
        print("BLI_Big.py FAILED:")
        for ln in result.stderr.strip().splitlines()[-30:]:
            print(f"  {ln}")
        sys.exit(1)
    return result.stdout


# =====================================================================
# 2. Parse all aero values from stdout
# =====================================================================

def parse_aero(raw: str) -> dict:
    """Extract aero values from BLI_Big.py printed output."""

    def _f(pattern, default=0.0):
        m = re.search(pattern, raw)
        return float(m.group(1).replace(",", "")) if m else default

    def _s(pattern, default=""):
        m = re.search(pattern, raw)
        return m.group(1).strip() if m else default

    d = {}
    # Top-level
    d["MTOW_kg"]    = _f(r"MTOW:\s+([\d,.]+)\s+kg")
    d["MTOW_lb"]    = _f(r"MTOW:\s+[\d,.]+\s+kg\s+\(\s*([\d,.]+)\s+lb\)")

    # Geometry
    d["wing_area"]  = _f(r"Wing Area:\s+([\d,.]+)\s+m")
    d["wing_span"]  = _f(r"Wing Span:\s+([\d,.]+)\s+m")
    d["AR"]         = _f(r"Aspect Ratio:\s+([\d,.]+)")
    d["taper"]      = _f(r"Taper Ratio:\s+([\d,.]+)")
    d["wing_loading_psf"] = _f(r"Wing Loading:\s+[\d,.]+\s+N/m\^2\s+\(\s*([\d,.]+)\s+psf\)")
    d["wing_loading_Nm2"] = _f(r"Wing Loading:\s+([\d,.]+)\s+N/m\^2")

    # Aero
    d["CL"]        = _f(r"CL:\s+([\d,.]+)")
    d["CD"]        = _f(r"CD:\s+([\d,.]+)")
    d["LD_cruise"] = _f(r"L/D \(cruise\):\s+([\d,.]+)")
    d["LD_climb"]  = _f(r"L/D \(climb.*?\):\s+([\d,.]+)")
    d["alpha"]     = _f(r"Alpha:\s+([\d,.]+)")
    d["drag_N"]    = _f(r"Cruise Drag:\s+([\d,.]+)\s+N")
    d["drag_lbf"]  = _f(r"Cruise Drag:\s+[\d,.]+\s+N\s+\(\s*([\d,.]+)\s+lbf\)")
    d["eff_drag_N"]= _f(r"Effective Drag \(w/ BLI\):\s+([\d,.]+)\s+N")
    d["bli_wake_N"]= _f(r"BLI Wake Fill Thrust:\s+([\d,.]+)\s+N")

    # Drag breakdown
    d["CD_raw"]    = _f(r"AeroBuildup CD \(raw\):\s+([\d,.]+)")
    d["CD_corr"]   = _f(r"After 10% correction:\s+([\d,.]+)")
    d["CD_misc"]   = _f(r"Misc drag CD \(CDA/S\):\s+([\d,.]+)")
    d["CD_total"]  = _f(r"Total CD \(corrected\):\s+([\d,.]+)")

    # Flight condition
    d["Mach"]      = _f(r"Mach Number:\s+([\d,.]+)")
    d["Re_MAC"]    = _f(r"Re \(MAC\):\s+([\d,.eE+]+)")
    d["q_Pa"]      = _f(r"Dynamic Pressure:\s+([\d,.]+)\s+Pa")
    d["cruise_ft"] = _f(r"Cruise Altitude:\s+([\d,.]+)\s+ft")

    return d


# =====================================================================
# 3. Figure 1 — Drag Waterfall Chart
# =====================================================================

def plot_drag_waterfall(d: dict, ax=None):
    """CD buildup waterfall: raw → +10% → +misc → total → −BLI → effective."""
    show_own = ax is None
    if show_own:
        fig, ax = plt.subplots(figsize=(10, 5.5))

    cd_raw  = d["CD_raw"]
    cd_corr = d["CD_corr"]     # raw × 1.10
    cd_misc = d["CD_misc"]
    cd_tot  = d["CD_total"]
    bli_pct = 0.10
    cd_eff  = cd_tot * (1 - bli_pct)

    # Waterfall steps: base, +correction, +misc, total, −BLI, effective
    labels = [
        "AeroBuildup\n(base)",
        "+10%\ncorrection",
        "+Misc drag\n(CDA/S)",
        "Total CD",
        "−10% BLI\ncredit",
        "Effective CD",
    ]
    values   = [cd_raw, cd_corr - cd_raw, cd_misc, cd_tot, -(cd_tot * bli_pct), cd_eff]
    cumul    = [0.0] * 6
    bottoms  = [0.0] * 6
    colours  = [C_BLUE, C_ORANGE, C_ORANGE, C_DARK, C_GREEN, C_TEAL]

    # Compute running total and bottoms for waterfall bars
    running = 0.0
    for i in range(len(values)):
        if i in (3, 5):  # Total / Effective — full bars from 0
            bottoms[i] = 0.0
            cumul[i]   = values[i]
        elif values[i] < 0:  # BLI reduction — drops from previous
            bottoms[i] = running + values[i]
            cumul[i]   = -values[i]
            running   += values[i]
        else:
            bottoms[i] = running
            cumul[i]   = values[i]
            running   += values[i]

    bars = ax.bar(labels, cumul, bottom=bottoms, color=colours,
                  edgecolor="white", linewidth=1.5, width=0.55, zorder=3)

    # Value labels on each bar
    for i, bar in enumerate(bars):
        y_pos = bottoms[i] + cumul[i] + cd_tot * 0.02
        sign = "" if i in (3, 5) else ("+" if values[i] > 0 else "")
        txt = f"{sign}{values[i]:.5f}"
        if i in (3, 5):
            txt = f"{values[i]:.5f}"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, txt,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=colours[i])

    # Connector lines
    for i in range(len(bars) - 1):
        if i == 2:  # skip connector before "Total" bar
            continue
        if i == 3:  # connector from Total to BLI
            top = bottoms[3] + cumul[3]
            ax.plot([bars[3].get_x() + bars[3].get_width(),
                     bars[4].get_x()],
                    [top, top], color=C_GREY, lw=0.8, ls="--", zorder=2)
            continue
        top = bottoms[i] + cumul[i]
        ax.plot([bars[i].get_x() + bars[i].get_width(),
                 bars[i + 1].get_x()],
                [top, top], color=C_GREY, lw=0.8, ls="--", zorder=2)

    ax.set_ylabel("Drag Coefficient (CD)", fontsize=11, fontweight="bold")
    ax.set_title("Drag Buildup & BLI Reduction", fontsize=14, fontweight="bold",
                 pad=12)
    ax.set_ylim(0, cd_tot * 1.25)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_own:
        fig.tight_layout()
        return fig


# =====================================================================
# 4. Figure 2 — L/D Comparison Bar Chart
# =====================================================================

def plot_ld_comparison(d: dict, ax=None):
    """Compare BLI_Big L/D against reference turboprops."""
    show_own = ax is None
    if show_own:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Reference aircraft L/D values (published / estimated)
    aircraft = [
        ("Beech 1900D",         13.5),
        ("DHC-8-100\n(Dash 8)", 14.0),
        ("Cessna\nSkyCourier",  13.0),
    ]

    # Effective L/D with BLI: drag reduced by 10% → L/D_eff = L/D / 0.90
    ld_eff = d["LD_cruise"] / 0.90
    aircraft.append(("Preferred\nSystem Concept", ld_eff))

    names  = [a[0] for a in aircraft]
    values = [a[1] for a in aircraft]
    colors = [C_GREY, C_GREY, C_GREY, C_GREEN]

    bars = ax.barh(names, values, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=11, fontweight="bold",
                color=C_DARK)

    ax.set_xlabel("Lift-to-Drag Ratio (L/D)", fontsize=11, fontweight="bold")
    ax.set_title("Cruise L/D — BLI_Big vs. Reference Turboprops",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    if show_own:
        fig.tight_layout()
        return fig


# =====================================================================
# 5. Figure 3 — Aero Summary Card
# =====================================================================

def plot_summary_card(d: dict, ax=None):
    """Clean summary card of key aero parameters for slide embedding."""
    show_own = ax is None
    if show_own:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, "Key Aerodynamic Characteristics",
            ha="center", va="top", fontsize=18, fontweight="bold", color=C_DARK)

    # Two-column layout
    left_x, right_x = 1.2, 6.2
    row_h = 1.1
    start_y = 8.0

    entries = [
        # (label, value, unit, column)
        ("CL (cruise)",          f"{d['CL']:.4f}",            "",      "L"),
        ("CD (total)",           f"{d['CD_total']:.5f}",      "",      "L"),
        ("L/D (cruise)",         f"{d['LD_cruise']:.1f}",     "",      "L"),
        ("L/D (climb)",          f"{d['LD_climb']:.1f}",      "(65% cruise)", "L"),
        ("Alpha",                f"{d['alpha']:.1f}",         "deg",   "L"),
        ("Effective L/D w/ BLI", f"{d['LD_cruise'] / 0.90:.1f}", "(+10% BLI)", "L"),

        ("Wing Area",            f"{d['wing_area']:.1f}",     "m²",    "R"),
        ("Aspect Ratio",         f"{d['AR']:.1f}",            "",      "R"),
        ("Taper Ratio",          f"{d['taper']:.2f}",         "",      "R"),
        ("Wing Loading",         f"{d['wing_loading_psf']:.1f}", "psf", "R"),
        ("Cruise Drag",          f"{d['drag_lbf']:.0f}",      "lbf",   "R"),
        ("Eff. Drag (w/ BLI)",   f"{d['eff_drag_N'] / 4.44822:.0f}", "lbf", "R"),
    ]

    row_l = 0
    row_r = 0
    for label, value, unit, col in entries:
        if col == "L":
            x = left_x
            y = start_y - row_l * row_h
            row_l += 1
        else:
            x = right_x
            y = start_y - row_r * row_h
            row_r += 1

        ax.text(x, y, label, fontsize=12, fontweight="bold", color=C_DARK,
                va="center")
        disp = f"{value} {unit}".strip()
        ax.text(x + 3.5, y, disp, fontsize=12, color=C_BLUE,
                va="center", fontweight="bold", family="monospace")

    # Divider line
    ax.plot([5.5, 5.5], [1.0, 8.7], color=C_GREY, lw=0.5, alpha=0.5)

    # Bottom note
    ax.text(5, 0.3,
            f"Cruise: 200 kt @ {d['cruise_ft']:.0f} ft  |  Mach {d['Mach']:.3f}  |  "
            f"BLI: −10% drag credit  |  Wingtip props: +15% η",
            ha="center", va="bottom", fontsize=9, color=C_GREY, style="italic")

    if show_own:
        fig.tight_layout()
        return fig


# =====================================================================
# 6. Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aero visuals for BLI_Big")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    raw = run_bli_big()
    d = parse_aero(raw)

    print("\n─── Parsed Aero Values ───")
    for k, v in d.items():
        print(f"  {k:25s} = {v}")

    out_dir = Path(__file__).parent / "n2_diagram_out" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Drag waterfall
    fig1, ax1 = plt.subplots(figsize=(10, 5.5))
    plot_drag_waterfall(d, ax1)
    fig1.tight_layout()
    fig1.savefig(out_dir / "aero_drag_waterfall.png", dpi=200, bbox_inches="tight")
    print(f"Saved → {out_dir / 'aero_drag_waterfall.png'}")

    # Figure 2: L/D comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_ld_comparison(d, ax2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "aero_ld_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved → {out_dir / 'aero_ld_comparison.png'}")

    # Figure 3: Summary card
    fig3, ax3 = plt.subplots(figsize=(10, 5.5))
    plot_summary_card(d, ax3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "aero_summary_card.png", dpi=200, bbox_inches="tight")
    print(f"Saved → {out_dir / 'aero_summary_card.png'}")

    # Figure 4: Combined 2-panel (waterfall + L/D) for single slide
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
    plot_drag_waterfall(d, ax4a)
    plot_ld_comparison(d, ax4b)
    fig4.suptitle("BLI_Big — Aerodynamic Performance", fontsize=16,
                  fontweight="bold", y=1.02)
    fig4.tight_layout()
    fig4.savefig(out_dir / "aero_combined.png", dpi=200, bbox_inches="tight")
    print(f"Saved → {out_dir / 'aero_combined.png'}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()

    print("\nDone — 4 figures saved to", out_dir)


if __name__ == "__main__":
    main()
