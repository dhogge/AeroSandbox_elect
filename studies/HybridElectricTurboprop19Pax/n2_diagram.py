"""
n2_diagram.py — Clean N² Diagram for BLI+WT Architecture
=========================================================

Generates a compact, readable N² (Design Structure Matrix) as a
standalone HTML file using only matplotlib (no OpenMDAO dependency).

Each discipline is ONE cell — the matrix shows which disciplines
feed data to which, with labeled coupling variables on hover.

8 blocks (BLI merged under Propulsion):
  Design Variables → Geometry → Aero → Propulsion & BLI →
  Weights → Field Perf → Stability → FuelEnergy

Feed-forward (upper triangle): direct coded variable dependencies.
Feedback (lower triangle): physical coupling loops closed
    simultaneously by the CasADi/Opti optimizer.

Usage:
    python n2_diagram.py            # generates n2_bli_big.html
    python n2_diagram.py --open     # generate and open in browser
"""

import argparse
import webbrowser
from pathlib import Path

# ===================================================================
# 1. Discipline names and coupling definitions
# ===================================================================

DISCIPLINES = [
    "Design\nVariables",
    "Geometry",
    "Aero",
    "Propulsion",
    "Weights",
    "Field\nPerf",
    "Stability",
    "Fuel &\nEnergy",
]

# Couplings: (from_index, to_index, [list of variables])
# Indices match DISCIPLINES list above (BLI merged into Propulsion & BLI)
COUPLINGS = [
    # DesVars → everything
    (0, 1, ["wing_span", "wing_root_chord", "vstab_span", "vstab_root_chord",
             "hstab_area", "hstab_AR", "prop_diameter", "bli_prop_diameter"]),
    (0, 2, ["cruise_alpha", "cruise_altitude"]),
    (0, 3, ["m_turboshaft", "hybrid_factor", "thrust_liftoff",
             "bli_thrust_liftoff", "prop_diameter", "cruise_altitude",
             "bli_motor_power", "bli_prop_diameter"]),
    (0, 4, ["TOGW", "wing_span", "fuel_mass", "battery_Wh",
             "m_turboshaft", "prop_diameter", "hstab_area"]),
    (0, 5, ["TOGW", "thrust_liftoff",
             "thrust_oei_reduced", "bli_thrust_liftoff"]),
    (0, 6, ["x_cg_battery", "thrust_oei_reduced",
             "wing_span", "hstab_area"]),
    (0, 7, ["fuel_mass", "battery_Wh", "hybrid_factor"]),

    # Geometry → downstream
    (1, 2, ["wing_area", "wing_AR"]),
    (1, 3, ["wingtip_prop_area", "bli_prop_area"]),
    (1, 4, ["wing_area", "wing_AR", "vstab_area"]),
    (1, 5, ["wing_area"]),
    (1, 6, ["wing_area", "wing_MAC", "vstab_area",
             "wing_AR → a_w (lift-curve slope)"]),

    # Aero → downstream
    (2, 3, ["drag_eff_cruise", "drag_bli_wake_fill"]),
    (2, 4, ["L/D → m_fuselage (Raymer)"]),
    (2, 5, ["L/D_cruise → L/D_climb"]),

    # Propulsion & BLI → downstream
    (3, 4, ["P_turboshaft", "P_electric",
             "m_bli_motor", "m_bli_esc", "m_bli_prop", "m_bli_nacelle"]),
    (3, 5, ["thrust_per_engine → OEI climb grad"]),
    (3, 7, ["fuel_burn_rate", "P_electric", "E_bli_climb"]),

    # Weights → downstream
    (4, 6, ["x_cg_aft", "x_cg_TOGW"]),

    # ══════════════════════════════════════════════════════════════
    # Feedback loops (below diagonal)
    # In BLI_Big.py the optimizer closes ALL of these simultaneously.
    # In a sequential MDA they would be iteration loops.
    # ══════════════════════════════════════════════════════════════

    # ── Back to DesVars (column 0) ──
    (4, 0, ["mass_total → TOGW"]),
    (7, 0, ["fuel_mass_typ → fuel_mass",
            "total_elec_energy → battery_Wh"]),
    (2, 0, ["lift ≈ weight (trim)"]),
    (5, 0, ["BFL → thrust_liftoff",
            "landing_dist → wing_area"]),
    (6, 0, ["SM → hstab_area",
            "V_mc → vstab / thrust_oei",
            "x_cg → x_cg_battery"]),

    # ── Weights ↔ Aero ──
    (4, 2, ["mass_total → CL_req"]),

    # ── Weights ↔ Propulsion & BLI ──
    (4, 3, ["mass_total → req thrust",
            "V_stall → V_liftoff → TO power"]),

    # ── Fuel/Energy ↔ Weights ──
    (7, 4, ["fuel_mass → mass_total"]),

    # ── Stability ↔ Geometry ──
    (6, 1, ["SM, V_h → vstab, hstab"]),

    # ── Field Perf ↔ Propulsion & BLI ──
    (5, 3, ["BFL → thrust / power"]),
]

# Colours
DIAG_COLOR   = "#2196F3"   # blue diagonal
FEED_FWD     = "#4CAF50"   # green upper triangle
FEEDBACK     = "#F44336"   # red lower triangle
BG_COLOR     = "#FAFAFA"
GRID_COLOR   = "#E0E0E0"


# ===================================================================
# 2. Build self-contained HTML
# ===================================================================

def _make_html() -> str:
    n = len(DISCIPLINES)

    # Build cell data: matrix[row][col] = list of variable names
    matrix = [[[] for _ in range(n)] for _ in range(n)]
    for src, tgt, variables in COUPLINGS:
        matrix[src][tgt] = variables  # N² convention: row=source, col=target

    # Diagonal labels
    diag_labels = DISCIPLINES

    html_parts = []
    html_parts.append(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>N² Diagram — BLI+WT 19-Pax Hybrid-Electric</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f5f5f5; display: flex; flex-direction: column;
    align-items: center; padding: 24px;
  }
  h1 { font-size: 1.8rem; color: #333; margin-bottom: 6px; }
  .subtitle { font-size: 1.05rem; color: #777; margin-bottom: 18px; }
  .legend { display:flex; gap:22px; margin-bottom:16px; font-size:1.0rem; color:#555; }
  .legend span { display:inline-flex; align-items:center; gap:5px; }
  .swatch { width:16px; height:16px; border-radius:2px; display:inline-block; }
  table { border-collapse: collapse; background: white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
  td {
    width: 100px; height: 100px; text-align: center; vertical-align: middle;
    font-size: 0.72rem; border: 1px solid GRID_CLR; position: relative;
    cursor: default; overflow: hidden;
  }
  td.diag {
    background: DIAG_CLR; color: white; font-weight: 700;
    font-size: 1.0rem; line-height: 1.3;
  }
  td.feed { background: FEED_CLR; color: white; font-weight: 700; font-size: 1.4rem; }
  td.back { background: BACK_CLR; color: white; font-weight: 700; font-size: 1.4rem; }
  td.empty { background: BG_CLR; }

  /* Tooltip */
  .tooltip {
    display: none; position: fixed; background: #333; color: #fff;
    padding: 8px 12px; border-radius: 6px; font-size: 0.78rem;
    max-width: 260px; z-index: 999; pointer-events: none;
    line-height: 1.45; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .tooltip b { color: #8cf; }
</style>
</head>
<body>
<h1>N² Dependency Matrix — BLI+WT 19-Pax Hybrid-Electric</h1>
<p class="subtitle">Source: BLI_Big.py &nbsp;|&nbsp; 20 design variables &nbsp;|&nbsp; 25 constraints</p>
<div class="legend">
  <span><span class="swatch" style="background:DIAG_CLR"></span> Discipline</span>
  <span><span class="swatch" style="background:FEED_CLR"></span> Feed-forward coupling</span>
  <span><span class="swatch" style="background:BACK_CLR"></span> Feedback (iteration loop)</span>
</div>
<div class="tooltip" id="tip"></div>
<table>
""".replace("DIAG_CLR", DIAG_COLOR)
       .replace("FEED_CLR", FEED_FWD)
       .replace("BACK_CLR", FEEDBACK)
       .replace("GRID_CLR", GRID_COLOR)
       .replace("BG_CLR", BG_COLOR))

    for row in range(n):
        html_parts.append("<tr>")
        for col in range(n):
            variables = matrix[row][col]
            if row == col:
                label = diag_labels[row].replace("\n", "<br>")
                html_parts.append(
                    f'<td class="diag">{label}</td>')
            elif variables:
                cls = "feed" if col > row else "back"
                var_text = "\\n".join(variables)
                src_name = DISCIPLINES[row].replace("\\n", " ")
                tgt_name = DISCIPLINES[col].replace("\\n", " ")
                tooltip = f"{src_name} → {tgt_name}|{var_text}"
                html_parts.append(
                    f'<td class="{cls}" data-tip="{tooltip}">{len(variables)}</td>')
            else:
                html_parts.append('<td class="empty"></td>')
        html_parts.append("</tr>")

    html_parts.append(r"""</table>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('td[data-tip]').forEach(td => {
  td.addEventListener('mouseenter', e => {
    const parts = td.dataset.tip.split('|');
    const header = parts[0];
    const vars = parts[1] ? parts[1].split('\\n') : [];
    tip.innerHTML = '<b>' + header + '</b><br>' +
      vars.map(v => '• ' + v).join('<br>');
    tip.style.display = 'block';
  });
  td.addEventListener('mousemove', e => {
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top  = (e.clientY + 14) + 'px';
  });
  td.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
</script>
</body>
</html>""")

    return "\n".join(html_parts)


# ===================================================================
# 3. Write and optionally open
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate N² diagram for BLI+WT hybrid-electric aircraft")
    parser.add_argument("--open", action="store_true",
                        help="Open in browser after generating")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output HTML filename")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.resolve()
    outfile = Path(args.outfile) if args.outfile else out_dir / "n2_bli_big.html"

    html = _make_html()
    outfile.write_text(html, encoding="utf-8")
    print(f"N² diagram written to {outfile}")

    if args.open:
        webbrowser.open(str(outfile))
    else:
        print("  Tip: re-run with --open to auto-launch the browser.")


if __name__ == "__main__":
    main()
