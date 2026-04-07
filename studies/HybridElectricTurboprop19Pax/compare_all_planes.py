"""
Compare All Aircraft Configurations
====================================

Runs every aircraft sizing script in this directory as a subprocess,
parses the printed results, and produces:
  1. A side-by-side comparison table (console)
  2. Bar-chart comparisons of key metrics (matplotlib)
  3. Payload-Range Energy Efficiency (PREE) ranking

PREE = (payload_weight * range) / (fuel_energy + battery_energy)
     Units: lb·nmi / kWh   (higher is better)

Usage:
    python compare_all_planes.py
"""

import subprocess
import sys
import re
import os
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration: which scripts to run and their short labels
# ---------------------------------------------------------------------------
# (script_filename, display_label, true_n_pax)
SCRIPTS = [
    ("skycourier_validation.py",       "SkyCourier\n(Conv.)",                  19),
    ("hybrid_electric_19pax.py",       "Hybrid\nWingtip",                      19),
    ("QuarterSpan_Big.py",            "Hybrid\nConventional",                  19),
    ("BLI_Big.py",                     "BLI +\nWingtip",                       19),
    ("small_pegasus.py",               "BLI + Wingtip\n+ Midspan Props",      19),
    ("dep.py",                         "Distributed\nElectric Propulsion",     19),
]

# Constants
FUEL_SPECIFIC_ENERGY_KWH_PER_KG = 43.02e6 / 3.6e6   # ~11.95 kWh/kg (Jet-A LHV)
BATTERY_SPECIFIC_ENERGY_KWH_PER_KG = 250 / 1000      # ~0.25 kWh/kg (Li-ion cell level)
LBM = 0.453592   # kg per lb
KNOT_TO_MPS = 0.514444
NMI_TO_M = 1852.0
LBF_PER_N = 0.224809
FT_PER_M = 3.28084

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class PlaneResult:
    label: str
    script: str
    success: bool = False
    error_msg: str = ""

    # Mission constants (parsed or default)
    n_pax: int = 0
    payload_lb: float = 0.0
    range_max_nmi: float = 0.0
    range_typical_nmi: float = 0.0

    # Weights (kg)
    mtow_kg: float = 0.0
    empty_kg: float = 0.0
    fuel_kg: float = 0.0
    battery_kg: float = 0.0

    # Geometry
    wing_span_m: float = 0.0
    wing_area_m2: float = 0.0
    aspect_ratio: float = 0.0

    # Performance
    L_over_D: float = 0.0
    cruise_fuel_burn_kghr: float = 0.0
    thermal_efficiency: float = 0.0
    hybridization: float = 0.0

    # Field Performance
    v_stall_kts: float = 0.0
    v_liftoff_kts: float = 0.0
    takeoff_ground_roll_ft: float = 0.0
    takeoff_total_distance_ft: float = 0.0
    balanced_field_length_ft: float = 0.0
    landing_distance_ft: float = 0.0
    climb_gradient_aeo_pct: float = 0.0
    climb_gradient_oei_pct: float = 0.0
    thrust_to_weight_to: float = 0.0

    # Cruise Aerodynamics
    CL_cruise: float = 0.0
    CD_cruise: float = 0.0
    cruise_drag_lbf: float = 0.0
    cruise_shaft_power_hp: float = 0.0
    cruise_throttle_pct: float = 0.0

    # Propulsion
    turboshaft_power_each_hp: float = 0.0
    turboshaft_power_total_hp: float = 0.0
    propeller_diameter_m: float = 0.0
    battery_capacity_kWh: float = 0.0

    # Geometry extras
    fuselage_length_m: float = 0.0
    taper_ratio: float = 0.0
    cruise_altitude_ft: float = 0.0

    # Fuel breakdown
    fuel_climb_kg: float = 0.0
    fuel_reserves_kg: float = 0.0

    # Derived
    @property
    def mtow_lb(self): return self.mtow_kg / LBM
    @property
    def empty_lb(self): return self.empty_kg / LBM
    @property
    def fuel_lb(self): return self.fuel_kg / LBM
    @property
    def battery_lb(self): return self.battery_kg / LBM
    @property
    def payload_kg(self): return self.payload_lb * LBM
    @property
    def empty_frac(self): return self.empty_kg / self.mtow_kg if self.mtow_kg else 0
    @property
    def fuel_frac(self): return self.fuel_kg / self.mtow_kg if self.mtow_kg else 0
    @property
    def battery_frac(self): return self.battery_kg / self.mtow_kg if self.mtow_kg else 0
    @property
    def payload_frac(self): return self.payload_kg / self.mtow_kg if self.mtow_kg else 0

    @property
    def fuel_energy_kWh(self):
        """Total chemical energy in carried fuel."""
        return self.fuel_kg * FUEL_SPECIFIC_ENERGY_KWH_PER_KG

    @property
    def battery_energy_kWh(self):
        """Usable battery energy (cell level, ~80% DoD assumed in sizing)."""
        return self.battery_kg * BATTERY_SPECIFIC_ENERGY_KWH_PER_KG

    @property
    def total_energy_kWh(self):
        return self.fuel_energy_kWh + self.battery_energy_kWh

    @property
    def PREE(self):
        """Payload-Range Energy Efficiency  [lb·nmi / kWh]."""
        if self.total_energy_kWh <= 0:
            return 0.0
        return (self.payload_lb * self.range_max_nmi) / self.total_energy_kWh

    @property
    def fuel_per_pax_nmi_lb(self):
        """Fuel burn per passenger per nautical mile [lb / (pax·nmi)]."""
        if self.n_pax <= 0 or self.range_max_nmi <= 0:
            return 0.0
        return self.fuel_lb / (self.n_pax * self.range_max_nmi)

    @property
    def wing_loading_psf(self):
        """Wing loading [lb/ft^2]."""
        area_ft2 = self.wing_area_m2 * FT_PER_M**2
        if area_ft2 <= 0:
            return 0.0
        return self.mtow_lb / area_ft2

    # --- NEW derived properties ---
    @property
    def co2_kg(self):
        """CO2 emissions for max-range mission [kg]. Jet-A: 3.16 kg CO2 / kg fuel."""
        return self.fuel_kg * 3.16

    @property
    def specific_range_nmi_per_lb(self):
        """Specific Range [nmi / lb fuel]."""
        if self.fuel_lb <= 0:
            return 0.0
        return self.range_max_nmi / self.fuel_lb

    @property
    def energy_per_seat_nmi(self):
        """Energy intensity [kWh / (seat · nmi)]."""
        if self.n_pax <= 0 or self.range_max_nmi <= 0:
            return 0.0
        return self.total_energy_kWh / (self.n_pax * self.range_max_nmi)

    @property
    def power_loading_lb_per_hp(self):
        """Power loading [lb / hp]."""
        if self.turboshaft_power_total_hp <= 0:
            return 0.0
        return self.mtow_lb / self.turboshaft_power_total_hp

    @property
    def co2_per_pax_nmi(self):
        """CO2 per passenger per nmi [kg / (pax·nmi)]."""
        if self.n_pax <= 0 or self.range_max_nmi <= 0:
            return 0.0
        return self.co2_kg / (self.n_pax * self.range_max_nmi)


# ---------------------------------------------------------------------------
# Regex parsers for the printed output of each script
# ---------------------------------------------------------------------------
def _float(text: str) -> float:
    """Extract a float from a string, stripping commas."""
    return float(text.replace(",", ""))


def parse_output(raw: str, label: str, script: str) -> PlaneResult:
    """Parse the stdout of an aircraft sizing script."""
    r = PlaneResult(label=label, script=script, success=True)

    # --- MTOW ---
    m = re.search(r"MTOW:\s+([\d.]+)\s+kg", raw)
    if m:
        r.mtow_kg = _float(m.group(1))

    # --- Empty Weight ---
    m = re.search(r"Empty Weight:\s+([\d.]+)\s+kg", raw)
    if m:
        r.empty_kg = _float(m.group(1))

    # --- Fuel Weight ---
    m = re.search(r"Fuel Weight:\s+([\d.]+)\s+kg", raw)
    if m:
        r.fuel_kg = _float(m.group(1))

    # --- Battery Weight ---
    m = re.search(r"Battery Weight:\s+([\d.]+)\s+kg", raw)
    if m:
        r.battery_kg = _float(m.group(1))

    # --- Wing Span ---
    m = re.search(r"Wing Span:\s+([\d.]+)\s+m", raw)
    if m:
        r.wing_span_m = _float(m.group(1))

    # --- Wing Area ---
    m = re.search(r"Wing Area:\s+([\d.]+)\s+m\^2", raw)
    if m:
        r.wing_area_m2 = _float(m.group(1))

    # --- Aspect Ratio ---
    m = re.search(r"Aspect Ratio:\s+([\d.]+)", raw)
    if m:
        r.aspect_ratio = _float(m.group(1))

    # --- L/D ---
    m = re.search(r"L/D:\s+([\d.]+)", raw)
    if m:
        r.L_over_D = _float(m.group(1))

    # --- Cruise Fuel Burn ---
    m = re.search(r"Cruise Fuel Burn:\s+([\d.]+)\s+kg/hr", raw)
    if m:
        r.cruise_fuel_burn_kghr = _float(m.group(1))

    # --- Thermal Efficiency ---
    m = re.search(r"Thermal Efficiency:\s+([\d.]+)%", raw)
    if m:
        r.thermal_efficiency = _float(m.group(1)) / 100.0

    # --- Hybridization ---
    m = re.search(r"Hybridization Factor:\s+([\d.]+)%", raw)
    if m:
        r.hybridization = _float(m.group(1)) / 100.0

    # --- Total Fuel (max range) -- fallback for fuel_kg ---
    if r.fuel_kg == 0:
        m = re.search(r"Total Fuel \(max range\):\s+([\d.]+)\s+kg", raw)
        if m:
            r.fuel_kg = _float(m.group(1))

    # --- Fallback: parse MTOW from weight breakdown line ---
    if r.mtow_kg == 0:
        m = re.search(r"MTOW\s+([\d.]+)\s+([\d.]+)", raw)
        if m:
            r.mtow_kg = _float(m.group(1))

    # --- Fallback: EMPTY WEIGHT from breakdown ---
    if r.empty_kg == 0:
        m = re.search(r"EMPTY WEIGHT\s+([\d.]+)\s+([\d.]+)", raw)
        if m:
            r.empty_kg = _float(m.group(1))

    # --- Fallback: Fuel from breakdown ---
    if r.fuel_kg == 0:
        m = re.search(r"^\s+Fuel\s+([\d.]+)\s+([\d.]+)", raw, re.MULTILINE)
        if m:
            r.fuel_kg = _float(m.group(1))

    # --- Fallback: Battery from breakdown ---
    if r.battery_kg == 0:
        m = re.search(r"^\s+Battery\s+([\d.]+)\s+([\d.]+)", raw, re.MULTILINE)
        if m:
            r.battery_kg = _float(m.group(1))

    # --- Payload from "Payload:" in Overall section ---
    m = re.search(r"Payload:\s+([\d.]+)\s+kg\s+\(\s*([\d.]+)\s+lb\)", raw)
    if m:
        r.payload_lb = _float(m.group(2))

    # fallback payload from breakdown
    if r.payload_lb == 0:
        m = re.search(r"^\s+Payload\s+([\d.]+)\s+([\d.]+)", raw, re.MULTILINE)
        if m:
            r.payload_lb = _float(m.group(2))  # lb column

    # --- Max Range from "Max Range Mission:" ---
    m = re.search(r"Max Range Mission:\s+(\d+)\s+nmi", raw)
    if m:
        r.range_max_nmi = float(m.group(1))

    # --- Typical Range ---
    m = re.search(r"Typical Mission:\s+(\d+)\s+nmi", raw)
    if m:
        r.range_typical_nmi = float(m.group(1))

    # ===================================================================
    # NEW PARSERS — Field Performance
    # ===================================================================
    # V_stall (SL):  handles both "V_stall (SL):" and "V_stall (SL, clean):" and "V_stall (SL, blown):"
    m = re.search(r"V_stall \(SL(?:, \w+)?\):\s+([\d.]+)\s+kts", raw)
    if m:
        r.v_stall_kts = _float(m.group(1))

    m = re.search(r"V_liftoff:\s+([\d.]+)\s+kts", raw)
    if m:
        r.v_liftoff_kts = _float(m.group(1))

    m = re.search(r"Takeoff Ground Roll:\s+([\d.]+)\s+ft", raw)
    if m:
        r.takeoff_ground_roll_ft = _float(m.group(1))

    m = re.search(r"Takeoff Total Distance:\s+([\d.]+)\s+ft", raw)
    if m:
        r.takeoff_total_distance_ft = _float(m.group(1))

    m = re.search(r"Balanced Field Length:\s+([\d.]+)\s+ft", raw)
    if m:
        r.balanced_field_length_ft = _float(m.group(1))

    m = re.search(r"Landing Total Distance:\s+([\d.]+)\s+ft", raw)
    if m:
        r.landing_distance_ft = _float(m.group(1))

    # Climb Gradient (AEO):  …  (xx.xx%)
    m = re.search(r"Climb Gradient \(AEO\):\s+[\d.]+\s+rad\s+\(\s*([\d.]+)%\)", raw)
    if m:
        r.climb_gradient_aeo_pct = _float(m.group(1))

    m = re.search(r"Climb Gradient \(OEI\):\s+[\d.]+\s+rad\s+\(\s*([\d.]+)%\)", raw)
    if m:
        r.climb_gradient_oei_pct = _float(m.group(1))

    m = re.search(r"Thrust/Weight \(TO\):\s+([\d.]+)", raw)
    if m:
        r.thrust_to_weight_to = _float(m.group(1))

    # ===================================================================
    # NEW PARSERS — Cruise Aerodynamics
    # ===================================================================
    m = re.search(r"CL:\s+([\d.]+)", raw)
    if m:
        r.CL_cruise = _float(m.group(1))

    m = re.search(r"CD:\s+([\d.]+)", raw)
    if m:
        r.CD_cruise = _float(m.group(1))

    # Cruise Drag:  xxxx N   ( xxxx lbf)
    m = re.search(r"Cruise Drag:\s+[\d.]+\s+N\s+\(\s*([\d.]+)\s+lbf\)", raw)
    if m:
        r.cruise_drag_lbf = _float(m.group(1))

    # "Cruise Shaft Power:" or "DEP Cruise Shaft Power:" — grab the first hp number
    m = re.search(r"(?:DEP )?Cruise Shaft Power:\s+([\d.]+)\s+hp", raw)
    if m:
        r.cruise_shaft_power_hp = _float(m.group(1))

    # Cruise Throttle:  xx.x%  (Python :8.1% format → "  81.2%")
    m = re.search(r"Cruise Throttle:\s+([\d.]+)%", raw)
    if m:
        r.cruise_throttle_pct = _float(m.group(1))

    # ===================================================================
    # NEW PARSERS — Propulsion
    # ===================================================================
    # "Turboshaft Power (each):" or "Core Shaft Power (each):"
    m = re.search(r"(?:Turboshaft|Core Shaft) Power \(each\):\s+([\d.]+)\s+hp", raw)
    if m:
        r.turboshaft_power_each_hp = _float(m.group(1))

    # "Turboshaft Power (total):" — some scripts print this
    m = re.search(r"Turboshaft Power \(total\):\s*([\d.]+)\s+hp", raw)
    if m:
        r.turboshaft_power_total_hp = _float(m.group(1))
    elif r.turboshaft_power_each_hp > 0:
        # Derive total from each × 2 (most scripts have 2 engines)
        m2 = re.search(r"Number of Engines:\s+(\d+)", raw)
        n_eng = int(m2.group(1)) if m2 else 2
        r.turboshaft_power_total_hp = r.turboshaft_power_each_hp * n_eng

    m = re.search(r"Propeller Diameter:\s+([\d.]+)\s+m", raw)
    if m:
        r.propeller_diameter_m = _float(m.group(1))
    else:
        # DEP variant
        m = re.search(r"DEP Propeller Diameter:\s+([\d.]+)\s+m", raw)
        if m:
            r.propeller_diameter_m = _float(m.group(1))

    m = re.search(r"Battery Capacity:\s+([\d.]+)\s+kWh", raw)
    if m:
        r.battery_capacity_kWh = _float(m.group(1))

    # ===================================================================
    # NEW PARSERS — Geometry extras
    # ===================================================================
    m = re.search(r"Fuselage Length:\s+([\d.]+)\s+m", raw)
    if m:
        r.fuselage_length_m = _float(m.group(1))

    m = re.search(r"Taper Ratio:\s+([\d.]+)", raw)
    if m:
        r.taper_ratio = _float(m.group(1))

    m = re.search(r"Cruise Altitude:\s+([\d.]+)\s+ft", raw)
    if m:
        r.cruise_altitude_ft = _float(m.group(1))

    # ===================================================================
    # NEW PARSERS — Fuel Breakdown
    # ===================================================================
    m = re.search(r"Fuel for Climb:\s+([\d.]+)\s+kg", raw)
    if m:
        r.fuel_climb_kg = _float(m.group(1))

    m = re.search(r"Fuel Reserves \(45 min\):\s+([\d.]+)\s+kg", raw)
    if m:
        r.fuel_reserves_kg = _float(m.group(1))

    return r


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_script(script_name: str, label: str, n_pax: int, work_dir: Path) -> PlaneResult:
    """Run a sizing script and return parsed results."""
    script_path = work_dir / script_name
    if not script_path.exists():
        r = PlaneResult(label=label, script=script_name)
        r.error_msg = f"File not found: {script_path}"
        return r

    print(f"\n{'='*60}")
    print(f"  Running: {script_name}  ({label.replace(chr(10), ' ')})")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per script
            cwd=str(work_dir),
            env={**os.environ, "MPLBACKEND": "Agg"},  # Prevent plot windows
        )
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            r = PlaneResult(label=label, script=script_name)
            r.error_msg = f"Exit code {result.returncode}"
            # Print last 20 lines of stderr for debugging
            err_lines = stderr.strip().splitlines()
            for line in err_lines[-20:]:
                print(f"  STDERR: {line}")
            return r

        # Parse
        parsed = parse_output(stdout, label, script_name)
        parsed.n_pax = n_pax  # Use authoritative value from config
        if parsed.mtow_kg == 0:
            parsed.success = False
            parsed.error_msg = "Could not parse MTOW from output"
            print("  WARNING: Could not parse results from output.")
            # Print last 30 lines of stdout for debugging
            for line in stdout.strip().splitlines()[-30:]:
                print(f"  > {line}")
        else:
            print(f"  OK: MTOW = {parsed.mtow_lb:,.0f} lb, "
                  f"Fuel = {parsed.fuel_lb:,.0f} lb, "
                  f"PREE = {parsed.PREE:.2f} lb·nmi/kWh")

        return parsed

    except subprocess.TimeoutExpired:
        r = PlaneResult(label=label, script=script_name)
        r.error_msg = "Timed out (>600s)"
        print("  TIMEOUT")
        return r
    except Exception as e:
        r = PlaneResult(label=label, script=script_name)
        r.error_msg = str(e)
        print(f"  ERROR: {e}")
        return r


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(results: list[PlaneResult]):
    """Print a nicely formatted comparison table to console."""
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    if not ok:
        print("\nNo successful runs to compare.")
        return

    # Column width
    cw = 14

    # Header
    labels_line1 = [r.label.split("\n")[0] if "\n" in r.label else r.label for r in ok]
    labels_line2 = [r.label.split("\n")[1] if "\n" in r.label else "" for r in ok]

    print("\n" + "=" * (30 + cw * len(ok)))
    print("   AIRCRAFT CONFIGURATION COMPARISON")
    print("=" * (30 + cw * len(ok)))

    def row(name, values, fmt="{:>13.0f}"):
        vals = "".join(fmt.format(v) for v in values)
        print(f"  {name:<28}{vals}")

    def row_str(name, values):
        vals = "".join(f"{v:>13}" for v in values)
        print(f"  {name:<28}{vals}")

    # Label rows
    row_str("", labels_line1)
    row_str("", labels_line2)
    print(f"  {'-'*28}" + "-" * (cw * len(ok)))

    # Mission
    print(f"\n  {'--- Mission ---'}")
    row_str("Passengers", [str(r.n_pax) for r in ok])
    row("Payload (lb)", [r.payload_lb for r in ok])
    row("Max Range (nmi)", [r.range_max_nmi for r in ok])
    row("Typical Range (nmi)", [r.range_typical_nmi for r in ok])

    # Weights
    print(f"\n  {'--- Weights ---'}")
    row("MTOW (lb)", [r.mtow_lb for r in ok])
    row("Empty Wt (lb)", [r.empty_lb for r in ok])
    row("Fuel (lb)", [r.fuel_lb for r in ok])
    row("Battery (lb)", [r.battery_lb for r in ok])

    # Weight fractions
    print(f"\n  {'--- Weight Fractions ---'}")
    row("Empty / MTOW", [r.empty_frac for r in ok], fmt="{:>13.3f}")
    row("Fuel / MTOW", [r.fuel_frac for r in ok], fmt="{:>13.3f}")
    row("Battery / MTOW", [r.battery_frac for r in ok], fmt="{:>13.3f}")
    row("Payload / MTOW", [r.payload_frac for r in ok], fmt="{:>13.3f}")

    # Geometry
    print(f"\n  {'--- Geometry ---'}")
    row("Wing Span (m)", [r.wing_span_m for r in ok], fmt="{:>13.1f}")
    row("Wing Area (m^2)", [r.wing_area_m2 for r in ok], fmt="{:>13.1f}")
    row("Aspect Ratio", [r.aspect_ratio for r in ok], fmt="{:>13.2f}")
    row("Wing Loading (psf)", [r.wing_loading_psf for r in ok], fmt="{:>13.1f}")
    row("Taper Ratio", [r.taper_ratio for r in ok], fmt="{:>13.2f}")
    row("Fuselage Length (m)", [r.fuselage_length_m for r in ok], fmt="{:>13.1f}")
    row("Cruise Altitude (ft)", [r.cruise_altitude_ft for r in ok], fmt="{:>13.0f}")

    # Performance
    print(f"\n  {'--- Performance ---'}")
    row("L/D (cruise)", [r.L_over_D for r in ok], fmt="{:>13.1f}")
    row("Thermal Eff.", [r.thermal_efficiency * 100 for r in ok], fmt="{:>12.1f}%")
    row("Hybridization", [r.hybridization * 100 for r in ok], fmt="{:>12.1f}%")
    row("Fuel Burn (kg/hr)", [r.cruise_fuel_burn_kghr for r in ok], fmt="{:>13.1f}")

    # Cruise Aerodynamics
    print(f"\n  {'--- Cruise Aerodynamics ---'}")
    row("CL (cruise)", [r.CL_cruise for r in ok], fmt="{:>13.4f}")
    row("CD (cruise)", [r.CD_cruise for r in ok], fmt="{:>13.5f}")
    row("Cruise Drag (lbf)", [r.cruise_drag_lbf for r in ok], fmt="{:>13.0f}")
    row("Cruise Shaft Power (hp)", [r.cruise_shaft_power_hp for r in ok], fmt="{:>13.0f}")
    row("Cruise Throttle", [r.cruise_throttle_pct for r in ok], fmt="{:>12.1f}%")

    # Propulsion
    print(f"\n  {'--- Propulsion ---'}")
    row("Engine Power ea (hp)", [r.turboshaft_power_each_hp for r in ok], fmt="{:>13.0f}")
    row("Engine Power tot (hp)", [r.turboshaft_power_total_hp for r in ok], fmt="{:>13.0f}")
    row("Propeller Diam (m)", [r.propeller_diameter_m for r in ok], fmt="{:>13.2f}")
    row("Battery Cap (kWh)", [r.battery_capacity_kWh for r in ok], fmt="{:>13.1f}")
    row("Power Loading (lb/hp)", [r.power_loading_lb_per_hp for r in ok], fmt="{:>13.2f}")

    # Field Performance
    print(f"\n  {'--- Field Performance ---'}")
    row("V_stall (kts)", [r.v_stall_kts for r in ok], fmt="{:>13.1f}")
    row("V_liftoff (kts)", [r.v_liftoff_kts for r in ok], fmt="{:>13.1f}")
    row("TO Ground Roll (ft)", [r.takeoff_ground_roll_ft for r in ok], fmt="{:>13.0f}")
    row("TO Total Dist (ft)", [r.takeoff_total_distance_ft for r in ok], fmt="{:>13.0f}")
    row("BFL (ft)", [r.balanced_field_length_ft for r in ok], fmt="{:>13.0f}")
    row("Landing Dist (ft)", [r.landing_distance_ft for r in ok], fmt="{:>13.0f}")
    row("Climb Grad AEO", [r.climb_gradient_aeo_pct for r in ok], fmt="{:>12.2f}%")
    row("Climb Grad OEI", [r.climb_gradient_oei_pct for r in ok], fmt="{:>12.2f}%")
    row("T/W (TO)", [r.thrust_to_weight_to for r in ok], fmt="{:>13.3f}")

    # Fuel Breakdown
    print(f"\n  {'--- Fuel Breakdown ---'}")
    row("Fuel for Climb (kg)", [r.fuel_climb_kg for r in ok], fmt="{:>13.0f}")
    row("Fuel Reserves 45min (kg)", [r.fuel_reserves_kg for r in ok], fmt="{:>13.0f}")

    # Energy efficiency
    print(f"\n  {'--- Energy Efficiency ---'}")
    row("Fuel Energy (kWh)", [r.fuel_energy_kWh for r in ok], fmt="{:>13.0f}")
    row("Battery Energy (kWh)", [r.battery_energy_kWh for r in ok], fmt="{:>13.1f}")
    row("Total Energy (kWh)", [r.total_energy_kWh for r in ok], fmt="{:>13.0f}")
    row("PREE (lb·nmi/kWh)", [r.PREE for r in ok], fmt="{:>13.2f}")
    row("Fuel/pax/nmi (lb)", [r.fuel_per_pax_nmi_lb for r in ok], fmt="{:>13.4f}")

    # Derived Efficiency / Emissions
    print(f"\n  {'--- Emissions & Derived Efficiency ---'}")
    row("CO2 Emissions (kg)", [r.co2_kg for r in ok], fmt="{:>13.0f}")
    row("CO2/pax/nmi (kg)", [r.co2_per_pax_nmi for r in ok], fmt="{:>13.4f}")
    row("Specific Range (nmi/lb)", [r.specific_range_nmi_per_lb for r in ok], fmt="{:>13.3f}")
    row("Energy/seat·nmi (kWh)", [r.energy_per_seat_nmi for r in ok], fmt="{:>13.3f}")

    print(f"\n  {'-'*28}" + "-" * (cw * len(ok)))
    print("  PREE = Payload-Range Energy Efficiency = (payload_wt × range) / total_energy")
    print("         Higher is better.")
    print("  Fuel/pax/nmi = total fuel per passenger per nmi at max range.")
    print("         Lower is better.")

    # Rank by PREE
    ranked = sorted(ok, key=lambda r: r.PREE, reverse=True)
    print(f"\n  {'--- PREE Ranking (best → worst) ---'}")
    for i, r in enumerate(ranked, 1):
        lbl = r.label.replace("\n", " ")
        print(f"    {i}. {lbl:<25} PREE = {r.PREE:.2f} lb·nmi/kWh")

    # Rank by fuel per pax-nmi
    ranked_fuel = sorted(ok, key=lambda r: r.fuel_per_pax_nmi_lb)
    print(f"\n  {'--- Fuel Efficiency Ranking (best → worst) ---'}")
    for i, r in enumerate(ranked_fuel, 1):
        lbl = r.label.replace("\n", " ")
        print(f"    {i}. {lbl:<25} {r.fuel_per_pax_nmi_lb:.4f} lb/(pax·nmi)")

    # Rank by CO2/pax/nmi
    ranked_co2 = sorted(ok, key=lambda r: r.co2_per_pax_nmi)
    print(f"\n  {'--- CO2 per Pax-NMI Ranking (best → worst) ---'}")
    for i, r in enumerate(ranked_co2, 1):
        lbl = r.label.replace("\n", " ")
        print(f"    {i}. {lbl:<25} {r.co2_per_pax_nmi:.4f} kg/(pax·nmi)")

    # Rank by BFL (shortest first)
    ranked_bfl = sorted([r for r in ok if r.balanced_field_length_ft > 0],
                        key=lambda r: r.balanced_field_length_ft)
    if ranked_bfl:
        print(f"\n  {'--- Balanced Field Length Ranking (shortest → longest) ---'}")
        for i, r in enumerate(ranked_bfl, 1):
            lbl = r.label.replace("\n", " ")
            print(f"    {i}. {lbl:<25} {r.balanced_field_length_ft:,.0f} ft")

    # Rank by Energy per seat-nmi (lowest first)
    ranked_esn = sorted(ok, key=lambda r: r.energy_per_seat_nmi)
    print(f"\n  {'--- Energy Intensity Ranking (best → worst) ---'}")
    for i, r in enumerate(ranked_esn, 1):
        lbl = r.label.replace("\n", " ")
        print(f"    {i}. {lbl:<25} {r.energy_per_seat_nmi:.3f} kWh/(seat·nmi)")

    if fail:
        print(f"\n  {'--- Failed Runs ---'}")
        for r in fail:
            lbl = r.label.replace("\n", " ")
            print(f"    {lbl}: {r.error_msg}")

    print("\n" + "=" * (30 + cw * len(ok)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparisons(results: list[PlaneResult]):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    ok = [r for r in results if r.success]
    if len(ok) < 2:
        print("  Need at least 2 successful runs to plot.")
        return

    labels = [r.label.replace("\n", "\n") for r in ok]
    x = range(len(ok))

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Aircraft Configuration Comparison", fontsize=24, fontweight="bold")

    colors_mtow = "#2196F3"
    colors_fuel = "#FF9800"
    colors_batt = "#4CAF50"
    colors_pree = "#9C27B0"
    colors_ld   = "#00BCD4"

    # --- 1. Weight Breakdown (stacked) ---
    ax = axes[0, 0]
    empty = [r.empty_lb for r in ok]
    fuel = [r.fuel_lb for r in ok]
    batt = [r.battery_lb for r in ok]
    payload = [r.payload_lb for r in ok]

    ax.bar(x, empty, label="Empty", color="#78909C")
    ax.bar(x, fuel, bottom=empty, label="Fuel", color=colors_fuel)
    ax.bar(x, batt, bottom=[e + f for e, f in zip(empty, fuel)], label="Battery", color=colors_batt)
    ax.bar(x, payload, bottom=[e + f + b for e, f, b in zip(empty, fuel, batt)], label="Payload", color="#E91E63")
    ax.set_ylabel("Weight (lb)")
    ax.set_title("Weight Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    ax.legend(fontsize=20)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- 2. Fuel Weight Comparison ---
    ax = axes[0, 1]
    bars = ax.bar(x, [r.fuel_lb for r in ok], color=colors_fuel, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fuel Weight (lb)")
    ax.set_title("Total Fuel (Max Range)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    for bar, r in zip(bars, ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{r.fuel_lb:,.0f}", ha="center", va="bottom", fontsize=20)

    # --- 3. PREE ---
    ax = axes[0, 2]
    bars = ax.bar(x, [r.PREE for r in ok], color=colors_pree, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("PREE (lb·nmi / kWh)")
    ax.set_title("Payload-Range Energy Efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    for bar, r in zip(bars, ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{r.PREE:.2f}", ha="center", va="bottom", fontsize=20)

    # --- 4. Weight Fractions ---
    ax = axes[1, 0]
    w = 0.2
    x_arr = list(x)
    ax.bar([xi - 1.5*w for xi in x_arr], [r.empty_frac for r in ok], w, label="Empty", color="#78909C")
    ax.bar([xi - 0.5*w for xi in x_arr], [r.fuel_frac for r in ok], w, label="Fuel", color=colors_fuel)
    ax.bar([xi + 0.5*w for xi in x_arr], [r.battery_frac for r in ok], w, label="Battery", color=colors_batt)
    ax.bar([xi + 1.5*w for xi in x_arr], [r.payload_frac for r in ok], w, label="Payload", color="#E91E63")
    ax.set_ylabel("Fraction of MTOW")
    ax.set_title("Weight Fractions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    ax.legend(fontsize=20)

    # --- 5. L/D and Fuel Burn ---
    ax = axes[1, 1]
    ax2 = ax.twinx()
    bars1 = ax.bar([xi - 0.2 for xi in x_arr], [r.L_over_D for r in ok], 0.4,
                   label="L/D", color=colors_ld, alpha=0.8)
    bars2 = ax2.bar([xi + 0.2 for xi in x_arr], [r.cruise_fuel_burn_kghr for r in ok], 0.4,
                    label="Fuel Burn", color=colors_fuel, alpha=0.8)
    ax.set_ylabel("L/D (cruise)", color=colors_ld)
    ax2.set_ylabel("Fuel Burn (kg/hr)", color=colors_fuel)
    ax.set_title("Cruise L/D & Fuel Burn")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=20, loc="upper left")

    # --- 6. Fuel per pax-nmi ---
    ax = axes[1, 2]
    bars = ax.bar(x, [r.fuel_per_pax_nmi_lb for r in ok], color="#F44336", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("lb / (pax · nmi)")
    ax.set_title("Fuel per Passenger per NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    for bar, r in zip(bars, ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f"{r.fuel_per_pax_nmi_lb:.4f}", ha="center", va="bottom", fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(__file__).parent / "comparison_results.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n  Saved comparison chart (page 1) to: {out_path}")
    plt.close(fig)

    # ==================================================================
    # FIGURE 2 — Field Performance, Propulsion, CO2 & Efficiency
    # ==================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(24, 14))
    fig2.suptitle("Aircraft Comparison — Field Performance, Propulsion & Emissions",
                  fontsize=24, fontweight="bold")

    colors_bfl  = "#795548"
    colors_co2  = "#E53935"
    colors_pwr  = "#1565C0"
    colors_esn  = "#00897B"
    colors_clmb = "#43A047"

    # --- 7. Balanced Field Length ---
    ax = axes2[0, 0]
    bfl_vals = [r.balanced_field_length_ft for r in ok]
    bars = ax.bar(x, bfl_vals, color=colors_bfl, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("BFL (ft)")
    ax.set_title("Balanced Field Length")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    for bar, v in zip(bars, bfl_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=20)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- 8. CO2 per pax-nmi ---
    ax = axes2[0, 1]
    co2_vals = [r.co2_per_pax_nmi for r in ok]
    bars = ax.bar(x, co2_vals, color=colors_co2, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("kg CO₂ / (pax · nmi)")
    ax.set_title("CO₂ per Passenger per NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    for bar, v in zip(bars, co2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=20)

    # --- 9. Power Loading & Cruise Shaft Power ---
    ax = axes2[0, 2]
    ax2b = ax.twinx()
    bars1 = ax.bar([xi - 0.2 for xi in x_arr], [r.power_loading_lb_per_hp for r in ok], 0.4,
                   label="Power Loading", color=colors_pwr, alpha=0.8)
    bars2 = ax2b.bar([xi + 0.2 for xi in x_arr], [r.cruise_shaft_power_hp for r in ok], 0.4,
                     label="Cruise Power", color=colors_fuel, alpha=0.8)
    ax.set_ylabel("Power Loading (lb/hp)", color=colors_pwr)
    ax2b.set_ylabel("Cruise Shaft Power (hp)", color=colors_fuel)
    ax.set_title("Power Loading & Cruise Power")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2b.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=20, loc="upper left")

    # --- 10. Climb Gradients (AEO vs OEI) ---
    ax = axes2[1, 0]
    w = 0.35
    ax.bar([xi - w/2 for xi in x_arr], [r.climb_gradient_aeo_pct for r in ok], w,
           label="AEO", color=colors_clmb, alpha=0.9)
    ax.bar([xi + w/2 for xi in x_arr], [r.climb_gradient_oei_pct for r in ok], w,
           label="OEI", color="#FFA726", alpha=0.9)
    ax.set_ylabel("Climb Gradient (%)")
    ax.set_title("Climb Gradients (AEO vs OEI)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    ax.legend(fontsize=20)

    # --- 11. Energy per seat-nmi ---
    ax = axes2[1, 1]
    esn_vals = [r.energy_per_seat_nmi for r in ok]
    bars = ax.bar(x, esn_vals, color=colors_esn, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("kWh / (seat · nmi)")
    ax.set_title("Energy Intensity per Seat-NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    for bar, v in zip(bars, esn_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=20)

    # --- 12. T/W and V_stall ---
    ax = axes2[1, 2]
    ax2c = ax.twinx()
    bars1 = ax.bar([xi - 0.2 for xi in x_arr], [r.thrust_to_weight_to for r in ok], 0.4,
                   label="T/W (TO)", color="#7B1FA2", alpha=0.8)
    bars2 = ax2c.bar([xi + 0.2 for xi in x_arr], [r.v_stall_kts for r in ok], 0.4,
                     label="V_stall", color="#0288D1", alpha=0.8)
    ax.set_ylabel("T/W at Takeoff", color="#7B1FA2")
    ax2c.set_ylabel("V_stall (kts)", color="#0288D1")
    ax.set_title("T/W (Takeoff) & Stall Speed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, ha="center")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2c.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=20, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path2 = Path(__file__).parent / "comparison_results_page2.png"
    fig2.savefig(str(out_path2), dpi=150, bbox_inches="tight")
    print(f"  Saved comparison chart (page 2) to: {out_path2}")
    plt.close(fig2)

    # ==================================================================
    # FIGURE 3 — Weight Breakdown Pie Charts for each aircraft
    # ==================================================================
    n_planes = len(ok)
    ncols = min(n_planes, 3)
    nrows = (n_planes + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
    fig3.suptitle("Weight Breakdown — Pie Charts (19 Pax Configurations)",
                  fontsize=24, fontweight="bold")

    # Flatten axes array for easy iteration
    if n_planes == 1:
        ax_list = [axes3]
    else:
        ax_list = axes3.flatten() if hasattr(axes3, 'flatten') else [axes3]

    pie_colors = ["#78909C", "#FF9800", "#4CAF50", "#E91E63"]
    pie_labels_names = ["Empty", "Fuel", "Battery", "Payload"]

    for i, r in enumerate(ok):
        ax = ax_list[i]
        sizes = [r.empty_lb, r.fuel_lb, r.battery_lb, r.payload_lb]
        # Filter out zero-weight categories
        filtered_sizes = []
        filtered_labels = []
        filtered_colors = []
        for s, lbl, c in zip(sizes, pie_labels_names, pie_colors):
            if s > 0:
                filtered_sizes.append(s)
                filtered_labels.append(f"{lbl}\n{s:,.0f} lb")
                filtered_colors.append(c)

        wedges, texts, autotexts = ax.pie(
            filtered_sizes, labels=filtered_labels, colors=filtered_colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.75,
            textprops={"fontsize": 20}
        )
        for at in autotexts:
            at.set_fontsize(20)
        plane_label = r.label.replace("\n", " ")
        ax.set_title(f"{plane_label}\nMTOW = {r.mtow_lb:,.0f} lb", fontsize=22, fontweight="bold")

    # Hide unused subplots
    for j in range(n_planes, len(ax_list)):
        ax_list[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path3 = Path(__file__).parent / "comparison_weight_pies.png"
    fig3.savefig(str(out_path3), dpi=150, bbox_inches="tight")
    print(f"  Saved weight breakdown pie charts to: {out_path3}")
    plt.close(fig3)

    # ==================================================================
    # FIGURE 4 — Fuel Consumption Summary
    # ==================================================================
    fig4, axes4 = plt.subplots(1, 3, figsize=(24, 8))
    fig4.suptitle("Fuel Consumption — 19 Pax Configurations",
                  fontsize=14, fontweight="bold")

    # --- Total Fuel Weight ---
    ax = axes4[0]
    bars = ax.bar(x, [r.fuel_lb for r in ok], color="#FF9800", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fuel Weight (lb)")
    ax.set_title("Total Fuel (Max Range Mission)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, ha="center")
    for bar, r in zip(bars, ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{r.fuel_lb:,.0f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- Cruise Fuel Burn Rate ---
    ax = axes4[1]
    bars = ax.bar(x, [r.cruise_fuel_burn_kghr for r in ok], color="#F44336", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fuel Burn (kg/hr)")
    ax.set_title("Cruise Fuel Burn Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, ha="center")
    for bar, r in zip(bars, ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{r.cruise_fuel_burn_kghr:.1f}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    # --- Fuel per pax-nmi ---
    ax = axes4[2]
    bars = ax.bar(x, [r.fuel_per_pax_nmi_lb for r in ok], color="#E91E63", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("lb / (pax · nmi)")
    ax.set_title("Fuel per Passenger per NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, ha="center")
    for bar, r in zip(bars, ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f"{r.fuel_per_pax_nmi_lb:.4f}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path4 = Path(__file__).parent / "comparison_fuel_consumption.png"
    fig4.savefig(str(out_path4), dpi=150, bbox_inches="tight")
    print(f"  Saved fuel consumption chart to: {out_path4}")
    plt.close(fig4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    work_dir = Path(__file__).parent.resolve()
    print("=" * 60)
    print("   AIRCRAFT CONFIGURATION COMPARISON RUNNER")
    print(f"   Working directory: {work_dir}")
    print("=" * 60)

    results = []
    for script_name, label, n_pax in SCRIPTS:
        r = run_script(script_name, label, n_pax, work_dir)
        results.append(r)

    # Print comparison table
    print_comparison_table(results)

    # Generate plots
    plot_comparisons(results)

    # Save CSV for further analysis
    csv_path = work_dir / "comparison_results.csv"
    with open(csv_path, "w") as f:
        headers = [
            "Configuration", "Script", "Passengers", "Payload (lb)",
            "Max Range (nmi)", "MTOW (lb)", "Empty Wt (lb)", "Fuel (lb)",
            "Battery (lb)", "EW Frac", "Fuel Frac", "Batt Frac",
            "Payload Frac", "Wing Span (m)", "Wing Area (m2)", "AR",
            "Wing Loading (psf)", "L/D", "Thermal Eff", "Hybridization",
            "Fuel Burn (kg/hr)", "Fuel Energy (kWh)", "Battery Energy (kWh)",
            "PREE (lb·nmi/kWh)", "Fuel/pax/nmi (lb)",
            # --- NEW columns ---
            "Taper Ratio", "Fuselage Length (m)", "Cruise Alt (ft)",
            "CL cruise", "CD cruise", "Cruise Drag (lbf)",
            "Cruise Shaft Power (hp)", "Cruise Throttle (%)",
            "Engine Power ea (hp)", "Engine Power tot (hp)",
            "Propeller Diam (m)", "Battery Cap (kWh)",
            "Power Loading (lb/hp)",
            "V_stall (kts)", "V_liftoff (kts)",
            "TO Ground Roll (ft)", "TO Total Dist (ft)",
            "BFL (ft)", "Landing Dist (ft)",
            "Climb Grad AEO (%)", "Climb Grad OEI (%)",
            "T/W (TO)",
            "Fuel Climb (kg)", "Fuel Reserves (kg)",
            "CO2 (kg)", "CO2/pax/nmi (kg)",
            "Specific Range (nmi/lb)", "Energy/seat·nmi (kWh)",
        ]
        f.write(",".join(headers) + "\n")
        for r in results:
            if not r.success:
                continue
            lbl = r.label.replace("\n", " ")
            vals = [
                lbl, r.script, r.n_pax, f"{r.payload_lb:.0f}",
                f"{r.range_max_nmi:.0f}", f"{r.mtow_lb:.0f}",
                f"{r.empty_lb:.0f}", f"{r.fuel_lb:.0f}",
                f"{r.battery_lb:.0f}", f"{r.empty_frac:.4f}",
                f"{r.fuel_frac:.4f}", f"{r.battery_frac:.4f}",
                f"{r.payload_frac:.4f}", f"{r.wing_span_m:.2f}",
                f"{r.wing_area_m2:.1f}", f"{r.aspect_ratio:.2f}",
                f"{r.wing_loading_psf:.1f}", f"{r.L_over_D:.1f}",
                f"{r.thermal_efficiency:.4f}", f"{r.hybridization:.4f}",
                f"{r.cruise_fuel_burn_kghr:.1f}", f"{r.fuel_energy_kWh:.0f}",
                f"{r.battery_energy_kWh:.1f}", f"{r.PREE:.3f}",
                f"{r.fuel_per_pax_nmi_lb:.5f}",
                # --- NEW values ---
                f"{r.taper_ratio:.2f}", f"{r.fuselage_length_m:.1f}",
                f"{r.cruise_altitude_ft:.0f}",
                f"{r.CL_cruise:.4f}", f"{r.CD_cruise:.5f}",
                f"{r.cruise_drag_lbf:.0f}",
                f"{r.cruise_shaft_power_hp:.0f}", f"{r.cruise_throttle_pct:.1f}",
                f"{r.turboshaft_power_each_hp:.0f}", f"{r.turboshaft_power_total_hp:.0f}",
                f"{r.propeller_diameter_m:.2f}", f"{r.battery_capacity_kWh:.1f}",
                f"{r.power_loading_lb_per_hp:.2f}",
                f"{r.v_stall_kts:.1f}", f"{r.v_liftoff_kts:.1f}",
                f"{r.takeoff_ground_roll_ft:.0f}", f"{r.takeoff_total_distance_ft:.0f}",
                f"{r.balanced_field_length_ft:.0f}", f"{r.landing_distance_ft:.0f}",
                f"{r.climb_gradient_aeo_pct:.2f}", f"{r.climb_gradient_oei_pct:.2f}",
                f"{r.thrust_to_weight_to:.3f}",
                f"{r.fuel_climb_kg:.0f}", f"{r.fuel_reserves_kg:.0f}",
                f"{r.co2_kg:.0f}", f"{r.co2_per_pax_nmi:.4f}",
                f"{r.specific_range_nmi_per_lb:.3f}", f"{r.energy_per_seat_nmi:.3f}",
            ]
            f.write(",".join(str(v) for v in vals) + "\n")
    print(f"  Saved CSV data to: {csv_path}")


if __name__ == "__main__":
    main()
