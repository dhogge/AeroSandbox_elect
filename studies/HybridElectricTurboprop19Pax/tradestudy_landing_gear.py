"""
Trade Study: Fixed vs Retractable Landing Gear — BLI + WT Config (BLI_Big)
==========================================================================

Compares two landing gear configurations on the BLI_Big aircraft:

  1. **Fixed gear** (baseline):
     - Gear weight = 70% of Raymer retractable (no retraction mechanism)
     - CDA_misc = 0.20 m² (includes fixed gear fairings drag)

  2. **Retractable gear**:
     - Gear weight = 120% of Raymer retractable (retraction mechanism, actuators, doors)
     - CDA_misc = 0.15 m² (gear retracts flush in cruise, removing ~0.05 m²)

Both configurations are optimised for minimum typical-mission fuel burn
(175 nmi) while satisfying all BLI_Big constraints (field length, OEI,
stability, etc.).

Usage:
    python tradestudy_landing_gear.py
"""

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

# MBSE requirements
from requirements import load_requirements, get_limit, validate_solution

# Propulsion models
from aerosandbox.library.power_turboshaft import (
    power_turboshaft,
    thermal_efficiency_turboshaft,
)
from aerosandbox.library.propulsion_electric import (
    mass_battery_pack,
    mass_motor_electric,
    mass_ESC,
)
from aerosandbox.library.propulsion_propeller import (
    propeller_shaft_power_from_thrust,
    mass_gearbox,
)

# Field length
from aerosandbox.library.field_lengths import field_length_analysis_torenbeek

# Weight estimation
from aerosandbox.library.weights import raymer_cargo_transport_weights as raymer_wt
from aerosandbox.library.weights import torenbeek_weights as torenbeek_wt
from aerosandbox.library.weights.raymer_miscellaneous import (
    mass_passenger,
    mass_seat,
    mass_lavatories,
)


##### Section: PEGASUS Wing Weight Model #####

def wing_weight_pegasus(
        wing_area=578.,
        wing_ar=11.08,
        wing_taper=0.547,
        wing_af_thickness=0.14,
        mtow=44000.,
        battery_weight_ratio=0.35,
        engine_inboard_weight=900.,
        engine_inboard_eta=0.392,
        engine_outboard_weight=1800.,
        engine_outboard_eta=0.99,
):
    estimate = (-763.654116272738) + 66.6533463708713 * wing_ar + 436.788492704573 * wing_taper +1.21715502571785 * wing_area + -2950.25309234943 * wing_af_thickness + 0.0163144157953134 * mtow + -35.3096443123149 * battery_weight_ratio + -0.0230719745324687 * engine_inboard_weight + -152.537262155881 * engine_inboard_eta + -0.078580212183211 * engine_outboard_weight + -217.296864650889 * engine_outboard_eta + (wing_ar - 11.089463482063) * ((wing_ar -11.089463482063) * 3.32515028103463) + (wing_ar - 11.089463482063) * ((wing_taper -0.549316967122807) * 80.0810304968525) + (wing_ar - 11.089463482063) * ((wing_area -578.352162689163) * 0.0679869849011377) + (wing_ar - 11.089463482063) * (( wing_af_thickness - 0.140046114226006) * -632.721055431577) + (wing_ar - 11.089463482063) * ((mtow - 45067.1269825181) * 0.00235424149828414) + (wing_ar - 11.089463482063) * ((battery_weight_ratio - 0.304891299969045) * -14.6950965049355) + (wing_ar -11.089463482063) * ((engine_inboard_weight - 901.045007855934) * -0.0045676649394544) + ( wing_ar - 11.089463482063) * ((engine_inboard_eta - 0.375467167887513) * -31.7948952878013) + (wing_ar - 11.089463482063) * ((engine_outboard_weight - 1798.4504628483) * -0.0130605611336943) + (wing_ar - 11.089463482063) * ((engine_outboard_eta -0.844749104111456) * -49.7660905560111) + (wing_taper - 0.549316967122807) * ((wing_taper - 0.549316967122807) * 208.043958071801) + (wing_taper - 0.549316967122807) * ((wing_area - 578.352162689163) * 0.399566153569777) + (wing_taper - 0.549316967122807) * (( wing_af_thickness - 0.140046114226006) * -4681.22845517446) + (wing_taper - 0.549316967122807) * ((mtow - 45067.1269825181) * 0.0148168755374293) + (wing_taper - 0.549316967122807) * ((battery_weight_ratio - 0.304891299969045) * -19.1906778245744) + (wing_taper -0.549316967122807) * ((engine_inboard_weight - 901.045007855934) * 0.0170718428289706) + ( wing_taper - 0.549316967122807) * ((engine_inboard_eta - 0.375467167887513) * -135.424552441368 ) + (wing_taper - 0.549316967122807) * ((engine_outboard_weight - 1798.4504628483) * -0.057278134745046) + (wing_taper - 0.549316967122807) * ((engine_outboard_eta -0.844749104111456) * -174.466282246233) + (wing_area - 578.352162689163) * ((wing_area - 578.352162689163) * -0.000188690713945717) + (wing_area - 578.352162689163) * (( wing_af_thickness - 0.140046114226006) * -2.30265260237845) + (wing_area - 578.352162689163) * ((mtow - 45067.1269825181) * 0.0000151058790758956) + (wing_area - 578.352162689163 ) * ((battery_weight_ratio - 0.304891299969045) * 0.410596598562827) + (wing_area -578.352162689163) * ((engine_inboard_weight - 901.045007855934) * -0.0000233386805120355) + ( wing_area - 578.352162689163) * ((engine_inboard_eta - 0.375467167887513) * -0.0881905271304493) + (wing_area - 578.352162689163) * ((engine_outboard_weight - 1798.4504628483 ) * -0.0000546839233665841) + (wing_area - 578.352162689163) * ((engine_outboard_eta -0.844749104111456) * -0.305520724576287) + (wing_af_thickness - 0.140046114226006) * (( wing_af_thickness - 0.140046114226006) * 36135.7252036525) + (wing_af_thickness -0.140046114226006) * ((mtow - 45067.1269825181) * -0.129022665411152) + ( wing_af_thickness - 0.140046114226006) * ((battery_weight_ratio - 0.304891299969045) * -156.216624212617) + (wing_af_thickness - 0.140046114226006) * ((engine_inboard_weight -901.045007855934) * 0.455452785051827) + (wing_af_thickness - 0.140046114226006) * (( engine_inboard_eta - 0.375467167887513) * 1651.39734196954) + (wing_af_thickness -0.140046114226006) * ((engine_outboard_weight - 1798.4504628483) * 0.437340687962727) + ( wing_af_thickness - 0.140046114226006) * ((engine_outboard_eta - 0.844749104111456) * 1793.27103669289) + (mtow - 45067.1269825181) * ((mtow - 45067.1269825181) * 0.000000075413179298) + (mtow - 45067.1269825181) * ((battery_weight_ratio -0.304891299969045) * -0.000406171183132496) + (mtow - 45067.1269825181) * (( engine_inboard_weight - 901.045007855934) * -0.0000003900624341104) + (mtow - 45067.1269825181) * ((engine_inboard_eta - 0.375467167887513) * -0.00296434684811031) + (mtow -45067.1269825181) * ((engine_outboard_weight - 1798.4504628483) * -0.0000010065281736412) + ( mtow - 45067.1269825181) * ((engine_outboard_eta - 0.844749104111456) * -0.00230199459363745) + (battery_weight_ratio - 0.304891299969045) * ((battery_weight_ratio -0.304891299969045) * 136.188540688368) + (battery_weight_ratio - 0.304891299969045) * (( engine_inboard_weight - 901.045007855934) * 0.123471075162514) + (battery_weight_ratio -0.304891299969045) * ((engine_inboard_eta - 0.375467167887513) * 90.7460647792041) + ( battery_weight_ratio - 0.304891299969045) * ((engine_outboard_weight - 1798.4504628483) * 0.0198872946728822) + (battery_weight_ratio - 0.304891299969045) * ((engine_outboard_eta -0.844749104111456) * 55.0181153025222) + (engine_inboard_weight - 901.045007855934) * (( engine_inboard_weight - 901.045007855934) * 0.0000236389247774922) + (engine_inboard_weight -901.045007855934) * ((engine_inboard_eta - 0.375467167887513) * -0.09370310791469) + ( engine_inboard_weight - 901.045007855934) * ((engine_outboard_weight - 1798.4504628483) * 0.0000166647330951795) + (engine_inboard_weight - 901.045007855934) * ((engine_outboard_eta -0.844749104111456) * 0.0410735845461825) + (engine_inboard_eta - 0.375467167887513) * (( engine_inboard_eta - 0.375467167887513) * -236.984105322582) + (engine_inboard_eta -0.375467167887513) * ((engine_outboard_weight - 1798.4504628483) * 0.0401404174385669) + ( engine_inboard_eta - 0.375467167887513) * ((engine_outboard_eta - 0.844749104111456) * -141.822798401132) + (engine_outboard_weight - 1798.4504628483) * ((engine_outboard_weight -1798.4504628483) * 0.0000168826355194513) + (engine_outboard_weight - 1798.4504628483) * (( engine_outboard_eta - 0.844749104111456) * -0.0816642024504058) + (engine_outboard_eta -0.844749104111456) * ((engine_outboard_eta - 0.844749104111456) * 487.31877454063)
    return estimate


# ======================================================================
#  Optimisation wrapper — parameterised by landing gear configuration
# ======================================================================

def run_bli_big(gear_type="fixed"):
    """
    Run the full BLI_Big optimisation.

    Parameters
    ----------
    gear_type : str
        "fixed"       — lighter gear (0.7×), higher CDA_misc (0.20 m²)
        "retractable" — heavier gear (1.0×), lower CDA_misc  (0.10 m²)

    Returns
    -------
    dict with all key results for comparison.
    """
    assert gear_type in ("fixed", "retractable")

    reqs = load_requirements()

    # --- Landing gear parameters ---
    if gear_type == "fixed":
        gear_weight_factor = 0.70        # No retraction mechanism
        CDA_misc = 0.20                  # Fixed gear fairings + protuberances
    else:
        gear_weight_factor = 1.20        # Retraction mechanism, actuators, doors add 20%
        CDA_misc = 0.10                  # Gear retracts flush, removing ~0.10 m²

    ##### Section: Mission Constants #####

    n_pax = 19
    n_crew = 2
    payload_mass = 6000 * u.lbm
    cruise_speed = 200 * u.knot
    field_length_req = 2600 * u.foot
    n_engines = 2
    design_range_max = 350 * u.naut_mile
    design_range_typical = 175 * u.naut_mile
    ultimate_load_factor = 1.5 * 3.0
    CL_max = 2.4
    g = 9.81

    wingtip_propeller_efficiency_bonus = 1.15
    generator_efficiency = 0.93

    fuel_density = 820
    fuel_specific_energy = 43.02e6

    battery_cell_specific_energy = 350
    battery_pack_cell_fraction = 0.70
    battery_max_dod = 0.80

    bli_drag_reduction_factor = 0.10
    bli_propeller_CoP = 0.80

    fuse_length = 16
    fuse_cabin_width = 1.9
    fuse_cabin_height = 1.85
    nose_length = 2.5
    cabin_length = 7.1
    tail_length = fuse_length - nose_length - cabin_length

    tail_arm = 7.0

    ##### Section: Optimization Setup #####

    opti = asb.Opti()

    ##### Section: Design Variables #####

    cruise_altitude = opti.variable(
        init_guess=7000 * u.foot, lower_bound=3000 * u.foot, upper_bound=10000 * u.foot
    )
    design_mass_TOGW = opti.variable(
        init_guess=7700, log_transform=True, lower_bound=4000, upper_bound=18000
    )
    wing_span = opti.variable(
        init_guess=17.7, lower_bound=10, upper_bound=25
    )
    wing_root_chord = opti.variable(
        init_guess=2.6, lower_bound=1.5, upper_bound=4.0
    )
    cruise_alpha = opti.variable(
        init_guess=3.0, lower_bound=-2, upper_bound=10
    )
    mass_turboshaft_per_engine = opti.variable(
        init_guess=130, log_transform=True, lower_bound=50, upper_bound=400
    )
    propeller_diameter = opti.variable(
        init_guess=2.8, lower_bound=1.0, upper_bound=3.0
    )
    hybridization_factor = opti.variable(
        init_guess=0.25, lower_bound=0.20, upper_bound=0.70
    )
    battery_capacity_Wh = opti.variable(
        init_guess=50000, log_transform=True, lower_bound=5000, upper_bound=500000
    )
    fuel_mass = opti.variable(
        init_guess=800, log_transform=True, lower_bound=50, upper_bound=3000
    )
    thrust_at_liftoff = opti.variable(
        init_guess=7700 * g * 0.30, log_transform=True, lower_bound=5000
    )
    bli_motor_power = opti.variable(
        init_guess=150000, log_transform=True, lower_bound=20000, upper_bound=500000
    )
    bli_propeller_diameter = opti.variable(
        init_guess=1.2, lower_bound=0.6, upper_bound=1.8
    )
    bli_thrust_at_liftoff = opti.variable(
        init_guess=2000, lower_bound=100, log_transform=True
    )
    x_cg_battery = opti.variable(
        init_guess=nose_length + 0.25 * cabin_length,
        lower_bound=nose_length + 0.5,
        upper_bound=nose_length + cabin_length - 0.5,
    )
    thrust_wingtip_oei_reduced = opti.variable(
        init_guess=5000, lower_bound=100, log_transform=True
    )
    vstab_span_val = opti.variable(
        init_guess=2.5, lower_bound=1.5, upper_bound=6.0
    )
    vstab_root_chord_val = opti.variable(
        init_guess=2.2, lower_bound=1.2, upper_bound=5
    )
    vstab_taper_ratio = 0.5
    vstab_tip_chord_val = vstab_root_chord_val * vstab_taper_ratio
    vstab_area = (vstab_root_chord_val + vstab_tip_chord_val) / 2 * vstab_span_val

    hstab_area = opti.variable(
        init_guess=8.0, lower_bound=3.0, upper_bound=15.0
    )
    hstab_aspect_ratio = opti.variable(
        init_guess=4.0, lower_bound=3.0, upper_bound=6.0
    )
    hstab_taper_ratio = 0.5

    hstab_span_val = (hstab_area * hstab_aspect_ratio) ** 0.5
    hstab_root_chord_val = 2 * hstab_area / (hstab_span_val * (1 + hstab_taper_ratio))
    hstab_tip_chord_val = hstab_root_chord_val * hstab_taper_ratio

    wing_taper_ratio = 0.45
    wing_tip_chord = wing_root_chord * wing_taper_ratio
    wing_mean_chord = (wing_root_chord + wing_tip_chord) / 2

    ##### Section: Aircraft Geometry #####

    fuse = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[0, 0, 0], width=0, height=0),
            asb.FuselageXSec(
                xyz_c=[nose_length * 0.5, 0, -0.05],
                width=fuse_cabin_width * 0.7, height=fuse_cabin_height * 0.7, shape=2.5,
            ),
            asb.FuselageXSec(
                xyz_c=[nose_length, 0, 0],
                width=fuse_cabin_width, height=fuse_cabin_height, shape=2.5,
            ),
            asb.FuselageXSec(
                xyz_c=[nose_length + cabin_length, 0, 0],
                width=fuse_cabin_width, height=fuse_cabin_height, shape=2.5,
            ),
            asb.FuselageXSec(
                xyz_c=[nose_length + cabin_length + tail_length * 0.6, 0, 0.25],
                width=fuse_cabin_width * 0.5, height=fuse_cabin_height * 0.5, shape=2.0,
            ),
            asb.FuselageXSec(xyz_c=[fuse_length, 0, 0.5], width=0.40, height=0.40),
        ],
    )

    wing_x_le = 0.40 * fuse_length - 0.25 * wing_root_chord
    wing_z_le = 0.5 * fuse_cabin_height

    flap = asb.ControlSurface(name="Flap", symmetric=True, deflection=0, hinge_point=0.75)
    aileron = asb.ControlSurface(name="Aileron", symmetric=False, deflection=0, hinge_point=0.75)

    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0], chord=wing_root_chord, twist=2,
                airfoil=asb.Airfoil("naca23018"), control_surfaces=[flap],
            ),
            asb.WingXSec(
                xyz_le=[
                    0.6 * wing_span / 2 * np.tand(3),
                    0.6 * wing_span / 2,
                    0.6 * wing_span / 2 * np.tand(2),
                ],
                chord=wing_root_chord * 0.70, twist=0,
                airfoil=asb.Airfoil("naca23015"), control_surfaces=[aileron],
            ),
            asb.WingXSec(
                xyz_le=[
                    wing_span / 2 * np.tand(3),
                    wing_span / 2,
                    wing_span / 2 * np.tand(2),
                ],
                chord=wing_tip_chord, twist=-1,
                airfoil=asb.Airfoil("naca23015"),
            ),
        ],
    ).translate([wing_x_le, 0, wing_z_le])

    elevator = asb.ControlSurface(name="Elevator", symmetric=True, deflection=0, hinge_point=0.70)
    hstab_x_le = wing_x_le + 0.25 * wing_root_chord + tail_arm - 0.25 * hstab_root_chord_val

    rudder = asb.ControlSurface(name="Rudder", symmetric=True, deflection=0, hinge_point=0.70)
    vstab_z_le = 0.3
    hstab_z_le = vstab_z_le + vstab_span_val
    vstab_x_le = hstab_x_le - vstab_span_val * np.tand(30)

    vstab = asb.Wing(
        name="Vertical Stabilizer",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0], chord=vstab_root_chord_val,
                airfoil=asb.Airfoil("naca0012"), control_surfaces=[rudder],
            ),
            asb.WingXSec(
                xyz_le=[vstab_span_val * np.tand(30), 0, vstab_span_val],
                chord=vstab_tip_chord_val, airfoil=asb.Airfoil("naca0010"),
            ),
        ],
    ).translate([vstab_x_le, 0, vstab_z_le])

    hstab = asb.Wing(
        name="Horizontal Stabilizer",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0], chord=hstab_root_chord_val,
                airfoil=asb.Airfoil("naca0012"), control_surfaces=[elevator],
            ),
            asb.WingXSec(
                xyz_le=[hstab_span_val / 2 * np.tand(10), hstab_span_val / 2, 0],
                chord=hstab_tip_chord_val, airfoil=asb.Airfoil("naca0010"),
            ),
        ],
    ).translate([hstab_x_le, 0, hstab_z_le])

    bli_propulsor = asb.Propulsor(
        name="BLI Pusher",
        xyz_c=[fuse_length + 0.3, 0, 0.5],
        xyz_normal=[1, 0, 0],
        radius=bli_propeller_diameter / 2,
        length=0.3,
    )

    airplane = asb.Airplane(
        name="HE-19 Hybrid Electric Turboprop + BLI",
        xyz_ref=[wing_x_le + 0.25 * wing_root_chord, 0, 0],
        wings=[wing, hstab, vstab],
        fuselages=[fuse],
        propulsors=[bli_propulsor],
    )

    ##### Section: Aerodynamic Analysis (Cruise) #####

    cruise_atmo = asb.Atmosphere(altitude=cruise_altitude)
    cruise_op_point = asb.OperatingPoint(
        atmosphere=cruise_atmo,
        velocity=cruise_speed,
        alpha=cruise_alpha,
    )

    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=cruise_op_point,
    ).run()

    wing_area = wing.area()

    drag_correction_factor = 1.10

    CD_misc = CDA_misc / wing_area
    CL_cruise = aero["CL"]
    CD_cruise = aero["CD"] * drag_correction_factor + CD_misc
    L_over_D_cruise = CL_cruise / CD_cruise

    q_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2
    drag_cruise = CD_cruise * q_cruise * wing_area

    drag_effective_cruise = drag_cruise * (1 - bli_drag_reduction_factor)
    drag_bli_wake_fill = drag_cruise * bli_drag_reduction_factor

    ##### Section: Weight Breakdown #####

    wing_to_hstab_distance = tail_arm
    m_hstab = raymer_wt.mass_hstab(
        hstab=hstab,
        design_mass_TOGW=design_mass_TOGW,
        ultimate_load_factor=ultimate_load_factor,
        wing_to_hstab_distance=wing_to_hstab_distance,
        fuselage_width_at_hstab_intersection=fuse_cabin_width,
    )

    wing_to_vstab_distance = tail_arm * 0.95
    m_vstab = raymer_wt.mass_vstab(
        vstab=vstab,
        design_mass_TOGW=design_mass_TOGW,
        ultimate_load_factor=ultimate_load_factor,
        wing_to_vstab_distance=wing_to_vstab_distance,
    )

    m_fuselage = raymer_wt.mass_fuselage(
        fuselage=fuse,
        design_mass_TOGW=design_mass_TOGW,
        ultimate_load_factor=ultimate_load_factor,
        L_over_D=L_over_D_cruise,
        main_wing=wing,
        n_cargo_doors=1,
        landing_gear_mounted_on_fuselage=False,
    )

    # Landing gear — key trade variable
    atmo_sl = asb.Atmosphere(altitude=0)
    V_stall_sl = np.sqrt(
        2 * design_mass_TOGW * g / (atmo_sl.density() * wing_area * CL_max)
    )

    m_mlg = raymer_wt.mass_main_landing_gear(
        main_gear_length=0.8,
        landing_speed=V_stall_sl * 1.3,
        design_mass_TOGW=design_mass_TOGW,
        n_wheels=4,
        n_shock_struts=2,
    ) * gear_weight_factor

    m_nlg = raymer_wt.mass_nose_landing_gear(
        nose_gear_length=0.6,
        design_mass_TOGW=design_mass_TOGW,
        n_wheels=2,
    ) * gear_weight_factor

    nacelle_length = 1.8
    nacelle_width = 0.7
    nacelle_height = 0.7

    m_nacelles = raymer_wt.mass_nacelles(
        nacelle_length=nacelle_length,
        nacelle_width=nacelle_width,
        nacelle_height=nacelle_height,
        ultimate_load_factor=ultimate_load_factor,
        mass_per_engine=mass_turboshaft_per_engine,
        n_engines=n_engines,
        engines_have_propellers=True,
    )

    m_instruments = raymer_wt.mass_instruments(
        fuselage=fuse,
        main_wing=wing,
        n_engines=n_engines,
        n_crew=n_crew,
        engine_is_turboprop=True,
    )

    m_electrical = raymer_wt.mass_electrical(
        system_electrical_power_rating=50000,
        electrical_routing_distance=fuse_length * 0.6,
        n_engines=n_engines,
    )

    m_furnishings = raymer_wt.mass_furnishings(
        n_crew=n_crew,
        mass_cargo=payload_mass,
        fuselage=fuse,
    )

    cabin_volume = (
        np.pi * (fuse_cabin_width / 2) * (fuse_cabin_height / 2) * cabin_length
    )

    m_ac = raymer_wt.mass_air_conditioning(
        n_crew=n_crew,
        n_pax=n_pax,
        volume_pressurized=cabin_volume,
        mass_uninstalled_avionics=200 * u.lbm,
    )

    m_anti_ice = raymer_wt.mass_anti_ice(design_mass_TOGW=design_mass_TOGW)

    fuel_volume = fuel_mass / fuel_density
    m_fuel_system = raymer_wt.mass_fuel_system(
        fuel_volume=fuel_volume,
        n_tanks=2,
        fraction_in_integral_tanks=1.0,
    )

    # Propulsion
    power_per_turboshaft = power_turboshaft(mass_turboshaft_per_engine) * .8
    m_turboshaft_total = mass_turboshaft_per_engine * n_engines

    electric_power_per_engine = (
        hybridization_factor / (1 - hybridization_factor) * power_per_turboshaft
    )

    motor_power_density = 5000  # W/kg (5 kW/kg, aircraft-class certified electric motor)
    m_motor_per_engine = electric_power_per_engine / motor_power_density
    m_motor_total = m_motor_per_engine * n_engines

    m_esc_per_engine = mass_ESC(electric_power_per_engine)
    m_esc_total = m_esc_per_engine * n_engines

    m_battery = mass_battery_pack(
        battery_capacity_Wh,
        battery_cell_specific_energy_Wh_kg=battery_cell_specific_energy,
        battery_pack_cell_fraction=battery_pack_cell_fraction,
    )

    total_power_per_propeller = power_per_turboshaft + electric_power_per_engine
    m_propeller_each = torenbeek_wt.mass_propeller(
        propeller_diameter=propeller_diameter,
        propeller_power=total_power_per_propeller,
        n_blades=4,
    ) * 0.35
    m_propellers_total = m_propeller_each * n_engines

    turboshaft_output_rpm = 33000
    propeller_rpm = 1700
    m_gearbox_each = mass_gearbox(
        power=total_power_per_propeller,
        rpm_in=turboshaft_output_rpm,
        rpm_out=propeller_rpm,
    )
    m_gearbox_total = m_gearbox_each * n_engines

    m_bli_motor = bli_motor_power / motor_power_density  # 5 kW/kg
    m_bli_esc = mass_ESC(bli_motor_power)
    m_bli_propeller = torenbeek_wt.mass_propeller(
        propeller_diameter=bli_propeller_diameter,
        propeller_power=bli_motor_power,
        n_blades=5,
    ) * 0.35
    m_bli_nacelle = 0.10 * m_bli_motor + 15

    wing_aspect_ratio = wing_span ** 2 / wing_area

    V_NE = cruise_speed * 1.40
    V_flap = V_stall_sl * 1.8

    suspended_mass_approx = design_mass_TOGW * 0.92

    m_wing_basic = torenbeek_wt.mass_wing_basic_structure(
        wing=wing,
        design_mass_TOGW=design_mass_TOGW,
        ultimate_load_factor=ultimate_load_factor,
        suspended_mass=suspended_mass_approx,
        never_exceed_airspeed=V_NE,
        main_gear_mounted_to_wing=False,
        k_e=0.90,
    )
    m_wing_hld = torenbeek_wt.mass_wing_high_lift_devices(
        wing=wing,
        max_airspeed_for_flaps=V_flap,
        flap_deflection_angle=30,
    )
    m_wing_spoilers = torenbeek_wt.mass_wing_spoilers_and_speedbrakes(
        wing=wing,
        mass_basic_wing=m_wing_basic,
    )
    m_wing = m_wing_basic + 1.2 * (m_wing_hld + m_wing_spoilers)

    m_pax = n_pax * mass_passenger
    m_seats = n_pax * mass_seat("passenger") + n_crew * mass_seat("flight_deck")
    m_lavs = mass_lavatories(n_pax, aircraft_type="short-haul")
    m_flight_controls = 0.02 * design_mass_TOGW

    ##### Section: Total Weight #####

    mass_empty = (
        m_wing + m_hstab + m_vstab + m_fuselage
        + m_mlg + m_nlg + m_nacelles
        + m_turboshaft_total + m_motor_total + m_esc_total
        + m_propellers_total + m_gearbox_total
        + m_fuel_system
        + m_bli_motor + m_bli_esc + m_bli_propeller + m_bli_nacelle
        + m_instruments + m_electrical + m_furnishings
        + m_ac + m_anti_ice + m_flight_controls
        + m_seats + m_lavs
    )

    mass_total = mass_empty + payload_mass + fuel_mass + m_battery

    ##### Section: CG, Neutral Point & Static Margin #####

    wing_MAC = (2 / 3) * wing_root_chord * (
        1 + wing_taper_ratio + wing_taper_ratio ** 2
    ) / (1 + wing_taper_ratio)

    wing_sweep_LE = 3
    y_MAC = (wing_span / 6) * (1 + 2 * wing_taper_ratio) / (1 + wing_taper_ratio)
    x_MAC_le = wing_x_le + y_MAC * np.tand(wing_sweep_LE)

    x_ac_wing = x_MAC_le + 0.25 * wing_MAC

    x_cg_wing = x_MAC_le + 0.40 * wing_MAC
    x_cg_hstab = hstab_x_le + 0.42 * (hstab_root_chord_val + hstab_tip_chord_val) / 2
    x_cg_vstab = vstab_x_le + 0.42 * (vstab_root_chord_val + vstab_tip_chord_val) / 2
    x_cg_fuselage = 0.45 * fuse_length
    x_cg_mlg = wing_x_le + 0.55 * wing_mean_chord
    x_cg_nlg = 0.80 * nose_length

    x_wingtip_le = wing_x_le + (wing_span / 2) * np.tand(wing_sweep_LE)
    x_cg_nacelles = x_wingtip_le + 0.30 * wing_tip_chord
    x_cg_turboshaft = x_cg_nacelles
    x_cg_motors_wingtip = x_cg_nacelles
    x_cg_esc_wingtip = x_cg_nacelles - 0.2
    x_cg_propellers = x_cg_nacelles - 0.4
    x_cg_gearbox = x_cg_nacelles

    x_cg_bli_motor = fuse_length - 0.2
    x_cg_bli_esc = fuse_length - 0.5
    x_cg_bli_prop = fuse_length + 0.3
    x_cg_bli_nacelle = fuse_length + 0.1

    x_cg_fuel = x_MAC_le + 0.35 * wing_MAC
    x_cg_fuel_system = x_cg_fuel
    x_cg_payload = nose_length + 0.50 * cabin_length
    x_cg_instruments = 0.15 * fuse_length
    x_cg_electrical = 0.40 * fuse_length
    x_cg_furnishings = nose_length + 0.50 * cabin_length
    x_cg_aircon = nose_length + 0.25 * cabin_length
    x_cg_anti_ice = wing_x_le
    x_cg_flight_controls = 0.40 * fuse_length
    x_cg_seats = nose_length + 0.50 * cabin_length
    x_cg_lavs = nose_length + cabin_length - 0.5

    moment_TOGW = (
        m_wing * x_cg_wing
        + m_hstab * x_cg_hstab
        + m_vstab * x_cg_vstab
        + m_fuselage * x_cg_fuselage
        + m_mlg * x_cg_mlg
        + m_nlg * x_cg_nlg
        + m_nacelles * x_cg_nacelles
        + m_turboshaft_total * x_cg_turboshaft
        + m_motor_total * x_cg_motors_wingtip
        + m_esc_total * x_cg_esc_wingtip
        + m_propellers_total * x_cg_propellers
        + m_gearbox_total * x_cg_gearbox
        + m_bli_motor * x_cg_bli_motor
        + m_bli_esc * x_cg_bli_esc
        + m_bli_propeller * x_cg_bli_prop
        + m_bli_nacelle * x_cg_bli_nacelle
        + m_fuel_system * x_cg_fuel_system
        + m_instruments * x_cg_instruments
        + m_electrical * x_cg_electrical
        + m_furnishings * x_cg_furnishings
        + m_ac * x_cg_aircon
        + m_anti_ice * x_cg_anti_ice
        + m_flight_controls * x_cg_flight_controls
        + m_seats * x_cg_seats
        + m_lavs * x_cg_lavs
        + fuel_mass * x_cg_fuel
        + m_battery * x_cg_battery
        + payload_mass * x_cg_payload
    )
    x_cg_TOGW = moment_TOGW / mass_total

    mass_zfw = mass_total - fuel_mass
    x_cg_aft = (moment_TOGW - fuel_mass * x_cg_fuel) / mass_zfw

    a_w = 2 * np.pi * wing_aspect_ratio / (2 + np.sqrt(4 + wing_aspect_ratio ** 2))
    a_h = 2 * np.pi * hstab_aspect_ratio / (2 + np.sqrt(4 + hstab_aspect_ratio ** 2))

    depsilon_dalpha = 2 * a_w / (np.pi * wing_aspect_ratio) * 0.85

    eta_h = 0.90

    x_ac_hstab = hstab_x_le + 0.25 * (hstab_root_chord_val + hstab_tip_chord_val) / 2

    tail_lift_effectiveness = a_h * eta_h * (1 - depsilon_dalpha) * hstab_area
    wing_lift_effectiveness = a_w * wing_area
    x_np_wing_tail = (
        wing_lift_effectiveness * x_ac_wing
        + tail_lift_effectiveness * x_ac_hstab
    ) / (wing_lift_effectiveness + tail_lift_effectiveness)

    K_f = 0.92
    fuselage_width_eff = fuse_cabin_width * 0.85
    delta_x_np_fuse = K_f * fuselage_width_eff ** 2 * fuse_length / wing_lift_effectiveness
    x_np = x_np_wing_tail - delta_x_np_fuse

    static_margin = (x_np - x_cg_aft) / wing_MAC
    static_margin_TOGW = (x_np - x_cg_TOGW) / wing_MAC

    ##### Section: Propulsion and Performance #####

    wingtip_propulsive_area = n_engines * np.pi / 4 * propeller_diameter ** 2
    bli_propulsive_area = np.pi / 4 * bli_propeller_diameter ** 2

    shaft_power_cruise_wingtip = propeller_shaft_power_from_thrust(
        thrust_force=drag_effective_cruise,
        area_propulsive=wingtip_propulsive_area,
        airspeed=cruise_speed,
        rho=cruise_atmo.density(),
        propeller_coefficient_of_performance=0.85 * wingtip_propeller_efficiency_bonus,
    )

    shaft_power_cruise_bli = propeller_shaft_power_from_thrust(
        thrust_force=drag_bli_wake_fill,
        area_propulsive=bli_propulsive_area,
        airspeed=cruise_speed,
        rho=cruise_atmo.density(),
        propeller_coefficient_of_performance=bli_propeller_CoP,
    )

    opti.subject_to(bli_motor_power >= shaft_power_cruise_bli * get_limit(reqs, "REQ-001"))

    bli_electric_demand_from_turboshaft = shaft_power_cruise_bli / generator_efficiency

    shaft_power_cruise_per_engine_total = (
        shaft_power_cruise_wingtip + bli_electric_demand_from_turboshaft
    ) / n_engines

    opti.subject_to(power_per_turboshaft >= shaft_power_cruise_per_engine_total * get_limit(reqs, "REQ-002"))

    shaft_power_cruise_total = shaft_power_cruise_wingtip + bli_electric_demand_from_turboshaft
    throttle_cruise = shaft_power_cruise_per_engine_total / power_per_turboshaft

    eta_thermal_cruise = thermal_efficiency_turboshaft(
        mass_turboshaft=mass_turboshaft_per_engine,
        throttle_setting=throttle_cruise,
    )

    fuel_burn_rate_cruise = shaft_power_cruise_total / (
        eta_thermal_cruise * fuel_specific_energy
    )

    fuel_reserve_time = 45 * 60
    fuel_reserves = fuel_burn_rate_cruise * fuel_reserve_time

    climb_time = 10 * 60
    climb_fuel_factor = 1.3
    fuel_for_climb = fuel_burn_rate_cruise * climb_fuel_factor * climb_time

    cruise_time_max = design_range_max / cruise_speed
    fuel_for_cruise_max = fuel_burn_rate_cruise * cruise_time_max

    opti.subject_to(fuel_mass >= fuel_for_cruise_max + fuel_reserves + fuel_for_climb)

    cruise_time_typical = design_range_typical / cruise_speed
    fuel_for_cruise_typical = fuel_burn_rate_cruise * cruise_time_typical
    fuel_mass_typical = fuel_for_cruise_typical + fuel_reserves + fuel_for_climb

    wingtip_thrust_at_liftoff = thrust_at_liftoff - bli_thrust_at_liftoff

    total_power_takeoff_per_engine = power_per_turboshaft + electric_power_per_engine
    shaft_power_takeoff_wingtip = total_power_takeoff_per_engine * n_engines

    V_liftoff = 1.2 * V_stall_sl

    wingtip_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
        thrust_force=wingtip_thrust_at_liftoff,
        area_propulsive=wingtip_propulsive_area,
        airspeed=V_liftoff,
        rho=atmo_sl.density(),
        propeller_coefficient_of_performance=0.80 * wingtip_propeller_efficiency_bonus,
    )

    opti.subject_to(shaft_power_takeoff_wingtip >= wingtip_shaft_power_from_thrust_liftoff)

    bli_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
        thrust_force=bli_thrust_at_liftoff,
        area_propulsive=bli_propulsive_area,
        airspeed=V_liftoff,
        rho=atmo_sl.density(),
        propeller_coefficient_of_performance=bli_propeller_CoP,
    )

    opti.subject_to(bli_motor_power >= bli_shaft_power_from_thrust_liftoff)

    electric_energy_wingtip_climb = electric_power_per_engine * n_engines * climb_time
    electric_energy_bli_climb = bli_motor_power * climb_time

    total_electric_energy_Wh = (
        electric_energy_wingtip_climb + electric_energy_bli_climb
    ) / 3600

    opti.subject_to(battery_capacity_Wh >= total_electric_energy_Wh / get_limit(reqs, "REQ-006"))

    ##### Section: Constraints #####

    opti.subject_to(wing_aspect_ratio >= get_limit(reqs, "REQ-007"))
    opti.subject_to(wing_aspect_ratio <= get_limit(reqs, "REQ-008"))

    lift_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2 * wing_area * CL_cruise
    typical_mission_TOGW = design_mass_TOGW - (fuel_mass - fuel_mass_typical)
    mid_cruise_weight = (typical_mission_TOGW - fuel_for_cruise_typical * 0.5) * g

    opti.subject_to(lift_cruise >= mid_cruise_weight * get_limit(reqs, "REQ-010"))
    opti.subject_to(lift_cruise <= mid_cruise_weight * get_limit(reqs, "REQ-011"))

    L_over_D_climb = L_over_D_cruise * 0.65

    field_results = field_length_analysis_torenbeek(
        design_mass_TOGW=design_mass_TOGW,
        thrust_at_liftoff=thrust_at_liftoff,
        lift_over_drag_climb=L_over_D_climb,
        CL_max=CL_max,
        s_ref=wing_area,
        n_engines=n_engines,
        atmosphere=atmo_sl,
        CD_zero_lift=0.04,
        obstacle_height=50 * u.foot,
    )

    opti.subject_to(field_results["takeoff_total_distance"] <= field_length_req)
    opti.subject_to(field_results["landing_total_distance"] <= field_length_req)
    opti.subject_to(field_results["balanced_field_length"] <= field_length_req)

    thrust_per_wingtip_engine = wingtip_thrust_at_liftoff / n_engines
    opti.subject_to(thrust_wingtip_oei_reduced <= thrust_per_wingtip_engine)

    thrust_oei = thrust_wingtip_oei_reduced + bli_thrust_at_liftoff
    thrust_over_weight_oei = thrust_oei / (design_mass_TOGW * g)
    climb_gradient_oei = thrust_over_weight_oei - 1 / L_over_D_climb
    opti.subject_to(climb_gradient_oei >= get_limit(reqs, "REQ-016"))

    y_engine = wing_span / 2
    yaw_moment_oei = thrust_wingtip_oei_reduced * y_engine * 1.10

    CL_vstab_max_rudder = 0.9
    l_vt = tail_arm * 0.95

    q_vmc = yaw_moment_oei / (vstab_area * CL_vstab_max_rudder * l_vt)
    V_mc = np.sqrt(2 * q_vmc / atmo_sl.density())

    opti.subject_to(V_mc <= V_stall_sl)

    l_h = tail_arm
    V_h_coefficient = (hstab_area * l_h) / (wing_area * wing_mean_chord)
    opti.subject_to(V_h_coefficient >= get_limit(reqs, "REQ-018"))

    opti.subject_to(static_margin >= get_limit(reqs, "REQ-019"))
    opti.subject_to(static_margin_TOGW <= get_limit(reqs, "REQ-020"))

    vstab_aspect_ratio = vstab_span_val ** 2 / vstab_area
    opti.subject_to(vstab_aspect_ratio >= get_limit(reqs, "REQ-021"))
    opti.subject_to(vstab_aspect_ratio <= get_limit(reqs, "REQ-022"))

    opti.subject_to(bli_propeller_diameter <= fuse_cabin_height * get_limit(reqs, "REQ-009"))

    opti.subject_to(bli_thrust_at_liftoff <= thrust_at_liftoff * get_limit(reqs, "REQ-023"))

    opti.subject_to(mass_total <= design_mass_TOGW)

    opti.subject_to(design_mass_TOGW <= get_limit(reqs, "REQ-025") * u.lbm)

    ##### Section: Objective #####

    opti.minimize(fuel_mass_typical)

    ##### Section: Solve #####

    sol = opti.solve(max_iter=1500)

    ##### Section: Extract Results #####

    TOGW = sol(design_mass_TOGW)
    m_empty_sol = sol(mass_empty)
    m_fuel_sol = sol(fuel_mass)
    m_batt_sol = sol(m_battery)
    b = sol(wing_span)
    S_wing = sol(wing_area)
    AR = b ** 2 / S_wing

    results = {
        "gear_type":                gear_type,
        "CDA_misc":                 CDA_misc,
        "gear_weight_factor":       gear_weight_factor,

        # Top-level
        "MTOW_kg":                  TOGW,
        "MTOW_lb":                  TOGW / u.lbm,
        "empty_weight_kg":          m_empty_sol,
        "empty_weight_lb":          m_empty_sol / u.lbm,
        "fuel_weight_kg":           m_fuel_sol,
        "fuel_weight_lb":           m_fuel_sol / u.lbm,
        "battery_weight_kg":        m_batt_sol,
        "battery_weight_lb":        m_batt_sol / u.lbm,
        "payload_kg":               payload_mass,

        # Landing gear weights
        "m_mlg_kg":                 sol(m_mlg),
        "m_nlg_kg":                 sol(m_nlg),
        "m_gear_total_kg":          sol(m_mlg) + sol(m_nlg),

        # Aerodynamics
        "CL_cruise":                sol(CL_cruise),
        "CD_cruise":                sol(CD_cruise),
        "L_over_D":                 sol(L_over_D_cruise),
        "drag_cruise_N":            sol(drag_cruise),
        "drag_effective_N":         sol(drag_effective_cruise),

        # Geometry
        "wing_span_m":              b,
        "wing_area_m2":             S_wing,
        "aspect_ratio":             AR,
        "cruise_altitude_ft":       sol(cruise_altitude) / u.foot,

        # Propulsion
        "power_turboshaft_hp":      sol(power_per_turboshaft) / u.horsepower,
        "hybridization":            sol(hybridization_factor),
        "propeller_dia_m":          sol(propeller_diameter),
        "bli_motor_kW":             sol(bli_motor_power) / 1000,
        "bli_prop_dia_m":           sol(bli_propeller_diameter),

        # Performance
        "fuel_burn_rate_kg_hr":     sol(fuel_burn_rate_cruise) * 3600,
        "fuel_typical_mission_kg":  sol(fuel_mass_typical),
        "cruise_throttle":          sol(throttle_cruise),
        "V_stall_kts":              sol(V_stall_sl) / u.knot,
        "V_mc_kts":                 sol(V_mc) / u.knot,
        "wing_loading_psf":         TOGW * g / S_wing / (u.lbf / u.foot ** 2),

        # Field performance
        "TO_dist_ft":               sol(field_results["takeoff_total_distance"]) / u.foot,
        "LDG_dist_ft":              sol(field_results["landing_total_distance"]) / u.foot,
        "BFL_ft":                   sol(field_results["balanced_field_length"]) / u.foot,
        "OEI_gradient":             sol(climb_gradient_oei),

        # Stability
        "static_margin_aft_pct":    sol(static_margin) * 100,
        "static_margin_TOGW_pct":   sol(static_margin_TOGW) * 100,

        # Weight fractions
        "empty_frac":               m_empty_sol / TOGW,
        "fuel_frac":                m_fuel_sol / TOGW,
        "battery_frac":             m_batt_sol / TOGW,
        "payload_frac":             payload_mass / TOGW,
        "gear_frac":                (sol(m_mlg) + sol(m_nlg)) / TOGW,

        # For weight breakdown table
        "m_wing_kg":                sol(m_wing),
        "m_hstab_kg":               sol(m_hstab),
        "m_vstab_kg":               sol(m_vstab),
        "m_fuselage_kg":            sol(m_fuselage),
        "m_nacelles_kg":            sol(m_nacelles),
    }

    return results


# ======================================================================
#  Run both configurations
# ======================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 72)
    print("  TRADE STUDY: Fixed vs Retractable Landing Gear (BLI_Big)")
    print("=" * 72)

    # --- Run fixed gear (baseline) ---
    print("\n>>> Solving FIXED gear configuration...")
    res_fixed = run_bli_big("fixed")
    print("    ...converged.\n")

    # --- Run retractable gear ---
    print(">>> Solving RETRACTABLE gear configuration...")
    res_retract = run_bli_big("retractable")
    print("    ...converged.\n")

    # ======================================================================
    #  Side-by-side comparison table
    # ======================================================================

    def delta(key, fmt=".1f", pct=False):
        """Return formatted (fixed, retract, delta, delta%)."""
        f = res_fixed[key]
        r = res_retract[key]
        d = r - f
        dp = d / abs(f) * 100 if f != 0 else 0
        sign = "+" if d >= 0 else ""
        if pct:
            return f"{f:{fmt}}", f"{r:{fmt}}", f"{sign}{d:{fmt}}", f"{sign}{dp:.1f}%"
        return f"{f:{fmt}}", f"{r:{fmt}}", f"{sign}{d:{fmt}}"

    print("=" * 80)
    print(f"  {'METRIC':<36} {'FIXED':>12} {'RETRACT':>12} {'DELTA':>10} {'%':>8}")
    print("=" * 80)

    rows = [
        ("--- Configuration ---",             None),
        ("  CDA_misc (m²)",                   "CDA_misc",        ".2f"),
        ("  Gear weight factor",              "gear_weight_factor", ".2f"),
        ("--- Weights ---",                    None),
        ("  MTOW (lb)",                        "MTOW_lb",         ".0f"),
        ("  Empty Weight (lb)",                "empty_weight_lb", ".0f"),
        ("  Fuel Weight (lb)",                 "fuel_weight_lb",  ".0f"),
        ("  Battery Weight (lb)",              "battery_weight_lb", ".0f"),
        ("  Main Gear (kg)",                   "m_mlg_kg",        ".1f"),
        ("  Nose Gear (kg)",                   "m_nlg_kg",        ".1f"),
        ("  Total Gear (kg)",                  "m_gear_total_kg", ".1f"),
        ("  Gear Fraction (%MTOW)",            "gear_frac",       ".3f"),
        ("--- Aerodynamics ---",               None),
        ("  CL (cruise)",                      "CL_cruise",       ".4f"),
        ("  CD (cruise)",                      "CD_cruise",       ".5f"),
        ("  L/D (cruise)",                     "L_over_D",        ".1f"),
        ("  Cruise Drag (N)",                  "drag_cruise_N",   ".0f"),
        ("  Effective Drag w/ BLI (N)",        "drag_effective_N", ".0f"),
        ("--- Geometry ---",                   None),
        ("  Wing Span (m)",                    "wing_span_m",     ".2f"),
        ("  Wing Area (m²)",                   "wing_area_m2",    ".1f"),
        ("  Aspect Ratio",                     "aspect_ratio",    ".2f"),
        ("  Cruise Altitude (ft)",             "cruise_altitude_ft", ".0f"),
        ("  Wing Loading (psf)",               "wing_loading_psf", ".1f"),
        ("--- Propulsion ---",                 None),
        ("  Turboshaft Power (hp, each)",      "power_turboshaft_hp", ".0f"),
        ("  Hybridization Factor",             "hybridization",   ".3f"),
        ("  BLI Motor Power (kW)",             "bli_motor_kW",    ".0f"),
        ("--- Performance ---",                None),
        ("  Fuel Burn Rate (kg/hr)",           "fuel_burn_rate_kg_hr", ".1f"),
        ("  Fuel (175 nmi typical, kg)",       "fuel_typical_mission_kg", ".1f"),
        ("  Cruise Throttle",                  "cruise_throttle", ".3f"),
        ("  V_stall (kts)",                    "V_stall_kts",     ".1f"),
        ("  V_mc (kts)",                       "V_mc_kts",        ".1f"),
        ("--- Field Performance ---",          None),
        ("  Takeoff Distance (ft)",            "TO_dist_ft",      ".0f"),
        ("  Landing Distance (ft)",            "LDG_dist_ft",     ".0f"),
        ("  Balanced Field Length (ft)",        "BFL_ft",          ".0f"),
        ("  OEI Climb Gradient",               "OEI_gradient",    ".4f"),
        ("--- Stability ---",                  None),
        ("  Static Margin, aft CG (% MAC)",    "static_margin_aft_pct",  ".1f"),
        ("  Static Margin, TOGW (% MAC)",      "static_margin_TOGW_pct", ".1f"),
        ("--- Weight Fractions ---",           None),
        ("  Empty Weight Fraction",            "empty_frac",      ".3f"),
        ("  Fuel Fraction",                    "fuel_frac",       ".3f"),
        ("  Battery Fraction",                 "battery_frac",    ".3f"),
        ("  Payload Fraction",                 "payload_frac",    ".3f"),
    ]

    for row in rows:
        if row[1] is None:
            print(f"\n  {row[0]}")
            continue
        label, key, fmt = row[0], row[1], row[2]
        f_val = res_fixed[key]
        r_val = res_retract[key]
        d_val = r_val - f_val
        if abs(f_val) > 1e-9:
            dp = (r_val - f_val) / abs(f_val) * 100
        else:
            dp = 0.0
        sign = "+" if d_val >= 0 else ""
        print(
            f"  {label:<36} {f_val:>12{fmt}} {r_val:>12{fmt}}"
            f" {sign}{d_val:>9{fmt}} {sign}{dp:>6.1f}%"
        )

    print("\n" + "=" * 80)

    # ======================================================================
    #  Key Findings Summary
    # ======================================================================
    print("\n  KEY FINDINGS")
    print("  " + "-" * 40)

    fuel_save_pct = (res_fixed["fuel_typical_mission_kg"] - res_retract["fuel_typical_mission_kg"]) / res_fixed["fuel_typical_mission_kg"] * 100
    ld_gain_pct = (res_retract["L_over_D"] - res_fixed["L_over_D"]) / res_fixed["L_over_D"] * 100
    wt_penalty = res_retract["m_gear_total_kg"] - res_fixed["m_gear_total_kg"]
    mtow_delta = res_retract["MTOW_lb"] - res_fixed["MTOW_lb"]

    print(f"  Retractable gear L/D improvement:  {ld_gain_pct:+.1f}%")
    print(f"  Gear weight penalty (retractable):  {wt_penalty:+.1f} kg")
    print(f"  MTOW change:                        {mtow_delta:+.0f} lb")
    print(f"  Typical mission fuel savings:       {fuel_save_pct:+.1f}%")

    if fuel_save_pct > 0:
        print("\n  -> Retractable gear REDUCES fuel burn (drag reduction outweighs weight penalty)")
    elif fuel_save_pct < 0:
        print("\n  -> Fixed gear REDUCES fuel burn (lighter weight outweighs drag penalty)")
    else:
        print("\n  -> No meaningful difference in fuel burn")

    # ======================================================================
    #  Visualisation
    # ======================================================================

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Trade Study: Fixed vs Retractable Landing Gear — BLI + WT Config",
        fontsize=14, fontweight="bold",
    )
    configs = ["Fixed", "Retractable"]
    colors = ["#2196F3", "#FF9800"]
    x = np.array([0, 1])
    bar_w = 0.45

    # 1. MTOW
    ax = axes[0, 0]
    vals = [res_fixed["MTOW_lb"], res_retract["MTOW_lb"]]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("lb")
    ax.set_title("MTOW")
    for i, v in enumerate(vals):
        ax.text(i, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    # 2. L/D
    ax = axes[0, 1]
    vals = [res_fixed["L_over_D"], res_retract["L_over_D"]]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_title("Cruise L/D")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # 3. Typical mission fuel
    ax = axes[0, 2]
    vals = [res_fixed["fuel_typical_mission_kg"], res_retract["fuel_typical_mission_kg"]]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("kg")
    ax.set_title("Fuel — 175 nmi Typical Mission")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # 4. Weight breakdown comparison (stacked bar)
    ax = axes[1, 0]
    categories = ["Gear", "Wing", "Fuselage", "Empennage", "Nacelles"]
    fixed_vals = [
        res_fixed["m_gear_total_kg"],
        res_fixed["m_wing_kg"],
        res_fixed["m_fuselage_kg"],
        res_fixed["m_hstab_kg"] + res_fixed["m_vstab_kg"],
        res_fixed["m_nacelles_kg"],
    ]
    retract_vals = [
        res_retract["m_gear_total_kg"],
        res_retract["m_wing_kg"],
        res_retract["m_fuselage_kg"],
        res_retract["m_hstab_kg"] + res_retract["m_vstab_kg"],
        res_retract["m_nacelles_kg"],
    ]
    x_cat = np.arange(len(categories))
    ax.bar(x_cat - 0.18, fixed_vals, 0.35, label="Fixed", color=colors[0],
           edgecolor="black", linewidth=0.5)
    ax.bar(x_cat + 0.18, retract_vals, 0.35, label="Retractable", color=colors[1],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_cat)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("kg")
    ax.set_title("Structural Weight Comparison")
    ax.legend(fontsize=8)

    # 5. Drag breakdown
    ax = axes[1, 1]
    vals = [res_fixed["CD_cruise"], res_retract["CD_cruise"]]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_title("Total CD (Cruise)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0002, f"{v:.5f}", ha="center", va="bottom", fontsize=9)

    # 6. Field performance
    ax = axes[1, 2]
    field_cats = ["TO Dist", "LDG Dist", "BFL"]
    fixed_fp = [res_fixed["TO_dist_ft"], res_fixed["LDG_dist_ft"], res_fixed["BFL_ft"]]
    retract_fp = [res_retract["TO_dist_ft"], res_retract["LDG_dist_ft"], res_retract["BFL_ft"]]
    x_fp = np.arange(len(field_cats))
    ax.bar(x_fp - 0.18, fixed_fp, 0.35, label="Fixed", color=colors[0],
           edgecolor="black", linewidth=0.5)
    ax.bar(x_fp + 0.18, retract_fp, 0.35, label="Retractable", color=colors[1],
           edgecolor="black", linewidth=0.5)
    ax.axhline(y=2600, color="red", linestyle="--", linewidth=1, label="2600 ft limit")
    ax.set_xticks(x_fp)
    ax.set_xticklabels(field_cats, fontsize=8)
    ax.set_ylabel("ft")
    ax.set_title("Field Performance")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("tradestudy_landing_gear.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved to tradestudy_landing_gear.png")
    try:
        plt.show()
    except Exception:
        pass
