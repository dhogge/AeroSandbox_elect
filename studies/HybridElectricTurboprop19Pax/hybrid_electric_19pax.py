"""
Conceptual Design of a Hybrid-Electric 19-Passenger Turboprop Aircraft
======================================================================

Uses AeroSandbox's optimization framework (Opti) with:
- AeroBuildup for aerodynamic analysis
- Raymer cargo/transport weight estimation + PEGASUS wing weight surrogate
- Turboshaft + electric motor parallel hybrid propulsion model
- Wingtip-mounted propellers with 15% propulsive efficiency bonus
- Torenbeek field length analysis

Requirements:
    - 19 passengers, 6000 lb payload
    - 200 kt cruise speed at 7000 ft
    - 2 parallel hybrid-electric turboprops with wingtip propellers
    - 2600 ft takeoff and landing distance
    - 350 nmi max range, optimized for 175 nmi typical mission

Architecture: Parallel hybrid -- turboshaft and electric motor both
drive the same propeller shaft via a combining gearbox. Electric boost
during takeoff/climb, turboshaft-only during cruise. Propellers are
mounted at the wingtips, providing a 15% propulsive efficiency bonus
from wingtip vortex energy recovery. Wing weight uses the PEGASUS
surrogate model which accounts for outboard engine bending relief.
"""

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

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
    """
    Estimate wing structural weight of the PEGASUS vehicle configuration for
    integration with vehicle sizing tools like FLOPS or LEAPS.

    All inputs in imperial units; returns total wing weight in lbs.

    :param wing_area: Planform area (Wimpress method), sq. ft.
    :param wing_ar: Aspect ratio (b^2 / S)
    :param wing_taper: Taper ratio (c_t / c_r)
    :param wing_af_thickness: Airfoil thickness-to-chord ratio
    :param mtow: Maximum takeoff weight, lbs
    :param battery_weight_ratio: Battery weight / MTOW
    :param engine_inboard_weight: Inboard engine weight, lbs
    :param engine_inboard_eta: Inboard engine normalized span location
    :param engine_outboard_weight: Outboard engine weight, lbs
    :param engine_outboard_eta: Outboard engine normalized span location
    :return: Total wing structural weight, lbs
    :rtype: float
    """
    estimate = (
        (-763.654116272738)
        + 66.6533463708713 * wing_ar
        + 436.788492704573 * wing_taper
        + 1.21715502571785 * wing_area
        + -2950.25309234943 * wing_af_thickness
        + 0.0163144157953134 * mtow
        + -35.3096443123149 * battery_weight_ratio
        + -0.0230719745324687 * engine_inboard_weight
        + -152.537262155881 * engine_inboard_eta
        + -0.078580212183211 * engine_outboard_weight
        + -217.296864650889 * engine_outboard_eta
        + (wing_ar - 11.089463482063) * ((wing_ar - 11.089463482063) * 3.32515028103463)
        + (wing_ar - 11.089463482063) * ((wing_taper - 0.549316967122807) * 80.0810304968525)
        + (wing_ar - 11.089463482063) * ((wing_area - 578.352162689163) * 0.0679869849011377)
        + (wing_ar - 11.089463482063) * ((wing_af_thickness - 0.140046114226006) * -632.721055431577)
        + (wing_ar - 11.089463482063) * ((mtow - 45067.1269825181) * 0.00235424149828414)
        + (wing_ar - 11.089463482063) * ((battery_weight_ratio - 0.304891299969045) * -14.6950965049355)
        + (wing_ar - 11.089463482063) * ((engine_inboard_weight - 901.045007855934) * -0.0045676649394544)
        + (wing_ar - 11.089463482063) * ((engine_inboard_eta - 0.375467167887513) * -31.7948952878013)
        + (wing_ar - 11.089463482063) * ((engine_outboard_weight - 1798.4504628483) * -0.0130605611336943)
        + (wing_ar - 11.089463482063) * ((engine_outboard_eta - 0.844749104111456) * -49.7660905560111)
        + (wing_taper - 0.549316967122807) * ((wing_taper - 0.549316967122807) * 208.043958071801)
        + (wing_taper - 0.549316967122807) * ((wing_area - 578.352162689163) * 0.399566153569777)
        + (wing_taper - 0.549316967122807) * ((wing_af_thickness - 0.140046114226006) * -4681.22845517446)
        + (wing_taper - 0.549316967122807) * ((mtow - 45067.1269825181) * 0.0148168755374293)
        + (wing_taper - 0.549316967122807) * ((battery_weight_ratio - 0.304891299969045) * -19.1906778245744)
        + (wing_taper - 0.549316967122807) * ((engine_inboard_weight - 901.045007855934) * 0.0170718428289706)
        + (wing_taper - 0.549316967122807) * ((engine_inboard_eta - 0.375467167887513) * -135.424552441368)
        + (wing_taper - 0.549316967122807) * ((engine_outboard_weight - 1798.4504628483) * -0.057278134745046)
        + (wing_taper - 0.549316967122807) * ((engine_outboard_eta - 0.844749104111456) * -174.466282246233)
        + (wing_area - 578.352162689163) * ((wing_area - 578.352162689163) * -0.000188690713945717)
        + (wing_area - 578.352162689163) * ((wing_af_thickness - 0.140046114226006) * -2.30265260237845)
        + (wing_area - 578.352162689163) * ((mtow - 45067.1269825181) * 0.0000151058790758956)
        + (wing_area - 578.352162689163) * ((battery_weight_ratio - 0.304891299969045) * 0.410596598562827)
        + (wing_area - 578.352162689163) * ((engine_inboard_weight - 901.045007855934) * -0.0000233386805120355)
        + (wing_area - 578.352162689163) * ((engine_inboard_eta - 0.375467167887513) * -0.0881905271304493)
        + (wing_area - 578.352162689163) * ((engine_outboard_weight - 1798.4504628483) * -0.0000546839233665841)
        + (wing_area - 578.352162689163) * ((engine_outboard_eta - 0.844749104111456) * -0.305520724576287)
        + (wing_af_thickness - 0.140046114226006) * ((wing_af_thickness - 0.140046114226006) * 36135.7252036525)
        + (wing_af_thickness - 0.140046114226006) * ((mtow - 45067.1269825181) * -0.129022665411152)
        + (wing_af_thickness - 0.140046114226006) * ((battery_weight_ratio - 0.304891299969045) * -156.216624212617)
        + (wing_af_thickness - 0.140046114226006) * ((engine_inboard_weight - 901.045007855934) * 0.455452785051827)
        + (wing_af_thickness - 0.140046114226006) * ((engine_inboard_eta - 0.375467167887513) * 1651.39734196954)
        + (wing_af_thickness - 0.140046114226006) * ((engine_outboard_weight - 1798.4504628483) * 0.437340687962727)
        + (wing_af_thickness - 0.140046114226006) * ((engine_outboard_eta - 0.844749104111456) * 1793.27103669289)
        + (mtow - 45067.1269825181) * ((mtow - 45067.1269825181) * 0.000000075413179298)
        + (mtow - 45067.1269825181) * ((battery_weight_ratio - 0.304891299969045) * -0.000406171183132496)
        + (mtow - 45067.1269825181) * ((engine_inboard_weight - 901.045007855934) * -0.0000003900624341104)
        + (mtow - 45067.1269825181) * ((engine_inboard_eta - 0.375467167887513) * -0.00296434684811031)
        + (mtow - 45067.1269825181) * ((engine_outboard_weight - 1798.4504628483) * -0.0000010065281736412)
        + (mtow - 45067.1269825181) * ((engine_outboard_eta - 0.844749104111456) * -0.00230199459363745)
        + (battery_weight_ratio - 0.304891299969045) * ((battery_weight_ratio - 0.304891299969045) * 136.188540688368)
        + (battery_weight_ratio - 0.304891299969045) * ((engine_inboard_weight - 901.045007855934) * 0.123471075162514)
        + (battery_weight_ratio - 0.304891299969045) * ((engine_inboard_eta - 0.375467167887513) * 90.7460647792041)
        + (battery_weight_ratio - 0.304891299969045) * ((engine_outboard_weight - 1798.4504628483) * 0.0198872946728822)
        + (battery_weight_ratio - 0.304891299969045) * ((engine_outboard_eta - 0.844749104111456) * 55.0181153025222)
        + (engine_inboard_weight - 901.045007855934) * ((engine_inboard_weight - 901.045007855934) * 0.0000236389247774922)
        + (engine_inboard_weight - 901.045007855934) * ((engine_inboard_eta - 0.375467167887513) * -0.09370310791469)
        + (engine_inboard_weight - 901.045007855934) * ((engine_outboard_weight - 1798.4504628483) * 0.0000166647330951795)
        + (engine_inboard_weight - 901.045007855934) * ((engine_outboard_eta - 0.844749104111456) * 0.0410735845461825)
        + (engine_inboard_eta - 0.375467167887513) * ((engine_inboard_eta - 0.375467167887513) * -236.984105322582)
        + (engine_inboard_eta - 0.375467167887513) * ((engine_outboard_weight - 1798.4504628483) * 0.0401404174385669)
        + (engine_inboard_eta - 0.375467167887513) * ((engine_outboard_eta - 0.844749104111456) * -141.822798401132)
        + (engine_outboard_weight - 1798.4504628483) * ((engine_outboard_weight - 1798.4504628483) * 0.0000168826355194513)
        + (engine_outboard_weight - 1798.4504628483) * ((engine_outboard_eta - 0.844749104111456) * -0.0816642024504058)
        + (engine_outboard_eta - 0.844749104111456) * ((engine_outboard_eta - 0.844749104111456) * 487.31877454063)
    )
    return estimate


##### Section: Mission Constants #####

n_pax = 19
n_crew = 2
payload_mass = 6000 * u.lbm              # 2722 kg
cruise_speed = 200 * u.knot              # 102.9 m/s
cruise_altitude = 7000 * u.foot          # 2134 m
field_length_req = 2600 * u.foot         # 792.5 m
n_engines = 2
design_range_max = 350 * u.naut_mile      # 648 km, max range (fuel sizing)
design_range_typical = 175 * u.naut_mile  # 324 km, typical mission (80% of flights)
ultimate_load_factor = 1.5 * 3.0         # FAR 23 commuter
CL_max = 2.2                             # With flaps, high wing
g = 9.81

# Wingtip propeller efficiency bonus (vortex energy recovery)
wingtip_propeller_efficiency_bonus = 1.15  # 15% propulsive efficiency gain

# Fuel properties (Jet-A)
fuel_density = 820          # kg/m^3
fuel_specific_energy = 43.02e6  # J/kg, lower heating value

# Battery properties
battery_cell_specific_energy = 350   # Wh/kg at cell level
battery_pack_cell_fraction = 0.70    # Pack-level derating
battery_max_dod = 0.80               # Max depth of discharge

# Fuselage geometry (fixed, Beech 1900D class)
fuse_length = 17.6             # m total
fuse_cabin_width = 1.37        # m external width
fuse_cabin_height = 1.47       # m external height
nose_length = 2.7              # m
cabin_length = 8.5             # m (19 pax, 2+1 abreast, ~0.76 m pitch)
tail_length = fuse_length - nose_length - cabin_length  # ~6.4 m

# Tail geometry (fixed)
tail_arm = 9.0                 # m, wing AC to tail AC
hstab_span_val = 6.1           # m
hstab_root_chord_val = 1.5     # m
hstab_tip_chord_val = 0.75     # m
vstab_span_val = 2.5           # m
vstab_root_chord_val = 2.2     # m
vstab_tip_chord_val = 1.1      # m

##### Section: Optimization Setup #####

opti = asb.Opti()

##### Section: Design Variables #####

design_mass_TOGW = opti.variable(
    init_guess=7700, log_transform=True, lower_bound=4000, upper_bound=12000
)
wing_span = opti.variable(
    init_guess=17.7, lower_bound=12, upper_bound=25
)
wing_root_chord = opti.variable(
    init_guess=2.6, lower_bound=1.5, upper_bound=4.0
)
cruise_alpha = opti.variable(
    init_guess=3.0, lower_bound=-2, upper_bound=10
)
mass_turboshaft_per_engine = opti.variable(
    init_guess=130, log_transform=True, lower_bound=50, upper_bound=300
)
propeller_diameter = opti.variable(
    init_guess=2.8, lower_bound=2.0, upper_bound=4.0
)
hybridization_factor = opti.variable(
    init_guess=0.25, lower_bound=0.10, upper_bound=0.50
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

# Derived wing geometry
wing_taper_ratio = 0.45
wing_tip_chord = wing_root_chord * wing_taper_ratio
wing_mean_chord = (wing_root_chord + wing_tip_chord) / 2

##### Section: Aircraft Geometry #####

# --- Fuselage ---
fuse = asb.Fuselage(
    name="Fuselage",
    xsecs=[
        asb.FuselageXSec(  # Nose tip
            xyz_c=[0, 0, 0],
            width=0, height=0,
        ),
        asb.FuselageXSec(  # Nose midpoint
            xyz_c=[nose_length * 0.5, 0, -0.05],
            width=fuse_cabin_width * 0.7,
            height=fuse_cabin_height * 0.7,
            shape=2.5,
        ),
        asb.FuselageXSec(  # Cabin start
            xyz_c=[nose_length, 0, 0],
            width=fuse_cabin_width,
            height=fuse_cabin_height,
            shape=2.5,
        ),
        asb.FuselageXSec(  # Cabin end
            xyz_c=[nose_length + cabin_length, 0, 0],
            width=fuse_cabin_width,
            height=fuse_cabin_height,
            shape=2.5,
        ),
        asb.FuselageXSec(  # Tail mid
            xyz_c=[nose_length + cabin_length + tail_length * 0.6, 0, 0.25],
            width=fuse_cabin_width * 0.5,
            height=fuse_cabin_height * 0.5,
            shape=2.0,
        ),
        asb.FuselageXSec(  # Tail tip
            xyz_c=[fuse_length, 0, 0.5],
            width=0.15,
            height=0.15,
        ),
    ],
)

# --- Main Wing (HIGH wing) ---
wing_x_le = 0.40 * fuse_length - 0.25 * wing_root_chord
wing_z_le = 0.5 * fuse_cabin_height  # High wing: on top of fuselage

flap = asb.ControlSurface(
    name="Flap", symmetric=True, deflection=0, hinge_point=0.75
)
aileron = asb.ControlSurface(
    name="Aileron", symmetric=False, deflection=0, hinge_point=0.75
)

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(  # Root
            xyz_le=[0, 0, 0],
            chord=wing_root_chord,
            twist=2,
            airfoil=asb.Airfoil("naca23018"),
            control_surfaces=[flap],
        ),
        asb.WingXSec(  # Kink at 60% span
            xyz_le=[
                0.6 * wing_span / 2 * np.tand(3),
                0.6 * wing_span / 2,
                0.6 * wing_span / 2 * np.tand(2),
            ],
            chord=wing_root_chord * 0.70,
            twist=0,
            airfoil=asb.Airfoil("naca23015"),
            control_surfaces=[aileron],
        ),
        asb.WingXSec(  # Tip
            xyz_le=[
                wing_span / 2 * np.tand(3),
                wing_span / 2,
                wing_span / 2 * np.tand(2),
            ],
            chord=wing_tip_chord,
            twist=-1,
            airfoil=asb.Airfoil("naca23012"),
        ),
    ],
).translate([wing_x_le, 0, wing_z_le])

# --- Horizontal Stabilizer ---
elevator = asb.ControlSurface(
    name="Elevator", symmetric=True, deflection=0, hinge_point=0.70
)

hstab_x_le = wing_x_le + 0.25 * wing_root_chord + tail_arm - 0.25 * hstab_root_chord_val
hstab_z_le = 0.3

hstab = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=hstab_root_chord_val,
            airfoil=asb.Airfoil("naca0012"),
            control_surfaces=[elevator],
        ),
        asb.WingXSec(
            xyz_le=[
                hstab_span_val / 2 * np.tand(10),
                hstab_span_val / 2,
                0,
            ],
            chord=hstab_tip_chord_val,
            airfoil=asb.Airfoil("naca0010"),
        ),
    ],
).translate([hstab_x_le, 0, hstab_z_le])

# --- Vertical Stabilizer ---
rudder = asb.ControlSurface(
    name="Rudder", symmetric=True, deflection=0, hinge_point=0.70
)

vstab_x_le = hstab_x_le - 0.5
vstab_z_le = 0.3

vstab = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_root_chord_val,
            airfoil=asb.Airfoil("naca0012"),
            control_surfaces=[rudder],
        ),
        asb.WingXSec(
            xyz_le=[
                vstab_span_val * np.tand(30),
                0,
                vstab_span_val,
            ],
            chord=vstab_tip_chord_val,
            airfoil=asb.Airfoil("naca0010"),
        ),
    ],
).translate([vstab_x_le, 0, vstab_z_le])

# --- Assemble Airplane ---
airplane = asb.Airplane(
    name="HE-19 Hybrid Electric Turboprop",
    xyz_ref=[wing_x_le + 0.25 * wing_root_chord, 0, 0],
    wings=[wing, hstab, vstab],
    fuselages=[fuse],
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

# Wing area from geometry
wing_area = wing.area()

# --- Miscellaneous Drag Correction ---
# AeroBuildup omits protuberance drag (antennas, air scoops, door gaps,
# cooling inlets/exits, gear fairings, exhaust stacks, etc.) and
# underestimates interference drag for non-streamlined commuter
# fuselages.
#
# Raymer Table 12.6: typical CDA_misc = 0.10-0.30 m^2 for turboprop
# commuters. We use 0.20 m^2 for a 19-pax commuter with fixed gear
# fairings, dual exhaust, antennas, and cooling openings.
CDA_misc = 0.20  # m^2 flat-plate equivalent drag area

# Interference / form-factor correction (fuselage-wing junction,
# fuselage-nacelle junction, surface roughness exceedances, non-ideal
# fuselage fineness ratio for a stubby commuter fuselage).
drag_correction_factor = 1.10  # 10% increase on AeroBuildup base drag

CD_misc = CDA_misc / wing_area
CL_cruise = aero["CL"]
CD_cruise = aero["CD"] * drag_correction_factor + CD_misc
L_over_D_cruise = CL_cruise / CD_cruise

q_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2
drag_cruise = CD_cruise * q_cruise * wing_area

##### Section: Weight Breakdown #####

# -- Structural --

# Wing weight computed below (after propulsion section, needs motor/battery masses)

# Horizontal stabilizer (Raymer)
wing_to_hstab_distance = tail_arm
m_hstab = raymer_wt.mass_hstab(
    hstab=hstab,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    wing_to_hstab_distance=wing_to_hstab_distance,
    fuselage_width_at_hstab_intersection=fuse_cabin_width,
)

# Vertical stabilizer (Raymer)
wing_to_vstab_distance = tail_arm * 0.95
m_vstab = raymer_wt.mass_vstab(
    vstab=vstab,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    wing_to_vstab_distance=wing_to_vstab_distance,
)

# Fuselage (Raymer)
m_fuselage = raymer_wt.mass_fuselage(
    fuselage=fuse,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    L_over_D=L_over_D_cruise,
    main_wing=wing,
    n_cargo_doors=1,
    landing_gear_mounted_on_fuselage=False,
)

# Landing gear
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
)

m_nlg = raymer_wt.mass_nose_landing_gear(
    nose_gear_length=0.6,
    design_mass_TOGW=design_mass_TOGW,
    n_wheels=2,
)

# Nacelles
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

# -- Systems --

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

# Fuel system
fuel_volume = fuel_mass / fuel_density
m_fuel_system = raymer_wt.mass_fuel_system(
    fuel_volume=fuel_volume,
    n_tanks=2,
    fraction_in_integral_tanks=1.0,
)

# -- Propulsion System (Hybrid-Electric) --

# Turboshaft
power_per_turboshaft = power_turboshaft(mass_turboshaft_per_engine)
m_turboshaft_total = mass_turboshaft_per_engine * n_engines

# Electric motor: sized as a fraction of total takeoff power
# total_takeoff_power = turboshaft_power + electric_power
# hybridization_factor = electric_power / total_takeoff_power
# So: electric_power = hybridization_factor / (1 - hybridization_factor) * turboshaft_power
electric_power_per_engine = (
    hybridization_factor / (1 - hybridization_factor) * power_per_turboshaft
)

m_motor_per_engine = mass_motor_electric(electric_power_per_engine, method="hobbyking")
m_motor_total = m_motor_per_engine * n_engines

m_esc_per_engine = mass_ESC(electric_power_per_engine)
m_esc_total = m_esc_per_engine * n_engines

# Battery
m_battery = mass_battery_pack(
    battery_capacity_Wh,
    battery_cell_specific_energy_Wh_kg=battery_cell_specific_energy,
    battery_pack_cell_fraction=battery_pack_cell_fraction,
)

# Propellers (Torenbeek)
total_power_per_propeller = power_per_turboshaft + electric_power_per_engine
m_propeller_each = torenbeek_wt.mass_propeller(
    propeller_diameter=propeller_diameter,
    propeller_power=total_power_per_propeller,
    n_blades=4,
)
m_propellers_total = m_propeller_each * n_engines

# Gearbox
turboshaft_output_rpm = 33000
propeller_rpm = 1700
m_gearbox_each = mass_gearbox(
    power=total_power_per_propeller,
    rpm_in=turboshaft_output_rpm,
    rpm_out=propeller_rpm,
)
m_gearbox_total = m_gearbox_each * n_engines

# Wing (PEGASUS surrogate model -- accounts for wingtip engine bending relief)
wing_aspect_ratio = wing_span ** 2 / wing_area
outboard_engine_weight_per_side = (m_motor_per_engine + m_esc_per_engine + m_propeller_each) / u.lbm

m_wing = wing_weight_pegasus(
    wing_area=wing_area / u.foot ** 2,
    wing_ar=wing_aspect_ratio,
    wing_taper=wing_taper_ratio,
    wing_af_thickness=0.18,
    mtow=design_mass_TOGW / u.lbm,
    battery_weight_ratio=m_battery / design_mass_TOGW,
    engine_inboard_weight=0.,
    engine_inboard_eta=0.,
    engine_outboard_weight=outboard_engine_weight_per_side,
    engine_outboard_eta=0.95,
) * u.lbm  # convert lbs -> kg

# -- Payload / Cabin --
m_pax = n_pax * mass_passenger
m_seats = n_pax * mass_seat("passenger") + n_crew * mass_seat("flight_deck")
m_lavs = mass_lavatories(n_pax, aircraft_type="short-haul")

# Flight controls (simplified: 2% MTOW typical for this class)
m_flight_controls = 0.02 * design_mass_TOGW

##### Section: Total Weight #####

mass_empty = (
    # Structure
    m_wing + m_hstab + m_vstab + m_fuselage
    + m_mlg + m_nlg + m_nacelles
    # Propulsion
    + m_turboshaft_total + m_motor_total + m_esc_total
    + m_propellers_total + m_gearbox_total
    + m_fuel_system
    # Systems
    + m_instruments + m_electrical + m_furnishings
    + m_ac + m_anti_ice + m_flight_controls
    # Cabin equipment
    + m_seats + m_lavs
)

mass_total = mass_empty + payload_mass + fuel_mass + m_battery

##### Section: Propulsion and Performance #####

# --- Cruise Power Balance (turboshaft only) ---
propulsive_area = n_engines * np.pi / 4 * propeller_diameter ** 2

shaft_power_cruise_total = propeller_shaft_power_from_thrust(
    thrust_force=drag_cruise,
    area_propulsive=propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=0.85 * wingtip_propeller_efficiency_bonus,
)

shaft_power_cruise_per_engine = shaft_power_cruise_total / n_engines

# Turboshaft must handle all cruise power
opti.subject_to(power_per_turboshaft >= shaft_power_cruise_per_engine * 1.05)

# Cruise throttle and fuel consumption
throttle_cruise = shaft_power_cruise_per_engine / power_per_turboshaft

eta_thermal_cruise = thermal_efficiency_turboshaft(
    mass_turboshaft=mass_turboshaft_per_engine,
    throttle_setting=throttle_cruise,
)

fuel_burn_rate_cruise = shaft_power_cruise_total / (
    eta_thermal_cruise * fuel_specific_energy
)

# 45-min VFR reserves
fuel_reserve_time = 45 * 60
fuel_reserves = fuel_burn_rate_cruise * fuel_reserve_time

# Climb fuel estimate (~10 min at higher power)
climb_time = 10 * 60  # seconds
climb_fuel_factor = 1.3  # Higher burn rate during climb
fuel_for_climb = fuel_burn_rate_cruise * climb_fuel_factor * climb_time

# Fuel for max range mission (350 nmi) -- sizes fuel tanks and MTOW
cruise_time_max = design_range_max / cruise_speed
fuel_for_cruise_max = fuel_burn_rate_cruise * cruise_time_max

opti.subject_to(fuel_mass >= fuel_for_cruise_max + fuel_reserves + fuel_for_climb)

# Fuel for typical mission (175 nmi) -- used for cruise optimization
cruise_time_typical = design_range_typical / cruise_speed
fuel_for_cruise_typical = fuel_burn_rate_cruise * cruise_time_typical
fuel_mass_typical = fuel_for_cruise_typical + fuel_reserves + fuel_for_climb

# --- Takeoff Power (Hybrid Boost) ---
total_power_takeoff_per_engine = power_per_turboshaft + electric_power_per_engine
shaft_power_takeoff_total = total_power_takeoff_per_engine * n_engines

V_liftoff = 1.2 * V_stall_sl

shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=thrust_at_liftoff,
    area_propulsive=propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=0.80 * wingtip_propeller_efficiency_bonus,
)

opti.subject_to(shaft_power_takeoff_total >= shaft_power_from_thrust_liftoff)

# --- Battery Sizing ---
electric_energy_for_climb = electric_power_per_engine * n_engines * climb_time  # Joules
electric_energy_for_climb_Wh = electric_energy_for_climb / 3600

opti.subject_to(battery_capacity_Wh >= electric_energy_for_climb_Wh / battery_max_dod)

##### Section: Constraints #####

# --- Wing Geometry Constraints ---
opti.subject_to(wing_aspect_ratio >= 8.0)   # Practical minimum for turboprop
opti.subject_to(wing_aspect_ratio <= 14.0)  # Practical maximum

# --- Cruise Lift = Weight (optimized for typical 175 nmi mission) ---
# On a typical mission the aircraft takes off lighter than MTOW because
# it carries only fuel_mass_typical instead of the full fuel_mass.
# Mid-cruise weight accounts for half the typical cruise fuel burned.
lift_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2 * wing_area * CL_cruise
typical_mission_TOGW = design_mass_TOGW - (fuel_mass - fuel_mass_typical)
mid_cruise_weight = (typical_mission_TOGW - fuel_for_cruise_typical * 0.5) * g

opti.subject_to(lift_cruise >= mid_cruise_weight * 0.99)
opti.subject_to(lift_cruise <= mid_cruise_weight * 1.01)

# --- Field Length ---
L_over_D_climb = L_over_D_cruise * 0.65  # Reduced L/D in takeoff config

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

# --- OEI Climb Gradient (FAR 23, 2 engines) ---
opti.subject_to(field_results["flight_path_angle_climb_one_engine_out"] >= 0.024)

# --- Mass Closure ---
opti.subject_to(mass_total <= design_mass_TOGW)

# --- MTOW Limit (FAR 23 commuter category: 19,000 lb) ---
opti.subject_to(design_mass_TOGW <= 19000 * u.lbm)

##### Section: Objective #####

# Minimize fuel burn on the typical 175 nmi mission (80% of flights).
# This favors aerodynamic efficiency (higher L/D) over minimum structure
# weight, unlike a min-MTOW objective which penalizes wing size.
opti.minimize(fuel_mass_typical)

##### Section: Solve #####

sol = opti.solve(max_iter=500)

##### Section: Results Summary #####

print("=" * 72)
print("   HE-19 HYBRID-ELECTRIC 19-PAX TURBOPROP -- DESIGN SUMMARY")
print("=" * 72)

# Extract solved values
TOGW = sol(design_mass_TOGW)
m_empty_sol = sol(mass_empty)
m_fuel_sol = sol(fuel_mass)
m_batt_sol = sol(m_battery)
b = sol(wing_span)
c_root = sol(wing_root_chord)
S_wing = sol(wing_area)
AR = b ** 2 / S_wing

print(f"\n{'--- Overall ---':^72}")
print(f"  MTOW:                    {TOGW:8.0f} kg  ({TOGW / u.lbm:8.0f} lb)")
print(f"  Empty Weight:            {m_empty_sol:8.0f} kg  ({m_empty_sol / u.lbm:8.0f} lb)")
print(f"  Payload:                 {payload_mass:8.0f} kg  ({payload_mass / u.lbm:8.0f} lb)")
print(f"  Fuel Weight:             {m_fuel_sol:8.0f} kg  ({m_fuel_sol / u.lbm:8.0f} lb)")
print(f"  Battery Weight:          {m_batt_sol:8.0f} kg  ({m_batt_sol / u.lbm:8.0f} lb)")
print(f"  Useful Load:             {payload_mass + m_fuel_sol + m_batt_sol:8.0f} kg")

print(f"\n{'--- Geometry ---':^72}")
print(f"  Wing Span:               {b:8.2f} m   ({b / u.foot:8.1f} ft)")
print(f"  Wing Area:               {S_wing:8.1f} m^2 ({S_wing / u.foot**2:8.0f} ft^2)")
print(f"  Aspect Ratio:            {AR:8.2f}")
print(f"  Root Chord:              {c_root:8.2f} m")
print(f"  Tip Chord:               {sol(wing_tip_chord):8.2f} m")
print(f"  Taper Ratio:             {wing_taper_ratio:8.2f}")
print(f"  Fuselage Length:         {fuse_length:8.1f} m   ({fuse_length / u.foot:8.1f} ft)")
print(f"  Wing Loading:            {TOGW * g / S_wing:8.1f} N/m^2 ({TOGW * g / S_wing / (u.lbf / u.foot**2):8.1f} psf)")

print(f"\n{'--- Aerodynamics (Cruise @ {:.0f} ft) ---'.format(cruise_altitude / u.foot):^72}")
print(f"  CL:                      {sol(CL_cruise):8.4f}")
print(f"  CD:                      {sol(CD_cruise):8.5f}")
print(f"  L/D:                     {sol(L_over_D_cruise):8.1f}")
print(f"  Alpha:                   {sol(cruise_alpha):8.1f} deg")
print(f"  Cruise Drag:             {sol(drag_cruise):8.0f} N   ({sol(drag_cruise) / u.lbf:8.0f} lbf)")

print(f"\n{'--- Propulsion (Parallel Hybrid) ---':^72}")
print(f"  Turboshaft Mass (each):  {sol(mass_turboshaft_per_engine):8.1f} kg  ({sol(mass_turboshaft_per_engine) / u.lbm:8.0f} lb)")
print(f"  Turboshaft Power (each): {sol(power_per_turboshaft) / u.horsepower:8.0f} hp  ({sol(power_per_turboshaft) / 1000:8.0f} kW)")
print(f"  Turboshaft Power (total):{sol(power_per_turboshaft) * n_engines / u.horsepower:8.0f} hp")
print(f"  Electric Motor (each):   {sol(electric_power_per_engine) / 1000:8.0f} kW  ({sol(electric_power_per_engine) / u.horsepower:8.0f} hp)")
print(f"  Hybridization Factor:    {sol(hybridization_factor):8.1%}")
print(f"  Total TO Power (both):   {sol(shaft_power_takeoff_total) / u.horsepower:8.0f} hp")
print(f"  Propeller Diameter:      {sol(propeller_diameter):8.2f} m   ({sol(propeller_diameter) / u.foot:8.1f} ft)")
print(f"  Battery Capacity:        {sol(battery_capacity_Wh) / 1000:8.1f} kWh")
print(f"  Battery Mass:            {m_batt_sol:8.1f} kg  ({m_batt_sol / u.lbm:8.0f} lb)")

print(f"\n{'--- Aerodynamic Drag Breakdown ---':^72}")
CD_aerobuildup = sol(aero["CD"])
CD_total = sol(CD_cruise)
CD_misc_sol = sol(CD_misc)
print(f"  AeroBuildup CD (raw):    {CD_aerobuildup:8.5f}")
print(f"  After 10% correction:    {CD_aerobuildup * 1.10:8.5f}")
print(f"  Misc drag CD (CDA/S):    {CD_misc_sol:8.5f}  (CDA_misc = {CDA_misc:.2f} m^2)")
print(f"  Total CD (corrected):    {CD_total:8.5f}")

print(f"\n{'--- Cruise Performance ---':^72}")
print(f"  Cruise Shaft Power:      {sol(shaft_power_cruise_total) / u.horsepower:8.0f} hp")
print(f"  Cruise Throttle:         {sol(throttle_cruise):8.1%}")
print(f"  Thermal Efficiency:      {sol(eta_thermal_cruise):8.1%}")
print(f"  Cruise Fuel Burn:        {sol(fuel_burn_rate_cruise) * 3600:8.1f} kg/hr")
print(f"  Fuel for Max Range:      {sol(fuel_for_cruise_max):8.0f} kg  (350 nmi)")
print(f"  Fuel for Typical Range:  {sol(fuel_for_cruise_typical):8.0f} kg  (175 nmi)")
print(f"  Fuel Reserves (45 min):  {sol(fuel_reserves):8.0f} kg")
print(f"  Fuel for Climb:          {sol(fuel_for_climb):8.0f} kg")
print(f"  Total Fuel (max range):  {sol(fuel_mass):8.0f} kg")

print(f"\n{'--- Field Performance ---':^72}")
print(f"  V_stall (SL):            {sol(field_results['V_stall']) / u.knot:8.1f} kts ({sol(field_results['V_stall']):8.1f} m/s)")
print(f"  V_liftoff:               {sol(field_results['V_liftoff']) / u.knot:8.1f} kts")
print(f"  Takeoff Ground Roll:     {sol(field_results['takeoff_ground_roll_distance']) / u.foot:8.0f} ft  ({sol(field_results['takeoff_ground_roll_distance']):8.0f} m)")
print(f"  Takeoff Total Distance:  {sol(field_results['takeoff_total_distance']) / u.foot:8.0f} ft  ({sol(field_results['takeoff_total_distance']):8.0f} m)")
print(f"  Balanced Field Length:   {sol(field_results['balanced_field_length']) / u.foot:8.0f} ft  ({sol(field_results['balanced_field_length']):8.0f} m)")
print(f"  Landing Total Distance:  {sol(field_results['landing_total_distance']) / u.foot:8.0f} ft  ({sol(field_results['landing_total_distance']):8.0f} m)")
print(f"  Climb Gradient (AEO):    {sol(field_results['flight_path_angle_climb']):8.4f} rad ({sol(field_results['flight_path_angle_climb']) * 100:8.2f}%)")
print(f"  Climb Gradient (OEI):    {sol(field_results['flight_path_angle_climb_one_engine_out']):8.4f} rad ({sol(field_results['flight_path_angle_climb_one_engine_out']) * 100:8.2f}%)")
print(f"  Thrust/Weight (TO):      {sol(thrust_at_liftoff) / (TOGW * g):8.3f}")

print(f"\n{'--- Weight Breakdown ---':^72}")
print(f"  {'Component':<28} {'Mass (kg)':>10} {'Mass (lb)':>10} {'% MTOW':>8}")
print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")

weight_items = [
    ("Wing", sol(m_wing)),
    ("H-Stab", sol(m_hstab)),
    ("V-Stab", sol(m_vstab)),
    ("Fuselage", sol(m_fuselage)),
    ("Main Landing Gear", sol(m_mlg)),
    ("Nose Landing Gear", sol(m_nlg)),
    ("Nacelles", sol(m_nacelles)),
    ("Turboshaft Engines", sol(m_turboshaft_total)),
    ("Electric Motors", sol(m_motor_total)),
    ("ESCs", sol(m_esc_total)),
    ("Propellers", sol(m_propellers_total)),
    ("Gearboxes", sol(m_gearbox_total)),
    ("Fuel System", sol(m_fuel_system)),
    ("Instruments", sol(m_instruments)),
    ("Electrical System", sol(m_electrical)),
    ("Furnishings", sol(m_furnishings)),
    ("Air Conditioning", sol(m_ac)),
    ("Anti-Ice", sol(m_anti_ice)),
    ("Flight Controls", sol(m_flight_controls)),
    ("Seats", sol(m_seats)),
    ("Lavatories", sol(m_lavs)),
]

struct_total = 0
for name, mass_val in weight_items:
    print(f"  {name:<28} {mass_val:10.1f} {mass_val / u.lbm:10.1f} {mass_val / TOGW * 100:7.1f}%")
    struct_total += mass_val

print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")
print(f"  {'EMPTY WEIGHT':<28} {m_empty_sol:10.1f} {m_empty_sol / u.lbm:10.1f} {m_empty_sol / TOGW * 100:7.1f}%")
print(f"  {'Payload':<28} {payload_mass:10.1f} {payload_mass / u.lbm:10.1f} {payload_mass / TOGW * 100:7.1f}%")
print(f"  {'Fuel':<28} {m_fuel_sol:10.1f} {m_fuel_sol / u.lbm:10.1f} {m_fuel_sol / TOGW * 100:7.1f}%")
print(f"  {'Battery':<28} {m_batt_sol:10.1f} {m_batt_sol / u.lbm:10.1f} {m_batt_sol / TOGW * 100:7.1f}%")
print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")
print(f"  {'MTOW':<28} {TOGW:10.1f} {TOGW / u.lbm:10.1f} {100:7.1f}%")
print(f"\n  Empty Weight Fraction:   {m_empty_sol / TOGW:.3f}")
print(f"  Fuel Fraction:           {m_fuel_sol / TOGW:.3f}")
print(f"  Battery Fraction:        {m_batt_sol / TOGW:.3f}")
print(f"  Payload Fraction:        {payload_mass / TOGW:.3f}")

print(f"\n{'--- Mission Profile ---':^72}")
print(f"  Typical Mission:         {design_range_typical / u.naut_mile:.0f} nmi ({design_range_typical / 1000:.0f} km)")
print(f"  Max Range Mission:       {design_range_max / u.naut_mile:.0f} nmi ({design_range_max / 1000:.0f} km)")
print(f"  Typical Takeoff Weight:  {sol(typical_mission_TOGW):.0f} kg ({sol(typical_mission_TOGW) / u.lbm:.0f} lb)")
print(f"  Max Takeoff Weight:      {TOGW:.0f} kg ({TOGW / u.lbm:.0f} lb)")

print(f"\n{'--- Active Constraints ---':^72}")
print(f"  BFL:       {sol(field_results['balanced_field_length']) / u.foot:.0f} ft vs {field_length_req / u.foot:.0f} ft limit")
print(f"  TO dist:   {sol(field_results['takeoff_total_distance']) / u.foot:.0f} ft vs {field_length_req / u.foot:.0f} ft limit")
print(f"  LDG dist:  {sol(field_results['landing_total_distance']) / u.foot:.0f} ft vs {field_length_req / u.foot:.0f} ft limit")
print(f"  OEI grad:  {sol(field_results['flight_path_angle_climb_one_engine_out']):.4f} vs 0.024 min")
print(f"  AR:        {AR:.2f} vs [8.0, 14.0] bounds")
print(f"  Hybrid:    {sol(hybridization_factor):.1%} vs [10%, 50%] bounds")

print("\n" + "=" * 72)

# --- Optional: Draw 3-view ---
try:
    sol_airplane = sol(airplane)
    sol_airplane.draw_three_view()
except Exception:
    pass  # Skip drawing if display not available
