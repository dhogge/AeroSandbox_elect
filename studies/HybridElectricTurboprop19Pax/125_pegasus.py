"""
Conceptual Design of a Hybrid-Electric 19-Passenger Turboprop Aircraft
======================================================================

Uses AeroSandbox's optimization framework (Opti) with:
- AeroBuildup for aerodynamic analysis
- Raymer cargo/transport weight estimation + PEGASUS wing weight surrogate
- Turboshaft + electric motor parallel hybrid propulsion model
- Wingtip-mounted propellers with 15% propulsive efficiency bonus
- Fully-electric BLI (Boundary Layer Ingestion) tail pusher prop
- Torenbeek field length analysis

Req
    - 19 passengers, 6000 lb payload
    - 200 kt cruise speed at 7000 ft
    - 2 parallel hybrid-electric turboprops with wingtip propellers
    - 1 fully-electric BLI pusher prop at tail (centerline)
    - 2600 ft takeoff and landing distance
    - 350 nmi max range, optimized for 175 nmi typical mission

Architecture: Parallel hybrid -- turboshaft and electric motor both
drive the same propeller shaft via a combining gearbox. Electric boost
during takeoff/climb, turboshaft-only during cruise. Propellers are
mounted at the wingtips, providing a 15% propulsive efficiency bonus
from wingtip vortex energy recovery. A third propulsor -- a fully-
electric tail-mounted pusher -- ingests the fuselage boundary layer
(BLI), reducing effective aircraft drag by 10%. The BLI pusher
operates in all flight phases and shares the battery with the wingtip
electric motors. Two additional small electric rotors are mounted at
half-span on each wing. These mid-span rotors are fully electric
(powered from the battery during takeoff/climb and generated
electricity from the turboshafts). They contribute thrust during
takeoff and climb, then fold/stow for cruise to eliminate drag.
Because the mid-span rotors can provide differential thrust during
OEI, they reduce the yaw moment the vertical stabilizer must
counter, allowing a significantly smaller vertical tail.
During cruise, the turboshafts generate electricity for the BLI
motor via the wingtip motors acting as generators, so the battery
only needs to cover takeoff/climb boost (not cruise BLI energy).
Wing weight uses the PEGASUS surrogate model which accounts for
outboard engine bending relief.
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
    estimate = (-763.654116272738) + 66.6533463708713 * wing_ar + 436.788492704573 * wing_taper +1.21715502571785 * wing_area + -2950.25309234943 * wing_af_thickness + 0.0163144157953134 * mtow + -35.3096443123149 * battery_weight_ratio + -0.0230719745324687 * engine_inboard_weight + -152.537262155881 * engine_inboard_eta + -0.078580212183211 * engine_outboard_weight + -217.296864650889 * engine_outboard_eta + (wing_ar - 11.089463482063) * ((wing_ar -11.089463482063) * 3.32515028103463) + (wing_ar - 11.089463482063) * ((wing_taper -0.549316967122807) * 80.0810304968525) + (wing_ar - 11.089463482063) * ((wing_area -578.352162689163) * 0.0679869849011377) + (wing_ar - 11.089463482063) * (( wing_af_thickness - 0.140046114226006) * -632.721055431577) + (wing_ar - 11.089463482063) * ((mtow - 45067.1269825181) * 0.00235424149828414) + (wing_ar - 11.089463482063) * ((battery_weight_ratio - 0.304891299969045) * -14.6950965049355) + (wing_ar -11.089463482063) * ((engine_inboard_weight - 901.045007855934) * -0.0045676649394544) + ( wing_ar - 11.089463482063) * ((engine_inboard_eta - 0.375467167887513) * -31.7948952878013) + (wing_ar - 11.089463482063) * ((engine_outboard_weight - 1798.4504628483) * -0.0130605611336943) + (wing_ar - 11.089463482063) * ((engine_outboard_eta -0.844749104111456) * -49.7660905560111) + (wing_taper - 0.549316967122807) * ((wing_taper - 0.549316967122807) * 208.043958071801) + (wing_taper - 0.549316967122807) * ((wing_area - 578.352162689163) * 0.399566153569777) + (wing_taper - 0.549316967122807) * (( wing_af_thickness - 0.140046114226006) * -4681.22845517446) + (wing_taper - 0.549316967122807) * ((mtow - 45067.1269825181) * 0.0148168755374293) + (wing_taper - 0.549316967122807) * ((battery_weight_ratio - 0.304891299969045) * -19.1906778245744) + (wing_taper -0.549316967122807) * ((engine_inboard_weight - 901.045007855934) * 0.0170718428289706) + ( wing_taper - 0.549316967122807) * ((engine_inboard_eta - 0.375467167887513) * -135.424552441368 ) + (wing_taper - 0.549316967122807) * ((engine_outboard_weight - 1798.4504628483) * -0.057278134745046) + (wing_taper - 0.549316967122807) * ((engine_outboard_eta -0.844749104111456) * -174.466282246233) + (wing_area - 578.352162689163) * ((wing_area - 578.352162689163) * -0.000188690713945717) + (wing_area - 578.352162689163) * (( wing_af_thickness - 0.140046114226006) * -2.30265260237845) + (wing_area - 578.352162689163) * ((mtow - 45067.1269825181) * 0.0000151058790758956) + (wing_area - 578.352162689163 ) * ((battery_weight_ratio - 0.304891299969045) * 0.410596598562827) + (wing_area -578.352162689163) * ((engine_inboard_weight - 901.045007855934) * -0.0000233386805120355) + ( wing_area - 578.352162689163) * ((engine_inboard_eta - 0.375467167887513) * -0.0881905271304493) + (wing_area - 578.352162689163) * ((engine_outboard_weight - 1798.4504628483 ) * -0.0000546839233665841) + (wing_area - 578.352162689163) * ((engine_outboard_eta -0.844749104111456) * -0.305520724576287) + (wing_af_thickness - 0.140046114226006) * (( wing_af_thickness - 0.140046114226006) * 36135.7252036525) + (wing_af_thickness -0.140046114226006) * ((mtow - 45067.1269825181) * -0.129022665411152) + ( wing_af_thickness - 0.140046114226006) * ((battery_weight_ratio - 0.304891299969045) * -156.216624212617) + (wing_af_thickness - 0.140046114226006) * ((engine_inboard_weight -901.045007855934) * 0.455452785051827) + (wing_af_thickness - 0.140046114226006) * (( engine_inboard_eta - 0.375467167887513) * 1651.39734196954) + (wing_af_thickness -0.140046114226006) * ((engine_outboard_weight - 1798.4504628483) * 0.437340687962727) + ( wing_af_thickness - 0.140046114226006) * ((engine_outboard_eta - 0.844749104111456) * 1793.27103669289) + (mtow - 45067.1269825181) * ((mtow - 45067.1269825181) * 0.000000075413179298) + (mtow - 45067.1269825181) * ((battery_weight_ratio -0.304891299969045) * -0.000406171183132496) + (mtow - 45067.1269825181) * (( engine_inboard_weight - 901.045007855934) * -0.0000003900624341104) + (mtow - 45067.1269825181) * ((engine_inboard_eta - 0.375467167887513) * -0.00296434684811031) + (mtow -45067.1269825181) * ((engine_outboard_weight - 1798.4504628483) * -0.0000010065281736412) + ( mtow - 45067.1269825181) * ((engine_outboard_eta - 0.844749104111456) * -0.00230199459363745) + (battery_weight_ratio - 0.304891299969045) * ((battery_weight_ratio -0.304891299969045) * 136.188540688368) + (battery_weight_ratio - 0.304891299969045) * (( engine_inboard_weight - 901.045007855934) * 0.123471075162514) + (battery_weight_ratio -0.304891299969045) * ((engine_inboard_eta - 0.375467167887513) * 90.7460647792041) + ( battery_weight_ratio - 0.304891299969045) * ((engine_outboard_weight - 1798.4504628483) * 0.0198872946728822) + (battery_weight_ratio - 0.304891299969045) * ((engine_outboard_eta -0.844749104111456) * 55.0181153025222) + (engine_inboard_weight - 901.045007855934) * (( engine_inboard_weight - 901.045007855934) * 0.0000236389247774922) + (engine_inboard_weight -901.045007855934) * ((engine_inboard_eta - 0.375467167887513) * -0.09370310791469) + ( engine_inboard_weight - 901.045007855934) * ((engine_outboard_weight - 1798.4504628483) * 0.0000166647330951795) + (engine_inboard_weight - 901.045007855934) * ((engine_outboard_eta -0.844749104111456) * 0.0410735845461825) + (engine_inboard_eta - 0.375467167887513) * (( engine_inboard_eta - 0.375467167887513) * -236.984105322582) + (engine_inboard_eta -0.375467167887513) * ((engine_outboard_weight - 1798.4504628483) * 0.0401404174385669) + ( engine_inboard_eta - 0.375467167887513) * ((engine_outboard_eta - 0.844749104111456) * -141.822798401132) + (engine_outboard_weight - 1798.4504628483) * ((engine_outboard_weight -1798.4504628483) * 0.0000168826355194513) + (engine_outboard_weight - 1798.4504628483) * (( engine_outboard_eta - 0.844749104111456) * -0.0816642024504058) + (engine_outboard_eta -0.844749104111456) * ((engine_outboard_eta - 0.844749104111456) * 487.31877454063)
    return estimate


##### Section: Mission Constants #####

n_pax = 14
n_crew = 2
payload_mass = 3500 * u.lbm              # 2722 kg
cruise_speed = 200 * u.knot              # 102.9 m/s
# cruise_altitude = 7000 * u.foot          # 2134 m
field_length_req = 2600 * u.foot         # 792.5 m
n_engines = 2
design_range_max = 350 * u.naut_mile      # 648 km, max range (fuel sizing)
design_range_typical = 175 * u.naut_mile  # (80% of flights)
ultimate_load_factor = 1.5 * 3.0         # FAR 23 commuter
CL_max = 2.4                             # With flaps, high wing
g = 9.81

# Wingtip propeller efficiency bonus (vortex energy recovery)
wingtip_propeller_efficiency_bonus = 1.15  # 15% propulsive efficiency gain

# Generator efficiency: wingtip motors act as generators during cruise?????????
# to power the BLI motor from the turboshaft (shaft → generator → BLI motor)
generator_efficiency = 0.93  # Motor as generator

# Fuel properties (Jet-A)
fuel_density = 820          # kg/m^3
fuel_specific_energy = 43.02e6  # J/kg

# Battery properties
battery_cell_specific_energy = 350   # Wh/kg at cell level
battery_pack_cell_fraction = 0.70    # Pack-level derating
battery_max_dod = 0.80               # Max depth of discharge

# BLI (Boundary Layer Ingestion) pusher prop properties
bli_drag_reduction_factor = 0.10     # 10% effective drag reduction from BLI wake ingestion
bli_propeller_CoP = 0.80             # Baseline CoP for BLI tail pusher (smaller prop, aft install)

# Fuselage geometry 
fuse_length = 14             # m total
fuse_cabin_width = 1.9        # m external width
fuse_cabin_height = 1.85      # m external height
nose_length = 2.5             
cabin_length = 7.1             
tail_length = fuse_length - nose_length - cabin_length  

# Tail geometry
tail_arm = 7.0                 # m, wing AC to tail AC

##### Section: Optimization Setup #####

opti = asb.Opti()

##### Section: Design Variables #####

cruise_altitude = opti.variable(
    init_guess=7000 * u.foot, lower_bound=3000 * u.foot, upper_bound=10000 * u.foot
)

design_mass_TOGW = opti.variable(
    init_guess=7700, log_transform=True, lower_bound=3000, upper_bound=12500
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
    init_guess=2.8, lower_bound=2.0, upper_bound=4.0
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

# BLI electric pusher prop design variables
bli_motor_power = opti.variable(
    init_guess=150000, log_transform=True, lower_bound=20000, upper_bound=500000
)  # Watts, rated electric motor power for BLI pusher
bli_propeller_diameter = opti.variable(
    init_guess=1.2, lower_bound=0.6, upper_bound=2.0
)  # meters, BLI pusher prop diameter (constrained by tail cone)
bli_thrust_at_liftoff = opti.variable(
    init_guess=2000, lower_bound=100, log_transform=True
)  # Newtons, BLI thrust contribution at liftoff

# Mid-span electric rotor design variables (2 rotors, one per wing half)
# These are small fully-electric rotors at 50% span. They contribute
# thrust during takeoff and climb, then fold/stow for cruise (zero drag).
# Powered by battery during climb and by turboshaft-generated electricity.
# Differential thrust in OEI helps reduce V_mc → smaller vertical tail.
n_midspan_rotors = 2  # One per wing half
midspan_motor_power = opti.variable(
    init_guess=80000, log_transform=True, lower_bound=10000, upper_bound=300000
)  # Watts, rated power per mid-span electric motor
midspan_propeller_diameter = opti.variable(
    init_guess=1.0, lower_bound=0.5, upper_bound=1.8
)  # meters, mid-span rotor diameter (must fit chord)
midspan_thrust_at_liftoff = opti.variable(
    init_guess=1500, lower_bound=100, log_transform=True
)  # Newtons, total mid-span thrust (both rotors) at liftoff
midspan_eta = 0.50  # Normalized spanwise location (half-span)

# Cabin floor, goes whever best for cg
x_cg_battery = opti.variable(
    init_guess=nose_length + 0.25 * cabin_length,
    lower_bound=nose_length + 0.5, 
    upper_bound=nose_length + cabin_length - 0.5,      
)  # from nose

# OEI operative-wingtip reduced thrust (optimizer can throttle back
# the surviving wingtip to cut yaw moment → smaller vtail)
thrust_wingtip_oei_reduced = opti.variable(
    init_guess=5000, lower_bound=100, log_transform=True
)  # Newtons – thrust the operative wingtip is set to during OEI

# Vertical stabilizer sizing (engine-out directional control)
vstab_span_val = opti.variable(
    init_guess=2.5, lower_bound=1.5, upper_bound=6.0
)
vstab_root_chord_val = opti.variable(
    init_guess=2.2, lower_bound=1.2, upper_bound=5
)
vstab_taper_ratio = 0.5
vstab_tip_chord_val = vstab_root_chord_val * vstab_taper_ratio
vstab_area = (vstab_root_chord_val + vstab_tip_chord_val) / 2 * vstab_span_val

# Horizontal stabilizer sizing (Longitudinal stability via Volume Coefficient)
# Optimization variables for hstab
hstab_area = opti.variable(
    init_guess=8.0, lower_bound=3.0, upper_bound=15.0
)
hstab_aspect_ratio = opti.variable(
    init_guess=4.0, lower_bound=3.0, upper_bound=6.0
)
hstab_taper_ratio = 0.5

# Derived hstab geometry
hstab_span_val = (hstab_area * hstab_aspect_ratio) ** 0.5
hstab_root_chord_val = 2 * hstab_area / (hstab_span_val * (1 + hstab_taper_ratio))
hstab_tip_chord_val = hstab_root_chord_val * hstab_taper_ratio

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
        asb.FuselageXSec(  # Tail tip (widened for BLI pusher prop spinner)
            xyz_c=[fuse_length, 0, 0.5],
            width=0.40,
            height=0.40,
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

# --- Vertical Stabilizer ---
rudder = asb.ControlSurface(
    name="Rudder", symmetric=True, deflection=0, hinge_point=0.70
)

# Vstab root LE: place so that mid-chord is at tail_arm from wing AC
vstab_x_le = wing_x_le + 0.25 * wing_root_chord + tail_arm - vstab_root_chord_val * 0.5
vstab_z_le = 0.3

# T-tail configuration: hstab sits on top of vstab
hstab_z_le = vstab_z_le + vstab_span_val

# Hstab root quarter-chord aligned with vstab tip quarter-chord
vstab_tip_le_x = vstab_x_le + vstab_span_val * np.tand(30)
hstab_x_le = vstab_tip_le_x + 0.25 * vstab_tip_chord_val - 0.25 * hstab_root_chord_val

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

# --- BLI Pusher Propulsor (tail-mounted, centerline) ---
bli_propulsor = asb.Propulsor(
    name="BLI Pusher",
    xyz_c=[fuse_length + 0.3, 0, 0.5],     # Just aft of fuselage tail tip
    xyz_normal=[1, 0, 0],                   # Thrust in +x (aft-facing pusher)
    radius=bli_propeller_diameter / 2,
    length=0.3,
)

# --- Mid-Span Electric Rotors (2×, one per wing half, stowed during cruise) ---
# These are small tractor props at 50% span on each wing. They fold flat
# against the nacelle during cruise so they add zero drag. They are placed
# at the leading edge of the wing at the half-span station.
midspan_y = midspan_eta * wing_span / 2
midspan_x = wing_x_le + midspan_eta * wing_span / 2 * np.tand(3)   # LE sweep
midspan_z = wing_z_le + midspan_eta * wing_span / 2 * np.tand(2)   # dihedral

midspan_rotor_left = asb.Propulsor(
    name="Mid-Span Rotor L",
    xyz_c=[midspan_x - 0.3, midspan_y, midspan_z],   # Slightly ahead of LE
    xyz_normal=[1, 0, 0],
    radius=midspan_propeller_diameter / 2,
    length=0.25,
)
midspan_rotor_right = asb.Propulsor(
    name="Mid-Span Rotor R",
    xyz_c=[midspan_x - 0.3, -midspan_y, midspan_z],
    xyz_normal=[1, 0, 0],
    radius=midspan_propeller_diameter / 2,
    length=0.25,
)

# --- Assemble Airplane ---
airplane = asb.Airplane(
    name="HE-19 Hybrid Electric Turboprop + BLI + Mid-Span Rotors",
    xyz_ref=[wing_x_le + 0.25 * wing_root_chord, 0, 0],
    wings=[wing, hstab, vstab],
    fuselages=[fuse],
    propulsors=[bli_propulsor, midspan_rotor_left, midspan_rotor_right],
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

# --- BLI Drag Reduction ---
# BLI ingests fuselage boundary layer wake → reduces effective aircraft
# drag by bli_drag_reduction_factor. The wingtip props only need to
# overcome the reduced effective drag. The BLI prop must run at cruise
# to fill the wake deficit, consuming battery power.
drag_effective_cruise = drag_cruise * (1 - bli_drag_reduction_factor)  # What wingtip props must overcome
drag_bli_wake_fill = drag_cruise * bli_drag_reduction_factor           # What BLI prop must fill

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
# Size now governed by V_mc constraint; no blanket scale factor.
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
) * 0.7  # Fixed gear: no retraction mechanism, ~70% of retractable weight

m_nlg = raymer_wt.mass_nose_landing_gear(
    nose_gear_length=0.6,
    design_mass_TOGW=design_mass_TOGW,
    n_wheels=2,
) * 0.7  # Fixed gear

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

# Propellers (Torenbeek) — ×0.35 for lightweight composite wingtip electric propellers
total_power_per_propeller = power_per_turboshaft + electric_power_per_engine
m_propeller_each = torenbeek_wt.mass_propeller(
    propeller_diameter=propeller_diameter,
    propeller_power=total_power_per_propeller,
    n_blades=4,
) * 0.35
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

# -- BLI Pusher Propulsion (fully electric, tail-mounted) --
m_bli_motor = mass_motor_electric(bli_motor_power, method="hobbyking")
m_bli_esc = mass_ESC(bli_motor_power)
m_bli_propeller = torenbeek_wt.mass_propeller(
    propeller_diameter=bli_propeller_diameter,
    propeller_power=bli_motor_power,
    n_blades=5,  # 5-blade BLI fan typical
) * 0.35  # Lightweight composite, same factor as wingtip props
m_bli_nacelle = 0.10 * m_bli_motor + 15  # Lightweight tail fairing + spinner

# -- Mid-Span Electric Rotors (fully electric, stowed during cruise) --
# Each rotor has: motor, ESC, folding propeller, nacelle/pylon, stow mechanism
m_midspan_motor_each = mass_motor_electric(midspan_motor_power, method="hobbyking")
m_midspan_motor_total = m_midspan_motor_each * n_midspan_rotors
m_midspan_esc_each = mass_ESC(midspan_motor_power)
m_midspan_esc_total = m_midspan_esc_each * n_midspan_rotors
m_midspan_propeller_each = torenbeek_wt.mass_propeller(
    propeller_diameter=midspan_propeller_diameter,
    propeller_power=midspan_motor_power,
    n_blades=3,  # Lightweight 3-blade folding prop
) * 0.30  # Composite folding blades, lighter than standard props
m_midspan_propeller_total = m_midspan_propeller_each * n_midspan_rotors
# Nacelle/pylon + fold/stow mechanism (actuator, latch, fairings)
m_midspan_nacelle_each = 0.10 * m_midspan_motor_each + 8  # Small nacelle
m_midspan_stow_each = 5.0  # kg, fold/stow actuator and mechanism per rotor
m_midspan_installed_each = (
    m_midspan_motor_each + m_midspan_esc_each + m_midspan_propeller_each
    + m_midspan_nacelle_each + m_midspan_stow_each
)
m_midspan_total = m_midspan_installed_each * n_midspan_rotors

# Wing (Torenbeek method -- more accurate than PEGASUS surrogate for this class)
# k_e = 0.90: wingtip-mounted propulsors provide significant bending relief
# (more than standard underwing engines at k_e=0.95, but these are relatively
# light motors rather than full turbofans, so not as aggressive as k_e=0.85).
wing_aspect_ratio = wing_span ** 2 / wing_area

# V_NE ~ 1.4× cruise for FAR 23 commuter turboprops
V_NE = cruise_speed * 1.40
# V_flap ~ 1.8× stall speed (sea level)
V_flap = V_stall_sl * 1.8

# Suspended mass: everything the wing carries (total mass minus wing itself)
# Use design TOGW as an approximation (wing mass is a small fraction)
suspended_mass_approx = design_mass_TOGW * 0.92  # ~8% wing weight fraction

m_wing_basic = torenbeek_wt.mass_wing_basic_structure(
    wing=wing,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    suspended_mass=suspended_mass_approx,
    never_exceed_airspeed=V_NE,
    main_gear_mounted_to_wing=False,
    k_e=0.90,  # Wingtip propulsors bending relief
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
    # Wingtip propulsion (hybrid)
    + m_turboshaft_total + m_motor_total + m_esc_total
    + m_propellers_total + m_gearbox_total
    + m_fuel_system
    # BLI pusher propulsion (electric)
    + m_bli_motor + m_bli_esc + m_bli_propeller + m_bli_nacelle
    # Mid-span electric rotors (stowed during cruise)
    + m_midspan_total
    # Systems
    + m_instruments + m_electrical + m_furnishings
    + m_ac + m_anti_ice + m_flight_controls
    # Cabin equipment
    + m_seats + m_lavs
)

mass_total = mass_empty + payload_mass + fuel_mass + m_battery

##### Section: CG, Neutral Point & Static Margin #####

#  Mean Aerodynamic Chord (MAC) for trapezoidal wing 
wing_MAC = (2 / 3) * wing_root_chord * (
    1 + wing_taper_ratio + wing_taper_ratio ** 2
) / (1 + wing_taper_ratio)

# Spanwise station & LE x-position of MAC
wing_sweep_LE = 3  # degrees (from wing geometry definition)
y_MAC = (wing_span / 6) * (1 + 2 * wing_taper_ratio) / (1 + wing_taper_ratio)
x_MAC_le = wing_x_le + y_MAC * np.tand(wing_sweep_LE)

# Wing aerodynamic center (quarter-chord of MAC)
x_ac_wing = x_MAC_le + 0.25 * wing_MAC

#  Component CG x-positions (x = 0 at nose, positive aft) 
x_cg_wing = x_MAC_le + 0.40 * wing_MAC
x_cg_hstab = hstab_x_le + 0.42 * (hstab_root_chord_val + hstab_tip_chord_val) / 2
x_cg_vstab = vstab_x_le + 0.42 * (vstab_root_chord_val + vstab_tip_chord_val) / 2
x_cg_fuselage = 0.45 * fuse_length
x_cg_mlg = wing_x_le + 0.55 * wing_mean_chord
x_cg_nlg = 0.80 * nose_length

# Wingtip-mounted propulsion (at wing tip LE + offset into nacelle)
x_wingtip_le = wing_x_le + (wing_span / 2) * np.tand(wing_sweep_LE)
x_cg_nacelles = x_wingtip_le + 0.30 * wing_tip_chord
x_cg_turboshaft = x_cg_nacelles
x_cg_motors_wingtip = x_cg_nacelles
x_cg_esc_wingtip = x_cg_nacelles - 0.2
x_cg_propellers = x_cg_nacelles - 0.4
x_cg_gearbox = x_cg_nacelles

# BLI tail-mounted propulsion
x_cg_bli_motor = fuse_length - 0.2
x_cg_bli_esc = fuse_length - 0.5
x_cg_bli_prop = fuse_length + 0.3
x_cg_bli_nacelle = fuse_length + 0.1

# Mid-span electric rotors (at half-span on each wing)
x_cg_midspan = midspan_x  # Already computed in geometry section

# Fuel, battery, payload, systems
x_cg_fuel = x_MAC_le + 0.35 * wing_MAC               # Integral wing tanks
# x_cg_battery is now an optimization variable (defined in Design Variables section)
x_cg_fuel_system = x_cg_fuel
x_cg_payload = nose_length + 0.50 * cabin_length      # Center of cabin
x_cg_instruments = 0.15 * fuse_length                  # Cockpit area
x_cg_electrical = 0.40 * fuse_length
x_cg_furnishings = nose_length + 0.50 * cabin_length
x_cg_aircon = nose_length + 0.25 * cabin_length
x_cg_anti_ice = wing_x_le                              # Wing LE area
x_cg_flight_controls = 0.40 * fuse_length
x_cg_seats = nose_length + 0.50 * cabin_length
x_cg_lavs = nose_length + cabin_length - 0.5

#  CG at MTOW (full fuel, full payload, all items) 
moment_TOGW = (
    # Structure
    m_wing * x_cg_wing
    + m_hstab * x_cg_hstab
    + m_vstab * x_cg_vstab
    + m_fuselage * x_cg_fuselage
    + m_mlg * x_cg_mlg
    + m_nlg * x_cg_nlg
    + m_nacelles * x_cg_nacelles
    # Wingtip propulsion
    + m_turboshaft_total * x_cg_turboshaft
    + m_motor_total * x_cg_motors_wingtip
    + m_esc_total * x_cg_esc_wingtip
    + m_propellers_total * x_cg_propellers
    + m_gearbox_total * x_cg_gearbox
    # BLI propulsion
    + m_bli_motor * x_cg_bli_motor
    + m_bli_esc * x_cg_bli_esc
    + m_bli_propeller * x_cg_bli_prop
    + m_bli_nacelle * x_cg_bli_nacelle
    # Mid-span electric rotors
    + m_midspan_total * x_cg_midspan
    # Systems
    + m_fuel_system * x_cg_fuel_system
    + m_instruments * x_cg_instruments
    + m_electrical * x_cg_electrical
    + m_furnishings * x_cg_furnishings
    + m_ac * x_cg_aircon
    + m_anti_ice * x_cg_anti_ice
    + m_flight_controls * x_cg_flight_controls
    + m_seats * x_cg_seats
    + m_lavs * x_cg_lavs
    # Operating items
    + fuel_mass * x_cg_fuel
    + m_battery * x_cg_battery
    + payload_mass * x_cg_payload
)
x_cg_TOGW = moment_TOGW / mass_total

#  CG at aft-loading (fuel depleted = zero-fuel weight) 
# Most-aft CG: critical case for longitudinal stability.
mass_zfw = mass_total - fuel_mass
x_cg_aft = (moment_TOGW - fuel_mass * x_cg_fuel) / mass_zfw

#  Neutral Point (stick-fixed) 
# Lift-curve slopes via Helmbold equation
a_w = 2 * np.pi * wing_aspect_ratio / (2 + np.sqrt(4 + wing_aspect_ratio ** 2))
a_h = 2 * np.pi * hstab_aspect_ratio / (2 + np.sqrt(4 + hstab_aspect_ratio ** 2))

# Downwash gradient at hstab (T-tail: 15% less downwash than conventional)
depsilon_dalpha = 2 * a_w / (np.pi * wing_aspect_ratio) * 0.85

# Dynamic pressure ratio at hstab (T-tail above wing wake)
eta_h = 0.90

# Hstab AC (quarter-chord of mean chord)
x_ac_hstab = hstab_x_le + 0.25 * (hstab_root_chord_val + hstab_tip_chord_val) / 2

# Wing-tail neutral point (exact linearised solution, Etkin formulation)
tail_lift_effectiveness = a_h * eta_h * (1 - depsilon_dalpha) * hstab_area
wing_lift_effectiveness = a_w * wing_area
x_np_wing_tail = (
    wing_lift_effectiveness * x_ac_wing
    + tail_lift_effectiveness * x_ac_hstab
) / (wing_lift_effectiveness + tail_lift_effectiveness)

# Fuselage destabilising shift (Munk-Multhopp, moves NP forward)
# K_f combines (k2-k1), taper correction, and unit conversion
K_f = 0.92
fuselage_width_eff = fuse_cabin_width * 0.85  # Effective average width
delta_x_np_fuse = K_f * fuselage_width_eff ** 2 * fuse_length / wing_lift_effectiveness
x_np = x_np_wing_tail - delta_x_np_fuse

#  Static Margin 
# SM > 0 => statically stable (nose-down moment with alpha increase)
static_margin = (x_np - x_cg_aft) / wing_MAC        # At aft CG (critical for stability)
static_margin_TOGW = (x_np - x_cg_TOGW) / wing_MAC  # At fwd CG (critical for trim)

##### Section: Propulsion and Performance #####

# --- Cruise Power Balance ---
# Wingtip props handle effective drag (after BLI reduction)
# BLI prop fills the wake deficit
wingtip_propulsive_area = n_engines * np.pi / 4 * propeller_diameter ** 2
bli_propulsive_area = np.pi / 4 * bli_propeller_diameter ** 2

# Wingtip shaft power (turboshaft-only at cruise)
shaft_power_cruise_wingtip = propeller_shaft_power_from_thrust(
    thrust_force=drag_effective_cruise,
    area_propulsive=wingtip_propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=0.85 * wingtip_propeller_efficiency_bonus,
)

# BLI shaft power (electric motor, fills wake deficit)
shaft_power_cruise_bli = propeller_shaft_power_from_thrust(
    thrust_force=drag_bli_wake_fill,
    area_propulsive=bli_propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=bli_propeller_CoP,
)

# BLI motor must handle cruise wake-filling power
opti.subject_to(bli_motor_power >= shaft_power_cruise_bli * 1.05)

# Turboshaft must handle wingtip cruise power PLUS BLI electric power.
# During cruise the wingtip motors act as generators: turboshaft drives
# the propeller AND siphons extra shaft power through the motor/generator
# to feed the BLI motor. Losses: generator_efficiency × motor_efficiency.
# The motor efficiency is embedded in the BLI motor's own ESC/motor chain;
# here we account for the generator-side loss on the wingtip motor.
bli_electric_demand_from_turboshaft = shaft_power_cruise_bli / generator_efficiency

shaft_power_cruise_per_engine_total = (
    shaft_power_cruise_wingtip + bli_electric_demand_from_turboshaft
) / n_engines

opti.subject_to(power_per_turboshaft >= shaft_power_cruise_per_engine_total * 1.05)

# Cruise throttle and fuel consumption
# Turboshaft burns fuel for wingtip thrust + BLI generation
shaft_power_cruise_total = shaft_power_cruise_wingtip + bli_electric_demand_from_turboshaft
throttle_cruise = shaft_power_cruise_per_engine_total / power_per_turboshaft

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

# --- Takeoff Power (Hybrid Boost + BLI + Mid-Span Rotors) ---
# Total liftoff thrust = wingtip props + BLI pusher + mid-span rotors
# thrust_at_liftoff is the TOTAL thrust from all propulsors
wingtip_thrust_at_liftoff = thrust_at_liftoff - bli_thrust_at_liftoff - midspan_thrust_at_liftoff

total_power_takeoff_per_engine = power_per_turboshaft + electric_power_per_engine
shaft_power_takeoff_wingtip = total_power_takeoff_per_engine * n_engines

V_liftoff = 1.2 * V_stall_sl

# Wingtip shaft power required at liftoff
wingtip_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=wingtip_thrust_at_liftoff,
    area_propulsive=wingtip_propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=0.80 * wingtip_propeller_efficiency_bonus,
)

opti.subject_to(shaft_power_takeoff_wingtip >= wingtip_shaft_power_from_thrust_liftoff)

# BLI shaft power required at liftoff
bli_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=bli_thrust_at_liftoff,
    area_propulsive=bli_propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=bli_propeller_CoP,
)

opti.subject_to(bli_motor_power >= bli_shaft_power_from_thrust_liftoff)

# Mid-span rotor shaft power required at liftoff
midspan_propulsive_area = n_midspan_rotors * np.pi / 4 * midspan_propeller_diameter ** 2
midspan_CoP = 0.75  # Slightly lower CoP for small folding props

midspan_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=midspan_thrust_at_liftoff,
    area_propulsive=midspan_propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=midspan_CoP,
)

# Each mid-span motor must handle its share of the liftoff power
opti.subject_to(
    midspan_motor_power * n_midspan_rotors >= midspan_shaft_power_from_thrust_liftoff
)

# Mid-span propeller diameter must fit within wing chord at half-span
# (chord at 50% span ≈ 85% of root chord for this taper)
midspan_chord_approx = wing_root_chord * 0.85
opti.subject_to(midspan_propeller_diameter <= midspan_chord_approx * 0.70)

# --- Battery Sizing ---
# During cruise, turboshafts generate BLI electricity via wingtip
# motors-as-generators → battery only needs to cover takeoff/climb:
#   (1) wingtip electric boost during climb
#   (2) BLI motor during climb
#   (3) mid-span rotors during climb (full power, stowed at cruise)
# No cruise BLI energy needed in the battery.
# No mid-span energy needed in cruise (rotors stowed).
electric_energy_wingtip_climb = electric_power_per_engine * n_engines * climb_time  # Joules
electric_energy_bli_climb = bli_motor_power * climb_time                            # Joules
electric_energy_midspan_climb = midspan_motor_power * n_midspan_rotors * climb_time # Joules

total_electric_energy_Wh = (
    electric_energy_wingtip_climb + electric_energy_bli_climb + electric_energy_midspan_climb
) / 3600

opti.subject_to(battery_capacity_Wh >= total_electric_energy_Wh / battery_max_dod)

##### Section: Constraints #####

# --- Wing Geometry Constraints ---
opti.subject_to(wing_aspect_ratio >= 6.0)   # Practical minimum for turboprop
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

# --- OEI Climb Gradient (FAR 23, 2 wingtip engines + BLI + mid-span rotors) ---
# When one wingtip engine fails, the remaining propulsors are:
#   - 1 operative wingtip engine (turboshaft + electric motor)
#   - BLI pusher (fully electric, centerline, always operative)
#   - 2 mid-span rotors (fully electric, both still operative)
# The pilot/FADEC can throttle the operative wingtip DOWN to reduce
# the asymmetric yaw moment, and use DIFFERENTIAL mid-span thrust to
# actively counter the yaw. The BLI pusher carries a larger share of
# OEI thrust. The optimizer chooses the best split that satisfies
# BOTH the climb-gradient AND V_mc constraints simultaneously.
thrust_per_wingtip_engine = wingtip_thrust_at_liftoff / n_engines

# Operative-wingtip thrust may not exceed full per-engine capability
opti.subject_to(thrust_wingtip_oei_reduced <= thrust_per_wingtip_engine)

# Mid-span differential thrust for OEI yaw control:
# The mid-span rotor on the dead-engine side increases thrust, and the
# rotor on the operative side decreases thrust. This creates a yaw
# moment that opposes the asymmetric wingtip moment.
# Let T_ms_each = midspan_thrust_at_liftoff / n_midspan_rotors (per rotor)
# Mid-span differential: ΔT_ms = optimizer variable (how much differential is applied)
midspan_thrust_per_rotor = midspan_thrust_at_liftoff / n_midspan_rotors

# OEI mid-span differential thrust variable: the asymmetric thrust offset
# applied to each mid-span rotor (dead side gets +delta, live side gets -delta)
# Bounded by rotor capability (can't go negative or exceed max)
midspan_oei_differential = opti.variable(
    init_guess=500, lower_bound=0, log_transform=True
)  # Newtons of differential per rotor

opti.subject_to(midspan_oei_differential <= midspan_thrust_per_rotor * 0.95)  # Can't exceed rotor max

# Total OEI thrust (all operative propulsors at their OEI settings)
# Both mid-span rotors produce thrust (they're electric, independent of the
# failed wingtip turboshaft). Total mid-span thrust is unchanged; only the
# left/right split changes for yaw control.
thrust_oei = (
    thrust_wingtip_oei_reduced            # Operative wingtip (reduced)
    + bli_thrust_at_liftoff               # BLI pusher (full, centerline)
    + midspan_thrust_at_liftoff           # Both mid-span (total unchanged)
)
thrust_over_weight_oei = thrust_oei / (design_mass_TOGW * g)
climb_gradient_oei = thrust_over_weight_oei - 1 / L_over_D_climb
opti.subject_to(climb_gradient_oei >= 0.024)  # FAR 23 minimum for 2-engine class

# --- Engine-Out Directional Control (V_mc <= V_stall, FAR 23.149) ---
# Wingtip propellers create a large yaw moment when one engine fails.
# The BLI pusher is on the centerline → zero yaw contribution.
# The mid-span rotors can apply differential thrust to REDUCE the net
# yaw moment that the vertical stabilizer must counter:
#
#   Net yaw moment = T_wingtip_reduced × y_wingtip × 1.10  (windmilling)
#                  - ΔT_midspan × y_midspan × 2            (differential, both rotors)
#
# The factor of 2 on differential: dead-side rotor increases by ΔT while
# live-side decreases by ΔT → net differential moment = 2 × ΔT × y_ms.
#
#   Yaw balance: net_yaw_moment = q_vmc × S_vt × CL_vt × l_vt
#
# Conservative: no credit for 5° bank toward operative engine;
# +10% for windmilling drag on the dead-engine propeller.
y_engine = wing_span / 2                              # Wingtip-mounted engine offset
y_midspan = midspan_eta * wing_span / 2               # Mid-span rotor offset

yaw_moment_wingtip = thrust_wingtip_oei_reduced * y_engine * 1.10  # +10% windmilling
yaw_moment_midspan_counter = midspan_oei_differential * y_midspan * 2  # Differential thrust counters yaw
yaw_moment_oei = yaw_moment_wingtip - yaw_moment_midspan_counter

# Net yaw moment must be non-negative (rudder always aids directional control)
opti.subject_to(yaw_moment_oei >= 0)

CL_vstab_max_rudder = 0.9      # Side-force coeff at max rudder deflection (~25°)
l_vt = tail_arm * 0.95         # CG to vstab aerodynamic center

q_vmc = yaw_moment_oei / (vstab_area * CL_vstab_max_rudder * l_vt)
V_mc = np.sqrt(2 * q_vmc / atmo_sl.density())

opti.subject_to(V_mc <= V_stall_sl)   # FAR 23.149: V_mc must not exceed V_s1

# --- Cruise Engine-Out Directional Control (mid-span rotors STOWED) ---
# During cruise the mid-span rotors are folded and inactive. If a wingtip
# engine fails at cruise, the vertical stabilizer must handle the yaw
# moment with NO mid-span differential thrust assistance.
#
# The surviving wingtip continues at cruise thrust (turboshaft only).
# The BLI pusher is on the centerline → zero yaw contribution.
# We require that at cruise dynamic pressure, the rudder can counter
# the full yaw moment from the surviving engine at cruise power.
#
# Yaw moment:  T_cruise_per_engine × y_engine × 1.10  (windmilling)
# Rudder auth: q_cruise × S_vt × CL_vt_max × l_vt
cruise_thrust_per_wingtip = drag_effective_cruise / n_engines

yaw_moment_cruise_oei = cruise_thrust_per_wingtip * y_engine * 1.10  # +10% windmilling
vstab_yaw_authority_cruise = q_cruise * vstab_area * CL_vstab_max_rudder * l_vt

# Vstab must handle cruise OEI with margin (rudder authority ≥ yaw moment)
opti.subject_to(vstab_yaw_authority_cruise >= yaw_moment_cruise_oei)

# Derived: V_mc at cruise altitude (for reporting only)
q_cruise_oei = yaw_moment_cruise_oei / (vstab_area * CL_vstab_max_rudder * l_vt)
V_mc_cruise = np.sqrt(2 * q_cruise_oei / cruise_atmo.density())

# --- Hstab Sizing (Volume Coefficient) ---
# Horizontal Tail Volume Coefficient (V_h) ensures longitudinal stability and control authority.
# V_h = (S_h * l_h) / (S_w * c_mac)
# Typical values for T-tail turboprops: 0.8 - 1.2 (Raymer Table 6.4)
# We use 0.90 as a minimum for this STOL configuration.
l_h = tail_arm  # Distance from wing AC to hstab AC
V_h_coefficient = (hstab_area * l_h) / (wing_area * wing_mean_chord)
opti.subject_to(V_h_coefficient >= 0.90)

#  Longitudinal Stability (Static Margin) 
# V_h above is a geometric floor; these enforce actual stability accounting
# for component CG locations
opti.subject_to(static_margin >= 0.05)  
opti.subject_to(static_margin_TOGW <= 0.20)      

# --- Vstab Sizing (Volume Coefficient for Directional Stability) ---
# The V_mc constraint only sizes the vstab for engine-out control.
# We also need a minimum vertical tail volume coefficient (V_v) for:
#   - Directional (weathercock) stability
#   - Crosswind landing capability
#   - Dutch roll damping
#
# V_v = (S_v * l_v) / (S_w * b)
# Typical values for single-engine/twin turboprops: 0.04-0.08 (Raymer Table 6.4)
# With mid-span differential thrust providing backup yaw authority, we
# can accept the LOW end of the range. Use 0.04 as the floor — this is
# already a significant reduction from the 0.07 a conventional twin
# turboprop would require.
l_v = tail_arm * 0.95  # CG to vstab aerodynamic center
V_v_coefficient = (vstab_area * l_v) / (wing_area * wing_span)
opti.subject_to(V_v_coefficient >= 0.04)

# --- Vstab Geometry Limits ---
vstab_aspect_ratio = vstab_span_val ** 2 / vstab_area
opti.subject_to(vstab_aspect_ratio >= 1.0)
opti.subject_to(vstab_aspect_ratio <= 2.5)

# --- BLI Propeller Diameter Constraint ---
# BLI prop must fit on the tail cone (bounded by fuselage tail height)
opti.subject_to(bli_propeller_diameter <= fuse_cabin_height * 0.8)

# --- BLI Thrust Sanity ---
# BLI thrust at liftoff must not exceed total liftoff thrust
opti.subject_to(bli_thrust_at_liftoff <= thrust_at_liftoff * 0.60)  # BLI ≤ 60% of total
opti.subject_to(wingtip_thrust_at_liftoff >= 0)  # Wingtips must produce positive thrust

# --- Mid-Span Rotor Thrust Sanity ---
# Mid-span rotors are supplementary; they shouldn't dominate total thrust
opti.subject_to(midspan_thrust_at_liftoff <= thrust_at_liftoff * 0.30)  # Mid-span ≤ 30% of total
opti.subject_to(midspan_thrust_at_liftoff >= 0)

# --- Mass Closure ---
opti.subject_to(mass_total <= design_mass_TOGW)

# --- MTOW Limit (FAR 23 commuter category: 19,000 lb) ---
opti.subject_to(design_mass_TOGW <= 12500 * u.lbm)

##### Section: Objective #####

# Minimize fuel burn on the typical 175 nmi mission (80% of flights).
# This favors aerodynamic efficiency (higher L/D) over minimum structure
# weight, unlike a min-MTOW objective which penalizes wing size.
opti.minimize(fuel_mass_typical)

##### Section: Solve #####

sol = opti.solve(max_iter=1500)

##### Section: Results Summary #####

print("=" * 72)
print("   HE-19 HYBRID-ELECTRIC 19-PAX TURBOPROP + BLI + MID-SPAN ROTORS")
print("              -- DESIGN SUMMARY --")
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
print(f"  Cruise Altitude:         {sol(cruise_altitude) / u.foot:8.0f} ft")


print(f"\n{'--- Geometry ---':^72}")
print(f"  Wing Span:               {b:8.2f} m   ({b / u.foot:8.1f} ft)")
print(f"  Wing Area:               {S_wing:8.1f} m^2 ({S_wing / u.foot**2:8.0f} ft^2)")
print(f"  Aspect Ratio:            {AR:8.2f}")
print(f"  Root Chord:              {c_root:8.2f} m")
print(f"  Tip Chord:               {sol(wing_tip_chord):8.2f} m")
print(f"  Taper Ratio:             {wing_taper_ratio:8.2f}")
print(f"  Fuselage Length:         {fuse_length:8.1f} m   ({fuse_length / u.foot:8.1f} ft)")
print(f"  Wing Loading:            {TOGW * g / S_wing:8.1f} N/m^2 ({TOGW * g / S_wing / (u.lbf / u.foot**2):8.1f} psf)")
print(f"  V-Stab Span:             {sol(vstab_span_val):8.2f} m")
print(f"  V-Stab Root Chord:       {sol(vstab_root_chord_val):8.2f} m")
print(f"  V-Stab Area:             {sol(vstab_area):8.2f} m^2")
print(f"  V-Stab V_v Coeff:        {sol(V_v_coefficient):8.3f}  (min 0.040)")
print(f"  H-Stab Span:             {sol(hstab_span_val):8.2f} m")
print(f"  H-Stab Area:             {sol(hstab_area):8.2f} m^2")
print(f"  H-Stab V_h Coeff:        {sol(V_h_coefficient):8.2f}")

print(f"\n{'--- CG & Stability ---':^72}")
print(f"  Wing MAC:                {sol(wing_MAC):8.2f} m")
print(f"  Wing AC (x):             {sol(x_ac_wing):8.2f} m from nose")
print(f"  CG at TOGW (x):         {sol(x_cg_TOGW):8.2f} m  ({sol((x_cg_TOGW - x_MAC_le) / wing_MAC) * 100:5.1f}% MAC)")
print(f"  CG at ZFW / aft (x):    {sol(x_cg_aft):8.2f} m  ({sol((x_cg_aft - x_MAC_le) / wing_MAC) * 100:5.1f}% MAC)")
print(f"  Neutral Point (x):      {sol(x_np):8.2f} m  ({sol((x_np - x_MAC_le) / wing_MAC) * 100:5.1f}% MAC)")
print(f"  Fuse NP shift (fwd):    {sol(delta_x_np_fuse):8.2f} m")
print(f"  Battery Position (x):   {sol(x_cg_battery):8.2f} m  ({(sol(x_cg_battery) - nose_length) / cabin_length * 100:5.1f}% cabin)")
print(f"  Static Margin (TOGW):   {sol(static_margin_TOGW) * 100:8.1f}% MAC")
print(f"  Static Margin (aft CG): {sol(static_margin) * 100:8.1f}% MAC")

print(f"\n{'--- Aerodynamics (Cruise @ {:.0f} ft) ---'.format(sol(cruise_altitude) / u.foot):^72}")

print(f"  CL:                      {sol(CL_cruise):8.4f}")
print(f"  CD:                      {sol(CD_cruise):8.5f}")
print(f"  L/D:                     {sol(L_over_D_cruise):8.1f}")
print(f"  Alpha:                   {sol(cruise_alpha):8.1f} deg")
print(f"  Cruise Drag:             {sol(drag_cruise):8.0f} N   ({sol(drag_cruise) / u.lbf:8.0f} lbf)")
print(f"  BLI Drag Reduction:      {bli_drag_reduction_factor:8.0%}")
print(f"  Effective Drag (w/ BLI): {sol(drag_effective_cruise):8.0f} N   ({sol(drag_effective_cruise) / u.lbf:8.0f} lbf)")
print(f"  BLI Wake Fill Thrust:    {sol(drag_bli_wake_fill):8.0f} N   ({sol(drag_bli_wake_fill) / u.lbf:8.0f} lbf)")

print(f"\n{'--- Propulsion (Wingtip Parallel Hybrid) ---':^72}")
print(f"  Turboshaft Mass (each):  {sol(mass_turboshaft_per_engine):8.1f} kg  ({sol(mass_turboshaft_per_engine) / u.lbm:8.0f} lb)")
print(f"  Turboshaft Power (each): {sol(power_per_turboshaft) / u.horsepower:8.0f} hp  ({sol(power_per_turboshaft) / 1000:8.0f} kW)")
print(f"  Turboshaft Power (total):{sol(power_per_turboshaft) * n_engines / u.horsepower:8.0f} hp")
print(f"  Electric Motor (each):   {sol(electric_power_per_engine) / 1000:8.0f} kW  ({sol(electric_power_per_engine) / u.horsepower:8.0f} hp)")
print(f"  Hybridization Factor:    {sol(hybridization_factor):8.1%}")
print(f"  Wingtip TO Power (both): {sol(shaft_power_takeoff_wingtip) / u.horsepower:8.0f} hp")
print(f"  Propeller Diameter:      {sol(propeller_diameter):8.2f} m   ({sol(propeller_diameter) / u.foot:8.1f} ft)")

print(f"\n{'--- Propulsion (BLI Electric Pusher) ---':^72}")
print(f"  BLI Motor Power:         {sol(bli_motor_power) / 1000:8.0f} kW  ({sol(bli_motor_power) / u.horsepower:8.0f} hp)")
print(f"  BLI Motor Mass:          {sol(m_bli_motor):8.1f} kg  ({sol(m_bli_motor) / u.lbm:8.0f} lb)")
print(f"  BLI ESC Mass:            {sol(m_bli_esc):8.1f} kg  ({sol(m_bli_esc) / u.lbm:8.0f} lb)")
print(f"  BLI Propeller Diameter:  {sol(bli_propeller_diameter):8.2f} m   ({sol(bli_propeller_diameter) / u.foot:8.1f} ft)")
print(f"  BLI Propeller Mass:      {sol(m_bli_propeller):8.1f} kg  ({sol(m_bli_propeller) / u.lbm:8.0f} lb)")
print(f"  BLI Nacelle Mass:        {sol(m_bli_nacelle):8.1f} kg  ({sol(m_bli_nacelle) / u.lbm:8.0f} lb)")
print(f"  BLI Thrust at Liftoff:   {sol(bli_thrust_at_liftoff):8.0f} N   ({sol(bli_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  BLI Cruise Shaft Power:  {sol(shaft_power_cruise_bli) / 1000:8.1f} kW  ({sol(shaft_power_cruise_bli) / u.horsepower:8.0f} hp)")

print(f"\n{'--- Propulsion (Mid-Span Electric Rotors, stowed in cruise) ---':^72}")
print(f"  Number of Rotors:        {n_midspan_rotors:8d}")
print(f"  Motor Power (each):      {sol(midspan_motor_power) / 1000:8.0f} kW  ({sol(midspan_motor_power) / u.horsepower:8.0f} hp)")
print(f"  Motor Mass (each):       {sol(m_midspan_motor_each):8.1f} kg  ({sol(m_midspan_motor_each) / u.lbm:8.0f} lb)")
print(f"  ESC Mass (each):         {sol(m_midspan_esc_each):8.1f} kg  ({sol(m_midspan_esc_each) / u.lbm:8.0f} lb)")
print(f"  Propeller Diameter:      {sol(midspan_propeller_diameter):8.2f} m   ({sol(midspan_propeller_diameter) / u.foot:8.1f} ft)")
print(f"  Propeller Mass (each):   {sol(m_midspan_propeller_each):8.1f} kg  ({sol(m_midspan_propeller_each) / u.lbm:8.0f} lb)")
print(f"  Installed Mass (each):   {sol(m_midspan_installed_each):8.1f} kg  ({sol(m_midspan_installed_each) / u.lbm:8.0f} lb)")
print(f"  Total Mid-Span Mass:     {sol(m_midspan_total):8.1f} kg  ({sol(m_midspan_total) / u.lbm:8.0f} lb)")
print(f"  Span Location (eta):     {midspan_eta:8.2f}  ({sol(midspan_y):8.2f} m from CL)")
print(f"  Total TO Thrust (both):  {sol(midspan_thrust_at_liftoff):8.0f} N   ({sol(midspan_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  OEI Differential/rotor:  {sol(midspan_oei_differential):8.0f} N   (yaw control)")

print(f"\n{'--- Battery (takeoff/climb only; cruise BLI from turboshaft) ---':^72}")
print(f"  Battery Capacity:        {sol(battery_capacity_Wh) / 1000:8.1f} kWh")
print(f"  Battery Mass:            {m_batt_sol:8.1f} kg  ({m_batt_sol / u.lbm:8.0f} lb)")
print(f"  Wingtip Climb Energy:    {sol(electric_energy_wingtip_climb) / 3600 / 1000:8.1f} kWh")
print(f"  BLI Climb Energy:        {sol(electric_energy_bli_climb) / 3600 / 1000:8.1f} kWh")
print(f"  Mid-Span Climb Energy:   {sol(electric_energy_midspan_climb) / 3600 / 1000:8.1f} kWh")
print(f"  Total Electric Energy:   {sol(total_electric_energy_Wh) / 1000:8.1f} kWh")

print(f"\n{'--- Aerodynamic Drag Breakdown ---':^72}")
CD_aerobuildup = sol(aero["CD"])
CD_total = sol(CD_cruise)
CD_misc_sol = sol(CD_misc)
print(f"  AeroBuildup CD (raw):    {CD_aerobuildup:8.5f}")
print(f"  After 10% correction:    {CD_aerobuildup * 1.10:8.5f}")
print(f"  Misc drag CD (CDA/S):    {CD_misc_sol:8.5f}  (CDA_misc = {CDA_misc:.2f} m^2)")
print(f"  Total CD (corrected):    {CD_total:8.5f}")

print(f"\n{'--- Cruise Performance ---':^72}")
print(f"  Wingtip Cruise Power:    {sol(shaft_power_cruise_wingtip) / u.horsepower:8.0f} hp  (thrust)")
print(f"  BLI Cruise Power:        {sol(shaft_power_cruise_bli) / u.horsepower:8.0f} hp  (electric)")
print(f"  BLI from Turboshaft:     {sol(bli_electric_demand_from_turboshaft) / u.horsepower:8.0f} hp  (gen @ {generator_efficiency:.0%} eff)")
print(f"  Total Turboshaft Cruise: {sol(shaft_power_cruise_total) / u.horsepower:8.0f} hp")
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
print(f"  Climb Gradient (OEI):    {sol(field_results['flight_path_angle_climb_one_engine_out']):8.4f} rad ({sol(field_results['flight_path_angle_climb_one_engine_out']) * 100:8.2f}%)  (Torenbeek, conservative)")
print(f"  Climb Gradient (OEI+BLI):{sol(climb_gradient_oei):8.4f} rad ({sol(climb_gradient_oei) * 100:8.2f}%)  (reduced wingtip + BLI + mid-span)")
print(f"  OEI Wingtip Thrust:      {sol(thrust_wingtip_oei_reduced):8.0f} N   ({sol(thrust_wingtip_oei_reduced) / u.lbf:8.0f} lbf)  [{sol(thrust_wingtip_oei_reduced)/sol(thrust_per_wingtip_engine)*100:.0f}% of max]")
print(f"  Thrust/Weight (TO):      {sol(thrust_at_liftoff) / (TOGW * g):8.3f}")
print(f"  Wingtip Thrust at TO:    {sol(wingtip_thrust_at_liftoff):8.0f} N   ({sol(wingtip_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  BLI Thrust at TO:        {sol(bli_thrust_at_liftoff):8.0f} N   ({sol(bli_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  Mid-Span Thrust at TO:   {sol(midspan_thrust_at_liftoff):8.0f} N   ({sol(midspan_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  OEI Mid-Span Diff/rotor: {sol(midspan_oei_differential):8.0f} N   (yaw moment relief)")
print(f"  Net OEI Yaw Moment:      {sol(yaw_moment_oei):8.0f} N·m (after mid-span differential)")
print(f"  --- Cruise OEI (mid-span STOWED) ---")
print(f"  Cruise OEI Thrust/eng:   {sol(cruise_thrust_per_wingtip):8.0f} N   ({sol(cruise_thrust_per_wingtip) / u.lbf:8.0f} lbf)")
print(f"  Cruise OEI Yaw Moment:   {sol(yaw_moment_cruise_oei):8.0f} N·m")
print(f"  Vstab Authority (cruise):{sol(vstab_yaw_authority_cruise):8.0f} N·m")
print(f"  V_mc (cruise alt):       {sol(V_mc_cruise) / u.knot:8.1f} kts vs {cruise_speed / u.knot:8.1f} kts V_cruise")

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
    ("Nacelles (wingtip)", sol(m_nacelles)),
    ("Turboshaft Engines", sol(m_turboshaft_total)),
    ("Wingtip Electric Motors", sol(m_motor_total)),
    ("Wingtip ESCs", sol(m_esc_total)),
    ("Wingtip Propellers", sol(m_propellers_total)),
    ("Gearboxes", sol(m_gearbox_total)),
    ("BLI Motor", sol(m_bli_motor)),
    ("BLI ESC", sol(m_bli_esc)),
    ("BLI Propeller", sol(m_bli_propeller)),
    ("BLI Nacelle", sol(m_bli_nacelle)),
    ("Mid-Span Motors (2x)", sol(m_midspan_motor_total)),
    ("Mid-Span ESCs (2x)", sol(m_midspan_esc_total)),
    ("Mid-Span Propellers (2x)", sol(m_midspan_propeller_total)),
    ("Mid-Span Nacelle+Stow (2x)", sol((m_midspan_nacelle_each + m_midspan_stow_each) * n_midspan_rotors)),
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
print(f"  OEI grad:  {sol(climb_gradient_oei):.4f} vs 0.024 min (reduced wingtip + BLI + mid-span)")
print(f"  V_mc (TO): {sol(V_mc) / u.knot:.1f} kts vs V_stall {sol(V_stall_sl) / u.knot:.1f} kts  (with mid-span diff)")
print(f"  V_mc (cr): {sol(V_mc_cruise) / u.knot:.1f} kts vs V_cruise {cruise_speed / u.knot:.1f} kts  (NO mid-span, vstab only)")
print(f"  Wingtip T: {sol(wingtip_thrust_at_liftoff):,.0f} N  (BLI offloads {sol(bli_thrust_at_liftoff):,.0f} N, mid-span adds {sol(midspan_thrust_at_liftoff):,.0f} N)")
print(f"  OEI wingtip (reduced): {sol(thrust_wingtip_oei_reduced):,.0f} N  ({sol(thrust_wingtip_oei_reduced)/sol(thrust_per_wingtip_engine)*100:.0f}% of full)")
print(f"  OEI mid-span diff:     {sol(midspan_oei_differential):,.0f} N/rotor  (vstab relief)")
print(f"  AR:        {AR:.2f} vs [6.0, 16.0] bounds")
print(f"  Hybrid:    {sol(hybridization_factor):.1%} vs [20%, 70%] bounds")
print(f"  BLI prop:  {sol(bli_propeller_diameter):.2f} m vs {fuse_cabin_height * 1.1:.2f} m max")
print(f"  Mid-span:  {sol(midspan_propeller_diameter):.2f} m dia, {sol(midspan_motor_power)/1000:.0f} kW ea")
print(f"  SM aft:    {sol(static_margin)*100:.1f}% MAC vs 5% min")
print(f"  SM fwd:    {sol(static_margin_TOGW)*100:.1f}% MAC vs 20% max")

print("\n" + "=" * 72)

#  Optional: Draw 3-view 
try:
    sol_airplane = sol(airplane)
    axs = sol_airplane.draw_three_view(show=False)

    # Draw CG marker on every panel
    cg_x = sol(x_cg_TOGW)
    cg_y = 0.0
    cg_z = 0.0
    for ax in axs.flat:
        ax.plot(
            [cg_x], [cg_y], [cg_z],
            marker="o", color="red", markersize=8, zorder=999,
            label="CG",
        )
    axs[0, 0].legend(fontsize=8, loc="upper right")

    import matplotlib.pyplot as plt
    plt.show()
except Exception:
    pass  # Skip drawing if display not available
