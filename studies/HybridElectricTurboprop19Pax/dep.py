"""
Conceptual Design of a Series Hybrid-Electric 19-Passenger DEP Aircraft
======================================================================

Uses AeroSandbox's optimization framework (Opti) with:
- AeroBuildup for aerodynamic analysis
- Torenbeek / Raymer weight estimation methods
- Series hybrid-electric propulsion: turbogenerator -> DC bus -> DEP motors
- Distributed Electric Propulsion (DEP): 12 wing-mounted electric propellers
- Battery provides supplemental electric boost during takeoff/climb
- Torenbeek field length analysis

Requirements:
    - 19 passengers, 4000 lb payload
    - 200 kt cruise speed at 7000 ft
    - 2 turbogenerators (integrated gas-turbine + generator units)
    - 12 DEP electric propellers along the wing leading edge (6 per side)
    - 2600 ft takeoff and landing distance
    - 350 nmi max range, optimized for 175 nmi typical mission

Architecture: Series Hybrid-Electric with Distributed Electric Propulsion
(DEP). Two turbogenerators — each an integrated gas-turbine core coupled
directly to a high-efficiency permanent-magnet generator — feed a common
high-voltage DC bus. A lithium-ion battery pack also connects to the bus
for supplemental power during takeoff and climb. The bus powers 12 small
electric motors (6 per wing half), each driving an individual propeller
mounted along the wing leading edge.

Benefits of DEP + series architecture:
  (1) Blown-wing effect -> higher CL_max -> shorter field length
  (2) Spanwise thrust distribution -> wing bending relief from dist. mass
  (3) Graceful degradation: loss of one motor = only 1/12 thrust loss
  (4) Very low V_mc: each motor contributes a small asymmetric yaw moment
  (5) Direct-drive electric motors -> no gearbox required per propeller
  (6) Series architecture decouples turbine RPM from propeller RPM
  (7) Turbogenerators run at optimum RPM regardless of flight condition
  (8) No mechanical drivetrain to propellers — all power transmitted electrically

During cruise, turbogenerators provide all power through the series-electric
chain. Battery is sized for takeoff/climb boost only (not cruise).
"""

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

# Propulsion models
# Note: power_turboshaft / thermal_efficiency_turboshaft are used to model
# the gas-turbine core inside the turbogenerator (same thermodynamic cycle).
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

n_pax = 19
n_crew = 2
payload_mass = 6000 * u.lbm              # 2722 kg
cruise_speed = 200 * u.knot              # 102.9 m/s
field_length_req = 2600 * u.foot         # 792.5 m
n_turbogen = 2                            # Turbogenerator units (gas-turbine + integrated generator)
n_dep_motors = 12                         # Total DEP motors (6 per wing half)
n_dep_per_side = n_dep_motors // 2
design_range_max = 350 * u.naut_mile      # 648 km, max range (fuel sizing)
design_range_typical = 175 * u.naut_mile  # (80% of flights)
ultimate_load_factor = 1.5 * 3.0         # FAR 23 commuter
CL_max_clean = 2.4                       # Clean wing, with flaps
g = 9.81

# DEP blown-wing CL augmentation
# During takeoff, DEP propellers blow air over the wing, tripling
# effective CL (~200% increase). This is a key benefit of DEP for STOL.
dep_blown_wing_CL_factor = 2
CL_max_blown = CL_max_clean * dep_blown_wing_CL_factor  # Effective CL_max at takeoff

# DEP propeller cruise coefficient of performance
dep_propeller_CoP_cruise = 0.85   # Slightly lower than large prop (small diameter)
dep_propeller_CoP_takeoff = 0.75  # Lower at low airspeed

# Turbogenerator parameters
# The turbogenerator is an integrated gas-turbine + permanent-magnet generator.
# The gas-turbine core is ~75% of total unit mass; the rest is the generator,
# power electronics, and housing.
turbogen_core_mass_fraction = 0.75   # Gas-turbine core mass / total turbogenerator mass
turbogen_elec_eff = 0.92             # Integrated generator + rectification efficiency

# Series-electric chain efficiencies (turbogenerator electrical output -> DEP motor shaft)
eta_bus = 0.98             # DC bus distribution losses (wiring, contactors)
eta_esc = 0.96             # Electronic speed controller
eta_dep_motor = 0.96      # DEP electric motor
eta_series_chain = turbogen_elec_eff * eta_bus * eta_esc * eta_dep_motor
# Overall series chain: ~0.845 (turbogenerator fuel-in to DEP motor shaft)

# Fuel properties (Jet-A)
fuel_density = 820
fuel_specific_energy = 43.02e6

# Battery properties
battery_cell_specific_energy = 350
battery_pack_cell_fraction = 0.70
battery_max_dod = 0.80

# Fuselage geometry
fuse_length = 16
fuse_cabin_width = 1.9
fuse_cabin_height = 1.85
nose_length = 2.5
cabin_length = 7.1
tail_length = fuse_length - nose_length - cabin_length

# Tail geometry
tail_arm = 7.0  # wing AC to tail AC

##### Section: Optimization Setup #####

opti = asb.Opti()

##### Section: Design Variables #####

cruise_altitude = opti.variable(
    init_guess=7000 * u.foot, lower_bound=3000 * u.foot, upper_bound=10000 * u.foot
)  # 19-pax commuter assumed pressurized; ceiling raised for efficiency

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
mass_turbogen_each = opti.variable(
    init_guess=170, log_transform=True, lower_bound=60, upper_bound=500
)  # Total mass of one turbogenerator unit (turbine core + integrated generator)

# DEP propeller diameter (all DEP props same size)
dep_propeller_diameter = opti.variable(
    init_guess=0.9, lower_bound=0.1, upper_bound=1.2
)

hybridization_factor = opti.variable(
    init_guess=0.25, lower_bound=0.10, upper_bound=0.60
)  # Battery bus power / total bus power during takeoff

battery_capacity_Wh = opti.variable(
    init_guess=50000, log_transform=True, lower_bound=5000, upper_bound=500000
)
fuel_mass = opti.variable(
    init_guess=800, log_transform=True, lower_bound=50, upper_bound=3000
)
thrust_at_liftoff = opti.variable(
    init_guess=7700 * g * 0.30, log_transform=True, lower_bound=5000
)

# Cabin floor, goes wherever best for CG
x_cg_battery = opti.variable(
    init_guess=nose_length + 0.25 * cabin_length,
    lower_bound=nose_length + 0.5,
    upper_bound=nose_length + cabin_length - 0.5,
)  # from nose

# Vertical stabilizer sizing (much smaller with DEP - low V_mc)
vstab_span_val = opti.variable(
    init_guess=2.0, lower_bound=1.0, upper_bound=5.0
)
vstab_root_chord_val = opti.variable(
    init_guess=1.8, lower_bound=1.0, upper_bound=4.0
)
vstab_taper_ratio = 0.5
vstab_tip_chord_val = vstab_root_chord_val * vstab_taper_ratio
vstab_area = (vstab_root_chord_val + vstab_tip_chord_val) / 2 * vstab_span_val

# Horizontal stabilizer sizing
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

#  Fuselage
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
            width=0.40,
            height=0.40,
        ),
    ],
)

#  Main Wing (HIGH wing)
wing_x_le = 0.40 * fuse_length - 0.25 * wing_root_chord
wing_z_le = 0.5 * fuse_cabin_height

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
            airfoil=asb.Airfoil("naca23015"),
        ),
    ],
).translate([wing_x_le, 0, wing_z_le])

#  Horizontal Stabilizer
elevator = asb.ControlSurface(
    name="Elevator", symmetric=True, deflection=0, hinge_point=0.70
)

# Initial hstab x position (based on tail arm)
hstab_x_le = wing_x_le + 0.25 * wing_root_chord + tail_arm - 0.25 * hstab_root_chord_val

#  Vertical Stabilizer
rudder = asb.ControlSurface(
    name="Rudder", symmetric=True, deflection=0, hinge_point=0.70
)

vstab_z_le = 0.3

# T-tail configuration: Place hstab on top of vstab (z direction)
hstab_z_le = vstab_z_le + vstab_span_val

# Position vstab so its tip LE aligns with the hstab root LE (proper T-tail)
vstab_x_le = hstab_x_le - vstab_span_val * np.tand(30)

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

#  DEP Propulsors (distributed along wing leading edge)
#  6 per wing half, from ~15% to ~90% semi-span
wing_sweep_LE = 3  # degrees (from wing geometry definition)

dep_propulsors = []
dep_eta_spans = []  # Store span locations for CG calculations
for i in range(n_dep_per_side):
    eta_span = 0.15 + 0.75 * i / (n_dep_per_side - 1)  # 15% to 90% semi-span
    dep_eta_spans.append(eta_span)
    y_pos = eta_span * wing_span / 2
    x_pos = wing_x_le + y_pos * np.tand(wing_sweep_LE) - 0.3  # Ahead of wing LE
    z_pos = wing_z_le + y_pos * np.tand(2)

    # Right wing
    dep_propulsors.append(asb.Propulsor(
        name=f"DEP R{i+1}",
        xyz_c=[x_pos, y_pos, z_pos],
        xyz_normal=[1, 0, 0],
        radius=dep_propeller_diameter / 2,
        length=0.3,
    ))
    # Left wing (mirror)
    dep_propulsors.append(asb.Propulsor(
        name=f"DEP L{i+1}",
        xyz_c=[x_pos, -y_pos, z_pos],
        xyz_normal=[1, 0, 0],
        radius=dep_propeller_diameter / 2,
        length=0.3,
    ))

#  Assemble Airplane
airplane = asb.Airplane(
    name="HE-19 Series Hybrid-Electric DEP",
    xyz_ref=[wing_x_le + 0.25 * wing_root_chord, 0, 0],
    wings=[wing, hstab, vstab],
    fuselages=[fuse],
    propulsors=dep_propulsors,
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

# Miscellaneous Drag Correction
# AeroBuildup omits protuberance drag (random stuff)
# DEP nacelle/pod drag partially captured by AeroBuildup propulsors
# DEP pods are small streamlined nacelles (~15 cm diameter each),
# contributing much less drag than a single large conventional nacelle.
CDA_misc = 0.15  # Misc protuberance + DEP pod drag (12 small streamlined pods)

# Drag correction factor
drag_correction_factor = 1.10  # 10% increase on AeroBuildup base drag

CD_misc = CDA_misc / wing_area
CL_cruise = aero["CL"]
CD_cruise = aero["CD"] * drag_correction_factor + CD_misc
L_over_D_cruise = CL_cruise / CD_cruise

q_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2
drag_cruise = CD_cruise * q_cruise * wing_area

# No BLI drag reduction in DEP configuration.
# All DEP propellers produce thrust to overcome full drag.
drag_effective_cruise = drag_cruise

##### Section: Weight Breakdown #####

# -> Structural ->

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
# Smaller than BLI/wingtip config because V_mc is very low with DEP
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
V_stall_clean = np.sqrt(
    2 * design_mass_TOGW * g / (atmo_sl.density() * wing_area * CL_max_clean)
)
V_stall_blown = np.sqrt(
    2 * design_mass_TOGW * g / (atmo_sl.density() * wing_area * CL_max_blown)
)

m_mlg = raymer_wt.mass_main_landing_gear(
    main_gear_length=0.8,
    landing_speed=V_stall_clean * 1.3,  # Conservative: clean-wing stall for engine-out landing
    design_mass_TOGW=design_mass_TOGW,
    n_wheels=4,
    n_shock_struts=2,
) * 0.7  # Fixed gear: no retraction mechanism ~70% lighter

m_nlg = raymer_wt.mass_nose_landing_gear(
    nose_gear_length=0.6,
    design_mass_TOGW=design_mass_TOGW,
    n_wheels=2,
) * 0.7  # Fixed gear

# -> Propulsion System (Series Hybrid-Electric + DEP) ->
# Turbogenerators are housed inside the fuselage (no external nacelles).

# Turbogenerator: integrated gas-turbine core + permanent-magnet generator
# Use the turboshaft power scaling for the gas-turbine core (same thermodynamic cycle)
mass_turbogen_core = mass_turbogen_each * turbogen_core_mass_fraction
power_turbogen_shaft = power_turboshaft(mass_turbogen_core)    # Core mechanical shaft power
power_turbogen_elec = power_turbogen_shaft * turbogen_elec_eff  # Electrical output per unit
m_turbogen_total = mass_turbogen_each * n_turbogen

# Series-electric power chain sizing
# During takeoff: turbogenerators + battery -> DC bus -> DEP motors
total_gen_bus_power = power_turbogen_elec * n_turbogen
total_bus_power_takeoff = total_gen_bus_power / (1 - hybridization_factor)
battery_bus_power_takeoff = total_bus_power_takeoff * hybridization_factor

# Per-DEP-motor electrical power from bus (all motors share equally)
dep_motor_bus_power_each = total_bus_power_takeoff * eta_bus / n_dep_motors
dep_motor_shaft_power_each = dep_motor_bus_power_each * eta_esc * eta_dep_motor

# DEP electric motors (direct-drive, no gearbox)
motor_power_density = 5000  # W/kg (5 kW/kg, aircraft-class certified electric motor)
m_dep_motor_each = dep_motor_bus_power_each / motor_power_density
m_dep_motors_total = m_dep_motor_each * n_dep_motors

# DEP ESCs
m_dep_esc_each = mass_ESC(dep_motor_bus_power_each)
m_dep_esc_total = m_dep_esc_each * n_dep_motors

# Battery
m_battery = mass_battery_pack(
    battery_capacity_Wh,
    battery_cell_specific_energy_Wh_kg=battery_cell_specific_energy,
    battery_pack_cell_fraction=battery_pack_cell_fraction,
)

# DEP propellers (small, lightweight composite, direct-drive)
m_dep_propeller_each = torenbeek_wt.mass_propeller(
    propeller_diameter=dep_propeller_diameter,
    propeller_power=dep_motor_shaft_power_each,
    n_blades=3,
) * 0.35  # Lightweight composite
m_dep_propellers_total = m_dep_propeller_each * n_dep_motors

# DEP pod/nacelle mass (small fairings for each motor, proportional to motor mass)
m_dep_pod_each = 0.10 * m_dep_motor_each + 2.0  # Small fairing + spinner
m_dep_pods_total = m_dep_pod_each * n_dep_motors

# Fuel system
fuel_volume = fuel_mass / fuel_density
m_fuel_system = raymer_wt.mass_fuel_system(
    fuel_volume=fuel_volume,
    n_tanks=2,
    fraction_in_integral_tanks=1.0,
)

# -> Systems ->

m_instruments = raymer_wt.mass_instruments(
    fuselage=fuse,
    main_wing=wing,
    n_engines=n_turbogen,
    n_crew=n_crew,
    engine_is_turboprop=False,  # Turbogenerators, not turboprops
)

m_electrical = raymer_wt.mass_electrical(
    system_electrical_power_rating=80000,  # Higher for series-electric (large bus)
    electrical_routing_distance=fuse_length * 0.8,  # More wiring for DEP
    n_engines=n_turbogen,
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

# -> Wing Weight ->

wing_aspect_ratio = wing_span ** 2 / wing_area

# V_NE ~ 1.4x cruise for FAR 23 commuter turboprops
V_NE = cruise_speed * 1.40
# V_flap ~ 1.8x stall speed (sea level, clean config)
V_flap = V_stall_clean * 1.8

suspended_mass_approx = design_mass_TOGW * 0.92

m_wing_basic = torenbeek_wt.mass_wing_basic_structure(
    wing=wing,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    suspended_mass=suspended_mass_approx,
    never_exceed_airspeed=V_NE,
    main_gear_mounted_to_wing=False,
    k_e=0.95,  # DEP distributed mass gives moderate bending relief
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

# -> Payload / Cabin ->
m_pax = n_pax * mass_passenger
m_seats = n_pax * mass_seat("passenger") + n_crew * mass_seat("flight_deck")
m_lavs = mass_lavatories(n_pax, aircraft_type="short-haul")

m_flight_controls = 0.02 * design_mass_TOGW

##### Section: Total Weight #####

mass_empty = (
    # Structure
    m_wing + m_hstab + m_vstab + m_fuselage
    + m_mlg + m_nlg
    # Series hybrid propulsion (turbogenerators)
    + m_turbogen_total
    # DEP system
    + m_dep_motors_total + m_dep_esc_total
    + m_dep_propellers_total + m_dep_pods_total
    + m_fuel_system
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

# Turbogenerators housed inside fuselage (aft cabin / tail cone area)
x_cg_turbogen = nose_length + cabin_length * 0.85  # Aft fuselage, near wing trailing edge

# DEP motors distributed along wing LE (average CG position)
# Average eta_span of DEP motors: mean of 0.15 to 0.90 = 0.525
dep_avg_eta = 0.525
x_cg_dep_motors = wing_x_le + dep_avg_eta * wing_span / 2 * np.tand(wing_sweep_LE) - 0.15
x_cg_dep_esc = x_cg_dep_motors + 0.1
x_cg_dep_propellers = x_cg_dep_motors - 0.2
x_cg_dep_pods = x_cg_dep_motors

# Fuel, battery, payload, systems
x_cg_fuel = x_MAC_le + 0.35 * wing_MAC               # Integral wing tanks
# x_cg_battery is an optimization variable (defined earlier)
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
    # Series hybrid propulsion (turbogenerators)
    + m_turbogen_total * x_cg_turbogen
    # DEP system
    + m_dep_motors_total * x_cg_dep_motors
    + m_dep_esc_total * x_cg_dep_esc
    + m_dep_propellers_total * x_cg_dep_propellers
    + m_dep_pods_total * x_cg_dep_pods
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
K_f = 0.92
fuselage_width_eff = fuse_cabin_width * 0.85  # Effective average width
delta_x_np_fuse = K_f * fuselage_width_eff ** 2 * fuse_length / wing_lift_effectiveness
x_np = x_np_wing_tail - delta_x_np_fuse

#  Static Margin
static_margin = (x_np - x_cg_aft) / wing_MAC        # At aft CG (critical for stability)
static_margin_TOGW = (x_np - x_cg_TOGW) / wing_MAC  # At fwd CG (critical for trim)

##### Section: Propulsion and Performance #####

#  Cruise Power Balance
# All DEP propellers produce thrust to overcome drag
# Power chain: turbogenerator -> bus -> ESC -> motor -> propeller
dep_propulsive_area = n_dep_motors * np.pi / 4 * dep_propeller_diameter ** 2

# DEP shaft power required to produce cruise thrust
shaft_power_cruise_dep = propeller_shaft_power_from_thrust(
    thrust_force=drag_effective_cruise,
    area_propulsive=dep_propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=dep_propeller_CoP_cruise,
)

# Turbogenerator power required at cruise (accounting for series chain losses)
turbogen_cruise_power_total = shaft_power_cruise_dep / eta_series_chain

# Each turbogenerator must handle its share (with 5% margin)
# power_turbogen_elec is electrical output per unit
opti.subject_to(
    power_turbogen_elec >= turbogen_cruise_power_total / n_turbogen * 1.05
)

# Cruise throttle and fuel consumption
# Throttle based on shaft power demanded from gas-turbine core
turbogen_cruise_shaft_total = turbogen_cruise_power_total / turbogen_elec_eff
throttle_cruise = turbogen_cruise_shaft_total / (power_turbogen_shaft * n_turbogen)

eta_thermal_cruise = thermal_efficiency_turboshaft(
    mass_turboshaft=mass_turbogen_core,    # Gas-turbine core mass for thermal eff scaling
    throttle_setting=throttle_cruise,
)

fuel_burn_rate_cruise = turbogen_cruise_shaft_total / (
    eta_thermal_cruise * fuel_specific_energy
)

# 45-min VFR reserves
fuel_reserve_time = 45 * 60
fuel_reserves = fuel_burn_rate_cruise * fuel_reserve_time

# Climb fuel estimate (~10 min at higher power)
climb_time = 10 * 60  # seconds
climb_fuel_factor = 1.3  # Higher burn rate during climb
fuel_for_climb = fuel_burn_rate_cruise * climb_fuel_factor * climb_time

# Fuel for max range mission (350 nmi) -> sizes fuel tanks and MTOW
cruise_time_max = design_range_max / cruise_speed
fuel_for_cruise_max = fuel_burn_rate_cruise * cruise_time_max

opti.subject_to(fuel_mass >= fuel_for_cruise_max + fuel_reserves + fuel_for_climb)

# Fuel for typical mission (175 nmi) -> used for cruise optimization
cruise_time_typical = design_range_typical / cruise_speed
fuel_for_cruise_typical = fuel_burn_rate_cruise * cruise_time_typical
fuel_mass_typical = fuel_for_cruise_typical + fuel_reserves + fuel_for_climb

#  Takeoff Power (Series Hybrid + Battery Boost)
# All thrust produced by DEP motors
total_dep_shaft_power_takeoff = dep_motor_shaft_power_each * n_dep_motors

V_liftoff = 1.2 * V_stall_blown  # Blown-wing stall speed (DEP active at takeoff)

# DEP shaft power required at liftoff
dep_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=thrust_at_liftoff,
    area_propulsive=dep_propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=dep_propeller_CoP_takeoff,
)

opti.subject_to(total_dep_shaft_power_takeoff >= dep_shaft_power_from_thrust_liftoff)

# DEP per-motor power must be rated for takeoff demand
dep_motor_power_required_each = dep_shaft_power_from_thrust_liftoff / n_dep_motors
opti.subject_to(dep_motor_shaft_power_each >= dep_motor_power_required_each)

#  Battery Sizing
# During cruise: turbogenerators provide ALL power (no battery drain)
# Battery only used during takeoff/climb for boost:
#   - Battery provides hybridization_factor of total bus power during climb
electric_energy_climb_Wh = battery_bus_power_takeoff * climb_time / 3600  # Wh

opti.subject_to(battery_capacity_Wh >= electric_energy_climb_Wh / battery_max_dod)

##### Section: Constraints #####

#  Wing Geometry Constraints
opti.subject_to(wing_aspect_ratio >= 6.0)   # Practical minimum
opti.subject_to(wing_aspect_ratio <= 14.0)  # DEP bending relief enables higher AR

#  Cruise Lift = Weight (optimized for typical 175 nmi mission)
lift_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2 * wing_area * CL_cruise
typical_mission_TOGW = design_mass_TOGW - (fuel_mass - fuel_mass_typical)
mid_cruise_weight = (typical_mission_TOGW - fuel_for_cruise_typical * 0.5) * g

opti.subject_to(lift_cruise >= mid_cruise_weight * 0.99)
opti.subject_to(lift_cruise <= mid_cruise_weight * 1.01)

#  Field Length (using blown-wing CL_max from DEP)
L_over_D_climb = L_over_D_cruise * 0.65  # Reduced L/D in takeoff config

field_results = field_length_analysis_torenbeek(
    design_mass_TOGW=design_mass_TOGW,
    thrust_at_liftoff=thrust_at_liftoff,
    lift_over_drag_climb=L_over_D_climb,
    CL_max=CL_max_blown,  # DEP blown-wing CL augmentation
    s_ref=wing_area,
    n_engines=n_turbogen,  # Critical failure = one turbogenerator loss
    atmosphere=atmo_sl,
    CD_zero_lift=0.04,
    obstacle_height=50 * u.foot,
)

opti.subject_to(field_results["takeoff_total_distance"] <= field_length_req)
opti.subject_to(field_results["landing_total_distance"] <= field_length_req)
opti.subject_to(field_results["balanced_field_length"] <= field_length_req)

#  Engine-Out Analysis (Loss of one turbogenerator)
# With series-electric, losing one turbogenerator means losing ~50% of
# generation capacity. The battery provides emergency supplemental power.
# All 12 DEP motors remain operative (just with less total bus power).
remaining_gen_power = power_turbogen_elec * 1  # One turbogenerator still operating
# Battery at full boost power available for limited time
remaining_bus_power = remaining_gen_power + battery_bus_power_takeoff
# Fraction of normal takeoff power remaining
power_fraction_oei = remaining_bus_power / total_bus_power_takeoff
thrust_oei = thrust_at_liftoff * power_fraction_oei  # All 12 motors, reduced power

thrust_over_weight_oei = thrust_oei / (design_mass_TOGW * g)
climb_gradient_oei = thrust_over_weight_oei - 1 / L_over_D_climb
opti.subject_to(climb_gradient_oei >= 0.024)  # FAR 23 minimum for 2-engine class

#  Engine-Out Directional Control (V_mc)
# With DEP, the worst single-motor failure creates minimal yaw.
# Loss of one DEP motor: only 1/12 of total thrust, at outermost position.
y_engine_max = 0.90 * wing_span / 2  # Outermost DEP motor
thrust_per_dep_motor = thrust_at_liftoff / n_dep_motors
yaw_moment_dep_oei = thrust_per_dep_motor * y_engine_max * 1.10  # +10% windmilling

CL_vstab_max_rudder = 0.9      # Side-force coeff at max rudder deflection
l_vt = tail_arm * 0.95         # CG to vstab aerodynamic center

q_vmc = yaw_moment_dep_oei / (vstab_area * CL_vstab_max_rudder * l_vt)
V_mc = np.sqrt(2 * q_vmc / atmo_sl.density())

opti.subject_to(V_mc <= V_stall_blown)   # FAR 23.149: V_mc must not exceed V_s1 (takeoff config, DEP active)

#  Hstab Sizing (Volume Coefficient)
l_h = tail_arm
V_h_coefficient = (hstab_area * l_h) / (wing_area * wing_mean_chord)
opti.subject_to(V_h_coefficient >= 0.90)

#  Longitudinal Stability (Static Margin)
opti.subject_to(static_margin >= 0.05)
opti.subject_to(static_margin_TOGW <= 0.20)

#  Vstab Geometry Limits
vstab_aspect_ratio = vstab_span_val ** 2 / vstab_area
opti.subject_to(vstab_aspect_ratio >= 1.0)
opti.subject_to(vstab_aspect_ratio <= 2.5)

#  DEP Propeller Diameter Constraint
# DEP props must not overlap: spacing = semi-span / n_dep_per_side
dep_spacing = (0.90 - 0.15) * wing_span / 2 / (n_dep_per_side - 1)
opti.subject_to(dep_propeller_diameter <= dep_spacing * 0.95)  # 5% clearance

# Mass Closure
opti.subject_to(mass_total <= design_mass_TOGW)

#  MTOW Limit (FAR 23 commuter category: 19,000 lb)
opti.subject_to(design_mass_TOGW <= 18000 * u.lbm)

##### Section: Objective #####

# Minimize fuel burn on the typical 175 nmi mission (80% of flights).
opti.minimize(fuel_mass_typical)

##### Section: Solve #####

sol = opti.solve(max_iter=1500)

##### Section: Results Summary #####

print("=" * 72)
print("   HE-19 SERIES HYBRID-ELECTRIC DEP -> DESIGN SUMMARY")
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

print(f"\n{' Overall ':^72}")
print(f"  MTOW:                    {TOGW:8.0f} kg  ({TOGW / u.lbm:8.0f} lb)")
print(f"  Empty Weight:            {m_empty_sol:8.0f} kg  ({m_empty_sol / u.lbm:8.0f} lb)")
print(f"  Payload:                 {payload_mass:8.0f} kg  ({payload_mass / u.lbm:8.0f} lb)")
print(f"  Fuel Weight:             {m_fuel_sol:8.0f} kg  ({m_fuel_sol / u.lbm:8.0f} lb)")
print(f"  Battery Weight:          {m_batt_sol:8.0f} kg  ({m_batt_sol / u.lbm:8.0f} lb)")
print(f"  Useful Load:             {payload_mass + m_fuel_sol + m_batt_sol:8.0f} kg")
print(f"  Cruise Altitude:         {sol(cruise_altitude) / u.foot:8.0f} ft")

print(f"\n{' Geometry ':^72}")
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
print(f"  H-Stab Span:             {sol(hstab_span_val):8.2f} m")
print(f"  H-Stab Area:             {sol(hstab_area):8.2f} m^2")
print(f"  H-Stab V_h Coeff:        {sol(V_h_coefficient):8.2f}")
print(f"  DEP Propeller Diameter:  {sol(dep_propeller_diameter):8.2f} m   ({sol(dep_propeller_diameter) / u.foot:8.1f} ft)")
print(f"  DEP Prop Spacing:        {sol(dep_spacing):8.2f} m")

print(f"\n{' CG & Stability ':^72}")
print(f"  Wing MAC:                {sol(wing_MAC):8.2f} m")
print(f"  Wing AC (x):             {sol(x_ac_wing):8.2f} m from nose")
print(f"  CG at TOGW (x):         {sol(x_cg_TOGW):8.2f} m  ({sol((x_cg_TOGW - x_MAC_le) / wing_MAC) * 100:5.1f}% MAC)")
print(f"  CG at ZFW / aft (x):    {sol(x_cg_aft):8.2f} m  ({sol((x_cg_aft - x_MAC_le) / wing_MAC) * 100:5.1f}% MAC)")
print(f"  Neutral Point (x):      {sol(x_np):8.2f} m  ({sol((x_np - x_MAC_le) / wing_MAC) * 100:5.1f}% MAC)")
print(f"  Fuse NP shift (fwd):    {sol(delta_x_np_fuse):8.2f} m")
print(f"  Battery Position (x):   {sol(x_cg_battery):8.2f} m  ({(sol(x_cg_battery) - nose_length) / cabin_length * 100:5.1f}% cabin)")
print(f"  Static Margin (TOGW):   {sol(static_margin_TOGW) * 100:8.1f}% MAC")
print(f"  Static Margin (aft CG): {sol(static_margin) * 100:8.1f}% MAC")

print(f"\n{' Aerodynamics (Cruise @ {:.0f} ft) '.format(sol(cruise_altitude) / u.foot):^72}")

print(f"  CL:                      {sol(CL_cruise):8.4f}")
print(f"  CD:                      {sol(CD_cruise):8.5f}")
print(f"  L/D:                     {sol(L_over_D_cruise):8.1f}")
print(f"  Alpha:                   {sol(cruise_alpha):8.1f} deg")
print(f"  Cruise Drag:             {sol(drag_cruise):8.0f} N   ({sol(drag_cruise) / u.lbf:8.0f} lbf)")
print(f"  CL_max (clean):          {CL_max_clean:8.2f}")
print(f"  CL_max (DEP blown):      {CL_max_blown:8.2f}  (x{dep_blown_wing_CL_factor:.2f} augmentation)")

print(f"\n{' Propulsion (Series Hybrid-Electric) ':^72}")
print(f"  Turbogenerator Mass (ea): {sol(mass_turbogen_each):8.1f} kg  ({sol(mass_turbogen_each) / u.lbm:8.0f} lb)")
print(f"  Turbine Core Mass (ea):   {sol(mass_turbogen_core):8.1f} kg")
print(f"  Core Shaft Power (each):  {sol(power_turbogen_shaft) / u.horsepower:8.0f} hp  ({sol(power_turbogen_shaft) / 1000:8.0f} kW)")
print(f"  Elec Output (each):       {sol(power_turbogen_elec) / u.horsepower:8.0f} hp  ({sol(power_turbogen_elec) / 1000:8.0f} kW)")
print(f"  Elec Output (total):      {sol(power_turbogen_elec) * n_turbogen / u.horsepower:8.0f} hp")
print(f"  Turbogen Elec Efficiency: {turbogen_elec_eff:8.1%}")
print(f"  Series Chain Efficiency:  {eta_series_chain:8.1%}")
print(f"  Hybridization Factor:     {sol(hybridization_factor):8.1%}")

print(f"\n{' DEP System ({n_dep_motors} motors) ':^72}")
print(f"  DEP Motor Power (each):  {sol(dep_motor_bus_power_each) / 1000:8.0f} kW  ({sol(dep_motor_bus_power_each) / u.horsepower:8.0f} hp)")
print(f"  DEP Motor Shaft (each):  {sol(dep_motor_shaft_power_each) / 1000:8.0f} kW")
print(f"  DEP Motor Mass (each):   {sol(m_dep_motor_each):8.1f} kg")
print(f"  DEP Motors Mass (total): {sol(m_dep_motors_total):8.1f} kg")
print(f"  DEP ESC Mass (total):    {sol(m_dep_esc_total):8.1f} kg")
print(f"  DEP Propeller Mass (ea): {sol(m_dep_propeller_each):8.1f} kg")
print(f"  DEP Props Mass (total):  {sol(m_dep_propellers_total):8.1f} kg")
print(f"  DEP Pods Mass (total):   {sol(m_dep_pods_total):8.1f} kg")
print(f"  Total DEP System Mass:   {sol(m_dep_motors_total + m_dep_esc_total + m_dep_propellers_total + m_dep_pods_total):8.1f} kg")

print(f"\n{' Battery (takeoff/climb boost only; cruise from turbogenerators) ':^72}")
print(f"  Battery Capacity:        {sol(battery_capacity_Wh) / 1000:8.1f} kWh")
print(f"  Battery Mass:            {m_batt_sol:8.1f} kg  ({m_batt_sol / u.lbm:8.0f} lb)")
print(f"  Battery Boost Power:     {sol(battery_bus_power_takeoff) / 1000:8.1f} kW")
print(f"  Climb Energy (battery):  {sol(electric_energy_climb_Wh) / 1000:8.1f} kWh")

print(f"\n{' Aerodynamic Drag Breakdown ':^72}")
CD_aerobuildup = sol(aero["CD"])
CD_total = sol(CD_cruise)
CD_misc_sol = sol(CD_misc)
print(f"  AeroBuildup CD (raw):    {CD_aerobuildup:8.5f}")
print(f"  After 10% correction:    {CD_aerobuildup * 1.10:8.5f}")
print(f"  Misc drag CD (CDA/S):    {CD_misc_sol:8.5f}  (CDA_misc = {CDA_misc:.2f} m^2)")
print(f"  Total CD (corrected):    {CD_total:8.5f}")

print(f"\n{' Cruise Performance ':^72}")
print(f"  DEP Cruise Shaft Power:  {sol(shaft_power_cruise_dep) / u.horsepower:8.0f} hp  (all {n_dep_motors} motors)")
print(f"  Turbogen Cruise Power:   {sol(turbogen_cruise_power_total) / u.horsepower:8.0f} hp  (elec, incl. series losses)")
print(f"  Turbogen Shaft Cruise:   {sol(turbogen_cruise_shaft_total) / u.horsepower:8.0f} hp  (core shaft)")
print(f"  Cruise Throttle:         {sol(throttle_cruise):8.1%}")
print(f"  Thermal Efficiency:      {sol(eta_thermal_cruise):8.1%}")
print(f"  Cruise Fuel Burn:        {sol(fuel_burn_rate_cruise) * 3600:8.1f} kg/hr")
print(f"  Fuel for Max Range:      {sol(fuel_for_cruise_max):8.0f} kg  (350 nmi)")
print(f"  Fuel for Typical Range:  {sol(fuel_for_cruise_typical):8.0f} kg  (175 nmi)")
print(f"  Fuel Reserves (45 min):  {sol(fuel_reserves):8.0f} kg")
print(f"  Fuel for Climb:          {sol(fuel_for_climb):8.0f} kg")
print(f"  Total Fuel (max range):  {sol(fuel_mass):8.0f} kg")

print(f"\n{' Field Performance ':^72}")
print(f"  V_stall (SL, clean):     {sol(V_stall_clean) / u.knot:8.1f} kts ({sol(V_stall_clean):8.1f} m/s)")
print(f"  V_stall (SL, blown):     {sol(V_stall_blown) / u.knot:8.1f} kts ({sol(V_stall_blown):8.1f} m/s)")
print(f"  V_liftoff:               {sol(field_results['V_liftoff']) / u.knot:8.1f} kts")
print(f"  Takeoff Ground Roll:     {sol(field_results['takeoff_ground_roll_distance']) / u.foot:8.0f} ft  ({sol(field_results['takeoff_ground_roll_distance']):8.0f} m)")
print(f"  Takeoff Total Distance:  {sol(field_results['takeoff_total_distance']) / u.foot:8.0f} ft  ({sol(field_results['takeoff_total_distance']):8.0f} m)")
print(f"  Balanced Field Length:   {sol(field_results['balanced_field_length']) / u.foot:8.0f} ft  ({sol(field_results['balanced_field_length']):8.0f} m)")
print(f"  Landing Total Distance:  {sol(field_results['landing_total_distance']) / u.foot:8.0f} ft  ({sol(field_results['landing_total_distance']):8.0f} m)")
print(f"  Climb Gradient (AEO):    {sol(field_results['flight_path_angle_climb']):8.4f} rad ({sol(field_results['flight_path_angle_climb']) * 100:8.2f}%)")
print(f"  Climb Gradient (OEI):    {sol(climb_gradient_oei):8.4f} rad ({sol(climb_gradient_oei) * 100:8.2f}%)  (1 turbogenerator out + battery)")
print(f"  Thrust/Weight (TO):      {sol(thrust_at_liftoff) / (TOGW * g):8.3f}")
print(f"  OEI Power Fraction:      {sol(power_fraction_oei):8.1%}")

print(f"\n{' Weight Breakdown ':^72}")
print(f"  {'Component':<28} {'Mass (kg)':>10} {'Mass (lb)':>10} {'% MTOW':>8}")
print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")

weight_items = [
    ("Wing", sol(m_wing)),
    ("H-Stab", sol(m_hstab)),
    ("V-Stab", sol(m_vstab)),
    ("Fuselage", sol(m_fuselage)),
    ("Main Landing Gear", sol(m_mlg)),
    ("Nose Landing Gear", sol(m_nlg)),
    ("Turbogenerators (in fuselage)", sol(m_turbogen_total)),
    ("DEP Motors (x{})".format(n_dep_motors), sol(m_dep_motors_total)),
    ("DEP ESCs (x{})".format(n_dep_motors), sol(m_dep_esc_total)),
    ("DEP Propellers (x{})".format(n_dep_motors), sol(m_dep_propellers_total)),
    ("DEP Pods/Nacelles", sol(m_dep_pods_total)),
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

print(f"\n{' Mission Profile ':^72}")
print(f"  Typical Mission:         {design_range_typical / u.naut_mile:.0f} nmi ({design_range_typical / 1000:.0f} km)")
print(f"  Max Range Mission:       {design_range_max / u.naut_mile:.0f} nmi ({design_range_max / 1000:.0f} km)")
print(f"  Typical Takeoff Weight:  {sol(typical_mission_TOGW):.0f} kg ({sol(typical_mission_TOGW) / u.lbm:.0f} lb)")
print(f"  Max Takeoff Weight:      {TOGW:.0f} kg ({TOGW / u.lbm:.0f} lb)")

print(f"\n{' Active Constraints ':^72}")
print(f"  BFL:       {sol(field_results['balanced_field_length']) / u.foot:.0f} ft vs {field_length_req / u.foot:.0f} ft limit")
print(f"  TO dist:   {sol(field_results['takeoff_total_distance']) / u.foot:.0f} ft vs {field_length_req / u.foot:.0f} ft limit")
print(f"  LDG dist:  {sol(field_results['landing_total_distance']) / u.foot:.0f} ft vs {field_length_req / u.foot:.0f} ft limit")
print(f"  OEI grad:  {sol(climb_gradient_oei):.4f} vs 0.024 min (1 turbogenerator out + battery)")
print(f"  V_mc:      {sol(V_mc) / u.knot:.1f} kts vs V_stall {sol(V_stall_blown) / u.knot:.1f} kts  (DEP blown, 1/{n_dep_motors} motor loss)")
print(f"  AR:        {AR:.2f} vs [6.0, 14.0] bounds")
print(f"  Hybrid:    {sol(hybridization_factor):.1%} vs [10%, 60%] bounds")
print(f"  DEP prop:  {sol(dep_propeller_diameter):.2f} m vs {sol(dep_spacing) * 0.95:.2f} m max (spacing)")
print(f"  SM aft:    {sol(static_margin)*100:.1f}% MAC vs 5% min")
print(f"  SM fwd:    {sol(static_margin_TOGW)*100:.1f}% MAC vs 20% max")
print(f"  Series eff:{eta_series_chain:.1%} (turbogenerator elec -> DEP motor shaft)")

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
