"""
Need requirements updates
"""

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

# Requirements traceability
from requirements import load_requirements, get_limit, validate_solution
reqs = load_requirements()

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

n_pax = 19
n_crew = 2
payload_mass = 6000 * u.lbm   
cruise_speed = 200 * u.knot  
# cruise_altitude = 7000 * u.foot 
field_length_req = 2600 * u.foot    
n_engines = 2
design_range_max = 350 * u.naut_mile     
design_range_typical = 175 * u.naut_mile  # (80% of flights)
ultimate_load_factor = 3.0 # FAR 23 commuter
CL_max = 2.4
g = 9.81

# Wingtip propeller efficiency bonus (vortex energy recovery)
wingtip_propeller_efficiency_bonus = 1.15  

# Generator efficiency: wingtip motors act as generators during cruise?????????
# to power the BLI motor from the turboshaft (shaft -> generator -> BLI motor)
generator_efficiency = 0.93  # Motor as generator???????????

# Fuel properties (Jet-A)
fuel_density = 820 
fuel_specific_energy = 43.02e6 

# Battery properties
battery_cell_specific_energy = 350
battery_pack_cell_fraction = 0.70 # cooling system etc
battery_max_dod = 0.80 


# NEEDS UPDATING -> Dr Blaesser comment of only reduces fuselage drag which is smaller, 5% more realistic
# BLI pusher properties
bli_drag_reduction_factor = 0.10 # 10% effective drag reduction from BLI wake ingestion
bli_propeller_CoP = 0.80 # litte less efficient

# Fuselage geometry 
# Vibes
fuse_length = 16    
fuse_cabin_width = 1.9  
fuse_cabin_height = 1.85   
nose_length = 2.5             
cabin_length = 7.1             
tail_length = fuse_length - nose_length - cabin_length  

# Tail geometry
tail_arm = 7.0  #wing AC to tail AC

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

# BLI electric pusher prop design variables
bli_motor_power = opti.variable(
    init_guess=150000, log_transform=True, lower_bound=20000, upper_bound=500000
)  # Watts

bli_propeller_diameter = opti.variable(
    init_guess=1.2, lower_bound=0.6, upper_bound=1.8
)
bli_thrust_at_liftoff = opti.variable(
    init_guess=2000, lower_bound=100, log_transform=True
)  
# Cabin floor, goes whever best for cg
x_cg_battery = opti.variable(
    init_guess=nose_length + 0.25 * cabin_length,
    lower_bound=nose_length + 0.5, 
    upper_bound=nose_length + cabin_length - 0.5,      
)  # from nose

# OEI wingtip reduced thrust (optimizer can throttle back
# the surviving wingtip to cut yaw moment -> smaller vtail)
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

#  BLI Pusher Propulsor (tail-mounted, centerline) 
bli_propulsor = asb.Propulsor(
    name="BLI Pusher",
    xyz_c=[fuse_length + 0.3, 0, 0.5],     # Just aft of fuselage tail tip
    xyz_normal=[1, 0, 0],                   # Thrust in +x (aft-facing pusher)
    radius=bli_propeller_diameter / 2,
    length=0.3,
)

#  Assemble Airplane 
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

# Wing area from geometry
wing_area = wing.area()

# Miscellaneous Drag Correction 
# AeroBuildup omits protuberance drag (random stuff)
CDA_misc = 0.20 

# Drag correction factor
drag_correction_factor = 1.10  # 10% increase on AeroBuildup base drag

CD_misc = CDA_misc / wing_area
CL_cruise = aero["CL"]
CD_cruise = aero["CD"] * drag_correction_factor + CD_misc
L_over_D_cruise = CL_cruise / CD_cruise

q_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2
drag_cruise = CD_cruise * q_cruise * wing_area


#  BLI Drag Reduction 
# BLI ingests fuselage boundary layer wake -> reduces effective aircraft
# drag by bli_drag_reduction_factor. The wingtip props only need to
# overcome the reduced effective drag. The BLI prop must run at cruise
# to fill the wake deficit, consuming battery power.
drag_effective_cruise = drag_cruise * (1 - bli_drag_reduction_factor)
drag_bli_wake_fill = drag_cruise * bli_drag_reduction_factor      

##### Section: Weight Breakdown #####

# Structural

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
# Size governed by V_mc constraint
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
) * 0.7  # Fixed gear: no retraction mechanism ~70% lighter

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

# Systems

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

# Propulsion System

# Turboshaft
# CORRECTIVE FACTOR, .8 applied because the p/w ratio is optimistic?
power_per_turboshaft = power_turboshaft(mass_turboshaft_per_engine) * .8
m_turboshaft_total = mass_turboshaft_per_engine * n_engines 

# Electric motor: sized as a fraction of total takeoff power
# total_takeoff_power = turboshaft_power + electric_power
# hybridization_factor = electric_power / total_takeoff_power
# So: electric_power = hybridization_factor / (1 - hybridization_factor) * turboshaft_power
electric_power_per_engine = (
    hybridization_factor / (1 - hybridization_factor) * power_per_turboshaft
)

# ASSUMPTION
motor_power_density = 5000  # W/kg (5 kW/kg, aircraft-class certified electric motor, magnix is the same pretty sure)
m_motor_per_engine = electric_power_per_engine / motor_power_density
m_motor_total = m_motor_per_engine * n_engines

m_esc_per_engine = mass_ESC(electric_power_per_engine)
m_esc_total = m_esc_per_engine * n_engines

# Battery
m_battery = mass_battery_pack(
    battery_capacity_Wh,
    battery_cell_specific_energy_Wh_kg=battery_cell_specific_energy,
    battery_pack_cell_fraction=battery_pack_cell_fraction,
)

# CORRECTION FACTOR
# Propellers (Torenbeek) — ×0.35 for lightweight composite propellers
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

# -> BLI Pusher Propulsion (fully electric, tail-mounted) ->
m_bli_motor = bli_motor_power / motor_power_density  # 5 kW/kg
m_bli_esc = mass_ESC(bli_motor_power)
m_bli_propeller = torenbeek_wt.mass_propeller(
    propeller_diameter=bli_propeller_diameter,
    propeller_power=bli_motor_power,
    n_blades=5,  # vibes
) * 0.35  # Lightweight composite, same factor as wingtip props
m_bli_nacelle = 0.10 * m_bli_motor + 15  # Lightweight tail fairing + spinner

wing_aspect_ratio = wing_span ** 2 / wing_area

# V_NE ~ 1.4× cruise for FAR 23 commuter turboprops
V_NE = cruise_speed * 1.40
# V_flap ~ 1.8× stall speed (sea level)
V_flap = V_stall_sl * 1.8

suspended_mass_approx = design_mass_TOGW * 0.92  


# NEEDS REFINING
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

# Payload / Cabin 
m_pax = n_pax * mass_passenger
m_seats = n_pax * mass_seat("passenger") + n_crew * mass_seat("flight_deck")
m_lavs = mass_lavatories(n_pax, aircraft_type="short-haul")

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

#  CG at aft-loading (fuel depleted) 
# Most-aft CG: critical case 
mass_zfw = mass_total - fuel_mass
x_cg_aft = (moment_TOGW - fuel_mass * x_cg_fuel) / mass_zfw

#  Neutral Point
# Lift-curve slopes via Helmbold
a_w = 2 * np.pi * wing_aspect_ratio / (2 + np.sqrt(4 + wing_aspect_ratio ** 2))
a_h = 2 * np.pi * hstab_aspect_ratio / (2 + np.sqrt(4 + hstab_aspect_ratio ** 2))

# Downwash gradient at hstab (15% less downwash than conventional)
depsilon_dalpha = 2 * a_w / (np.pi * wing_aspect_ratio) * 0.85

# Dynamic pressure ratio at hstab (T-tail above wing wake)
eta_h = 0.90

# Hstab AC (quarter-chord of mean chord)
x_ac_hstab = hstab_x_le + 0.25 * (hstab_root_chord_val + hstab_tip_chord_val) / 2

# Wing-tail neutral point (Etkin)
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
static_margin = (x_np - x_cg_aft) / wing_MAC        # At aft CG (stability)
static_margin_TOGW = (x_np - x_cg_TOGW) / wing_MAC  # At fwd CG (trim)

##### Section: Propulsion and Performance #####

#  Cruise Power Balance 
# Wingtip props handle effective drag (after BLI reduction)
# BLI prop fills the wake deficit
wingtip_propulsive_area = n_engines * np.pi / 4 * propeller_diameter ** 2
bli_propulsive_area = np.pi / 4 * bli_propeller_diameter ** 2

# Need electric boost for econ mission
# Wingtip shaft power (turboshaft-only at cruise)
shaft_power_cruise_wingtip = propeller_shaft_power_from_thrust(
    thrust_force=drag_effective_cruise,
    area_propulsive=wingtip_propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=0.85 * wingtip_propeller_efficiency_bonus,
)

# BLI shaft power
shaft_power_cruise_bli = propeller_shaft_power_from_thrust(
    thrust_force=drag_bli_wake_fill,
    area_propulsive=bli_propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=bli_propeller_CoP,
)

opti.subject_to(bli_motor_power >= shaft_power_cruise_bli * get_limit(reqs, "REQ-001"))  # REQ-001

# Turboshaft must handle wingtip cruise power PLUS BLI electric power.
# During cruise the wingtip motors act as generators: turboshaft drives
# the propeller AND siphons extra shaft power through the motor/generator
bli_electric_demand_from_turboshaft = shaft_power_cruise_bli / generator_efficiency

shaft_power_cruise_per_engine_total = (
    shaft_power_cruise_wingtip + bli_electric_demand_from_turboshaft
) / n_engines

opti.subject_to(power_per_turboshaft >= shaft_power_cruise_per_engine_total * get_limit(reqs, "REQ-002"))  # REQ-002

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

# Fuel for max range mission (350 nmi) -> sizes fuel tanks and MTOW
cruise_time_max = design_range_max / cruise_speed
fuel_for_cruise_max = fuel_burn_rate_cruise * cruise_time_max

opti.subject_to(fuel_mass >= fuel_for_cruise_max + fuel_reserves + fuel_for_climb)  # REQ-003

# Fuel for typical mission (175 nmi) -> used for cruise optimization
cruise_time_typical = design_range_typical / cruise_speed
fuel_for_cruise_typical = fuel_burn_rate_cruise * cruise_time_typical
fuel_mass_typical = fuel_for_cruise_typical + fuel_reserves + fuel_for_climb

#  Takeoff Power (Hybrid Boost + BLI) 
# Total liftoff thrust = wingtip props + BLI pusher
# thrust_at_liftoff is the TOTAL thrust from all propulsors
wingtip_thrust_at_liftoff = thrust_at_liftoff - bli_thrust_at_liftoff

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

opti.subject_to(shaft_power_takeoff_wingtip >= wingtip_shaft_power_from_thrust_liftoff)  # REQ-004

# BLI shaft power required at liftoff
bli_shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=bli_thrust_at_liftoff,
    area_propulsive=bli_propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=bli_propeller_CoP,
)

opti.subject_to(bli_motor_power >= bli_shaft_power_from_thrust_liftoff)  # REQ-005

#  Battery Sizing 
# During cruise, turboshafts generate BLI electricity via wingtip
# motors-as-generators -> battery only needs to cover takeoff/climb:
electric_energy_wingtip_climb = electric_power_per_engine * n_engines * climb_time  # Joules
electric_energy_bli_climb = bli_motor_power * climb_time                            # Joules

total_electric_energy_Wh = (
    electric_energy_wingtip_climb + electric_energy_bli_climb
) / 3600

opti.subject_to(battery_capacity_Wh >= total_electric_energy_Wh / get_limit(reqs, "REQ-006"))  # REQ-006

##### Section: Constraints #####

#  Wing Geometry Constraints 
opti.subject_to(wing_aspect_ratio >= get_limit(reqs, "REQ-007"))  # REQ-007 Practical minimum for turboprop
opti.subject_to(wing_aspect_ratio <= get_limit(reqs, "REQ-008"))  # REQ-008 Practical maximum

# Cruise Lift = Weight (optimized for typical 175 nmi mission) 
# On a typical mission the aircraft takes off lighter than MTOW because
# it carries only fuel_mass_typical instead of the full fuel_mass.
lift_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2 * wing_area * CL_cruise
typical_mission_TOGW = design_mass_TOGW - (fuel_mass - fuel_mass_typical)
mid_cruise_weight = (typical_mission_TOGW - fuel_for_cruise_typical * 0.5) * g

opti.subject_to(lift_cruise >= mid_cruise_weight * get_limit(reqs, "REQ-010"))  # REQ-010
opti.subject_to(lift_cruise <= mid_cruise_weight * get_limit(reqs, "REQ-011"))  # REQ-011

# Field Length 
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

opti.subject_to(field_results["takeoff_total_distance"] <= field_length_req)  # REQ-012
opti.subject_to(field_results["landing_total_distance"] <= field_length_req)  # REQ-013
opti.subject_to(field_results["balanced_field_length"] <= field_length_req)   # REQ-014

# When one wingtip engine fails, the remaining propulsors are:
# 1 operative wingtip engine (turboshaft + electric motor)
# BLI pusher (fully electric, centerline, always operative)
# Need help on more failure analysis here:
# Example what if the electric bus fails
# The pilot can throttle the operative wingtip DOWN to reduce
# the asymmetric yaw moment, relying on the centerline BLI pusher to
# carry a larger share of the OEI thrust.  The optimizer chooses the
# best split that satisfies BOTH the climb-gradient AND V_mc constraints simultaneously.
thrust_per_wingtip_engine = wingtip_thrust_at_liftoff / n_engines

# Operative-wingtip thrust may not exceed full per-engine capability
opti.subject_to(thrust_wingtip_oei_reduced <= thrust_per_wingtip_engine)  # REQ-015

thrust_oei = thrust_wingtip_oei_reduced + bli_thrust_at_liftoff  # reduced wingtip + full BLI
thrust_over_weight_oei = thrust_oei / (design_mass_TOGW * g)
climb_gradient_oei = thrust_over_weight_oei - 1 / L_over_D_climb
opti.subject_to(climb_gradient_oei >= get_limit(reqs, "REQ-016"))  # REQ-016 FAR 23 minimum for 2-engine class

# Engine-Out Directional Control (V_mc <= V_stall, FAR 23) 
# Wingtip propellers create a large yaw moment when one engine fails.
# The BLI pusher is on the centerline -> zero yaw contribution.
# Because the operative wingtip can be throttled back (the optimizer
# chooses `thrust_wingtip_oei_reduced`), the yaw moment is set by
# THAT reduced thrust, not the full per-engine value.  The BLI
# pusher fills the remaining thrust needed for OEI climb.
#
# +10% for windmilling drag on the dead-engine propeller.
y_engine = wing_span / 2  
# ASSUMPTIONS/ CORRECTION FACTOR
yaw_moment_oei = thrust_wingtip_oei_reduced * y_engine * 1.10  # +10% windmilling

CL_vstab_max_rudder = 0.9      # Side-force coeff at max rudder deflection (~25°)
l_vt = tail_arm * 0.95         # CG to vstab aerodynamic center

q_vmc = yaw_moment_oei / (vstab_area * CL_vstab_max_rudder * l_vt)
V_mc = np.sqrt(2 * q_vmc / atmo_sl.density())

opti.subject_to(V_mc <= V_stall_sl)   # REQ-017 FAR 23.149: V_mc must not exceed V_s1

#  Hstab Sizing (Volume Coefficient) 
# Horizontal Tail Volume Coefficient (V_h) ensures longitudinal stability and control authority.
# V_h = (S_h * l_h) / (S_w * c_mac)
# Typical values for turboprops 0.8 - 1.2 from raymer
# Skycourier .9
l_h = tail_arm  
V_h_coefficient = (hstab_area * l_h) / (wing_area * wing_mean_chord)
opti.subject_to(V_h_coefficient >= get_limit(reqs, "REQ-018"))  # REQ-018

# Vertical Tail Volume Coefficient (V_v) ensures directional stability.
# V_v = (S_v * l_v) / (S_w * b)
l_v = tail_arm
V_v_coefficient = (vstab_area * l_v) / (wing_area * wing_span)

#  Longitudinal Stability (Static Margin) 
# V_h above is a geometric floor; these enforce actual stability accounting
# for component CG locations
# EH
opti.subject_to(static_margin >= get_limit(reqs, "REQ-019"))         # REQ-019
opti.subject_to(static_margin_TOGW <= get_limit(reqs, "REQ-020"))  # REQ-020      

#Vstab Geometry Limits 
vstab_aspect_ratio = vstab_span_val ** 2 / vstab_area
opti.subject_to(vstab_aspect_ratio >= get_limit(reqs, "REQ-021"))  # REQ-021
opti.subject_to(vstab_aspect_ratio <= get_limit(reqs, "REQ-022"))  # REQ-022

# BLI Propeller Diameter Constraint 
# BLI prop must fit on the tail cone (bounded by fuselage tail height)
# Need to look into physics more here
opti.subject_to(bli_propeller_diameter <= fuse_cabin_height * get_limit(reqs, "REQ-009"))  # REQ-009

opti.subject_to(bli_thrust_at_liftoff <= thrust_at_liftoff * get_limit(reqs, "REQ-023"))  # REQ-023 

# Mass Closure 
opti.subject_to(mass_total <= design_mass_TOGW)  # REQ-024

#  MTOW Limit (FAR 23 commuter category: 19,000 lb) 
opti.subject_to(design_mass_TOGW <= get_limit(reqs, "REQ-025") * u.lbm)  # REQ-025

##### Section: Objective #####

# Minimize fuel burn on the typical 175 nmi mission (80% of flights).
opti.minimize(fuel_mass_typical)

##### Section: Solve #####

sol = opti.solve(max_iter=1500)

##### Section: Results Summary #####

print("=" * 72)
print("   HE-19 HYBRID-ELECTRIC 19-PAX TURBOPROP + BLI -> DESIGN SUMMARY")
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
print(f"  V-Stab V_v Coeff:        {sol(V_v_coefficient):8.3f}")
print(f"  H-Stab Span:             {sol(hstab_span_val):8.2f} m")
print(f"  H-Stab Area:             {sol(hstab_area):8.2f} m^2")
print(f"  H-Stab V_h Coeff:        {sol(V_h_coefficient):8.2f}")

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

# Flight condition
rho_cruise = sol(cruise_atmo.density())
q_cruise_sol = sol(q_cruise)
mu_cruise = sol(cruise_atmo.dynamic_viscosity())
Re_mac = rho_cruise * cruise_speed * sol(wing_MAC) / mu_cruise
Mach_cruise = cruise_speed / sol(cruise_atmo.speed_of_sound())
print(f"  Cruise Speed:            {cruise_speed:8.1f} m/s  ({cruise_speed / u.knot:8.0f} kt)")
print(f"  Mach Number:             {Mach_cruise:8.3f}")
print(f"  Dynamic Pressure:        {q_cruise_sol:8.0f} Pa   ({q_cruise_sol / (u.lbf / u.foot**2):8.1f} psf)")
print(f"  Air Density:             {rho_cruise:8.4f} kg/m^3")
print(f"  Re (MAC):                {Re_mac:8.2e}")

# Lift & drag
print(f"  CL:                      {sol(CL_cruise):8.4f}")
print(f"  CD:                      {sol(CD_cruise):8.5f}")
print(f"  L/D (cruise):            {sol(L_over_D_cruise):8.1f}")
print(f"  L/D (climb, 65%%):       {sol(L_over_D_climb):8.1f}")
print(f"  Alpha:                   {sol(cruise_alpha):8.1f} deg")
print(f"  Lift (cruise):           {sol(lift_cruise):8.0f} N   ({sol(lift_cruise) / u.lbf:8.0f} lbf)")
print(f"  Cruise Drag:             {sol(drag_cruise):8.0f} N   ({sol(drag_cruise) / u.lbf:8.0f} lbf)")
print(f"  BLI Drag Reduction:      {bli_drag_reduction_factor:8.0%}")
print(f"  Effective Drag (w/ BLI): {sol(drag_effective_cruise):8.0f} N   ({sol(drag_effective_cruise) / u.lbf:8.0f} lbf)")
print(f"  BLI Wake Fill Thrust:    {sol(drag_bli_wake_fill):8.0f} N   ({sol(drag_bli_wake_fill) / u.lbf:8.0f} lbf)")

# Stability derivatives
print(f"  CL_alpha (wing):         {sol(a_w):8.4f} /rad")
print(f"  CL_alpha (hstab):        {sol(a_h):8.4f} /rad")
print(f"  de/da (downwash):        {sol(depsilon_dalpha):8.4f}")
print(f"  eta_h (tail dp ratio):   {eta_h:8.2f}")
print(f"  Cm (AeroBuildup):        {sol(aero['Cm']):8.4f}")

print(f"\n{' Propulsion (Wingtip Parallel Hybrid) ':^72}")
print(f"  Turboshaft Mass (each):  {sol(mass_turboshaft_per_engine):8.1f} kg  ({sol(mass_turboshaft_per_engine) / u.lbm:8.0f} lb)")
print(f"  Turboshaft Power (each): {sol(power_per_turboshaft) / u.horsepower:8.0f} hp  ({sol(power_per_turboshaft) / 1000:8.0f} kW)")
print(f"  Turboshaft Power (total):{sol(power_per_turboshaft) * n_engines / u.horsepower:8.0f} hp")
print(f"  Electric Motor (each):   {sol(electric_power_per_engine) / 1000:8.0f} kW  ({sol(electric_power_per_engine) / u.horsepower:8.0f} hp)")
print(f"  Hybridization Factor:    {sol(hybridization_factor):8.1%}")
print(f"  Wingtip TO Power (both): {sol(shaft_power_takeoff_wingtip) / u.horsepower:8.0f} hp")
print(f"  Propeller Diameter:      {sol(propeller_diameter):8.2f} m   ({sol(propeller_diameter) / u.foot:8.1f} ft)")

print(f"\n{' Propulsion (BLI Electric Pusher) ':^72}")
print(f"  BLI Motor Power:         {sol(bli_motor_power) / 1000:8.0f} kW  ({sol(bli_motor_power) / u.horsepower:8.0f} hp)")
print(f"  BLI Motor Mass:          {sol(m_bli_motor):8.1f} kg  ({sol(m_bli_motor) / u.lbm:8.0f} lb)")
print(f"  BLI ESC Mass:            {sol(m_bli_esc):8.1f} kg  ({sol(m_bli_esc) / u.lbm:8.0f} lb)")
print(f"  BLI Propeller Diameter:  {sol(bli_propeller_diameter):8.2f} m   ({sol(bli_propeller_diameter) / u.foot:8.1f} ft)")
print(f"  BLI Propeller Mass:      {sol(m_bli_propeller):8.1f} kg  ({sol(m_bli_propeller) / u.lbm:8.0f} lb)")
print(f"  BLI Nacelle Mass:        {sol(m_bli_nacelle):8.1f} kg  ({sol(m_bli_nacelle) / u.lbm:8.0f} lb)")
print(f"  BLI Thrust at Liftoff:   {sol(bli_thrust_at_liftoff):8.0f} N   ({sol(bli_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  BLI Cruise Shaft Power:  {sol(shaft_power_cruise_bli) / 1000:8.1f} kW  ({sol(shaft_power_cruise_bli) / u.horsepower:8.0f} hp)")

print(f"\n{' Battery (takeoff/climb only; cruise BLI from turboshaft) ':^72}")
print(f"  Battery Capacity:        {sol(battery_capacity_Wh) / 1000:8.1f} kWh")
print(f"  Battery Mass:            {m_batt_sol:8.1f} kg  ({m_batt_sol / u.lbm:8.0f} lb)")
print(f"  Wingtip Climb Energy:    {sol(electric_energy_wingtip_climb) / 3600 / 1000:8.1f} kWh")
print(f"  BLI Climb Energy:        {sol(electric_energy_bli_climb) / 3600 / 1000:8.1f} kWh")
print(f"  Total Electric Energy:   {sol(total_electric_energy_Wh) / 1000:8.1f} kWh")

print(f"\n{' Aerodynamic Drag Breakdown ':^72}")
CD_aerobuildup = sol(aero["CD"])
CD_total = sol(CD_cruise)
CD_misc_sol = sol(CD_misc)
print(f"  AeroBuildup CD (raw):    {CD_aerobuildup:8.5f}")
print(f"  After 10% correction:    {CD_aerobuildup * 1.10:8.5f}")
print(f"  Misc drag CD (CDA/S):    {CD_misc_sol:8.5f}  (CDA_misc = {CDA_misc:.2f} m^2)")
print(f"  Total CD (corrected):    {CD_total:8.5f}")

print(f"\n{' Cruise Performance ':^72}")
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

print(f"\n{' Field Performance ':^72}")
print(f"  V_stall (SL):            {sol(field_results['V_stall']) / u.knot:8.1f} kts ({sol(field_results['V_stall']):8.1f} m/s)")
print(f"  V_liftoff:               {sol(field_results['V_liftoff']) / u.knot:8.1f} kts")
print(f"  Takeoff Ground Roll:     {sol(field_results['takeoff_ground_roll_distance']) / u.foot:8.0f} ft  ({sol(field_results['takeoff_ground_roll_distance']):8.0f} m)")
print(f"  Takeoff Total Distance:  {sol(field_results['takeoff_total_distance']) / u.foot:8.0f} ft  ({sol(field_results['takeoff_total_distance']):8.0f} m)")
print(f"  Balanced Field Length:   {sol(field_results['balanced_field_length']) / u.foot:8.0f} ft  ({sol(field_results['balanced_field_length']):8.0f} m)")
print(f"  Landing Total Distance:  {sol(field_results['landing_total_distance']) / u.foot:8.0f} ft  ({sol(field_results['landing_total_distance']):8.0f} m)")
print(f"  Climb Gradient (AEO):    {sol(field_results['flight_path_angle_climb']):8.4f} rad ({sol(field_results['flight_path_angle_climb']) * 100:8.2f}%)")
print(f"  Climb Gradient (OEI):    {sol(field_results['flight_path_angle_climb_one_engine_out']):8.4f} rad ({sol(field_results['flight_path_angle_climb_one_engine_out']) * 100:8.2f}%)  (Torenbeek, conservative)")
print(f"  Climb Gradient (OEI+BLI):{sol(climb_gradient_oei):8.4f} rad ({sol(climb_gradient_oei) * 100:8.2f}%)  (reduced wingtip + BLI)")
print(f"  OEI Wingtip Thrust:      {sol(thrust_wingtip_oei_reduced):8.0f} N   ({sol(thrust_wingtip_oei_reduced) / u.lbf:8.0f} lbf)  [{sol(thrust_wingtip_oei_reduced)/sol(thrust_per_wingtip_engine)*100:.0f}% of max]")
print(f"  Thrust/Weight (TO):      {sol(thrust_at_liftoff) / (TOGW * g):8.3f}")
print(f"  Wingtip Thrust at TO:    {sol(wingtip_thrust_at_liftoff):8.0f} N   ({sol(wingtip_thrust_at_liftoff) / u.lbf:8.0f} lbf)")
print(f"  BLI Thrust at TO:        {sol(bli_thrust_at_liftoff):8.0f} N   ({sol(bli_thrust_at_liftoff) / u.lbf:8.0f} lbf)")

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
print(f"  OEI grad:  {sol(climb_gradient_oei):.4f} vs 0.024 min (reduced wingtip + BLI)")
print(f"  V_mc:      {sol(V_mc) / u.knot:.1f} kts vs V_stall {sol(V_stall_sl) / u.knot:.1f} kts")
print(f"  Wingtip T: {sol(wingtip_thrust_at_liftoff):,.0f} N  (BLI offloads {sol(bli_thrust_at_liftoff):,.0f} N)")
print(f"  OEI wingtip (reduced): {sol(thrust_wingtip_oei_reduced):,.0f} N  ({sol(thrust_wingtip_oei_reduced)/sol(thrust_per_wingtip_engine)*100:.0f}% of full)")
print(f"  AR:        {AR:.2f} vs [6.0, 16.0] bounds")
print(f"  Hybrid:    {sol(hybridization_factor):.1%} vs [20%, 70%] bounds")
print(f"  BLI prop:  {sol(bli_propeller_diameter):.2f} m vs {fuse_cabin_height * 1.1:.2f} m max")
print(f"  SM aft:    {sol(static_margin)*100:.1f}% MAC vs 5% min")
print(f"  SM fwd:    {sol(static_margin_TOGW)*100:.1f}% MAC vs 40% max")

print("\n" + "=" * 72)

##### Section: AVL Validation (Post-Solve) #####
# Run Athena Vortex Lattice on the solved geometry to cross-check
# AeroBuildup lift/moment predictions. AVL computes inviscid (induced)
# aerodynamics only viscous drag must still come from AeroBuildup or
# flat-plate estimates. This block is optional: if AVL is not installed,
# it is silently skipped.

try:
    sol_airplane_avl = sol(airplane)
    avl_op_point = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=sol(cruise_altitude)),
        velocity=cruise_speed,
        alpha=sol(cruise_alpha),
    )

    avl = asb.AVL(
        airplane=sol_airplane_avl,
        op_point=avl_op_point,
        avl_command=r"C:\Users\dylan\Downloads\avl352.exe",
        timeout=10,
    )
    avl_aero = avl.run()

    # Extract AeroBuildup values for comparison
    ab_CL = sol(CL_cruise)
    ab_CD = sol(CD_cruise)                   # includes 10% correction + misc
    ab_CDi = sol(aero["CD"])                 # raw AeroBuildup (profile + induced)
    ab_Cm = sol(aero["Cm"]) if "Cm" in aero else None

    avl_CL = avl_aero["CL"]
    avl_CDi = avl_aero["CD"]                 # inviscid induced drag only
    avl_Cm = avl_aero.get("Cm", None)

    print(f"\n{'':=^72}")
    print(f"{'  AVL VALIDATION (Athena Vortex Lattice)  ':^72}")
    print(f"{'':=^72}")
    print(f"\n  {'Parameter':<28} {'AeroBuildup':>12} {'AVL':>12} {'Delta':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'CL':<28} {ab_CL:12.4f} {avl_CL:12.4f} {(avl_CL - ab_CL):+10.4f}")
    print(f"  {'CD (induced only)':<28} {ab_CDi:12.5f} {avl_CDi:12.5f} {(avl_CDi - ab_CDi):+10.5f}")
    if ab_Cm is not None and avl_Cm is not None:
        print(f"  {'Cm (pitch moment)':<28} {ab_Cm:12.4f} {avl_Cm:12.4f} {(avl_Cm - ab_Cm):+10.4f}")
    print(f"\n  {'CD total (AB corrected)':<28} {ab_CD:12.5f}")
    print(f"  {'CD visc estimate (AB-AVL)':<28} {(ab_CD - avl_CDi):12.5f}")
    print(f"  {'L/D (AVL CL / AB CD tot)':<28} {(avl_CL / ab_CD):12.1f}   (AB: {sol(L_over_D_cruise):.1f})")

    # Stability derivatives (bonus — if available)
    try:
        avl_stab = asb.AVL(
            airplane=sol_airplane_avl,
            op_point=avl_op_point,
            avl_command=r"C:\Users\dylan\Downloads\avl352.exe",
            timeout=10,
        ).run_with_stability_derivatives()

        if "CLa" in avl_stab:
            print(f"\n  {'--- AVL Stability Derivatives ---':^60}")
            for key in ["CLa", "Cma", "Cmq", "CYb", "Clb", "Cnb", "Cnr"]:
                if key in avl_stab:
                    print(f"  {key:<28} {avl_stab[key]:12.4f}")
    except Exception:
        pass  # stability derivatives not critical

    print()
    print("  Note: AVL computes inviscid (vortex-lattice) aerodynamics only.")
    print("  Profile / viscous drag is NOT included in AVL CD.")
    print("=" * 72)

except FileNotFoundError:
    print("\n  [AVL validation skipped — 'avl' executable not found on PATH]")
    print("  Install AVL from https://web.mit.edu/drela/Public/web/avl/")
    print("  and ensure it is on your system PATH, or edit avl_command above.\n")
except Exception as _avl_err:
    print(f"\n  [AVL validation skipped: {_avl_err}]\n")

#  Optional: Draw 3-view 
try:
    sol_airplane = sol(airplane)
    axs = sol_airplane.draw_three_view(show=False)


    # Draw CG 
    cg_x = sol(x_cg_TOGW)
    cg_y = 0.0
    cg_z = 0.0
    for ax in axs.flat:
        ax.plot(
            [cg_x], [cg_y], [cg_z],
            marker="o", color="red", markersize=3, zorder=999,
            label="CG",
        )

    # Draw Neutral Point 
    np_x = sol(x_np)
    np_y = 0.0
    np_z = 0.0
    for ax in axs.flat:
        ax.plot(
            [np_x], [np_y], [np_z],
            marker="o", color="blue", markersize=3, zorder=998,
            label="NP",
        )

    axs[0, 0].legend(fontsize=8, loc="upper right")

    import matplotlib.pyplot as plt
    plt.show()
except Exception:
    pass  # Skip drawing if display not available

##### Section: Requirements Traceability #####

# Post-solve: validate all constraints against requirements.yaml
try:
    _sv = {
        "REQ-001": sol(bli_motor_power) / sol(shaft_power_cruise_bli),
        "REQ-002": sol(power_per_turboshaft) / sol(shaft_power_cruise_per_engine_total),
        # REQ-003, REQ-004, REQ-005: expression-only (limit is null)
        "REQ-007": sol(wing_aspect_ratio),
        "REQ-008": sol(wing_aspect_ratio),
        "REQ-009": sol(bli_propeller_diameter) / fuse_cabin_height,
        "REQ-010": sol(lift_cruise) / sol(mid_cruise_weight),
        "REQ-011": sol(lift_cruise) / sol(mid_cruise_weight),
        "REQ-012": sol(field_results["takeoff_total_distance"]),
        "REQ-013": sol(field_results["landing_total_distance"]),
        "REQ-014": sol(field_results["balanced_field_length"]),
        "REQ-016": sol(climb_gradient_oei),
        "REQ-018": sol(V_h_coefficient),
        "REQ-019": sol(static_margin),
        "REQ-020": sol(static_margin_TOGW),
        "REQ-021": sol(vstab_aspect_ratio),
        "REQ-022": sol(vstab_aspect_ratio),
        "REQ-023": sol(bli_thrust_at_liftoff) / sol(thrust_at_liftoff),
        "REQ-025": sol(design_mass_TOGW) / u.lbm,
    }
    validate_solution(reqs, _sv)
except Exception as _e:
    print(f"  [requirements validation skipped: {_e}]")

##### Section: Constraint Diagram #####
try:
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq
    from pathlib import Path
    import numpy as _np  # plain numpy for constraint-diagram arrays

    print(f"\n{'='*72}")
    print(f"  Generating Constraint Diagram (Power Loading)...")
    print(f"{'='*72}")

    # ---- Unit helpers ----
    _psf_to_Pa = float(u.lbf / u.foot**2)        # 1 psf in Pa
    _W_per_hplb = float(u.horsepower / u.lbf)     # W per (hp/lb)  ≈ 1643.9

    # ---- Propeller efficiencies (same as used in the optimizer) ----
    eta_p_TO     = 0.80 * wingtip_propeller_efficiency_bonus   # takeoff/climb
    eta_p_cruise = 0.85 * wingtip_propeller_efficiency_bonus   # cruise
    eta_p_ceil   = eta_p_cruise                                # ceiling ≈ cruise

    # ---- Solved Design Point ----
    WS_design_Pa  = TOGW * g / S_wing
    WS_design_psf = WS_design_Pa / _psf_to_Pa
    TW_design     = sol(thrust_at_liftoff) / (TOGW * g)
    V_stall_solved = sol(field_results['V_stall'])            # m/s
    V_stall_kts    = V_stall_solved / u.knot
    V_lof_design   = 1.2 * V_stall_solved
    cruise_alt_solved = sol(cruise_altitude)
    cruise_alt_ft     = cruise_alt_solved / u.foot
    LD_cruise_val  = sol(L_over_D_cruise)
    LD_climb_val   = LD_cruise_val * 0.65                     # climb config

    # Design-point P/W  (installed power at liftoff → W/N → hp/lb)
    PW_design_SI = TW_design * V_lof_design / eta_p_TO       # W/N
    PW_design_hplb = PW_design_SI / _W_per_hplb              # hp/lb

    # ---- Atmospheres ----
    rho_sl_val      = 1.225
    rho_cruise_val  = float(asb.Atmosphere(altitude=cruise_alt_solved).density())
    rho_ceiling_val = float(asb.Atmosphere(altitude=10000 * u.foot).density())

    # ---- Drag-polar approximation ----
    CD0_approx = 0.04 * 1.10 + 0.20 / S_wing
    CL_cr_sol  = sol(CL_cruise)
    if CL_cr_sol > 0.01:
        k_drag = (CD_total - CD0_approx) / (CL_cr_sol ** 2)
    else:
        k_drag = 1.0 / (_np.pi * AR * 0.75)
    e_oswald = 1.0 / (_np.pi * AR * k_drag) if k_drag > 0 else 0.75

    print(f"  CD0 (approx):  {CD0_approx:.5f}")
    print(f"  k (induced):   {k_drag:.5f}")
    print(f"  e (Oswald):    {e_oswald:.3f}")
    print(f"  L/D cruise:    {LD_cruise_val:.2f}")
    print(f"  L/D climb:     {LD_climb_val:.2f}")
    print(f"  eta_p TO:      {eta_p_TO:.4f}")
    print(f"  eta_p cruise:  {eta_p_cruise:.4f}")

    # ---- W/S sweep ----
    WS_psf_arr = _np.linspace(10, 120, 500)
    WS_Pa_arr  = WS_psf_arr * _psf_to_Pa
    V_stall_arr = _np.sqrt(2.0 * WS_Pa_arr / (rho_sl_val * CL_max))
    V_lof_arr   = 1.2 * V_stall_arr

    # ================================================================
    # 1. Takeoff  (BFL = 2600 ft)  — solve for T/W then convert to P/W
    # ================================================================
    BFL_m       = float(field_length_req)
    obs_ht      = float(50 * u.foot)
    CD0_field   = 0.04
    mu_friction = 0.02
    max_braking = 0.37
    inertia_t   = 4.5
    min_gamma_oei = 0.024

    def _TW_for_BFL(ws_pa):
        """Torenbeek Eq. 5-89 BFL inversion."""
        def _res(tw):
            V_st  = _np.sqrt(2.0 * ws_pa / (rho_sl_val * CL_max))
            V_lof = 1.2 * V_st
            mu_eff  = mu_friction + 0.72 * CD0_field / CL_max
            accel_g = tw - mu_eff
            if accel_g <= 1e-6:
                return 1e6
            gamma_oei = tw * (n_engines - 1) / n_engines - 1.0 / LD_climb_val
            gamma_bar = 0.06 + (gamma_oei - min_gamma_oei)
            bfl = ((V_lof ** 2 / (2.0 * g * (1.0 + gamma_bar / max_braking))) *
                   (1.0 / accel_g + 1.0 / max_braking) *
                   (1.0 + (2.0 * g * obs_ht) / V_lof ** 2) +
                   inertia_t * V_lof)
            return bfl - BFL_m
        try:
            return brentq(_res, 0.02, 5.0)
        except Exception:
            return float('nan')

    TW_takeoff_arr = _np.array([_TW_for_BFL(ws) for ws in WS_Pa_arr])
    PW_takeoff_arr = TW_takeoff_arr * V_lof_arr / eta_p_TO / _W_per_hplb

    # ================================================================
    # 2. OEI Climb  (γ = 2.4 %)  — T/W is constant → P/W varies with V_lof
    # ================================================================
    gamma_oei_req = 0.024
    TW_oei = float(n_engines) / (n_engines - 1) * (gamma_oei_req + 1.0 / LD_climb_val)
    PW_oei_arr = TW_oei * V_lof_arr / eta_p_TO / _W_per_hplb

    # ================================================================
    # 3. AEO Climb  (γ = 5.0 %)  — T/W is constant → P/W varies with V_lof
    # ================================================================
    gamma_aeo_req = 0.05
    TW_aeo = gamma_aeo_req + 1.0 / LD_climb_val
    PW_aeo_arr = TW_aeo * V_lof_arr / eta_p_TO / _W_per_hplb

    # ================================================================
    # 4. Service Ceiling  (10 000 ft, RoC ≥ 100 ft/min)
    # ================================================================
    RoC_ceil   = float(100 * u.foot) / 60.0
    V_cr       = float(cruise_speed)
    q_ceil     = 0.5 * rho_ceiling_val * V_cr ** 2
    TW_ceil_arr = (q_ceil * CD0_approx / WS_Pa_arr
                   + WS_Pa_arr * k_drag / q_ceil
                   + RoC_ceil / V_cr)
    PW_ceil_arr = TW_ceil_arr * V_cr / eta_p_ceil / _W_per_hplb

    # ================================================================
    # 5. Cruise  (200 kts @ solved altitude)
    # ================================================================
    q_cr_cruise   = 0.5 * rho_cruise_val * V_cr ** 2
    TW_cruise_arr = q_cr_cruise * CD0_approx / WS_Pa_arr + WS_Pa_arr * k_drag / q_cr_cruise
    PW_cruise_arr = TW_cruise_arr * V_cr / eta_p_cruise / _W_per_hplb

    # ================================================================
    # 6. Stall-speed limit  (vertical line)
    # ================================================================
    WS_stall_Pa  = 0.5 * rho_sl_val * V_stall_solved ** 2 * CL_max
    WS_stall_psf = WS_stall_Pa / _psf_to_Pa

    # ================================================================
    # 7. Landing  (LFL = 2600 ft, vertical line)
    # ================================================================
    landing_dist_sol = float(sol(field_results['landing_total_distance']))
    if landing_dist_sol > 0:
        WS_landing_psf = WS_design_psf * (BFL_m / landing_dist_sol)
    else:
        WS_landing_psf = 120.0

    # ================================================================
    #  PLOT
    # ================================================================
    fig_cd, ax_cd = plt.subplots(figsize=(12, 8))

    ax_cd.plot(WS_psf_arr, PW_takeoff_arr, color='darkblue', linewidth=2.5,
               label=f"Takeoff (BFL = {field_length_req / u.foot:.0f} ft)")

    ax_cd.plot(WS_psf_arr, PW_oei_arr, color='darkgreen', linewidth=2.5,
               label=f"OEI Climb (\u03b3 = {gamma_oei_req * 100:.1f}%)")

    ax_cd.plot(WS_psf_arr, PW_aeo_arr, color='cyan', linewidth=2.5,
               label=f"AEO Climb (\u03b3 = {gamma_aeo_req * 100:.1f}%)")

    ax_cd.plot(WS_psf_arr, PW_ceil_arr, color='purple', linewidth=2.5,
               label="Service Ceiling (10,000 ft)")

    ax_cd.plot(WS_psf_arr, PW_cruise_arr, color='orange', linewidth=2.5,
               label=f"Cruise ({cruise_speed / u.knot:.0f} kts @ {cruise_alt_ft:,.0f} ft)")

    # Stall speed  (red dashed vertical)
    ax_cd.axvline(x=WS_stall_psf, color='red', linestyle='--', linewidth=2,
                  label=f"Stall Speed ({V_stall_kts:.1f} kts): W/S = {WS_stall_psf:.1f}")

    # Landing  (green dashed vertical)
    ax_cd.axvline(x=WS_landing_psf, color='green', linestyle='--', linewidth=2,
                  label=f"Landing (LFL = {field_length_req / u.foot:.0f} ft): W/S = {WS_landing_psf:.1f}")

    # ---- Feasible region ----
    PW_envelope = _np.maximum.reduce([
        _np.nan_to_num(PW_takeoff_arr, nan=100.0),
        PW_oei_arr,
        PW_aeo_arr,
        PW_ceil_arr,
        PW_cruise_arr,
    ])
    WS_upper_lim  = min(WS_stall_psf, WS_landing_psf)
    feasible_mask  = WS_psf_arr <= WS_upper_lim
    y_top = max(PW_design_hplb * 3.0, 0.20)
    ax_cd.fill_between(
        WS_psf_arr[feasible_mask],
        PW_envelope[feasible_mask],
        y_top,
        color='green', alpha=0.15,
        label="Feasible Region",
    )

    # ---- Design point ----
    ax_cd.plot(WS_design_psf, PW_design_hplb, 'o',
               color='yellow', markersize=15,
               markeredgecolor='black', markeredgewidth=2,
               zorder=10,
               label=f"Design Point (W/S={WS_design_psf:.1f}, P/W={PW_design_hplb:.4f})")

    ax_cd.annotate(
        f"Design Point\nW/S = {WS_design_psf:.1f} lb/ft\u00b2\nP/W = {PW_design_hplb:.4f} hp/lb",
        xy=(WS_design_psf, PW_design_hplb),
        xytext=(WS_design_psf + 12, PW_design_hplb + y_top * 0.12),
        fontsize=10, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='black', alpha=0.8),
    )

    ax_cd.set_xlabel("Wing Loading W/S (lb/ft\u00b2)", fontsize=12)
    ax_cd.set_ylabel("Power Loading P/W (hp/lb)", fontsize=12)
    ax_cd.set_title("Constraint Diagram", fontsize=16, fontweight='bold')
    ax_cd.set_xlim(10, 120)
    ax_cd.set_ylim(0, y_top)
    ax_cd.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_cd.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path_cd = Path(__file__).parent / "constraint_diagram_BLI_Big.png"
    fig_cd.savefig(str(out_path_cd), dpi=150, bbox_inches='tight')
    print(f"  Saved constraint diagram to: {out_path_cd}")
    print(f"  Design Point  W/S = {WS_design_psf:.1f} psf,  P/W = {PW_design_hplb:.4f} hp/lb")
    plt.show()
    plt.close(fig_cd)

except Exception as _e:
    import traceback
    traceback.print_exc()
    print(f"  [Constraint diagram skipped: {_e}]")
