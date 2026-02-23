"""
Validation Case: Cessna 408 SkyCourier -- Conventional 19-Pax Turboprop
=======================================================================

Validates the AeroSandbox weight/aero/propulsion framework by modeling
a conventional (non-hybrid) turboprop configured to Cessna SkyCourier
requirements and comparing optimizer output to known real-aircraft data.

Real Aircraft Reference (Cessna 408 SkyCourier, passenger variant):
    MTOW:            19,000 lb (8,618 kg)
    Empty Weight:    ~12,325 lb (~5,591 kg)
    Fuel Capacity:   2,725 L (~2,189 kg)
    Cruise Speed:    200 KTAS at FL150
    Engines:         2x P&W PT6A-65SC, 1,100 shp each (235 kg dry)
    Propellers:      McCauley 4-blade, 110 in (2.79 m), 82 kg each
    Wing Span:       72 ft 3 in (22.0 m)
    Wing Area:       441 ft² (41.0 m²)
    AR:              11.8 (strut-braced high wing)
    TO Distance:     3,300 ft (over 50 ft, pax variant)
    LDG Distance:    ~3,010 ft
    Range (max pld): ~400 nmi
    Landing Gear:    Fixed tricycle
    Pressurization:  Unpressurized
    Tail:            T-tail
"""

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

# Propulsion models
from aerosandbox.library.power_turboshaft import (
    power_turboshaft,
    thermal_efficiency_turboshaft,
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

##### Section: Mission Constants #####

n_pax = 19
n_crew = 2
payload_mass = 6000 * u.lbm              # 2722 kg
cruise_speed = 200 * u.knot              # 102.9 m/s
cruise_altitude = 15000 * u.foot         # 4572 m
field_length_req = 3300 * u.foot         # 1006 m (real SkyCourier pax variant)
n_engines = 2
design_range_max = 400 * u.naut_mile     # 741 km (max payload range)
design_range_typical = 200 * u.naut_mile # 370 km (typical mission)
ultimate_load_factor = 1.5 * 3.0         # FAR 23 commuter
CL_max = 2.4                            # High wing + full flaps
g = 9.81

# Fuel properties (Jet-A)
fuel_density = 820          # kg/m^3
fuel_specific_energy = 43.02e6  # J/kg, lower heating value

# Fuselage geometry (Cessna 408 SkyCourier)
fuse_length = 16.7             # m (54.8 ft)
fuse_cabin_width = 2.13        # m (7.0 ft external, ~square cross section)
fuse_cabin_height = 2.13       # m (7.0 ft external)
nose_length = 3.0              # m
cabin_length = 8.0             # m (19 pax, 2+1 abreast)
tail_length = fuse_length - nose_length - cabin_length  # ~5.7 m

# Tail geometry (T-tail)
tail_arm = 8.0                 # m, wing AC to tail AC
hstab_span_val = 7.0           # m
hstab_root_chord_val = 1.5     # m
hstab_tip_chord_val = 0.75     # m
vstab_span_val = 3.5           # m (tall for T-tail)
vstab_root_chord_val = 2.5     # m
vstab_tip_chord_val = 1.2      # m

# Strut-braced wing reduces wing bending moments → ~30% lighter wing
strut_wing_weight_factor = 0.70

# Fixed landing gear: lighter (no retraction mechanism) but draggier
fixed_gear_weight_factor = 0.70

##### Section: Optimization Setup #####

opti = asb.Opti()

##### Section: Design Variables (8 -- no hybrid) #####

design_mass_TOGW = opti.variable(
    init_guess=8600, log_transform=True, lower_bound=5000, upper_bound=12000
)
wing_span = opti.variable(
    init_guess=22.0, lower_bound=15, upper_bound=28
)
wing_root_chord = opti.variable(
    init_guess=2.8, lower_bound=1.5, upper_bound=5.0
)
cruise_alpha = opti.variable(
    init_guess=3.0, lower_bound=-2, upper_bound=10
)
mass_turboshaft_per_engine = opti.variable(
    init_guess=235, log_transform=True, lower_bound=80, upper_bound=400
)
propeller_diameter = opti.variable(
    init_guess=2.79, lower_bound=2.0, upper_bound=3.5
)
fuel_mass = opti.variable(
    init_guess=1200, log_transform=True, lower_bound=100, upper_bound=3000
)
thrust_at_liftoff = opti.variable(
    init_guess=8600 * g * 0.30, log_transform=True, lower_bound=5000
)

# Derived wing geometry
wing_taper_ratio = 0.45
wing_tip_chord = wing_root_chord * wing_taper_ratio
wing_mean_chord = (wing_root_chord + wing_tip_chord) / 2

##### Section: Aircraft Geometry #####

# --- Fuselage (larger cross-section than Beech 1900D) ---
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

# --- Main Wing (HIGH wing, strut-braced) ---
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
                0.6 * wing_span / 2 * np.tand(5),  # 5° dihedral
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
                wing_span / 2 * np.tand(5),
            ],
            chord=wing_tip_chord,
            twist=-1,
            airfoil=asb.Airfoil("naca23012"),
        ),
    ],
).translate([wing_x_le, 0, wing_z_le])

# --- Horizontal Stabilizer (T-tail: on top of vertical) ---
elevator = asb.ControlSurface(
    name="Elevator", symmetric=True, deflection=0, hinge_point=0.70
)

hstab_x_le = wing_x_le + 0.25 * wing_root_chord + tail_arm - 0.25 * hstab_root_chord_val
hstab_z_le = vstab_span_val + 0.3  # T-tail: on top of V-stab

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

# --- Vertical Stabilizer (T-tail, tall) ---
rudder = asb.ControlSurface(
    name="Rudder", symmetric=True, deflection=0, hinge_point=0.70
)

vstab_x_le = hstab_x_le - 1.0
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
    name="Cessna 408 SkyCourier (Validation)",
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
# Higher CDA_misc than HE-19 due to:
#   - Fixed landing gear and fairings (~0.15 m²)
#   - Larger fuselage cross-section (more protuberances)
#   - Antennas, scoops, door gaps, cooling inlets (~0.20 m²)
CDA_misc = 0.40  # m² flat-plate equivalent drag area

drag_correction_factor = 1.10  # 10% on AeroBuildup base drag

CD_misc = CDA_misc / wing_area
CL_cruise = aero["CL"]
CD_cruise = aero["CD"] * drag_correction_factor + CD_misc
L_over_D_cruise = CL_cruise / CD_cruise

q_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2
drag_cruise = CD_cruise * q_cruise * wing_area

##### Section: Weight Breakdown #####

# -- Structural --

# Wing (Torenbeek, with strut-bracing reduction factor)
m_wing_cantilever = torenbeek_wt.mass_wing(
    wing=wing,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    suspended_mass=design_mass_TOGW * 0.90,
    never_exceed_airspeed=cruise_speed * 1.4,
    max_airspeed_for_flaps=cruise_speed * 0.55,
    main_gear_mounted_to_wing=True,
)
m_wing = m_wing_cantilever * strut_wing_weight_factor

# Horizontal stabilizer (Raymer)
wing_to_hstab_distance = tail_arm
m_hstab = raymer_wt.mass_hstab(
    hstab=hstab,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    wing_to_hstab_distance=wing_to_hstab_distance,
    fuselage_width_at_hstab_intersection=fuse_cabin_width,
)

# Vertical stabilizer (Raymer) -- heavier for T-tail (carries hstab loads)
wing_to_vstab_distance = tail_arm * 0.95
m_vstab_base = raymer_wt.mass_vstab(
    vstab=vstab,
    design_mass_TOGW=design_mass_TOGW,
    ultimate_load_factor=ultimate_load_factor,
    wing_to_vstab_distance=wing_to_vstab_distance,
)
m_vstab = m_vstab_base * 1.25  # T-tail penalty: vstab carries hstab loads

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

# Landing gear (fixed -- lighter than retractable)
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
) * fixed_gear_weight_factor

m_nlg = raymer_wt.mass_nose_landing_gear(
    nose_gear_length=0.6,
    design_mass_TOGW=design_mass_TOGW,
    n_wheels=2,
) * fixed_gear_weight_factor

# Nacelles
nacelle_length = 2.0
nacelle_width = 0.8
nacelle_height = 0.8

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
    system_electrical_power_rating=40000,  # Conventional (no hybrid bus)
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

# -- Propulsion (Conventional Turboprop -- no electric) --

# Turboshaft
power_per_turboshaft = power_turboshaft(mass_turboshaft_per_engine)
m_turboshaft_total = mass_turboshaft_per_engine * n_engines

# Propellers (Torenbeek)
m_propeller_each = torenbeek_wt.mass_propeller(
    propeller_diameter=propeller_diameter,
    propeller_power=power_per_turboshaft,
    n_blades=4,
)
m_propellers_total = m_propeller_each * n_engines

# Gearbox (PT6A output shaft → propeller)
turboshaft_output_rpm = 33000
propeller_rpm = 1700
m_gearbox_each = mass_gearbox(
    power=power_per_turboshaft,
    rpm_in=turboshaft_output_rpm,
    rpm_out=propeller_rpm,
)
m_gearbox_total = m_gearbox_each * n_engines

# -- Payload / Cabin --
m_pax = n_pax * mass_passenger
m_seats = n_pax * mass_seat("passenger") + n_crew * mass_seat("flight_deck")
m_lavs = mass_lavatories(n_pax, aircraft_type="short-haul")

# Flight controls (2% MTOW)
m_flight_controls = 0.02 * design_mass_TOGW

##### Section: Total Weight #####

mass_empty = (
    # Structure
    m_wing + m_hstab + m_vstab + m_fuselage
    + m_mlg + m_nlg + m_nacelles
    # Propulsion
    + m_turboshaft_total
    + m_propellers_total + m_gearbox_total
    + m_fuel_system
    # Systems
    + m_instruments + m_electrical + m_furnishings
    + m_ac + m_anti_ice + m_flight_controls
    # Cabin equipment
    + m_seats + m_lavs
)

mass_total = mass_empty + payload_mass + fuel_mass

##### Section: Propulsion and Performance #####

# --- Cruise Power Balance (turboshaft only) ---
propulsive_area = n_engines * np.pi / 4 * propeller_diameter ** 2

shaft_power_cruise_total = propeller_shaft_power_from_thrust(
    thrust_force=drag_cruise,
    area_propulsive=propulsive_area,
    airspeed=cruise_speed,
    rho=cruise_atmo.density(),
    propeller_coefficient_of_performance=0.85,
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
climb_fuel_factor = 1.3
fuel_for_climb = fuel_burn_rate_cruise * climb_fuel_factor * climb_time

# Fuel for max range (400 nmi) -- sizes tanks and MTOW
cruise_time_max = design_range_max / cruise_speed
fuel_for_cruise_max = fuel_burn_rate_cruise * cruise_time_max

opti.subject_to(fuel_mass >= fuel_for_cruise_max + fuel_reserves + fuel_for_climb)

# Fuel for typical mission (200 nmi)
cruise_time_typical = design_range_typical / cruise_speed
fuel_for_cruise_typical = fuel_burn_rate_cruise * cruise_time_typical
fuel_mass_typical = fuel_for_cruise_typical + fuel_reserves + fuel_for_climb

# --- Takeoff Power (turboshaft only, no electric boost) ---
shaft_power_takeoff_total = power_per_turboshaft * n_engines

V_liftoff = 1.2 * V_stall_sl

shaft_power_from_thrust_liftoff = propeller_shaft_power_from_thrust(
    thrust_force=thrust_at_liftoff,
    area_propulsive=propulsive_area,
    airspeed=V_liftoff,
    rho=atmo_sl.density(),
    propeller_coefficient_of_performance=0.80,
)

opti.subject_to(shaft_power_takeoff_total >= shaft_power_from_thrust_liftoff)

##### Section: Constraints #####

# --- Wing Geometry ---
wing_aspect_ratio = wing_span ** 2 / wing_area
opti.subject_to(wing_aspect_ratio >= 8.0)
opti.subject_to(wing_aspect_ratio <= 14.0)  # Strut-braced allows higher AR

# --- Cruise Lift = Weight (typical mission) ---
lift_cruise = 0.5 * cruise_atmo.density() * cruise_speed ** 2 * wing_area * CL_cruise
typical_mission_TOGW = design_mass_TOGW - (fuel_mass - fuel_mass_typical)
mid_cruise_weight = (typical_mission_TOGW - fuel_for_cruise_typical * 0.5) * g

opti.subject_to(lift_cruise >= mid_cruise_weight * 0.99)
opti.subject_to(lift_cruise <= mid_cruise_weight * 1.01)

# --- Field Length ---
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

# --- OEI Climb Gradient (FAR 23, 2 engines) ---
opti.subject_to(field_results["flight_path_angle_climb_one_engine_out"] >= 0.024)

# --- Mass Closure ---
opti.subject_to(mass_total <= design_mass_TOGW)

# --- MTOW Limit (FAR 23 commuter category) ---
opti.subject_to(design_mass_TOGW <= 19000 * u.lbm)

##### Section: Objective #####

# Minimize MTOW for a conventional design -- compare to real 19,000 lb.
opti.minimize(design_mass_TOGW)

##### Section: Solve #####

sol = opti.solve(max_iter=500)

##### Section: Results Summary #####

print("=" * 72)
print("   CESSNA 408 SKYCOURIER VALIDATION -- DESIGN SUMMARY")
print("=" * 72)

# Extract solved values
TOGW = sol(design_mass_TOGW)
m_empty_sol = sol(mass_empty)
m_fuel_sol = sol(fuel_mass)
b = sol(wing_span)
c_root = sol(wing_root_chord)
S_wing = sol(wing_area)
AR = b ** 2 / S_wing

print(f"\n{'--- Overall ---':^72}")
print(f"  MTOW:                    {TOGW:8.0f} kg  ({TOGW / u.lbm:8.0f} lb)")
print(f"  Empty Weight:            {m_empty_sol:8.0f} kg  ({m_empty_sol / u.lbm:8.0f} lb)")
print(f"  Payload:                 {payload_mass:8.0f} kg  ({payload_mass / u.lbm:8.0f} lb)")
print(f"  Fuel Weight:             {m_fuel_sol:8.0f} kg  ({m_fuel_sol / u.lbm:8.0f} lb)")
print(f"  Useful Load:             {payload_mass + m_fuel_sol:8.0f} kg")

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

print(f"\n{'--- Propulsion (Conventional Turboprop) ---':^72}")
print(f"  Turboshaft Mass (each):  {sol(mass_turboshaft_per_engine):8.1f} kg  ({sol(mass_turboshaft_per_engine) / u.lbm:8.0f} lb)")
print(f"  Turboshaft Power (each): {sol(power_per_turboshaft) / u.horsepower:8.0f} hp  ({sol(power_per_turboshaft) / 1000:8.0f} kW)")
print(f"  Turboshaft Power (total):{sol(power_per_turboshaft) * n_engines / u.horsepower:8.0f} hp")
print(f"  Propeller Diameter:      {sol(propeller_diameter):8.2f} m   ({sol(propeller_diameter) / u.foot:8.1f} ft)")
print(f"  Propeller Diameter:      {sol(propeller_diameter) / 0.0254:8.0f} in")

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
print(f"  Fuel for Max Range:      {sol(fuel_for_cruise_max):8.0f} kg  (400 nmi)")
print(f"  Fuel for Typical Range:  {sol(fuel_for_cruise_typical):8.0f} kg  (200 nmi)")
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
    ("Wing (strut-braced)", sol(m_wing)),
    ("H-Stab", sol(m_hstab)),
    ("V-Stab (T-tail)", sol(m_vstab)),
    ("Fuselage", sol(m_fuselage)),
    ("Main Landing Gear (fixed)", sol(m_mlg)),
    ("Nose Landing Gear (fixed)", sol(m_nlg)),
    ("Nacelles", sol(m_nacelles)),
    ("Turboshaft Engines", sol(m_turboshaft_total)),
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

for name, mass_val in weight_items:
    print(f"  {name:<28} {mass_val:10.1f} {mass_val / u.lbm:10.1f} {mass_val / TOGW * 100:7.1f}%")

print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")
print(f"  {'EMPTY WEIGHT':<28} {m_empty_sol:10.1f} {m_empty_sol / u.lbm:10.1f} {m_empty_sol / TOGW * 100:7.1f}%")
print(f"  {'Payload':<28} {payload_mass:10.1f} {payload_mass / u.lbm:10.1f} {payload_mass / TOGW * 100:7.1f}%")
print(f"  {'Fuel':<28} {m_fuel_sol:10.1f} {m_fuel_sol / u.lbm:10.1f} {m_fuel_sol / TOGW * 100:7.1f}%")
print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")
print(f"  {'MTOW':<28} {TOGW:10.1f} {TOGW / u.lbm:10.1f} {100:7.1f}%")
print(f"\n  Empty Weight Fraction:   {m_empty_sol / TOGW:.3f}")
print(f"  Fuel Fraction:           {m_fuel_sol / TOGW:.3f}")
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

# =====================================================================
# VALIDATION: Compare to Real Cessna 408 SkyCourier
# =====================================================================
print("\n" + "=" * 72)
print("   VALIDATION vs REAL CESSNA 408 SKYCOURIER")
print("=" * 72)

real = {
    "MTOW (lb)":          (TOGW / u.lbm,               19000),
    "Empty Wt (lb)":      (m_empty_sol / u.lbm,        12325),
    "Wing Span (ft)":     (b / u.foot,                  72.25),
    "Wing Area (ft²)":    (S_wing / u.foot**2,          441),
    "Aspect Ratio":       (AR,                          11.8),
    "Engine SHP (each)":  (sol(power_per_turboshaft) / u.horsepower, 1100),
    "Engine Mass (kg)":   (sol(mass_turboshaft_per_engine),          235),
    "Prop Dia (in)":      (sol(propeller_diameter) / 0.0254,         110),
    "Fuel (lb)":          (m_fuel_sol / u.lbm,          2130),
}

print(f"\n  {'Parameter':<22} {'Model':>10} {'Real':>10} {'Error':>10}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
for param, (model_val, real_val) in real.items():
    err_pct = (model_val - real_val) / real_val * 100
    print(f"  {param:<22} {model_val:10.0f} {real_val:10.0f} {err_pct:+9.1f}%")

print("\n  Notes:")
print("  - Model uses cantilever wing weight * 0.70 strut factor")
print("  - Model uses fixed gear weight * 0.70 retract factor")
print("  - Real SkyCourier fuel capacity is for full tanks (~900 nmi ferry)")
print("    Model sizes fuel for 400 nmi + reserves only")
print("  - Differences in empty weight reflect items not modeled")
print("    (cargo handling system, airstairs, etc.)")

print("\n" + "=" * 72)

# --- Optional: Draw 3-view ---
try:
    sol_airplane = sol(airplane)
    sol_airplane.draw_three_view()
except Exception:
    pass  # Skip drawing if display not available
