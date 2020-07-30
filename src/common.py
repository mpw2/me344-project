# ------------------------------------------
#
# common.py
#
#  - contains global variables
#
# ------------------------------------------


# input/output parameters
fout_path = None
fin_path = None
fparam_path = None

# physical parameters
mu = None  # dynamic viscosity
gamma = None  # cp/cv ratio
# Ma = None  # Mach number
# Re = None  # Reynolds number
Pr = None  # Prandtl number
R_g = None  # gas constant
k = None  # heat transfer coefficient
D = None  # scalar diffusivity coefficient

# domain parameters
Lx = None
Ly = None
Lz = None

# inlet conditions
jet_height_y = None  # extent of jet in y-dim
jet_height_z = None  # extent of jet in z-dim
U_jet = None  # jet inlet streamwise velocity
V_jet = None  # jet inlet normal velocity
W_jet = None  # jet inlet spanwise velocity
P_jet = None  # jet inlet pressure
T_jet = None  # jet inlet temperature
Rho_jet = None  # jet inlet density
Phi_jet = None  # jet inlet scalar concentration

# ambient conditions
U_inf = None  # reference streamwise velocity
V_inf = None  # reference normal velocity
W_inf = None  # reference spanwise velocity
P_inf = None  # reference pressure
T_inf = None  # reference temperature
Rho_inf = None  # reference density
Phi_inf = None  # reference scalar concentration

# grid parameters
nx_global = None  # global grid size
ny_glboal = None
nz_global = None
nx = None  # local grid size
ny = None
nz = None
xg_global = None  # global grid coordinates
yg_global = None
zg_global = None
xg = None  # local grid coordinates
yg = None
zg = None
i0_global = None  # lower extent streamwise index
i1_global = None  # upper extent streamwise index

# timestep parameters
CFL_ref = None
dt = None
t = None
tstep = None

# variable arrays
Q = None  # conserved variable vector
Qo = None  # previous time step
Rho = None  # density
U = None  # streamwise velocity
V = None  # normal velocity
W = None  # spanwise velocity
P = None  # pressure
E_t = None  # total energy
Phi = None  # scalar concentration

# rhs terms
E = None
F = None
G = None
dEdx = None
dFdy = None
dGdz = None

# constants
NVARS = 6  # number of transported variables

# sponge
sponge_fac = None
sponge_strength = 100  # magnitude of sponge damping
a_sponge = 2  # order of damping func
x_sponge = 0.2  # nondimensional sponge length
y_sponge = 0.2  # nondimensional sponge length
z_sponge = 0.0  # nondimensional sponge length
Qref = None  # reference condition

# Runge-Kutta steps
rk_step_1 = None
rk_step_2 = None


#
