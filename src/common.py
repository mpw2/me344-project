"""common.py
Contains global variables
"""
import numpy as np


# MPI objects/parameters
MPI = None
comm = None
nprocs = None
myrank = None

# input/output parameters
fout_path = None
fin_path = None
fparam_path = None

# physical parameters
mu = np.float64(0.0)  # dynamic viscosity
gamma = np.float64(0.0)  # cp/cv ratio
Pr = np.float64(0.0)  # Prandtl number
R_g = np.float64(0.0)  # gas constant
k = np.float64(0.0)  # heat transfer coefficient
D = np.float64(0.0)  # scalar diffusivity coefficient

# domain parameters
Lx = np.float64(0.0)
Ly = np.float64(0.0)
Lz = np.float64(0.0)

# inlet conditions
jet_height_y = np.float64(0.0)  # extent of jet in y-dim
jet_height_z = np.float64(0.0)  # extent of jet in z-dim
U_jet = np.float64(0.0)  # jet inlet streamwise velocity
V_jet = np.float64(0.0)  # jet inlet normal velocity
W_jet = np.float64(0.0)  # jet inlet spanwise velocity
P_jet = np.float64(0.0)  # jet inlet pressure
T_jet = np.float64(0.0)  # jet inlet temperature
Rho_jet = np.float64(0.0)  # jet inlet density
Phi_jet = np.float64(0.0)  # jet inlet scalar concentration

# ambient conditions
U_inf = np.float64(0.0)  # reference streamwise velocity
V_inf = np.float64(0.0)  # reference normal velocity
W_inf = np.float64(0.0)  # reference spanwise velocity
P_inf = np.float64(0.0)  # reference pressure
T_inf = np.float64(0.0)  # reference temperature
Rho_inf = np.float64(0.0)  # reference density
Phi_inf = np.float64(0.0)  # reference scalar concentration

# grid parameters
nx_global = None  # global grid size
ny_global = None
nz_global = None
xg_global = None  # global grid coordinates
yg_global = None
zg_global = None

i0_global = None  # lower extent streamwise index
i1_global = None  # upper extent streamwise index

nx = None  # local grid size
ny = None
nz = None
xg = None  # local grid coordinates
yg = None
zg = None

# timestep parameters
CFL_ref = np.float64(0.0)
dt = np.float64(0.0)
t = np.float64(0.0)
tstep = None
nsteps = None
nsave = None
nmonitor = None

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
mu_sgs = None # sgs viscosity

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
sponge_strength = 1.0  # magnitude of sponge damping
a_sponge = 3.0  # order of damping func
x_sponge = 0.3  # nondimensional sponge length
y_sponge = 0.2  # nondimensional sponge length
z_sponge = 0.0  # nondimensional sponge length
Qref = None  # reference condition

# Runge-Kutta steps
rk_step_bits = np.int8(0b000)


#
