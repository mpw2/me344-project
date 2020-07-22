# Import packages and utilities
import numpy as np
from sys import argv

# ====================================

# input/output parameters
fout_path = None
fin_path = None
fparam_path = None

# physical parameters
mu = None  # dynamic viscosity
gamma = None # cp/cv ratio
# Ma = None # Mach number
# Re = None # Reynolds number
Pr = None # Prandtl number
R_g = None # gas constant
k = None # heat transfer coefficient

# domain parameters
Lx = None
Ly = None
Lz = None

# inlet conditions
jet_height_y = None # extent of jet in y-dim
jet_height_z = None # extent of jet in z-dim
U_jet = None # jet inlet streamwise velocity
V_jet = None # jet inlet normal velocity
W_jet = None # jet inlet spanwise velocity
P_jet = None # jet inlet pressure
T_jet = None # jet inlet temperature
Rho_jet = None #

# ambient conditions
U_inf = None
V_inf = None
W_inf = None 
P_inf = None # reference pressure
T_inf = None
Rho_inf = None # reference density

# grid parameters
nx = None
ny = None
nz = None
xg = None 
yg = None
zg = None

# timestep parameters
CFL_ref = None
dt = None
t = None
tstep = None

# variable arrays
Q = None  # conserved variable vector
Qo = None
Rho = None # density
U = None # streamwise velocity
V = None # normal velocity
W = None # spanwise velocity
P = None # pressure
E_t = None # total energy

# rhs terms
E = None
F = None
G = None
dEdx = None
dFdy = None
dGdz = None

# constants
NVARS = 5 # number of transported variables


# sponge
sponge_fac = None
sponge_strength = 100 # magnitude of sponge damping
a_sponge = 2 # order of damping func
x_sponge = 0.2 # nondimensional sponge length
y_sponge = 0.2 # nondimensional sponge length
z_sponge = 0.0 # nondimensional sponge length

Qref = None # reference condition

