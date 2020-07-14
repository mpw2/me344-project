
import numpy as np

# input/output parameters
fout_path = None
fin_path = None
fparam_path = None

# physical parameters
mu = None  # dynamic viscosity
gamma = None # cp/cv ratio
Ma = None # Mach number
Re = None # Reynolds number
Pr = None # Prandtl number
R_g = None # gas constant
k = None # heat transfer coefficient
Rho_ref = None # reference density

# grid parameters
nx = None
ny = None

Lx = None
Ly = None
xg = None 
yg = None

# tiemstep parameters
CFL_ref = None
dt = None

# variable arrays
Q = None  # conserved variable vector
Rho = None # density
U = None # streamwise velocity
V = None # normal velocity
P = None # pressure
E_t = None # total energy

# constants
nvars = 4 # number of transported variables



