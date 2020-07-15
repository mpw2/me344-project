
import numpy as np

from boundary_conditions        import *
from equations          import *
from initialization     import *
from input_output       import *
from main               import *
from monitor            import *
from mpi                import *
from time_integration   import *

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
Rho_ref = None # reference density
Patm = None # atmospheric pressure
Ujet = None # jet velocity

# domain parameters
Lx = None
Ly = None
jet_height = None

# grid parameters
nx = None
ny = None
xg = None 
yg = None

# timestep parameters
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
NVARS = 4 # number of transported variables



