# -----------------------------------------------------------------------------
# mpi.py
#
# Description:
#  - Contains variables and functions for parallelization with mpi.
#
# -----------------------------------------------------------------------------

from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = None
myrank = None


def init():
    """Initialize parallel processes"""
    # MPI_init() called automatically on import
    # Initialize MPI parameter variables
    global nprocs, myrank, comm
    nprocs = comm.Get_size()
    myrank = comm.Get_rank()


def finalize():
    """Finalize parallel processes"""
    # MPI_Finalize() called automatically at exit
    # nothing to do
    pass


#
