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
    MPI.init()
    mpi.nprocs = mpi.comm.Get_size()
    mpi.myrank = mpi.comm.Get_rank()


def finalize():
    """Finalize parallel processes"""
    MPI.finalize()


#
