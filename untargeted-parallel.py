from hypergraph import *
from ricciutil import *
from mpi4py import MPI

COMM = MPI.COMM_WORLD


def manager(npr, verbose=False):
    return


def worker(w, verbose=False):
    return


if __name__ == "__main__":
    # TODO: Deal with this thought:
    """
    The curvature is scale agnostic. But that doesn't make it stay stable necessarily
    For the petersen graph, it should be the same. But it moves around.
    Is that an us problem? Is that something else we've got going?
    """
    ITTS = 100
    start = time.time()

    verbose = False
    absolute_change = True  # False is relative change
    maximum_error = True  # False is average error
    gurobi_flag = os.environ.get("GUROBI_FLAG")
    approx_emd = os.environ.get("APPROX")
    directed_flag = os.environ.get("DIRECTED_FLAG")
    if approx_emd is None:  # Make the default False
        approx_emd = "False"
    approx_emd = eval(approx_emd)
    if gurobi_flag is None:  # make gurobi flag default false
        gurobi_flag = "False"
    gurobi_flag = eval(gurobi_flag)
    if directed_flag is None:  # make directed flag default false
        directed_flag = "False"
    directed_flag = eval(directed_flag)

    if approx_emd:
        gurobi_flag = False

    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    if RANK == 0:
        manager(SIZE, verbose=False)
    else:
        worker(RANK, verbose=False)
    print(f"node {RANK} made it to the end")
