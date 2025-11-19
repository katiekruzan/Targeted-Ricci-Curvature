"""Written by Katie Kruzan in November 2025. this can be run with the following commandline
mpiexec -n 4 python .\tests-parallel.py
"""

from hypergraph import *
from ricciutil import *
import os
from mpi4py import MPI
import time
import pandas as pd
import csv

COMM = MPI.COMM_WORLD

# process functions


def early_analysis(src_graph: Hypergraph, verbose: bool):
    """Get the information on the graph we're working on

    :param Hypergraph src_graph: The graph we're analyzing
    :param bool verbose: verbose flag
    """
    connected = src_graph.is_weakly_connected()
    strconnect = src_graph.is_strongly_connected()
    max_degree, min_degree, avg_degree = src_graph.calculate_degrees()

    if verbose:
        print("type of graph", type(src_graph))
        # Printing the number of (hyper)edges in our network.
        print("Number of edges:", len(src_graph.hyperedges))
        # Printing the number of nodes in the network.
        print("Number of nodes", len(src_graph.nodes))
        print("The actual nodes:", src_graph.nodes)
        print("The actual edges with weights:", src_graph.weights)

        print(
            "The hypergraph is weakly connected."
            if connected
            else "The hypergraph is not weakly connected."
        )

        print(f"Max Degree: {max_degree}")
        print(f"Min Degree: {min_degree}")
        print(f"Average Degree: {avg_degree}")

    write_scorecard("----- Graph Statistics -----")
    write_scorecard(f"Type of Graph: {type(src_graph)}")
    write_scorecard(f"Number of edges: {len(src_graph.hyperedges)}")
    write_scorecard(f"Number of nodes: {len(src_graph.nodes)}")
    if connected:
        write_scorecard("The hypergraph is weakly connected.")
    else:
        write_scorecard("The hypergraph is not weakly connected.")
    if strconnect:
        write_scorecard("The hypergraph is strongly connected.")
    else:
        write_scorecard("The hypergraph is not strongly connected.")
    write_scorecard(f"Max Degree: {max_degree}")
    # Quick note: directed can have these be 0
    write_scorecard(f"Min Degree: {min_degree}")
    # for directed graphs, this should be equal.
    write_scorecard(f"Average Degree: {avg_degree}")
    write_scorecard("----------------------------")
    return


def set_up_one_direction(src_graph: Hypergraph, targ_graph: Hypergraph, op_flag=False): 
    '''Setting up the one direction stuff. But in a separate function to help organize.
    Returns the target and source distance matrices (also persists them), and the
    set of missing edges from source and target

    :param Hypergraph src_graph: The source graph
    :param Hypergraph targ_graph: The target graph
    :param bool op_flag: used to indicated if we should add the op prefix, defaults to False
    :returns: target_distance_matrix, distance_matrix, missing_from_src, missing_from_targ
    
    '''
    
    print("working on distance matrices")
    distance_matrix = src_graph.floyd_warshall()
    matfilename = "outputfiles/"
    if op_flag:
        matfilename += "op_"
    matfilename += "source_dist_fw.csv"
    save_matrix_csv(distance_matrix, matfilename)

    if verbose:
        clock_time("Time to make the source distance matrix")

    target_distance_matrix = targ_graph.floyd_warshall()
    matfilename = "outputfiles/"
    if op_flag:
        matfilename += "op_"
    matfilename += "target_dist_fw.csv"
    save_matrix_csv(target_distance_matrix, matfilename)

    if verbose:
        clock_time("Time to make the target distance matrix")

    missing_from_src, missing_from_targ = [], []

    if set(targ_graph.hyperedges) != set(src_graph.hyperedges):
        print(set(targ_graph.hyperedges) - set(src_graph.hyperedges))
        # logging the edges that are different
        missing_from_src = set(targ_graph.hyperedges) - \
            set(src_graph.hyperedges)
        missing_from_targ = set(src_graph.hyperedges) - \
            set(targ_graph.hyperedges)

        print("Taking care of missing edges")
        # add edges that are in the target but not the source
        src_graph.add_missing_edges_shortest_path(
            targ_graph, distance_matrix, verbose)
        targ_graph.add_missing_edges_shortest_path(
            src_graph, target_distance_matrix, verbose
        )
        if verbose:
            clock_time("time to add missing edges")

        # recalculate the matrices
        distance_matrix = src_graph.floyd_warshall()

        target_distance_matrix = targ_graph.floyd_warshall()
        if verbose:
            clock_time("time to recalc the distances")

    print("len missing from source", len(missing_from_src))
    print("len missing from targ", len(missing_from_targ))

    return target_distance_matrix, distance_matrix, missing_from_src, missing_from_targ


# Manager Functions
def calculate_target_orc_manager(
    npr: int,
    distance_matrix: list[list],
    graph: Hypergraph,
    verbose: bool,
    op_flag=False,
):
    """The function to calculate the staring info for the target graph

    :param int npr: number of processors
    :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
    :param Hypergraph graph: the target graph
    :param bool verbose: verbose flag
    :param bool op_flag: option to mark the file as 'op' (used in second half of
                           script), defaults to False
    """
    if op_flag:
        file_name = f"outputfiles/op_dataset_target_graph_orc.csv"
    else:
        file_name = f"outputfiles/dataset_target_graph_orc.csv"

    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)
        # Check if the file is empty to write headers
        if file.tell() == 0:
            writer.writerow(["Hyperedge ID", "ORC", "Weight"])

        # split the edges. Will have the remaining edges be split on the first r processors
        edges = list(graph.hyperedges.keys())
        njobs = len(edges)
        chunksize = njobs // (npr - 1)
        remainder = njobs % (npr - 1)
        jobcnt = 0

        while jobcnt < npr - 1:
            # send the jobs
            for i in range(1, npr):
                jobcnt = jobcnt + 1  # notably, will basically be equal to i
                jobstosend = []
                if jobcnt <= remainder:
                    jobstosend = edges[
                        (jobcnt - 1) * chunksize: jobcnt * chunksize
                    ] + [edges[-jobcnt]]
                else:
                    jobstosend = edges[(jobcnt - 1) *
                                       chunksize: jobcnt * chunksize]
                SLICE = (jobstosend, distance_matrix, graph, verbose)
                COMM.send(SLICE, dest=i, tag=33)
                if verbose:
                    print(
                        "-> manager sends job",
                        jobcnt,
                        "to worker",
                        i,
                        "number of jobs",
                        len(SLICE[0]),
                    )
            if verbose:
                clock_time("manager sent all the jobs")
            # receive the jobs // sync the graphs.
            for i in range(1, npr):
                newgraph, jobs = COMM.recv(source=i, tag=11)
                if verbose:
                    clock_time(f"gathered data from processor: {i}")
                    print(
                        "-> manager received data from worker",
                        i,
                        "number of jobs",
                        len(jobs),
                    )
                for e in jobs:  # sync up the graph
                    graph.add_ricci_curvature(
                        e, newgraph.ricci_curvature[e][-1])
                    graph.add_weights(e, newgraph.weights[e][-1])
                    writer.writerow(
                        [e, newgraph.ricci_curvature[e]
                            [-1], newgraph.weights[e][-1]]
                    )
    return


def update_orc_and_weights_iter_manager(
    npr: int,
    distance_matrix: list[list],
    graph: Hypergraph,
    targ_graph: Hypergraph,
    iteration: int,
    verbose: bool,
    file_format="csv",
    op_flag=False,
) -> None:
    """The main function of this whole sheboodle. Run the whole process for the given itteration

    :param int npr: the number of processors
    :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
    :param Hypergraph graph: the source graph we're looking at (or at least its current itteration)
    :param Hypergraph targ_graph: The target graph
    :param int iteration: the round we're on
    :param bool verbose: verbose flag
    :param str file_format: defaults to 'csv'
    :param bool op_flag: option to mark the file as 'op' (used in second half of
                           script), defaults to False
    """
    if op_flag:
        file_name = f"outputfiles/op_dataset_targeted_curvature_iteration_{iteration}.{file_format}"
    else:
        file_name = f"outputfiles/dataset_targeted_curvature_iteration_{iteration}.{file_format}"

    with open(file_name, "a", newline="") as file:
        if file_format == "csv":
            writer = csv.writer(file)
            # Check if the file is empty to write headers
            if file.tell() == 0:
                writer.writerow(
                    ["Hyperedge ID", "ORC: (based on t-1 weights)", "Weight:t"]
                )

            # split the edges. Will have the remaining edges be split on the first r processors
            edges = list(graph.hyperedges.keys())
            njobs = len(graph.hyperedges)
            chunksize = njobs // (npr - 1)
            remainder = njobs % (npr - 1)
            jobcnt = 0

            while jobcnt < npr - 1:
                # send the jobs
                for i in range(1, npr):
                    jobcnt = jobcnt + 1  # notably, will basically be equal to i
                    jobstosend = []
                    if jobcnt <= remainder:
                        jobstosend = edges[
                            (jobcnt - 1) * chunksize: jobcnt * chunksize
                        ] + [edges[-jobcnt]]
                    else:
                        jobstosend = edges[
                            (jobcnt - 1) * chunksize: jobcnt * chunksize
                        ]
                    SLICE = (
                        jobstosend,
                        distance_matrix,
                        graph,
                        targ_graph,
                        verbose,
                        iteration,
                    )
                    COMM.send(SLICE, dest=i, tag=44)
                    if verbose:
                        print(
                            "-> manager sends job",
                            jobcnt,
                            "to worker",
                            i,
                            "number of jobs",
                            len(SLICE[0]),
                        )
                if verbose:
                    clock_time(f"manager has sent all the jobs")
                # receive the jobs // sync the graphs.
                for i in range(1, npr):
                    newgraph, jobs = COMM.recv(source=i, tag=11)
                    if verbose:
                        print(
                            "-> manager received data from worker",
                            i,
                            "number of jobs",
                            len(jobs),
                        )
                    for e in jobs:  # sync up the graph
                        graph.add_ricci_curvature(
                            e, newgraph.ricci_curvature[e][-1])
                        graph.add_weights(e, newgraph.weights[e][-1])
                        writer.writerow(
                            [
                                e,
                                newgraph.ricci_curvature[e][-1],
                                newgraph.weights[e][-1],
                            ]
                        )
    return


def one_direction_of_work_manager(
    npr: int, src_graph: Hypergraph, targ_graph: Hypergraph, tot_its=100, op_flag=False
):
    """Doing the itterations from one graph to another

    :param int npr: the number of processors
    :param Hypergraph src_graph: The starting graph
    :param Hypergraph targ_graph: The target graph
    :param int tot_its: The maximum number of itterations to allow for this
      process, defaults to 100
    :param bool op_flag: This is true if we want to add op_ as a prefix on the
      files. Used to determine if we're going the opposite direction, defaults to False
    """
    # One Direction of Work
    targ_distance_matrix, distance_matrix, missing_from_src, missing_from_targ = (
        set_up_one_direction(src_graph, targ_graph, op_flag)
    )

    if verbose:
        clock_time("time to set up")
    print("starting ricci curvature")

    calculate_target_orc_manager(
        npr, targ_distance_matrix, targ_graph, verbose, op_flag=op_flag
    )

    if verbose:
        clock_time("Time to calc target ORC")

    update_orc_and_weights_iter_manager(
        npr,
        distance_matrix,
        src_graph,
        targ_graph,
        iteration=0,
        verbose=verbose,
        op_flag=op_flag,
    )

    if verbose:
        clock_time("Time to calc source ORC")

    for i in range(1, tot_its + 1):
        print("Working on itteration", i)
        distance_matrix_i = src_graph.floyd_warshall()
        if verbose:
            clock_time(f"finished distance matrix {i}")
        if i > 1:
            if len(missing_from_src) > 0 or len(missing_from_targ) > 0:
                # We're gonna to the reset here
                src_graph.add_missing_edges_shortest_path(
                    targ_graph, distance_matrix, verbose
                )
                targ_graph.add_missing_edges_shortest_path(
                    src_graph, targ_distance_matrix, verbose
                )
        update_orc_and_weights_iter_manager(
            npr,
            distance_matrix_i,
            src_graph,
            targ_graph,
            iteration=i,
            verbose=verbose,
            op_flag=op_flag,
        )
        clock_time(f"Time for ORC {i}")

        def missing_reset():
            # We will do a "reset" here
            if len(missing_from_src) > 0 or len(missing_from_targ) > 0:
                # We're gonna to the reset here
                # first delete all the edges
                for e in missing_from_src:
                    src_graph.remove_hyperedge(e)
                for e in missing_from_targ:
                    targ_graph.remove_hyperedge(e)

        allstable = True
        if i == 1:
            # TODO: fix this weirdness
            missing_reset()
            # take care of the getting started case
            continue
        errorlist = []
        for e in src_graph.hyperedges:
            clist = src_graph.ricci_curvature[e]
            old = clist[-2]
            new = clist[-1]
            if old != 0:
                if absolute_change:
                    error = abs(old - new)
                else:
                    error = abs((old - new) / old)  # relative change
            else:
                error = abs(old - new)
                if not absolute_change:
                    error = error / old
            if maximum_error:
                if absolute_change and (error > 0.01):
                    clock_time(f"unstable for edge {e} with error {error}")
                    allstable = False
                    missing_reset()
                    break
                if (not absolute_change) and (error > 0.05):  # relative change
                    clock_time(f"unstable for edge {e} with error {error}")
                    allstable = False
                    missing_reset()
                    break
            else:
                errorlist.append(error)
                missing_reset()
        # find the average error
        if not maximum_error:  # AKA we're in average error zone
            # assumed to be in absolute error zone
            avg_err = np.average(errorlist)
            if avg_err > 0.0001:
                clock_time(f"unstable with average error {avg_err}")
                allstable = False
        if allstable:
            # turn off all workers.
            for k in range(1, npr):
                COMM.send(-1, dest=k, tag=44)
            print("STABILIZED! Source to target distance is ", i)
            write_scorecard("\n\n----- Results -----")
            write_scorecard(f"Source to target distance is {i}")
            break
    return


# Worker functions
def calculate_target_orc_worker():
    '''Corresponds to calculate_target_orc_manager function
    '''
    specs = COMM.recv(source=0, tag=33)
    if specs == -1:
        return
    jobs, distance_matrix, graph, verbose = specs

    for hyperedge_id in jobs:
        if isinstance(graph, UndirectedHypergraph):
            orc = graph.earthmover_distance_hyperedge_combinations(
                hyperedge_id, distance_matrix, approx_emd, gurobi_flag, verbose
            )
        elif isinstance(graph, DirectedHypergraph):
            orc = 1 - graph.earthmover_distance_distance_matrix(
                hyperedge_id, distance_matrix, approx_emd, gurobi_flag, verbose=False
            )
        normalized_orc = ricci_normalizing(orc)
        graph.add_ricci_curvature(hyperedge_id, normalized_orc)
    if verbose:
        clock_time("finished worker now sending over")
    COMM.send((graph, jobs), dest=0, tag=11)
    return


def update_orc_and_weights_iter_worker() -> bool:
    """The actual worker processor doing the work of the orc itteration. the
    larger description is in update_orc_and_weights_iter_manager(). But will throw
    a false if its supposed to stop

    :return bool: This is how the loop will end and continue to the next part of the worker script
    """
    specs = COMM.recv(source=0, tag=44)
    if specs == -1:
        return False
    jobs, distance_matrix, graph, targ_graph, verbose, itteration = specs

    for hyperedge_id in jobs:
        if isinstance(graph, UndirectedHypergraph):
            orc = graph.earthmover_distance_hyperedge_combinations(
                hyperedge_id, distance_matrix, approx_emd, gurobi_flag, verbose
            )
        elif isinstance(graph, DirectedHypergraph):
            orc = 1 - graph.earthmover_distance_distance_matrix(
                hyperedge_id, distance_matrix, approx_emd, gurobi_flag, verbose=False
            )
        normalized_orc = ricci_normalizing(orc)
        graph.add_ricci_curvature(hyperedge_id, normalized_orc)
        weight = graph.weights[hyperedge_id][-1]
        if itteration != 0:  # update the weights
            orc_targ = targ_graph.ricci_curvature[hyperedge_id][-1]
            if weight != 0:
                # simple version
                step = 1
                wtplus1 = weight * (1 - step * (normalized_orc - orc_targ))
                normalized_weight = wtplus1
            else:
                normalized_weight = 0
            graph.add_weights(hyperedge_id, normalized_weight)
    COMM.send((graph, jobs), dest=0, tag=11)
    return True


def one_direction_of_work_worker(tot_its=100):
    '''This corresponds to the one_direction_of_work manager

    :param int tot_its: number of max iterations before considering divergence, defaults to 100
    '''
    calculate_target_orc_worker()
    update_orc_and_weights_iter_worker()
    cont = True
    cnt = 1
    while cont and (cnt <= tot_its):
        cont = update_orc_and_weights_iter_worker()
        cnt = cnt + 1
    return


def manager(npr: int, verbose=False):
    '''This is the main function for the manager node

    :param int npr: number of worker nodes we're working with
    :param bool verbose: verbose flag, defaults to False
    '''
    clean_output(verbose)
    # source_filename = os.environ.get('SOURCE_FILENAME')
    # target_filename = os.environ.get('TARGET_FILENAME')
    source_filename = "petersen/petersengraph.csv"
    target_filename = "petersen/petersengraphExtraEdge.csv"

    data_target = pd.read_csv(
        f"inputfiles/{target_filename}", dtype={"source": str, "target": str}, sep=","
    )
    data_source = pd.read_csv(
        f"inputfiles/{source_filename}", dtype={"source": str, "target": str}, sep=","
    )

    # Scorecard writing
    write_scorecard("----- Targeted Ricci Curvature -----")
    write_scorecard(f"target filename: {target_filename}")
    write_scorecard(f"source filename: {source_filename}")
    if absolute_change:
        write_scorecard("Absolute vs Relative change: absolute change")
    else:
        write_scorecard("Absolute vs Relative change: relative change")
    if maximum_error:
        write_scorecard("Max vs Avg error: Maximum")
    else:
        write_scorecard("Max vs Avg error: Avg")
    if approx_emd:
        write_scorecard("Type of EMD: Approx")
    else:
        write_scorecard("Type of EMD: Exact")
    if gurobi_flag:
        write_scorecard("EMD Solver: Gurobi")
    else:
        write_scorecard("EMD Solver: Python Optimal Transport")
    if verbose:
        clock_time("Time to read the data in seconds")

    if directed_flag:
        source_graph = DirectedHypergraph()
        target_graph = DirectedHypergraph()
    else:
        source_graph = UndirectedHypergraph()
        target_graph = UndirectedHypergraph()

    print("building source")
    source_graph.build_from_dataframe(data_source, verbose)

    print("building target")
    target_graph.build_from_dataframe(data_target, verbose)

    if verbose:
        clock_time("Time to build the graphs")

    if not (source_graph.is_2_uniform() and target_graph.is_2_uniform()):
        print(
            "This has not been fully fleshed out for hypergraphs. Please give a 2-uniform graph"
        )
        quit()

    early_analysis(source_graph, verbose)
    if verbose:
        clock_time("Time to analyze graphs")

    one_direction_of_work_manager(npr, source_graph, target_graph, tot_its=ITS)

    clock_time("Time for source->target")

    write_scorecard("\n")

    # Go the other way
    print("Now checking Target to Source....")

    if directed_flag:
        source_graph = DirectedHypergraph()
        target_graph = DirectedHypergraph()
    else:
        source_graph = UndirectedHypergraph()
        target_graph = UndirectedHypergraph()

    # swap them
    print("building source")
    source_graph.build_from_dataframe(data_target, verbose)
    print("building target")
    target_graph.build_from_dataframe(data_source, verbose)

    one_direction_of_work_manager(
        npr, source_graph, target_graph, tot_its=ITS, op_flag=True
    )

    clock_time("Time for final")

    # tell the jobs to sleep (at the very end)
    for i in range(1, npr):
        SLICE = -33
        COMM.send(SLICE, dest=i, tag=55)
        if verbose:
            print(f"-> manager sends {SLICE} to worker", i)
    return


def worker(w: int, verbose=False):
    '''This is the main function for the worker node

    :param int w: The rank of the specific worker node we're using. Used primarily in pulling the information
    :param bool verbose: verbose flag, defaults to False
    '''
    one_direction_of_work_worker(tot_its=ITS)
    one_direction_of_work_worker(tot_its=ITS)
    while True:
        specs = COMM.recv(source=0, tag=55)
        if specs == -33:
            if verbose:
                print(f"Worker {w} goes to sleep")
            break
    return


if __name__ == "__main__":
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
    if approx_emd and directed_flag:
        print("Approx EMD not implemented for Directed graphs.")
        quit()

    start = time.time()
    # moves this to the util
    set_start(start)

    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    ITS = 100
    if RANK == 0:
        manager(SIZE, verbose)
    else:
        worker(RANK, verbose)
    print(f"node {RANK} made it to the end")
