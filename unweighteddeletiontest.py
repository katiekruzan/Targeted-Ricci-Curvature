"""
The goal for this test is "Take an unweighted random graph, delete some subset of edges

Then do Ricci Flow, if the edge becomes small enough (say less than .1), then set it to 0."

For now, I will set this to be run sequentially, and perhaps this will help withfuture sequential runs

"""

from hypergraph import *
from ricciutil import *
import csv

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

def calculate_target_orc(
    distance_matrix: list[list],
    graph: Hypergraph,
    verbose: bool,
    op_flag=False):
    """The function to calculate the staring info for the target graph
    
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

        edges = list(graph.hyperedges.keys())
        jobs = edges

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
            writer.writerow([hyperedge_id, graph.ricci_curvature[hyperedge_id]
                            [-1], graph.weights[hyperedge_id][-1]]
                    )
    if verbose:
        clock_time("finished target EMD calculations")
    return


def update_orc_and_weights_iter(
    distance_matrix: list[list],
    graph: Hypergraph,
    targ_graph: Hypergraph,
    iteration: int,
    verbose: bool,
    file_format="csv",
    op_flag=False,
) -> None:
    """The main function of this whole sheboodle. Run the whole process for the given itteration

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
            jobs = edges

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
                if iteration != 0: #update the weights
                    orc_targ = targ_graph.ricci_curvature[hyperedge_id][-1]
                    if weight != 0:
                        # simple version
                        step = 1
                        wtplus1 = weight * (1 - step * (normalized_orc - orc_targ))
                        if wtplus1 < CUTOFF:
                            wtplus1 = 0
                        normalized_weight = wtplus1
                    else:
                        normalized_weight = 0
                    graph.add_weights(hyperedge_id, normalized_weight)
                writer.writerow([hyperedge_id, graph.ricci_curvature[hyperedge_id]
                                [-1], graph.weights[hyperedge_id][-1]]
                        )       
    return


def test_one_direction(src_graph:Hypergraph, targ_graph:Hypergraph, tot_its=100, op_flag=False):

    targ_distance_matrix, distance_matrix, missing_from_src, missing_from_targ = (
            set_up_one_direction(src_graph, targ_graph, op_flag)
        )

    if verbose:
        clock_time("time to set up")
    print("starting ricci curvature")

    calculate_target_orc(targ_distance_matrix, targ_graph, verbose, op_flag)
    
    update_orc_and_weights_iter(
        distance_matrix,
        src_graph,
        targ_graph,
        iteration=0,
        verbose=verbose,
        op_flag=op_flag
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
        # print(len(targ_graph.hyperedges))
        # print(len(src_graph.hyperedges))
        update_orc_and_weights_iter(
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

        # the test for stability will be when the edgesets with nonzero edge weights will be equal
        if i == 1:
            # TODO: fix this weirdness
            missing_reset()
            # take care of the getting started case
            continue
        
        missing_reset()
        # print(len(src_graph.weights))
        # print(type(src_graph.weights))

        # print(len(targ_graph.weights))
        # quit()
        nonzeroedges_src = [key for key, value_list in src_graph.weights.items() if value_list[-1]!=0]
        nonzeroedges_targ = [key for key, value_list in targ_graph.weights.items() if value_list[-1]!=0]
        if set(nonzeroedges_src) == set(nonzeroedges_targ):
            print("STABILIZED! Source to target distance is ", i)
            write_scorecard("\n\n----- Results -----")
            write_scorecard(f"Source to target distance is {i}")
            break
        else:
            print("sets not equal. ")
            print("In Src and not Targ: ")
            print(set(nonzeroedges_src)- set(nonzeroedges_targ))
        # break
    return


def runtest(verbose = False):
    '''This will be the function where I'm running this new test

    :param int npr: number of worker nodes we're working with
    :param bool verbose: verbose flag, defaults to False
    '''
    clean_output(verbose)
    # source_filename = os.environ.get('SOURCE_FILENAME')
    # target_filename = os.environ.get('TARGET_FILENAME')
    source_filename = "petersen/petersengraph.csv"
    target_filename = "petersen/petersengraphdeledges.csv"

    data_target = pd.read_csv(
        f"inputfiles/{target_filename}", dtype={"source": str, "target": str}, sep=","
    )
    data_source = pd.read_csv(
        f"inputfiles/{source_filename}", dtype={"source": str, "target": str}, sep=","
    )

    # Ensure the input graphs are unweighted.
    data_target['weight'] = 1.0
    data_source['weight'] = 1.0

    # Scorecard writing
    write_scorecard("----- Unweighted Deletetion Test -----")
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

    write_scorecard('***Source Analysis***')
    source_graph.run_early_analysis(verbose)

    write_scorecard('***Target Analysis***')
    target_graph.run_early_analysis(verbose)

    if verbose:
        clock_time("Time to analyze graphs")

    test_one_direction(source_graph, target_graph)
    return   



if __name__ == "__main__":
    CUTOFF = 0.1
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

    runtest(verbose)