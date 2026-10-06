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


def test_one_direction(src_graph:Hypergraph, targ_graph:Hypergraph, op_flag=False):

    targ_distance_matrix, distance_matrix, missing_from_src, missing_from_targ = (
            set_up_one_direction(src_graph, targ_graph, op_flag)
        )

    if verbose:
        clock_time("time to set up")
    print("starting ricci curvature")

    calculate_target_orc(targ_distance_matrix, targ_graph, verbose, op_flag)
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
    target_filename = "petersen/petersengraphExtraEdge.csv"

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

    runtest(verbose)