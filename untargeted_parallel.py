from hypergraph import *
from ricciutil import *
from mpi4py import MPI
import csv

COMM = MPI.COMM_WORLD


# Manager functions
def update_orc_and_weights_iter_manager(
    npr: int, hypergraph: Hypergraph, dist_matrix: list[list], iteration: int, graphname: str, verbose=False
):
    '''The main function of this whole sheboodle. Run the whole process for the given itteration

    :param int npr: the number of processors
    :param Hypergraph hypergraph: the graph we're looking at
    :param list[list] dist_matrix: matrix of minimal distances from the floyd_warshall function
    :param int iteration: the round we're on
    :param str graphname: This name is used in the filename
    :param bool verbose: verbose flag, defaults to False
    '''
    file_name = f"outputfiles/dataset_untargeted_curvature_{graphname}_iteration_{iteration}.csv"
    updated_weights = {}

    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(
                ["Hyperedge ID", "ORC: (based on t-1 weights)", "Weight:t"])

        # split the edges. Just do it the simple way. Just have the last one take the rest
        edges = list(hypergraph.hyperedges.keys())
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
                SLICE = (jobstosend, dist_matrix,
                         hypergraph, iteration, verbose)
                COMM.send(SLICE, dest=i, tag=333)
                if verbose:
                    print(
                        "-> manager sends job",
                        jobcnt,
                        "to worker",
                        i,
                        "number of jobs",
                        len(SLICE[0]),
                    )
            # receive the jobs // sync the graphs.
            for i in range(1, npr):
                newgraph, updated_weightspt, jobs = COMM.recv(source=i, tag=15)
                if verbose:
                    print(
                        "-> manager received data from worker",
                        i,
                        "number of jobs",
                        len(jobs),
                    )

                updated_weights.update(updated_weightspt)
                for e in jobs:  # sync up the graph
                    hypergraph.add_ricci_curvature(
                        e, newgraph.ricci_curvature[e][-1])

            for hyperedge_id, new_weight in updated_weights.items():
                normalized_wt = new_weight
                writer.writerow(
                    [
                        hyperedge_id,
                        hypergraph.ricci_curvature[hyperedge_id][-1],
                        normalized_wt,
                    ]
                )
                hypergraph.add_weights(hyperedge_id, normalized_wt)
    return


def one_direction_of_work_manager(npr: int, source_file: str, graphname: str, verbose=False):
    '''Organizes everything that should be done in one direction. This Runs the 
    whole process with one graph. Getting the whole curvature and all the way down.

    :param int npr: the number of processors
    :param str source_file: filepath name to where the source graph lies. Should be a csv
    :param str graphname: This name is used in the filename to persist info about the graph
    :param bool verbose: verbose flag, defaults to False
    :return: The final matrix and the source graph
    '''
    if directed_flag:
        source_graph = DirectedHypergraph()
    else:
        source_graph = UndirectedHypergraph()

    print("building the graph")

    df = pd.read_csv(source_file)
    source_graph.build_from_dataframe(df, verbose)

    write_scorecard(f"Type of graph: {type(source_graph)}")
    write_scorecard(f"Number of edges: {len(source_graph.hyperedges)}")
    write_scorecard(f"Number of nodes: {len(source_graph.nodes)}")

    for i in range(ITTS):
        print("Starting itteration ", i)

        # TODO: persist the most recent distance matrix
        dist_matrix = source_graph.floyd_warshall()
        # print('finished floyd warshal')
        update_orc_and_weights_iter_manager(
            npr, source_graph, dist_matrix, i, graphname, verbose
        )
        clock_time(f"Time for ORC {i}")

        allstable = True
        if i < 2:
            # Take care of the getting started case
            continue
        errorlist = []
        for e in source_graph.hyperedges:
            clist = source_graph.ricci_curvature[e]
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
                    # if verbose:
                    clock_time(f"unstable for edge {e} with error {error}")
                    allstable = False
                    break
                if (not absolute_change) and (error > 0.05):  # relative change
                    clock_time(f"unstable for edge {e} with error {error}")
                    allstable = False
                    break
            else:
                errorlist.append(error)
        if not maximum_error:  # AKA we're in average error zone
            # assumed to be in absolute error zone
            avg_err = np.average(errorlist)
            if avg_err > 0.0001:
                clock_time(f"unstable with average error {avg_err}")
                allstable = False
        if allstable:
            # turn off all workers.
            for k in range(1, npr):
                COMM.send(-1, dest=k, tag=333)
            print("STABILIZED! Source to target distance is ", i)
            write_scorecard("\n\n----- Results -----")
            write_scorecard(f"Source to target distance is {i}")
            break

    final_dist_matrix = source_graph.floyd_warshall()
    return final_dist_matrix, source_graph


# Worker Functions
def update_orc_and_weights_iter_worker(w: int) -> bool:
    '''The worker that is actually computing the EMDs and the new weights

    :param int w: RANK of the worker that we're on
    :return bool: True if the process should continue, False otherwise
    '''
    updated_weights = {}
    specs = COMM.recv(source=0, tag=333)
    if specs == -1:
        return False
    jobs, dist_matrix, graph, iteration, verbose = specs

    if verbose:
        print("worker", w, "starting job")
    for hyperedge_id in jobs:
        nodes = graph.hyperedges[hyperedge_id]
        if len(nodes) < 2:  # hyper edges with less than 2 edges
            continue

        if isinstance(graph, UndirectedHypergraph):
            u, v = (
                nodes[0],
                nodes[1],
            )  # Notably will only work for graphs, not hyper graphs
            orc = 1 - graph.earthmover_distance_distance_matrix(
                (u, v, hyperedge_id), dist_matrix, approx_emd, gurobi_flag, verbose
            )
        elif isinstance(graph, DirectedHypergraph):
            orc = 1 - graph.earthmover_distance_distance_matrix(
                hyperedge_id, dist_matrix, approx_emd, gurobi_flag, verbose
            )
        # Normalize the curvature
        normalized_orc = ricci_normalizing(orc)

        graph.add_ricci_curvature(hyperedge_id, normalized_orc)

        weight = graph.weights[hyperedge_id][-1]

        if iteration != 0:
            if weight != 0:
                step = 1
                wtplus1 = weight * (1 - step * normalized_orc)
            else:
                wtplus1 = weight
            # TODO: see if this makes sense to track updated weights separately
            # NOTE:The only thing here, is I *think* weights are allowed to be 0
            updated_weights[hyperedge_id] = max(wtplus1, 1e-8)
        else:
            updated_weights[hyperedge_id] = weight
    if verbose:
        print("worker", w, "finished this round")
    COMM.send((graph, updated_weights, jobs), dest=0, tag=15)

    return True


def one_direction_of_work_worker(w: int, tot_its=100):
    '''This corresponds to the one_direction_of_work manager

    :param int w: RANK of the worker that we're on
    :param int tot_its: number of max iterations before considering divergence, defaults to 100
    '''
    update_orc_and_weights_iter_worker(w)
    cont = True
    cnt = 1
    while cont and (cnt <= tot_its):
        cont = update_orc_and_weights_iter_worker(w)
        cnt = cnt + 1
    return


def build_distance_vectors(dist_dict: list[list], nodes_order: list, node_index: dict) -> np.array:
    '''Lining everything up in order to get things set for the norms

    :param list[list] dist_dict: distance matrices in the floyd_warshall form
    :param list nodes_order: a list of the nodes in the order we'll set with
    :param dict node_index: The dictionary that will say what position it is in the dist_dict
    :return np.array: Numpy array of the vectors
    '''
    vectors = []
    for src in nodes_order:
        vec = [dist_dict[node_index[src]][node_index[tgt]]
               for tgt in nodes_order]
        vectors.append(vec)
    return np.array(vectors)


def manager(npr: int, verbose=False):
    '''This is the main function for the manager node

    :param int npr: number of worker nodes we're working with
    :param bool verbose: verbose flag, defaults to False
    '''
    clean_output(verbose)

    source_filename = os.environ.get('SOURCE_FILENAME')
    target_filename = os.environ.get('TARGET_FILENAME')
    # source_filename = "petersen/petersengraph.csv"
    # target_filename = "petersen/petersengraphExtraEdge.csv"

    path1 = "inputfiles/" + source_filename
    path2 = "inputfiles/" + target_filename

    # Scorecard Writing
    write_scorecard("----- Untargeted Ricci Curvature -----")
    write_scorecard(f"Graph 1 filename: {path1}")
    write_scorecard(f"Graph 2 filename: {path2}")

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

    clock_time("Time to read the data in seconds")

    dist_G1, G1 = one_direction_of_work_manager(npr, path1, "Graph1", verbose)
    write_scorecard('Finished the first graph')
    order_G1 = sorted(G1.nodes)

    D1 = build_distance_vectors(dist_G1, order_G1, G1.node_index)
    np.savetxt("outputfiles/Graph1FinalDistance.txt",
               D1, delimiter=" ", fmt="%f")

    dist_G2, G2 = one_direction_of_work_manager(npr, path2, "Graph2", verbose)
    write_scorecard('Finished the second graph')

    D2 = build_distance_vectors(dist_G2, order_G1, G2.node_index)
    np.savetxt("outputfiles/Graph2FinalDistance.txt",
               D2, delimiter=" ", fmt="%f")

    write_scorecard('The final distance:')
    write_scorecard(str(np.linalg.norm(D1 - D2)))
    write_scorecard("New Distance")
    dist = 0
    for v in range(len(D1)):
        top = np.dot(D1[v], D2[v])
        bot = np.linalg.norm(D1) * np.linalg.norm(D2)
        dist = dist + (top/bot)
    write_scorecard(str(dist))

    # tell the jobs to sleep (at the very end)
    for i in range(1, npr):
        SLICE = -33
        COMM.send(SLICE, dest=i, tag=55)
        if verbose:
            print(f'-> manager sends {SLICE} to worker', i)
    return


def worker(w: int, verbose=False):
    '''This is the main function for the worker node

    :param int w: The rank of the specific worker node we're using. Used primarily in pulling the information
    :param bool verbose: verbose flag, defaults to False
    '''
    one_direction_of_work_worker(w, tot_its=ITTS)
    one_direction_of_work_worker(w, tot_its=ITTS)
    while True:
        specs = COMM.recv(source=0, tag=55)
        if specs == -33:
            if verbose:
                print(f'Worker {w} goes to sleep')
            break
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
    set_start(start)

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
