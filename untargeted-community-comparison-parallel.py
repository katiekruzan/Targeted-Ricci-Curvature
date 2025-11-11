from hypergraph import *
from ricciutil import *
from mpi4py import MPI
import csv
from sklearn.metrics import rand_score, adjusted_rand_score


COMM = MPI.COMM_WORLD


def delete_hyperedges(hypergraph:Hypergraph, percentage=0.08):
    '''We want to get the *highest* value weighted edges and delete them

    :param _type_ hypergraph: _description_
    :param float percentage: _description_, defaults to 0.08
    '''    
    total_hyperedges = len(hypergraph.hyperedges)
    del_hyperedges = int(percentage * total_hyperedges)
    recent_weights = {he: hypergraph.weights[he][-1] for he in hypergraph.hyperedges.keys()}
    hyperedges_to_remove = sorted(recent_weights, key=recent_weights.get, reverse=True)[:del_hyperedges]
    for he in hyperedges_to_remove:
        hypergraph.remove_hyperedge(he)
    

def write_hypergraph_stats(hypergraph:Hypergraph):
    write_scorecard(f"Number of hyperedges: {len(hypergraph.hyperedges)}")
    write_scorecard(f"Number of nodes: {len(hypergraph.nodes)}")
    connected = hypergraph.is_weakly_connected()
    write_scorecard("The hypergraph is weakly connected:" if connected else "The hypergraph is not weakly connected.")
    components = hypergraph.connected_components()
    write_scorecard(f"Connected Components: {components}")
    write_scorecard(f"No. of modules: {len(set(components.values()))}")
    # # Listing all hyperedges
    return components
            

# Manager Functions
def update_orc_and_weights_iter_manager(
    npr:int, hypergraph: Hypergraph, dist_matrix:list[list], iteration:int, graphname:str, verbose=False
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
            writer.writerow(["Hyperedge ID", "ORC: (based on t-1 weights)", "Weight:t"])

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
                        (jobcnt - 1) * chunksize : jobcnt * chunksize
                    ] + [edges[-jobcnt]]
                else:
                    jobstosend = edges[(jobcnt - 1) * chunksize : jobcnt * chunksize]
                SLICE = (jobstosend, dist_matrix, hypergraph, iteration, verbose)
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
                    hypergraph.add_ricci_curvature(e, newgraph.ricci_curvature[e][-1])

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


def one_direction_of_work_manager(npr, source_file, graphname):
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

        update_orc_and_weights_iter_manager(
            npr, source_graph, dist_matrix, i, graphname, verbose
        )
        clock_time(f"Time for ORC {i}")
        
        if i%2 ==0: # then we become surgeons
            print('time for surgery')
            delete_hyperedges(source_graph, percentage=0.08)
    
    communities = write_hypergraph_stats(source_graph)
    return communities

# Worker functions
def update_orc_and_weights_iter_worker(w):
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
            #TODO: see if this makes sense to track updated weights separately
            updated_weights[hyperedge_id] = max(wtplus1, 1e-8)
        else:
            updated_weights[hyperedge_id] = weight
    if verbose:
        print("worker", w, "finished this round")
    COMM.send((graph, updated_weights, jobs), dest=0, tag=15)

    return True


def one_direction_of_work_worker(w, tot_its = 100):   
    update_orc_and_weights_iter_worker(w) 
    cont=True
    cnt = 1
    while cont and (cnt<tot_its):
        cont = update_orc_and_weights_iter_worker(w)
        cnt = cnt + 1
    return 

def manager(npr:int, verbose=False):
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
    write_scorecard("----- Untargeted Community Comparison Ricci Curvature -----")
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

    communities1 = one_direction_of_work_manager(npr, path1, 'Graph1')
    write_scorecard('Finished the first graph\n')
    
    communities2 = one_direction_of_work_manager(npr, path2, 'Graph2')
    write_scorecard('Finished the second graph')
    
    # Now to do the adjusted random index. We will need to label each of them.
    labels1 = []
    labels2 = []
    for node in communities1.keys():
        labels1.append(communities1[node])
        labels2.append(communities2[node])
        
    write_scorecard('\nThe final community comparisons')
    write_scorecard(f'Rand Score: {rand_score(labels1, labels2)}')
    write_scorecard(f'Adjusted Rand Score: {adjusted_rand_score(labels1, labels2)}')
    
    # tell the jobs to sleep (at the very end)
    # send the jobs
    for i in range(1, npr):
        SLICE = -33
        COMM.send(SLICE, dest = i, tag=55)
        if verbose:
            print(f'-> manager sends {SLICE} to worker', i)
    return

def worker(w:int, verbose=False):
    one_direction_of_work_worker(w, tot_its=ITTS)
    one_direction_of_work_worker(w, tot_its=ITTS)
    # Turn off the workers
    print('worker is workin')
    while True:
        specs = COMM.recv(source = 0, tag = 55)
        if specs == -33: 
            if verbose:
                print(f'Worker {w} goes to sleep')
            break
    return


if __name__ == "__main__":
    """
    We're going to run this for a set number of itterations. As I don't think this is the same 
    sense of convergence as we want. Looking at the paper.
    """
    
    ITTS = 10
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