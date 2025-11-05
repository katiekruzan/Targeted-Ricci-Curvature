''' Written by Katie Kruzan in November 2025. this can be run with the following commandline
mpiexec -n 4 python .\tests-parallel.py
'''
  
from hypergraph import *
import os
from mpi4py import MPI
import time
import pandas as pd
import csv

COMM = MPI.COMM_WORLD

# helper functions
def clean_output(verbose:bool) -> None:
    '''puts all the files (other than the README) in the outputfiles/ folder into a subfolder

    :param bool verbose: Can turn on to print the file names it has moved.
    '''
    files = os.listdir('outputfiles')
    now = time.time()
    if not os.path.isdir(f'outputfiles/{now}'):
        os.makedirs(f'outputfiles/{now}')
    for f in files:
        if f=='README.md' or os.path.isdir(f'outputfiles/{f}'):
            continue
        else:
            try:
                os.rename(f'outputfiles/{f}', f'outputfiles/{now}/{f}')
                if verbose:
                    print(f'moving {f} to {now}/{f}')
            except:
                print(f'Had issues moving {f} to a new folder')
    return

def write_scorecard(line:str)-> None:
  '''function used to write the key results to a scorecard (for posterity)

  :param str line: text to be put onto the scorecard
  '''    
  with open('outputfiles/scorecard.txt', 'a+') as f:
      f.write(line)
      f.write('\n')
      
def clock_time(message:str)-> None:
    '''Get the time from the start of the process and write it to the scorecard with some message

    :param str message: The message to put before the time being spent
    '''
    now = time.time()
    rt = now-start
    write_scorecard(f'{message}: {rt}')
    return
  
def save_matrix_csv(matrix:list[list], filename:str) -> None:
  '''Function to save the matrix as a CSV file

  :param list[list] matrix: matrix to be written
  :param str filename: place to write it
  '''    
  pd.DataFrame(matrix).to_csv(filename, index=False, header=False)
  return
  
def ricci_normalizing(R: float)->float: 
  '''
  Using the normalization function sigma(R)/sigma(1) 
  Where sigma(x) is the standard sigmoid function 1/(1+\exp(-x))

  :param float R: the ORC value to be normalized 
  :return float: The normalized ORC value
  ''' 
  # return (1/(1+ np.exp(-R)))
  return (1/(1+ np.exp(-R)))

# process functions
def early_analysis(src_graph:Hypergraph, verbose:bool):
    '''Get the information on the graph we're working on

    :param Hypergraph src_graph: The graph we're analyzing
    :param bool verbose: verbose flag
    '''
    connected = src_graph.is_weakly_connected()
    strconnect = src_graph.is_strongly_connected()
    max_degree, min_degree, avg_degree = src_graph.calculate_degrees()

    if verbose:
        print('type of graph', type(src_graph))
        print("Number of edges:",len(src_graph.hyperedges)) #Printing the number of (hyper)edges in our network.
        print("Number of nodes",len(src_graph.nodes)) #Printing the number of nodes in the network.
        print('The actual nodes:', src_graph.nodes)
        print('The actual edges with weights:', src_graph.weights)
        
        print("The hypergraph is weakly connected." if connected else "The hypergraph is not weakly connected.")
        
        print(f"Max Degree: {max_degree}")
        print(f"Min Degree: {min_degree}")
        print(f"Average Degree: {avg_degree}")
        
    write_scorecard('----- Graph Statistics -----')
    write_scorecard(f'Type of Graph: {type(src_graph)}')
    write_scorecard(f'Number of edges: {len(src_graph.hyperedges)}')
    write_scorecard(f'Number of nodes: {len(src_graph.nodes)}')
    if connected: write_scorecard('The hypergraph is weakly connected.')
    else: write_scorecard('The hypergraph is not weakly connected.')
    if strconnect: write_scorecard('The hypergraph is strongly connected.')
    else: write_scorecard('The hypergraph is not strongly connected.')
    write_scorecard(f"Max Degree: {max_degree}")
    write_scorecard(f"Min Degree: {min_degree}") # Quick note: directed can have these be 0
    write_scorecard(f"Average Degree: {avg_degree}") # for directed graphs, this should be equal.
    write_scorecard('----------------------------')
    return
  
def set_up_one_direction(src_graph:Hypergraph, targ_graph:Hypergraph, op_flag=False):
  '''Setting up the one direction stuff. But in a separate function to help organize.
  Returns the target and source distance matrices (also persists them), and the 
  set of missing edges from source and target

  :param Hypergraph src_graph: The source graph
  :param Hypergraph targ_graph: The target graph
  :param bool op_flag: used to indicated if we should add the 'op_' prefix, defaults to False
  :return _type_: target_distance_matrix, distance_matrix, missing_from_src, missing_from_targ
  '''  
  print('working on distance matrices')
  distance_matrix = src_graph.floyd_warshall()
  matfilename = 'outputfiles/'
  if op_flag: matfilename += 'op_'
  matfilename += 'source_dist_fw.csv'
  save_matrix_csv(distance_matrix, matfilename)
  
  if verbose: clock_time('Time to make the source distance matrix')

  target_distance_matrix = targ_graph.floyd_warshall()
  matfilename = 'outputfiles/'
  if op_flag: matfilename += 'op_'
  matfilename += 'target_dist_fw.csv'
  save_matrix_csv(target_distance_matrix, matfilename)
  
  if verbose: clock_time('Time to make the target distance matrix')
  
  missing_from_src, missing_from_targ = [], []

  if set(targ_graph.hyperedges) != set(src_graph.hyperedges):
      print(set(targ_graph.hyperedges) - set(src_graph.hyperedges))
      # logging the edges that are different
      missing_from_src = set(targ_graph.hyperedges) - set(src_graph.hyperedges)
      missing_from_targ = set(src_graph.hyperedges) - set(targ_graph.hyperedges)
      
      print ('Taking care of missing edges')
      # add edges that are in the target but not the source
      src_graph.add_missing_edges_shortest_path(targ_graph, distance_matrix, verbose)
      targ_graph.add_missing_edges_shortest_path(src_graph, target_distance_matrix, verbose)
      if verbose: clock_time('time to add missing edges')
      
      # recalculate the matrices
      distance_matrix = src_graph.floyd_warshall()
      
      target_distance_matrix = targ_graph.floyd_warshall()
      if verbose: clock_time('time to recalc the distances')
  
  print('len missing from source', len(missing_from_src))
  print('len missing from targ', len(missing_from_targ))
  
  return target_distance_matrix, distance_matrix, missing_from_src, missing_from_targ


# Manager Functions

def calculate_target_orc_manager(npr:int, distance_matrix: list[list], graph:Hypergraph, verbose:bool, op_flag=False):
  '''The function to calculate the staring info for the target graph

  :param int npr: number of processors
  :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
  :param Hypergraph graph: the target graph
  :param bool verbose: verbose flag
  :param bool op_flag: option to mark the file as 'op' (used in second half of
                         script), defaults to False
  '''  
  if op_flag:
      file_name = f'outputfiles/op_dataset_target_graph_orc.csv'
  else:
      file_name = f'outputfiles/dataset_target_graph_orc.csv'
  
  with open(file_name, 'a', newline='') as file:
      writer = csv.writer(file)
      # Check if the file is empty to write headers
      #TODO: maybe fix the top writing?
      if file.tell() == 0:
          writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
          
      # split the edges. Will have the remaining edges be split on the first r processors
      edges = list(graph.hyperedges.keys())
      njobs = len(edges)
      chunksize = njobs//(npr-1)
      remainder = njobs % (npr-1)
      jobcnt = 0
      
      while jobcnt < npr-1:
          # send the jobs
          for i in range(1, npr):
              jobcnt = jobcnt + 1 # notably, will basically be equal to i
              jobstosend = []
              if jobcnt <= remainder:
                  jobstosend = edges[(jobcnt-1) * chunksize:jobcnt * chunksize] + [edges[-jobcnt]]
              else: 
                  jobstosend = edges[(jobcnt-1) * chunksize: jobcnt * chunksize]
              SLICE = (jobstosend, distance_matrix, graph, file_name, verbose)
              COMM.send(SLICE, dest = i, tag=33)
              if verbose:
                  print('-> manager sends job', jobcnt, 'to worker', i, 'number of jobs', len(SLICE[0]))
          if verbose: clock_time('manager sent all the jobs')
          # receive the jobs // sync the graphs.
          for i in range(1, npr):
              newgraph, jobs = COMM.recv(source=i, tag=11)
              if verbose: 
                  clock_time(f'gathered data from processor: {i}')
                  print('-> manager received data from worker', i, 'number of jobs', len(jobs))
              for e in jobs: #sync up the graph
                  graph.add_ricci_curvature(e, newgraph.ricci_curvature[e][-1])
                  graph.add_weights(e, newgraph.weights[e][-1])
                  writer.writerow([e, newgraph.ricci_curvature[e][-1], newgraph.weights[e][-1]]) 
  return



def one_direction_of_work_manager(npr:int, src_graph:Hypergraph, targ_graph:Hypergraph, tot_its = 100, op_flag=False):
  '''Doing the itterations from one graph to another

  :param int npr: the number of processors
  :param Hypergraph src_graph: The starting graph
  :param Hypergraph targ_graph: The target graph
  :param int tot_its: The maximum number of itterations to allow for this 
    process, defaults to 100
  :param bool op_flag: This is true if we want to add op_ as a prefix on the 
    files. Used to determine if we're going the opposite direction, defaults to False
  '''
  # One Direction of Work
  targ_distance_matrix, distance_matrix, missing_from_src, missing_from_targ = set_up_one_direction(src_graph, targ_graph, op_flag)
  
  if verbose: clock_time('time to set up')
  print('starting ricci curvature')
  
  calculate_target_orc_manager(npr, targ_distance_matrix, targ_graph, verbose, op_flag=op_flag)
  return

# Worker functions
def calculate_target_orc_worker():
    specs = COMM.recv(source = 0, tag = 33)
    if specs == -1: 
        return
    jobs, distance_matrix, graph, file_name, verbose = specs

    for hyperedge_id in jobs:
        if isinstance(graph, UndirectedHypergraph):
            orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix, verbose)
        elif isinstance(graph, DirectedHypergraph): 
            orc = 1 - graph.earthmover_distance_distance_matrix(hyperedge_id, distance_matrix, verbose=False)
        if RND1 and verbose:
            clock_time('finished ORC')
        normalized_orc = ricci_normalizing(orc)
        graph.add_ricci_curvature(hyperedge_id, normalized_orc)
    if verbose: clock_time('finished worker now sending over')
    COMM.send((graph,jobs) , dest=0, tag=11)
    return

def one_direction_of_work_worker(tot_its = 100):   
  calculate_target_orc_worker()
  return 


def manager(npr, verbose=True):
  clean_output(verbose)
  # source_filename = os.environ.get('SOURCE_FILENAME')
  # target_filename = os.environ.get('TARGET_FILENAME')
  source_filename = 'petersen/petersengraph.csv'
  target_filename = 'petersen/petersengraphExtraEdge.csv'
  
  data_target = pd.read_csv(f'inputfiles/{target_filename}', dtype ={'source': str, 'target':str}, sep=',')  
  data_source = pd.read_csv(f'inputfiles/{source_filename}', dtype ={'source': str, 'target':str}, sep=',')  
  
  # Scorecard writing
  write_scorecard('----- Targeted Ricci Curvature -----')
  write_scorecard(f'target filename: {target_filename}')
  write_scorecard(f'source filename: {source_filename}')
  if absolute_change:
      write_scorecard('Absolute vs Relative change: absolute change')
  else:
      write_scorecard('Absolute vs Relative change: relative change')
  if maximum_error:
      write_scorecard('Max vs Avg error: Maximum')
  else: 
      write_scorecard('Max vs Avg error: Avg')
  if approx_emd:
      write_scorecard('Type of EMD: Approx')
  else: 
      write_scorecard('Type of EMD: Exact')
  if gurobi_flag:
      write_scorecard('EMD Solver: Gurobi')
  else: 
      write_scorecard('EMD Solver: Python Optimal Transport')
  if verbose: clock_time('Time to read the data in seconds')
  
  if directed_flag:
      source_graph = DirectedHypergraph()
      target_graph = DirectedHypergraph()
  else:
      source_graph = UndirectedHypergraph()
      target_graph = UndirectedHypergraph() 
      
  print('building source')          
  source_graph.build_from_dataframe(data_source, verbose)
  
  print('building target')
  target_graph.build_from_dataframe(data_target, verbose)
  
  if verbose: clock_time('Time to build the graphs')
    
  if not (source_graph.is_2_uniform() and target_graph.is_2_uniform()) :
      print('This has not been fully fleshed out for hypergraphs. Please give a 2-uniform graph')
      quit()
  
  early_analysis(source_graph, verbose)
  if verbose: clock_time('Time to analyze graphs')
  
  one_direction_of_work_manager(npr, source_graph, target_graph, tot_its = ITS)
  return

def worker(w, verbose = True):
  one_direction_of_work_worker(tot_its = ITS)
  return  


if __name__ == "__main__": 
  verbose = False
  absolute_change = True # False is relative change
  maximum_error = True # False is average error
  gurobi_flag = os.environ.get('GUROBI_FLAG') 
  approx_emd = os.environ.get('APPROX')
  directed_flag = os.environ.get('DIRECTED_FLAG') 
  if approx_emd is None: # Make the default False
      approx_emd = 'False'
  approx_emd = eval(approx_emd)
  if gurobi_flag is None: # make gurobi flag default false
      gurobi_flag = 'False'
  gurobi_flag = eval(gurobi_flag) 
  if directed_flag is None: # make directed flag default false
      directed_flag = 'False'
  directed_flag = eval(directed_flag) 
  if approx_emd: gurobi_flag = False
  
  start = time.time()
  
  RANK = COMM.Get_rank()
  SIZE = COMM.Get_size()
  ITS = 100
  if RANK == 0:
      manager(SIZE, verbose)
  else: 
      worker(RANK, verbose)
  print(f'node {RANK} made it to the end')