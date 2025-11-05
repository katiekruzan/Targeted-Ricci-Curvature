''' Written by Katie Kruzan in November 2025. this can be run with the following commandline
mpiexec -n 4 python .\tests-parallel.py
'''
  
from hypergraph import *
import os
from mpi4py import MPI
import time
import pandas as pd

COMM = MPI.COMM_WORLD

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
  
  
  return

def worker(w, verbose = True):
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