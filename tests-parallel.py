''' Written by Katie Kruzan in November 2025. this can be run with the following commandline
mpiexec -n 4 python .\tests-parallel.py
'''
  
from hypergraph import *
import os
from mpi4py import MPI
import time

COMM = MPI.COMM_WORLD

def manager(npr, verbose=True):
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