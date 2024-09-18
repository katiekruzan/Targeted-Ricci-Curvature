'''
Idea: do a hypergraph and then subclasses
'''

class Hypergraph:
  def __init__(self):
        '''Initializing the hypergraph'''
        self.nodes = set() # arbitrary, not defined type as of now.
        self.hyperedges = {} #dict from hyperedge id to lists of nodes in that edge
        self.weights = {} # dict that had hyperedge ids to weights
        self.ricci_curvature = {} #dict with hyperedge id to list of ricci curvatures


class UndirectedHypergraph(Hypergraph):
  def tmp(self):
    return
  
class DirectedHypergraph(Hypergraph):
  def tmp2(self):
    return
  
if __name__ == "__main__": 
  print('hey')