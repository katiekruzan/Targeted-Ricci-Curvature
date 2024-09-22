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
            
    def add_node(self, node:any) -> None:
        '''Function to add a node to the hypergraph. The type is not set'''
        self.nodes.add(node)
        return
    
    def add_ricci_curvature(self, hyperedge_id:str, orc)-> None:
        '''Function to add ollivier ricci curvature for all hyperedges for every iteration.
            Seems to be appending onto a list.'''
        if hyperedge_id not in self.ricci_curvature:
            self.ricci_curvature[hyperedge_id] = []  # Initialize with an empty list if key doesn't exist
        self.ricci_curvature[hyperedge_id].append(orc)
        return
        
    def add_weights(self, hyperedge_id:str, weights) -> None:
        '''Function to add weights for all hyperedges for every iteration.
            Seems to be appending to a list.'''
        if weights is not None:
            self.weights[hyperedge_id].append(weights)
            
    def floyd_warshall(self):
        # TODO: double check this works for both and with weights
        node_list = list(self.nodes)
        index = {node: idx for idx, node in enumerate(node_list)}
        n = len(node_list)
        dist = [[float('inf') for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            dist[i][i] = 0
        
        for hyperedge_id, nodes in self.hyperedges.items():
            weight = self.weights[hyperedge_id][-1]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    idx_i = index[nodes[i]]
                    idx_j = index[nodes[j]]
                    if dist[idx_i][idx_j] > weight:
                        dist[idx_i][idx_j] = weight
                        dist[idx_j][idx_i] = weight  # Graph is undirected

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        dist[j][i] = dist[i][j]

        # Replace 'inf' with 0 for pairs of nodes that have no path between them
        for i in range(n):
            for j in range(n):
                if dist[i][j] == float('inf'):
                    dist[i][j] = 0 

        return dist


class UndirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, nodes:list):
        """Add a hyperedge to the hypergraph. Automatically adds missing nodes."""
        # Ensure nodes is a list
        if not isinstance(nodes, list):
            raise ValueError("Nodes must be provided as a list")
        
        # Check if hyperedge already exists
        if hyperedge_id in self.hyperedges:
            print(f"Hyperedge {hyperedge_id} already exists with nodes {self.hyperedges[hyperedge_id]}")
            return
        # Add missing nodes to the node set
        for node in nodes:
            if node not in self.nodes:
                self.add_node(node)

        # Add the hyperedge
        self.hyperedges[hyperedge_id] = nodes
        self.weights[hyperedge_id] = [1] #init the weights to 1
        return
  
class DirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, tail_set:set, head_set:set):
        '''Function to add a hyperedge to the hypergraph'''
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]=[1] # init the weight to 1
  
if __name__ == "__main__": 
    
    directed_flag = True 
    verbose = True
    
    if directed_flag:
        if verbose:
            print('directed')
    else:
        if verbose:
            print('undirected')
            
    print('hey')