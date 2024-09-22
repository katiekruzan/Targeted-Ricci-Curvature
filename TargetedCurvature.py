'''
Idea: do a hypergraph and then subclasses
'''

import pandas as pd

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
            
    def is_weakly_connected(self)-> bool:
        '''Check if the underlying graph is weakly connected
        '''
        # I think this is saying an empty graph is weakly connected
        if not self.nodes: 
            return True

        if isinstance(self, DirectedHypergraph):
            edges = self.get_underlying_edges()
        else: edges = self.hyperedges
        
        visited = set()

        def dfs(node):
            '''Depth First Search'''
            if node in visited:
                return
            visited.add(node)
            for edge in edges:
                if node in edge:
                    for next_node in edge:
                        if next_node != node:
                            dfs(next_node)

        # Start DFS from any node
        start_node = next(iter(self.nodes))
        dfs(start_node)

        return visited == self.nodes        
    
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
    def add_hyperedge(self, hyperedge_id:str, nodes:list, verbose=True):
        """Add a hyperedge to the hypergraph. Automatically adds missing nodes."""
        # Ensure nodes is a list
        # print(nodes)
        # quit()
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
        if verbose:
            f'Adding hyperedge {hyperedge_id} with nodes {nodes}'
        self.hyperedges[hyperedge_id] = nodes
        self.weights[hyperedge_id] = [1] #init the weights to 1
        return
    
    def build_from_dataframe(self, df:pd.DataFrame, verbose=True):
        '''Build hypergraph from a DataFrame'''
        print(self.nodes)
        print(self.hyperedges)
        # quit()
        # make an edge from each row in the csv
        #TODO: make this more general?
        for _, row in df.iterrows():
            node1 = row[0] #start
            node2 = row[1] #end
            edgeid = node1 + '_to_' + node2
            # print(node1, node2, edgeid)
            # quit()
            self.add_hyperedge(edgeid, [node1, node2], verbose)
        return
  
class DirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, tail_set:set, head_set:set):
        '''Function to add a hyperedge to the hypergraph'''
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]=[1] # init the weight to 1
        
    def get_underlying_edges(self) -> set:
        '''Function to get the edges from the hyperedges.
            Extract all edges from the hyperedges. 
            These are all possible connections (aka turn hypergraph into simple graph)
        '''
        edges = set()
        for tail_set, head_set in self.hyperedges.values():
            for tail in tail_set:
                for head in head_set:
                    edge = frozenset([tail, head])
                    edges.add(edge)
        return edges
    
  
if __name__ == "__main__": 
    directed_flag = False 
    verbose = True
    
    data = pd.read_csv('inputfiles/london_system.csv', header=None, sep='\t')
    
    #TODO: make a quick little test to make sure its a simple graph
    
    if directed_flag:
        graph = DirectedHypergraph()
        if verbose:
            print('directed')
    else:
        graph = UndirectedHypergraph()
        graph.build_from_dataframe(data, verbose)
        if verbose:
            print('undirected')
            print("Number of edges:",len(graph.hyperedges)) #Printing the number of hyperedges or papers in our network.
            print("Number of nodes",len(graph.nodes)) #Printing the number of nodes or authors in the network.
            # print('The actual nodes:', graph.nodes)
            connected = graph.is_weakly_connected()
            print("The hypergraph is weakly connected:" if connected else "The hypergraph is not weakly connected.")


            
    # print('hey')