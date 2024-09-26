'''
Idea: do a hypergraph and then subclasses
'''

import pandas as pd
import csv

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
            # TODO: Get is_weakly_connected to work for Directed
            edges = self.get_underlying_edges()
        else: edges = self.hyperedges
        
        print(edges)
        
        visited = set()

        def dfs(node):
            '''Depth First Search'''
            if node in visited:
                return
            visited.add(node)
            for edge in edges.values():
                if node in edge:
                    for next_node in edge:
                        if next_node != node:
                            dfs(next_node)

        # Start DFS from any node
        start_node = next(iter(self.nodes))
        dfs(start_node)
        
        print(visited)

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
    
    

    
    def earthmover_distance_gurobi_distance_matrix(self, node_A, node_B, distance_matrix):
        '''
        We will do this over two separate nodes. The directed script does all combos together, 
        but might make sense to just do 2 nodes for now.
        '''
        # TODO: actually figure out the probability distributions. 
        if node_A not in self.nodes or node_B not in self.nodes:
            print(f"Node {node_A} or {node_B} does not exist in the hypergraph.")
            return None  # Return None if either node does not exist
        
        '''Function to calculate EMD using the distance matrix (Optimized)'''
        # Get the probability distributions for the specified hyperedge.
        mu_A, mu_B = self.calculate_probability_distributions(hyperedge_id)

        # Convert distributions from dictionary to list format and print for debugging
        nodes_A = sorted(mu_A.keys())
        nodes_B = sorted(mu_B.keys())
        distribution1 = [mu_A[node] for node in nodes_A]
        distribution2 = [mu_B[node] for node in nodes_B]
    
        # Print the distributions to verify correctness
        print("Nodes in mu_A:", nodes_A)
        print("Nodes in mu_B:", nodes_B)
        print("Distribution mu_A:", distribution1)
        print("Distribution mu_B:", distribution2)

        # Check if distributions sum to the same value
        total_mass_A = sum(distribution1)
        total_mass_B = sum(distribution2)
        print("Total mass in mu_A:", total_mass_A)
        print("Total mass in mu_B:", total_mass_B)
    
        if abs(total_mass_A - total_mass_B) > 1e-6:
            raise ValueError('The total mass of the distributions mu_A and mu_B are not equal.')
        

        # Create a mapping of nodes to their indices in the distance matrix.
        node_to_index = {node: idx for idx, node in enumerate(self.nodes)}

        
        try:
            model = Model("EarthMoverDistance")

            # Set up the log file
            log_filename = f"gurobi_log_{hyperedge_id}.log"
            model.setParam('LogFile', log_filename)

            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)

            # Update the objective function to use the distance matrix.
            model.setObjective(quicksum(distance_matrix[node_to_index[x]][node_to_index[y]] * variables[x, y]
                                for x in mu_A for y in mu_B), GRB.MINIMIZE)

            # Add constraints
            for x in mu_A:
                model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

            for y in mu_B:
                model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")

            start_time = time.time()
            model.optimize()
            end_time = time.time()

            time_taken = end_time - start_time

            if model.status == GRB.OPTIMAL:
                total_cost = model.getObjective().getValue()
                print("Total EMD Cost:", total_cost)
                print("Time taken to find the optimal solution: {:.4f} seconds".format(time_taken))

                for x in mu_A:
                    for y in mu_B:
                        amount_moved = variables[x, y].X
                        if amount_moved > 0:
                            print(f"Move {amount_moved} from {x} to {y}")
                return total_cost
            else:
                print("No optimal solution found.")
                return None
            
        except Exception as e:
            print(f"Gurobi Error: {e}")
            return None



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
            node1 = row[0].strip() #start
            node2 = row[1].strip() #end
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
    
def save_matrix_csv(matrix, filename:str) -> None:
    '''Function to save the matrix as a CSV file'''    
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)

def update_orc_and_weights_iter(distance_matrix, iteration, file_format='csv'):
    file_name = f'dataset_targeted_curvature_iteration_{iteration}.{file_format}'
    
    with open(file_name, 'a', newline='') as file:
        if file_format == 'csv':
            writer = csv.writer(file)
            # Check if the file is empty to write headers
            if file.tell() == 0:
                writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
            
            for hyperedge_id in graph.hyperedges:
                # TODO: Figure out the difference in Directed//Undirected
                orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix)
                # add the value to our graph
                graph.add_ricci_curvature(hyperedge_id, orc)
                # update the weights
                weight = graph.weights[hyperedge_id][-1]
                
                if weight != 0:
                    # TODO: This is where the weights are being updated now -- where they will have to be fixed
                    weight = weight * (1 - orc)
                    normalized_weight = adjusted_sigmoid_0_to_1(weight)
                else:
                    normalized_weight == 0

                graph.add_weights(hyperedge_id, normalized_weight)
                
                writer.writerow([hyperedge_id, orc, normalized_weight])
                
def adjusted_sigmoid_0_to_1(x):
    # Clip x to a range that prevents overflow in exp.
    # The range of -709 to 709 is chosen based on the practical limits of np.exp()
    x_clipped = np.clip(x, -709, 709)
    a, b = 0, 1  # Define the target range
    return a + (b - a) / (1 + np.exp(-x_clipped))
    
  
if __name__ == "__main__": 
    directed_flag = False 
    verbose = True
    
    # TODO: make a simple graph data set. We need 2 graphs 
    data1 = pd.read_csv('inputfiles/petersengraph.csv', header=None, sep=',')
    data2 = pd.read_csv('inputfiles/petersengraphExtraEdge.csv', header=None, sep=',')
    # print(data1.info())
    # print(data1.head())
    
    # quit()
    
    #TODO: make a quick little test to make sure its a simple graph
    
    if directed_flag:
        graph = DirectedHypergraph()
        if verbose:
            print('directed')
    else:
        graph = UndirectedHypergraph()
        graph.build_from_dataframe(data1, verbose)
        if verbose:
            print('undirected')
            print("Number of edges:",len(graph.hyperedges)) #Printing the number of hyperedges or papers in our network.
            print("Number of nodes",len(graph.nodes)) #Printing the number of nodes or authors in the network.
            # print('The actual nodes:', graph.nodes)

            connected = graph.is_weakly_connected()
            print("The hypergraph is weakly connected:" if connected else "The hypergraph is not weakly connected.")
            
            # TODO: get the stats of the graph (max, min, avg)

    distance_matrix = graph.floyd_warshall()
    save_matrix_csv(distance_matrix, 'outputfiles/undirected_testing_fw.csv')

    print('starting ricci curvature')

    #TODO: Same idea as in the directed Hypergraph script
    
    #TODO: check to see if the guys are the same
    # update_orc_and_weights_iter0(distance_matrix,iteration=0)
    
    print('Itteration 0 done')
 
            
    # print('hey')