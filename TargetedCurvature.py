'''
Idea: do a hypergraph and then subclasses
'''

import pandas as pd
import csv
import numpy as np
from itertools import combinations
from gurobipy import Model, GRB, quicksum

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
            
    def is_2_uniform(self) -> bool:
        # need to check size of each edge is 2
        # find a way to do this quicker
        for edges in self.hyperedges.items():
            if len(edges) != 2:
                return False
        return True
            
    def is_weakly_connected(self)-> bool:
        '''Check if the underlying graph is weakly connected
        '''
        # I think this is saying an empty graph is weakly connected
        if not self.nodes: 
            return True

        if isinstance(self, DirectedHypergraph):
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
  
    
    def floyd_warshall(self) -> list[list]:
        # Initialize the distance matrix with "infinite" distances
        # Assume self.nodes is a list or set of nodes
        node_list = list(self.nodes) # Convert to list to ensure consistent ordering
        node_count = len(node_list)
        
        # Create a mapping of node to index
        node_index = {node: idx for idx, node in enumerate(node_list)}

        # Initialize a 2D list (matrix) with "infinite" distances
        dist = [[float('inf') for _ in range(node_count)] for _ in range(node_count)]
        
        # Set the diagonal to 0 (distance from each node to itself)
        for i in range(node_count):
            dist[i][i] = 0
        
        # Set the distance for directly connected nodes based on edge weights
        for hyperedge_id, nodes in self.hyperedges.items():
            if isinstance(self, UndirectedHypergraph):
                tail_set, head_set = nodes, nodes
            else: 
                tail_set, head_set = nodes
            
            # set distances within tail set and head set to 0
            #TODO: test this when it comes to actual hypergraphs
            '''
            for tail in tail_set:
                for another_tail in tail_set:
                    if tail != another_tail:
                        dist[node_index[tail]][node_index[another_tail]] = 0
            for head in head_set:
                for another_head in head_set:
                    if head != another_head:
                        dist[node_index[head]][node_index[another_head]] = 0
            '''
          
            for tail in tail_set:
                for head in head_set:
                    # Update the distance with the weight of the edge
                    # Assuming edge_id is used to access weights; adjust accordingly
                    dist[node_index[tail]][node_index[head]] = min(dist[node_index[tail]][node_index[head]],self.weights[hyperedge_id][-1])  # Using the last weight in the list
        
        # Floyd-Warshall algorithm to update distances
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]] < dist[node_index[i]][node_index[j]]:
                        dist[node_index[i]][node_index[j]] = dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]]

        # Replace 'inf' with 0 for pairs of nodes that have no path between them
        for i in range(node_count):
            for j in range(node_count):
                if dist[i][j] == float('inf'):
                    dist[i][j] = 0 

        return dist
        
        
    def calculate_degrees(self):
        '''
        Return the max degree, min degree, and average degree values. For Directed, we're get (in, out) pairs
        '''
        degrees = []
        
        # Iterate over each node in the hypergraph
        for node in self.nodes:
            degrees.append(self.node_degree(node))

        degrees = np.array(degrees)
        
        # If there are no nodes or degrees calculated, handle the case gracefully
        if len(degrees) == 0:
            max_degree = 0
            min_degree = 0
            avg_degree = 0.0
        else:
            max_degree = np.max(degrees, axis=0)
            min_degree = np.min(degrees, axis=0)
            avg_degree = np.average(degrees, axis=0)
        
        return max_degree, min_degree, avg_degree


class UndirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, nodes:list, verbose=True):
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
        if verbose:
            f'Adding hyperedge {hyperedge_id} with nodes {nodes}'
        self.hyperedges[hyperedge_id] = nodes
        self.weights[hyperedge_id] = [1] #init the weights to 1
        return
    
    def build_from_dataframe(self, df:pd.DataFrame, verbose=True):
        '''Build hypergraph from a DataFrame'''
        # make an edge from each row in the csv
        #TODO: make this more general?
        for _, row in df.iterrows():
            node1 = row[0].strip() #start
            node2 = row[1].strip() #end
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, [node1, node2], verbose)
            if verbose:
                print(f'Added hyperedge {edgeid} between {node1} and {node2}')
        return
    
    def node_degree(self, node):
        """Calculate the degree of a node. Degree is the number of hyperedges containing this node."""
        if node not in self.nodes:
            raise ValueError("Node does not exist in the graph.")
        return sum(node in hyperedge for hyperedge in self.hyperedges.values())
    
    
    def find_hyperedges_containing_nodes(self, *nodes):
        '''
        Find hyperedges that contain any of the specified nodes.
        # Ensure input is treated as a list even if a single node is passed
        if isinstance(nodes, str):
            nodes = [nodes]  # Convert single string node to a list
        '''
        nodes_set = set(nodes)  # Convert list to set for efficient intersection checks

        # Handle different types of inputs
        for node in nodes:
            if isinstance(node, (list, set, tuple)):  # If the input is any kind of collection
                nodes_set.update(node)  # Add all elements to the set
            else:
                nodes_set.add(node)  # Add the single element to the set

        found_hyperedges = []
        # Ensure all nodes in the set are in our nodes list
        if not nodes_set.issubset(self.nodes):
            print("Some nodes are not in the hypergraph.")
        
        # Iterate through all hyperedges
        for hyperedge_id, hyperedge_nodes in self.hyperedges.items():
            if nodes_set.intersection(hyperedge_nodes):  # Check if intersection is not empty
                found_hyperedges.append(hyperedge_id)
        
        return found_hyperedges
    
    def neighbours(self, node):
        """
        Find all nodes that share at least one hyperedge with the specified node.

        :param node: The node for which to find neighbors.
        :return: A set of neighboring nodes.
        """
        if node not in self.nodes:
            return set()  # Return an empty set if the node does not exist

        neighbours = set()
        # Iterate through all hyperedges
        for hyperedge in self.hyperedges.values():
            if node in hyperedge:
                neighbours.update(hyperedge)  # Add all nodes in the hyperedge

        neighbours.discard(node)  # Remove the node itself from the set of neighbours
        return neighbours
    
    def node_probability(self, node):
        alpha = 0.1  # Self-transition probability factor
        probability_distribution = {n: 0.0 for n in self.nodes}  # Initialize probabilities

        if node not in self.nodes:
            raise ValueError("Node does not exist in the hypergraph.")

        # Calculate the denominator: sum of (|f| - 1) for all f containing node
        denominator = 0
        hyperedges_containing_node = self.find_hyperedges_containing_nodes(node)
        for hyperedge_id in hyperedges_containing_node:
            hyperedge = self.hyperedges[hyperedge_id]
            denominator += (len(hyperedge) - 1)

        if denominator == 0:
            # If the denominator is zero, we should handle this edge case gracefully.
            probability_distribution[node] = 1.0
            return probability_distribution

        # Calculate the numerator for each neighbor node j and update their probabilities
        for neighbour in self.neighbours(node):
            hyperedges_containing_both = self.find_hyperedges_containing_nodes(node, neighbour)
            numerator = len(hyperedges_containing_both)
            '''
            for hyperedge_id in hyperedges_containing_both:
                hyperedge = self.hyperedges[hyperedge_id]
                numerator += (len(hyperedge))
            '''
            # Update the probability of transitioning to the neighbor
            probability_distribution[neighbour] = (1 - alpha) * numerator / denominator

        # Assign the self-loop probability
        probability_distribution[node] = alpha

        # Normalization step
        
        total_probability = sum(probability_distribution.values())
        for n in probability_distribution:
            probability_distribution[n] /= total_probability
        
        return probability_distribution
    
    
    def earthmover_distance_gurobi_distance_matrix(self, node_A, node_B, distance_matrix):
        if node_A not in self.nodes or node_B not in self.nodes:
            print(f"Node {node_A} or {node_B} does not exist in the hypergraph.")
            return None  # Return None if either node does not exist
        
        # Get the probability distributions for the two specified nodes.
        mu_A = self.node_probability(node_A)
        mu_B = self.node_probability(node_B)

        # Convert distributions from dictionary to list format 
        nodes_A = sorted(mu_A.keys())
        nodes_B = sorted(mu_B.keys())
        distribution1 = [mu_A[node] for node in nodes_A]
        distribution2 = [mu_B[node] for node in nodes_B]

        # Check if distributions sum to the same value
        total_mass_A = sum(distribution1)
        total_mass_B = sum(distribution2)
    
        if abs(total_mass_A - total_mass_B) > 1e-6:
            # TODO: improve error message
            raise ValueError('The total mass of the distributions mu_A and mu_B are not equal. For')
        

        # Create a mapping of nodes to their indices in the distance matrix.
        node_to_index = {node: idx for idx, node in enumerate(list(self.nodes))}

        try:
            # Create a new model in Gurobi.
            model = Model("EarthMoverDistance")

            # Set up the log file
            #log_filename = f"gurobi_log_{hyperedge_id}.log"
            # Set up the log file
            '''
            log_filename = f"gurobi_log_{hyperedge_id}.log"
            model.setParam('LogFile', log_filename)
            '''
            #model.setParam('OutputFlag', 1)
            # Create variables for the linear program.
            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)

            # Set the objective of the linear program to minimize the total cost.
            model.setObjective(quicksum(distance_matrix[node_to_index[x]][node_to_index[y]] * variables[x, y]
                                for x in mu_A for y in mu_B), GRB.MINIMIZE)

            # Add constraints to ensure the conservation of mass.
            for x in mu_A:
                model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

            for y in mu_B:
                model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")

            # Start the timer, solve the model, and calculate the time taken.
            model.optimize()

            # Check the model status and process the results.
            if model.status == GRB.OPTIMAL:
                total_cost = model.getObjective().getValue()
                return total_cost
            else:
                #TODO: add more info for this error
                print(f"No optimal solution found for nodes {node_A} and {node_B}")
                return None

        except Exception as e:
            print(f"Gurobi Error: {e}\n for nodes {node_A} and {node_B}")
            return None

    
    
    def earthmover_distance_hyperedge_combinations(self, hyperedge_id, distance_matrix):
        """
        This buddy gets the average EMD across the whole edge
        :param hyperedge_id: The identifier for the hyperedge.
        :return: The average EMD for all permutations of node pairs, or None if the hyperedge does not exist or has errors.
        """
        if hyperedge_id not in self.hyperedges:
            print(f"Hyperedge {hyperedge_id} does not exist.")
            return None
        
        nodes = self.hyperedges[hyperedge_id]

        if len(nodes) < 2:
            return 1
        
        sum_emd = 0
        pair_count = 0
        # Generate all combinations of pairs of nodes
        for node_A, node_B in combinations(nodes, 2):
            emd = self.earthmover_distance_gurobi_distance_matrix(node_A, node_B, distance_matrix)
            if emd is not None:
                sum_emd += emd
                pair_count += 1

        if pair_count > 0:
            # Compute the average EMD
            average_emd = sum_emd /pair_count
            weight = self.weights[hyperedge_id][-1]
            if weight == 0:
                return 1 - average_emd
            else:
                return 1 - average_emd/weight
        else:
            print(f"No valid EMD computations were possible. For hyperedge {hyperedge_id}")
            return None
    
  
class DirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, tail_set:set, head_set:set):
        '''Function to add a hyperedge to the hypergraph, if the nodes are not there, will add the nodes'''
        # Add missing nodes to the node set
        for node in tail_set.union(head_set):
            if node not in self.nodes:
                self.add_node(node)
        
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]=[1] # init the weight to 1
        

    def build_from_dataframe(self, df:pd.DataFrame, verbose=True):
        '''Build hypergraph from a DataFrame'''
        # make an edge from each row in the csv
        for _, row in df.iterrows():
            node1 = row[0].strip() #start
            node2 = row[1].strip() #end
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, set(node1), set(node2))
            if verbose:
                print(f'Added hyperedge {edgeid} with head set {node1} and tail set {node2}')
        return
    
            
    def get_underlying_edges(self) -> set:
        '''Function to get the edges from the hyperedges.
            We're basically going to make it look like an undirected graph
        '''
        edges = dict()
        
        for key in self.hyperedges.keys():
            tail, head = self.hyperedges[key]
            edges[key] = list(set(tail.union(head)))
        return edges
    
    def node_degree(self, node):
        """Calculate the degree of a node. Degree is the number of hyperedges containing this node.
        Will always return a numpy array (in-deg, out-deg)
        """
        if node not in self.nodes:
            raise ValueError("Node does not exist in the graph.")
        
        d_in_x = 0
        for _, (_, head_set) in self.hyperedges.items():
            if node in head_set:
                d_in_x += 1
        
        d_out_x = 0
        for _, (tail_set, _) in self.hyperedges.items():
            if node in tail_set:
                d_out_x += 1
        
        return [d_in_x, d_out_x]
    
    
    
def save_matrix_csv(matrix, filename:str) -> None:
    '''Function to save the matrix as a CSV file'''    
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)

def update_orc_and_weights_iter(distance_matrix:list[list], graph:Hypergraph, targ_graph:Hypergraph,  iteration:int, file_format='csv'):
    file_name = f'outputfiles/dataset_targeted_curvature_iteration_{iteration}.{file_format}'
    
    with open(file_name, 'a', newline='') as file:
        if file_format == 'csv':
            writer = csv.writer(file)
            # Check if the file is empty to write headers
            if file.tell() == 0:
                writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
            
            for hyperedge_id in graph.hyperedges:
                # TODO: Figure out the difference in Directed//Undirected
                #TODO: right now, implement Undirected
                if isinstance(graph, UndirectedHypergraph):
                    orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix)
                # add the value to our graph
                graph.add_ricci_curvature(hyperedge_id, orc)
                # update the weights
                weight = graph.weights[hyperedge_id][-1]
                alpha = .5
                beta = 0
                orc_targ = targ_graph.ricci_curvature[hyperedge_id][-1]
                
                if weight != 0:
                    # TODO: This is where the weights are being updated now -- where they will have to be fixed
                    wtplus1 = weight*((1 + (alpha * beta)/4) - (alpha/4)*(orc - orc_targ + beta))
                    normalized_weight = adjusted_sigmoid_0_to_1(wtplus1)
                else:
                    normalized_weight == 0

                graph.add_weights(hyperedge_id, normalized_weight)
                
                writer.writerow([hyperedge_id, orc, normalized_weight])
 
def calculate_target_orc(distance_matrix: list[list], graph:Hypergraph, file_format='csv'):
    file_name = f'outputfiles/dataset_target_graph_orc.{file_format}'
    
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # Check if the file is empty to write headers
        if file.tell() == 0:
            writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
        
        for hyperedge_id in graph.hyperedges:
            if isinstance(graph, UndirectedHypergraph):
                orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix)
            graph.add_ricci_curvature(hyperedge_id, orc)
            writer.writerow([hyperedge_id, orc])  
    return
               
def adjusted_sigmoid_0_to_1(x):
    # Clip x to a range that prevents overflow in exp.
    # The range of -709 to 709 is chosen based on the practical limits of np.exp()
    x_clipped = np.clip(x, -709, 709)
    a, b = 0, 1  # Define the target range
    return a + (b - a) / (1 + np.exp(-x_clipped))
    
  
if __name__ == "__main__": 
    directed_flag = False
    verbose = True
    
    #TODO: make a clean up function for the output files
    
    #For now, the nodes have to be labeled the same way. We're going to assume the hyperedges are going to be labeled the same.
    data1 = pd.read_csv('inputfiles/petersengraph.csv', header=None, sep=',')
    data2 = pd.read_csv('inputfiles/petersengraphExtraEdge.csv', header=None, sep=',')
        
    if directed_flag:
        source_graph = DirectedHypergraph()
        target_graph = DirectedHypergraph()
    else:
        source_graph = UndirectedHypergraph()
        target_graph = UndirectedHypergraph()          
              
    source_graph.build_from_dataframe(data1, verbose)
    target_graph.build_from_dataframe(data2, verbose)
    if not (source_graph.is_2_uniform() and target_graph.is_2_uniform()) :
        print('This has not been fully fleshed out for hypergraphs. Please give a 2-uniform graph')
        quit()
         
    if verbose:
        print('type of graph', type(source_graph))
        print("Number of edges:",len(source_graph.hyperedges)) #Printing the number of hyperedges or papers in our network.
        print("Number of nodes",len(source_graph.nodes)) #Printing the number of nodes or authors in the network.
        print('The actual nodes:', source_graph.nodes)
        
        connected = source_graph.is_weakly_connected()
        print("The hypergraph is weakly connected:" if connected else "The hypergraph is not weakly connected.")
        
        max_degree, min_degree, avg_degree = source_graph.calculate_degrees()
        print(f"Max Degree: {max_degree}")
        print(f"Min Degree: {min_degree}")
        print(f"Average Degree: {avg_degree}")

    distance_matrix = source_graph.floyd_warshall()
    save_matrix_csv(distance_matrix, 'outputfiles/undirected_testing_fw.csv')
    
    target_distance_matrix = target_graph.floyd_warshall()
    save_matrix_csv(target_distance_matrix, 'outputfiles/undirected_target_graph_fw.csv')

    # print('starting ricci curvature')

    #TODO: Same idea as in the undirected Hypergraph script
    
    #TODO: check to see if the guys are the same
    calculate_target_orc(target_distance_matrix, target_graph)
    update_orc_and_weights_iter(distance_matrix,source_graph, target_graph, iteration=0)
    # quit()
    
    #TODO: So i think we want to run this guy until we are stable (within some error bound?) We also want to be able to mess with the covergence params
    total_iterations = 5
    for i in range(1, total_iterations + 1):
        print('Working on itteration', i)
        distance_matrix_i = source_graph.floyd_warshall()
        filename = f'outputfiles/distance_matrix_normalized_weights_{i}.csv'
        save_matrix_csv(distance_matrix_i, filename)
        update_orc_and_weights_iter(distance_matrix, source_graph, target_graph, iteration=i)
            
    # print('hey')