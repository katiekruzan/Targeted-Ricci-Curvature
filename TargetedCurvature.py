'''
Idea: do a hypergraph and then subclasses
'''

import pandas as pd
import csv
import numpy as np
from itertools import combinations
from gurobipy import Model, GRB, quicksum
import time
import os

class Hypergraph:
    def __init__(self):
            '''Initializing the hypergraph'''
            self.nodes = set() # arbitrary, not defined type as of now.
            self.hyperedges = {} #dict from hyperedge id to lists of nodes in that edge
            self.weights = {} # dict that had hyperedge ids to weights
            self.ricci_curvature = {} #dict with hyperedge id to list of ricci curvatures
            self.node_index = {}
            
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
    
    def update_node_index(self) -> None:
        self.node_index = {node: idx for idx, node in enumerate(list(self.nodes))}
            
    def normalize_weights(self) -> None:
        ''' Normalize the weights of the edges as suggested in the proposal (divide by the total weight)'''
        all_weights = [self.weights[e][-1] for e in self.weights.keys() if self.weights[e][-1] != np.inf]
        tot_weight = sum(all_weights)
        # print(tot_weight)
        for edge_id in self.hyperedges.keys():
            recent_w = self.weights[edge_id][-1]
            new_w = recent_w/tot_weight
            self.add_weights(edge_id, new_w)
            
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
        return visited == self.nodes   
  
    
    def floyd_warshall(self) -> list[list]:
        # Initialize the distance matrix with "infinite" distances
        # Assume self.nodes is a list or set of nodes
        node_list = list(self.nodes) # Convert to list to ensure consistent ordering
        node_count = len(node_list)
        
        # Create a mapping of node to index
        self.update_node_index()
        node_index = self.node_index

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
    def add_hyperedge(self, hyperedge_id:str, nodes:list, weight_list = [1], verbose=True):
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
        self.weights[hyperedge_id] = weight_list #init the weights to [1]
        return
    
    def add_missing_target_edges(self, targ_graph:Hypergraph, targ_dist_mat, verbose):
        for e in set(targ_graph.hyperedges) - set(self.hyperedges):
            # TODO: Change when getting to hypergraphs
            node1 = targ_graph.hyperedges[e][0]
            node2 = targ_graph.hyperedges[e][1]
            dist = targ_dist_mat[targ_graph.node_index[node1]][targ_graph.node_index[node2]]
            self.add_hyperedge(e, targ_graph.hyperedges[e], [dist], verbose)
            
    def add_missing_source_edges(self, src_graph:Hypergraph, verbose):
        for e in set(src_graph.hyperedges) - set(self.hyperedges):
            self.add_hyperedge(e, src_graph.hyperedges[e], [0], verbose)
        
    
    def build_from_dataframe(self, df:pd.DataFrame, verbose=True):
        '''Build hypergraph from a DataFrame'''
        # make an edge from each row in the csv
        #TODO: make this more general? Be able to catch errors
        for _, row in df.iterrows():
            node1 = row['source'].strip() #start
            node2 = row['target'].strip() #end
            weight = float(row['weight'])
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, [node1, node2], [weight], verbose)
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
        #TODO: Note this is wrong for directed graphs
        alpha = 0.1  # Self-transition probability factor
        probability_distribution = {n: 0.0 for n in self.nodes}  # Initialize probabilities

        if node not in self.nodes:
            raise ValueError("Node does not exist in the hypergraph.")

        # Calculate the denominator: sum of (|f| - 1) for all f containing node
        denominator = 0
        hyperedges_containing_node = self.find_hyperedges_containing_nodes(node) # the part that's no good for directed
        for hyperedge_id in hyperedges_containing_node:
            hyperedge = self.hyperedges[hyperedge_id]
            denominator += (len(hyperedge) - 1)
        
        '''
        # Calculate the denominator: sum of weight(f) for all f containing node
        denominator = 0
        hyperedges_containing_node = self.find_hyperedges_containing_nodes(node)
        for hyperedge_id in hyperedges_containing_node:
            hyperedge_weight = self.weights[hyperedge_id]
            denominator += hyperedge_weight
        '''
            
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
    
    
    def earthmover_distance_gurobi_distance_matrix(self, node_A, node_B, distance_matrix, verbose):
        if node_A not in self.nodes or node_B not in self.nodes:
            print(f"Node {node_A} or {node_B} does not exist in the hypergraph.")
            return None  # Return None if either node does not exist
        
        # Get the probability distributions for the two specified nodes.
        mu_A = self.node_probability(node_A)
        mu_B = self.node_probability(node_B)
        if verbose:
            print('The node', node_A, 'has distribution', mu_A)
            print('The node', node_B, 'has distribution', mu_B)


        # Convert distributions from dictionary to list format 
        nodes_A = sorted(mu_A.keys())
        nodes_B = sorted(mu_B.keys())
        distribution1 = [mu_A[node] for node in nodes_A]
        distribution2 = [mu_B[node] for node in nodes_B]

        # Check if distributions sum to the same value
        total_mass_A = sum(distribution1)
        total_mass_B = sum(distribution2)
        if verbose:
            print(f'mass for {node_A} is {total_mass_A} and mass for {node_B} is {total_mass_B}')
    
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
            
            # Should make it less verbose
            if not verbose:
                model.Params.LogToConsole = 0

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
                print(f"The probability distributions are A: {mu_A} \nAnd B: {mu_B}")
                print('Model Status', model.status)
                print(f"The masses are {total_mass_A} for A and {total_mass_B} for B")
                return None

        except Exception as e:
            print(f"Gurobi Error: {e}\n for nodes {node_A} and {node_B}")
            return None

    
    
    def earthmover_distance_hyperedge_combinations(self, hyperedge_id, distance_matrix, verbose):
        """
        This buddy gets the average EMD across the whole edge
        :param hyperedge_id: The identifier for the hyperedge.
        :return: The average EMD for all permutations of node pairs, or None if the hyperedge does not exist or has errors.
        """
        if hyperedge_id not in self.hyperedges:
            print(f"Hyperedge {hyperedge_id} does not exist.")
            return None
        
        nodes = self.hyperedges[hyperedge_id]
        # print(nodes)

        if len(nodes) < 2:
            return 1
        
        sum_emd = 0
        pair_count = 0
        # Generate all combinations of pairs of nodes
        for node_A, node_B in combinations(nodes, 2):
            emd = self.earthmover_distance_gurobi_distance_matrix(node_A, node_B, distance_matrix, verbose)
            print(f'emd is {emd} on nodes {node_A} and {node_B}')
            
            if emd is not None:
                sum_emd += emd
                pair_count += 1
            else:
                print('emd is none on nodes ', node_A, 'and', node_B)

        if pair_count > 0:
            # Compute the average EMD
            average_emd = sum_emd /pair_count
            weight = self.weights[hyperedge_id][-1]
            if weight == 0:
                # this is the orc #TODO: check to see if this makes sense for weight =0 in a real way
                return 1 - average_emd
            else:
                # this is the orc. This is the EMD/dist(u,v) and dist(u,v) will just be the weight of the edge
                return 1 - average_emd/weight
        else:
            print(f"No valid EMD computations were possible. For hyperedge {hyperedge_id}")
            print(emd)
            return None
    
  
class DirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, tail_set:set, head_set:set, weight_list = [1]):
        '''Function to add a hyperedge to the hypergraph, if the nodes are not there, will add the nodes'''
        # Add missing nodes to the node set
        for node in tail_set.union(head_set):
            if node not in self.nodes:
                self.add_node(node)
        
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]=[weight_list] # init the weight to 1
        
    def add_missing_target_edges(self, targ_graph:Hypergraph):
        #TODO: check if this works for Digraphs
        for e in set(targ_graph.hyperedges) - set(self.hyperedges):
            self.add_hyperedge(e, targ_graph.hyperedges[e][0], targ_graph.hyperedges[e][1], targ_graph.weights[e])

    def build_from_dataframe(self, df:pd.DataFrame, verbose=True):
        '''Build hypergraph from a DataFrame'''
        # make an edge from each row in the csv
        for _, row in df.iterrows():
            node1 = row['source'].strip() #start
            node2 = row['target'].strip() #end
            weight = float(row['weight'].strip())
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, set(node1), set(node2), [weight])
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


def update_orc_and_weights_iter(distance_matrix:list[list], graph:Hypergraph, targ_graph:Hypergraph,  iteration:int, verbose, file_format='csv', op_flag = False):
    if op_flag:
        file_name = f'outputfiles/op_dataset_targeted_curvature_iteration_{iteration}.{file_format}'
    else:
        file_name = f'outputfiles/dataset_targeted_curvature_iteration_{iteration}.{file_format}'
    
    with open(file_name, 'a', newline='') as file:
        if file_format == 'csv':
            writer = csv.writer(file)
            # Check if the file is empty to write headers
            if file.tell() == 0:
                writer.writerow(['Hyperedge ID', 'ORC: (based on t-1 weights)', 'Weight:t'])
            
            for hyperedge_id in graph.hyperedges:
                # TODO: Figure out the difference in Directed//Undirected
                if isinstance(graph, UndirectedHypergraph):
                    orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix, verbose=False)
                # quit()
                if iteration ==2:
                    print(hyperedge_id, orc)
                    # quit()
                # Normalize the curvature
                # normalized_orc = ricci_normalizing(orc)
                # un-normalizing
                normalized_orc = orc
                # add the value to our graph
                graph.add_ricci_curvature(hyperedge_id, normalized_orc)
                # grab the latest weight the weights
                weight = graph.weights[hyperedge_id][-1]
                if iteration != 0:
                    alpha = 1
                    beta = 0
                    orc_targ = targ_graph.ricci_curvature[hyperedge_id][-1]
                    
                    if weight != 0:
                        wtplus1 = weight*((1 + (alpha * beta)/4) - (alpha/4)*(normalized_orc - orc_targ + beta))
                        # normalized_weight = adjusted_sigmoid_0_to_1(wtplus1)
                        normalized_weight = wtplus1
                    else:
                        normalized_weight = 0

                    graph.add_weights(hyperedge_id, normalized_weight)
                
                    writer.writerow([hyperedge_id, normalized_orc, normalized_weight])
                else: 
                    writer.writerow([hyperedge_id, normalized_orc, weight])
 
 
def calculate_target_orc(distance_matrix: list[list], graph:Hypergraph, verbose, file_format='csv', op_flag=False):
    if op_flag:
        file_name = f'outputfiles/op_dataset_target_graph_orc.{file_format}'
    else:
        file_name = f'outputfiles/dataset_target_graph_orc.{file_format}'
    
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # Check if the file is empty to write headers
        if file.tell() == 0:
            writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
        
        for hyperedge_id in graph.hyperedges:
            if isinstance(graph, UndirectedHypergraph):
                orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix, verbose = False)
            # normalizing
            # normalized_orc = ricci_normalizing(orc)
            # un-normalizing
            normalized_orc = orc
            # print('hyperedge:', hyperedge_id, 'orc: ', orc, 'normalized orc: ', normalized_orc)
            graph.add_ricci_curvature(hyperedge_id, normalized_orc)
            weight = graph.weights[hyperedge_id][-1]
            writer.writerow([hyperedge_id, normalized_orc, weight])  
    return
             
               
def adjusted_sigmoid_0_to_1(x):
    # Clip x to a range that prevents overflow in exp.
    # The range of -709 to 709 is chosen based on the practical limits of np.exp()
    x_clipped = np.clip(x, -709, 709)
    a, b = -1, 1  # Define the target range
    return a + (b - a) / (1 + np.exp(-x_clipped))

def ricci_normalizing(x):
    return ((1 - np.exp(-x))/(1+ np.exp(-x)))


def clean_output(verbose):
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
    
    
  
if __name__ == "__main__": 
    # TODO: implement Ricci for Directed
    # Check if the ratio of the weights is more or less tha same 
    # average absolute difference and see if that's small
    # try a network such that the sum of the two weights are the same
    '''
    
    1. don't normalize, then look through and see if its scale invariant, see if there's a multiplier
    
    2. Then normalize and then check for this c.
    
    If the ratios are basically the same, then it works
    
    but if the ratios at the end are different than the starting weights find a multiplier 
    for what value of c is the distance between the weights might be the smallest
    linear search
     
    update by Friday
    '''
    directed_flag = False
    verbose = True
    
    clean_output(verbose)
    
    #For now, the nodes have to be labeled the same way. We're going to assume the hyperedges are going to be labeled the same.
    # This section works
    
    '''
    The data needs to come in as a csv with three columns labeled 'source', 'target', and 'weight'
    This will be read as a pandas dataframe. 
    And the nodes must be labeled the same in both graphs for this to work
    '''
    
    # data_target = pd.read_csv('inputfiles/ERgraph50nodesweight1.csv', dtype ={'source': str, 'target':str}, sep=',')  
    # data_source = pd.read_csv('inputfiles/ERgraph50nodesincr.csv', dtype ={'source': str, 'target':str}, sep=',')  
    # data_target = pd.read_csv('inputfiles/petersengraph.csv', dtype ={'source': str, 'target':str}, sep=',')  
    # data_source = pd.read_csv('inputfiles/petersengraph_bigedges.csv', dtype ={'source': str, 'target':str}, sep=',')  
    # data_target = pd.read_csv('inputfiles/petersengraph.csv', dtype ={'source': str, 'target':str}, sep=',')  
    # data_source = pd.read_csv('inputfiles/petersengraph_newweights.csv', dtype ={'source': str, 'target':str}, sep=',')  
    data_target = pd.read_csv('inputfiles/petersengraph.csv', dtype ={'source': str, 'target':str}, sep=',')  
    data_source = pd.read_csv('inputfiles/petersengraph_newbigweights.csv', dtype ={'source': str, 'target':str}, sep=',')  

    
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
    
    if not (source_graph.is_2_uniform() and target_graph.is_2_uniform()) :
        print('This has not been fully fleshed out for hypergraphs. Please give a 2-uniform graph')
        quit()
         
    if verbose:
        print('type of graph', type(source_graph))
        print("Number of edges:",len(source_graph.hyperedges)) #Printing the number of (hyper)edges in our network.
        print("Number of nodes",len(source_graph.nodes)) #Printing the number of nodes in the network.
        print('The actual nodes:', source_graph.nodes)
        print('The actual edges with weights:', source_graph.weights)
        
        connected = source_graph.is_weakly_connected()
        print("The hypergraph is weakly connected:" if connected else "The hypergraph is not weakly connected.")
        
        max_degree, min_degree, avg_degree = source_graph.calculate_degrees()
        print(f"Max Degree: {max_degree}")
        print(f"Min Degree: {min_degree}")
        print(f"Average Degree: {avg_degree}")

    # Normalize the edge weights. Doing so as suggested in the proposal (basically divide the weights by total weight (excluding inf))
    # print('Normalizing the weights')
    # source_graph.normalize_weights()
    # target_graph.normalize_weights()
    # if verbose:
    #     print(source_graph.weights)
    #     print(target_graph.weights)
    
    print('working on distance matrices')
    distance_matrix = source_graph.floyd_warshall()
    save_matrix_csv(distance_matrix, 'outputfiles/undirected_source_dist_fw.csv')
    
    target_distance_matrix = target_graph.floyd_warshall()
    save_matrix_csv(target_distance_matrix, 'outputfiles/undirected_target_dist_fw.csv')
    
    # TODO: Check this works (not testing it right now)
    if set(target_graph.hyperedges) != set(source_graph.hyperedges):
        print ('Taking care of missing edges')
        # add edges that are in the target but not the source
        source_graph.add_missing_target_edges(target_graph, target_distance_matrix, verbose)
        
    print('starting ricci curvature')
    #TODO: check to see if the guys are Known Node Correspondence. 
    # Maybe just check that all the nodes are labeled the same.
    
    calculate_target_orc(target_distance_matrix, target_graph, verbose)
    update_orc_and_weights_iter(distance_matrix, source_graph, target_graph, iteration=0, verbose=verbose)
    # quit()
    
    total_iterations = 100
    for i in range(1, total_iterations + 1):
        print('Working on itteration', i)
        distance_matrix_i = source_graph.floyd_warshall()
        save_matrix_csv(distance_matrix_i, f'outputfiles/distance_matrix_source_itteration_{i}.csv')
        update_orc_and_weights_iter(distance_matrix_i, source_graph, target_graph, iteration=i, verbose=verbose)
        allstable = True
        finustab = None
        for e in source_graph.hyperedges:
            wlist = source_graph.weights[e]
            old = wlist[-2]
            new = wlist[-1]
            # print(e, old, new)
            if old != 0:
                error = abs((old-new)/old)
            else: error = abs(old-new)
            if error > 0.01:
                if verbose:
                    print('unstable for edge ', e, ' with error ', error)
                finustab = e
                allstable = False
                break
        if allstable:
            print('STABILIZED! Source to target distance is ',i)
            break
            
    if not allstable:
        # print(target_graph.weights)
        print(source_graph.weights[finustab])
    
    # quit()
    # Go the other way
    print('Now checking Target to Source....')
    if directed_flag:
        source_graph = DirectedHypergraph()
        target_graph = DirectedHypergraph()
    else:
        source_graph = UndirectedHypergraph()
        target_graph = UndirectedHypergraph()          
    
    # swap them
    print('building source')
    source_graph.build_from_dataframe(data_target, verbose)
    print('building target')
    target_graph.build_from_dataframe(data_source, verbose)
        
    # source_graph.normalize_weights()
    # target_graph.normalize_weights()
        
    distance_matrix = source_graph.floyd_warshall()    
    target_distance_matrix = target_graph.floyd_warshall()
    
    if set(target_graph.hyperedges) != set(source_graph.hyperedges):
        print ('Taking care of missing edges')
        # add edges that are in the target but not the source
        source_graph.add_missing_target_edges(target_graph, target_distance_matrix, verbose)
        target_graph.add_missing_source_edges(source_graph, verbose)
    
    # print(source_graph.hyperedges)
    calculate_target_orc(target_distance_matrix, target_graph, verbose, op_flag=True)
    update_orc_and_weights_iter(distance_matrix, source_graph, target_graph, iteration=0, verbose=verbose, op_flag=True)
    # verbose = True
    for i in range(1, total_iterations + 1):
        print('Working on itteration', i)
        distance_matrix_i = source_graph.floyd_warshall()
        update_orc_and_weights_iter(distance_matrix_i, source_graph, target_graph, iteration=i, verbose=False, op_flag=True)
        allstable = True
        finustab = None
        for e in source_graph.hyperedges:
            wlist = source_graph.weights[e]
            old = wlist[-2]
            new = wlist[-1]
            error = abs((old-new)/old) #error as a percentage of old
            # error = abs(old-new)
            if error > 0.01:
                if verbose:
                    print('unstable for edge ', e, ' with error ', error)
                finustab = e
                allstable = False
                break
        if allstable:
            print('STABILIZED! Source to target distance is ',i)
            break
            
    if not allstable:
        # print(target_graph.weights)
        print(source_graph.weights[finustab])
    