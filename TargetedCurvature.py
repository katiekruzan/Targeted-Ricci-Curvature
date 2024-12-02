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
from numbers import Number

class Hypergraph:
    def __init__(self):       
        '''Initializing the hypergraph
        '''
        self.nodes = set() # arbitrary, not defined type as of now.
        self.hyperedges = {} #dict from hyperedge id to lists of nodes in that edge
        self.weights = {} # dict that had hyperedge ids to weights
        self.ricci_curvature = {} #dict with hyperedge id to list of ricci curvatures (floats)
        self.node_index = {}
            
    def add_node(self, node:any) -> None:
        '''Function to add a node to the hypergraph. The type is not set

        :param any node: The node to be added
        '''      
        self.nodes.add(node)
        return
    
    def add_ricci_curvature(self, hyperedge_id:str, orc:float)-> None:
        '''Function to add ollivier ricci curvature for all hyperedges for each iteration.
            It will be appending to a list

        :param str hyperedge_id: The id of the hyperedge of interest
        :param float orc: the curvature to be appended
        '''        
        if hyperedge_id not in self.ricci_curvature:
            self.ricci_curvature[hyperedge_id] = []  # Initialize with an empty list if key doesn't exist
        self.ricci_curvature[hyperedge_id].append(orc)
        return
        
    def add_weights(self, hyperedge_id:str, weight: Number) -> None:
        '''Function to add weights for all hyperedges for each iteration.
            It will be appending to a list

        :param str hyperedge_id: the id of the hyperedge of interest
        :param Number weights: a number for the weight
        ''' 
        if weight is not None:
            self.weights[hyperedge_id].append(weight)
    
    def update_node_index(self) -> None:
        '''The goal is to ensure there is a static node index for the graph. This function generates it
        '''
        self.node_index = {node: idx for idx, node in enumerate(list(self.nodes))}
            
    def is_2_uniform(self) -> bool:
        '''Check if size of each edge is 2

        :return bool: true if graph is 2 uniform
        '''
        # TODO: find a way to do this quicker
        for edges in self.hyperedges.items():
            if len(edges) != 2:
                return False
        return True
            
    def is_weakly_connected(self)-> bool:
        '''Check if the underlying graph is weakly connected
        # TODO: think if there is a quicker way other than DFS to do this

        :return bool: True if weakly connected
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
        '''Use the Floyd-Warshall algorithm to find the shortest distances between
           each pair of vertices. Right now, if you cannot get from one node to another,
           the distance will be 'inf'. This will be relavant in the case of directed, not strongly connected graphs.

        :return list[list]: a matrix with the shortest distances
        '''        
    
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
        # #TODO: See if this makes sense for directed
        for i in range(node_count):
            dist[i][i] = 0
        
        # Set the distance for directly connected nodes based on edge weights
        for hyperedge_id, nodes in self.hyperedges.items():
            # print(hyperedge_id)
            # print(nodes)
            if isinstance(self, UndirectedHypergraph):
                tail_set, head_set = nodes, nodes
            else: 
                tail_set, head_set = nodes
            
            #TODO: test this when it comes to actual hypergraphs
            '''
            # set distances within tail set and head set to 0
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
                    dist[node_index[tail]][node_index[head]] = min(dist[node_index[tail]][node_index[head]],self.weights[hyperedge_id][-1])  # Using the last weight in the list
        
        # Floyd-Warshall algorithm to update distances
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]] < dist[node_index[i]][node_index[j]]:
                        dist[node_index[i]][node_index[j]] = dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]]
        
        # Replace 'inf' with 0 for pairs of nodes that have no path between them
        #TODO: Check if this is a good idea. I think this is odd, so will drop for now. Like probably? but also nodes have dist 0 to themselves
        '''
        for i in range(node_count):
            for j in range(node_count):
                if dist[i][j] == float('inf'):
                    dist[i][j] = 0 
        '''

        return dist
        
        
    def calculate_degrees(self):
        '''
        Return the max degree, min degree, and average degree values. 
        For Directed, we get (in, out) pairs. Confirmed works on Directed

        :return: max degree, min degree, and average degree 
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
    def add_hyperedge(self, hyperedge_id:str, nodes:list, weight_list = [1], verbose=True)-> None:
        '''Add a hyperedge to the hypergraph. Automatically adds missing nodes.

        :param str hyperedge_id: the name you would like to be used for the hyperedge
        :param list nodes: a list of the adjacent nodes
        :param list weight_list: Should start as a list with a single element (expect for odd cases), defaults to [1]
        :param bool verbose: verbose flag, defaults to True
        :raises ValueError: If the nodes is not a list, will raise this error
        '''        
        
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
    
    def add_missing_target_edges(self, targ_graph:Hypergraph, targ_dist_mat: list[list], verbose:bool)->None:
        '''This will be used in the case where the target graph has edges this graph does not. We must add them.
        Right now, initializing them to the shortest distance in the current graph.

        :param Hypergraph targ_graph: The target graph
        :param list[list] targ_dist_mat: the distance matrix of that graph from floyd warshall
        :param bool verbose: verbose flag
        '''
        for e in set(targ_graph.hyperedges) - set(self.hyperedges):
            # TODO: Change when getting to hypergraphs
            node1 = targ_graph.hyperedges[e][0]
            node2 = targ_graph.hyperedges[e][1]
            dist = targ_dist_mat[targ_graph.node_index[node1]][targ_graph.node_index[node2]]
            self.add_hyperedge(e, targ_graph.hyperedges[e], [dist], verbose)
            
    def add_missing_source_edges(self, src_graph:Hypergraph, verbose:bool) -> None:
        '''This will be used in the case where the source graph has edges this graph does not. We must add them.
        Right now, initializing them to a weight of 0.
        #TODO: check to see if this makes sense

        :param Hypergraph src_graph: the source graph in question
        :param bool verbose: the verbose flag
        '''
        for e in set(src_graph.hyperedges) - set(self.hyperedges):
            self.add_hyperedge(e, src_graph.hyperedges[e], [0], verbose)
        
    
    def build_from_dataframe(self, df:pd.DataFrame, verbose=True):
        '''Build hypergraph from a DataFrame

        :param pd.DataFrame df: has the columns 'source', 'target', and 'weight'
        :param bool verbose: verbose flag, defaults to True
        '''        
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
            raise ValueError('The total mass of the distributions mu_A and mu_B are not equal.')
        

        # Create a mapping of nodes to their indices in the distance matrix.
        node_to_index = self.node_index

        try:
            # Create a new model in Gurobi.
            model = Model("EarthMoverDistance")

            # Set up the log file
            #log_filename = f"gurobi_log_{hyperedge_id}.log"
            # Set up the log file
            # '''
            # log_filename = f"gurobi_log_{hyperedge_id}.log"
            # model.setParam('LogFile', log_filename)
            # '''
            #model.setParam('OutputFlag', 1)
            # Create variables for the linear program.
            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)
            
            # Should make it less verbose
            # if not verbose:
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
                print(f"No optimal solution found for nodes {node_A} and {node_B}")
                print(f"The probability distributions are A: {mu_A} \nAnd B: {mu_B}")
                print('Model Status', model.status)
                print(f"The masses are {total_mass_A} for A and {total_mass_B} for B")
                return None

        except Exception as e:
            print(f"Gurobi Error: {e}\n for nodes {node_A} and {node_B}")
            return None

    
    
    def earthmover_distance_hyperedge_combinations(self, hyperedge_id:str, distance_matrix:list[list], verbose:bool):
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
            emd = self.earthmover_distance_gurobi_distance_matrix(node_A, node_B, distance_matrix, verbose)
            
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
                # this is the orc 
                # #TODO: check to see if this makes sense for weight =0 in a real way
                return 1 - average_emd
            else:
                # this is the orc. This is the EMD/dist(u,v) and dist(u,v) will just be the weight of the edge
                return 1 - average_emd/weight
        else:
            print(f"No valid EMD computations were possible. For hyperedge {hyperedge_id} with EMD {emd}")
            return None
    
  
class DirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, tail_set:set, head_set:set, weight_list = [1]):
        '''Function to add a hyperedge to the hypergraph, if the nodes are not there, will add the nodes'''
        # Add missing nodes to the node set
        for node in tail_set.union(head_set):
            if node not in self.nodes:
                self.add_node(node)
        
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]= weight_list
        
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
            weight = float(row['weight'])
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

    
    def calculate_probability_distributions(self, hyperedge_id):
        #TODO: double check for correctness
        '''Function to calculate the probability distributions over all nodes based on the hyperedge'''
        tail_set, head_set = self.hyperedges[hyperedge_id]

        # Initialize mu_A and mu_B only for nodes in the tail set and head set respectively
       
        mu_A_in = {node: 0 for node in self.nodes}
        for node in tail_set:
            d_x_in = self.node_degree(node)[0]
            if d_x_in != 0:
                mu_A_in[node] = 0
            else:
                mu_A_in[node] = 1 / len(tail_set)

       
        mu_B_out = {node: 0 for node in self.nodes}
        for node in head_set:
            d_x_out = self.node_degree(node)[1]
            if d_x_out != 0:
                mu_B_out[node] = 0
            else:
                mu_B_out[node] = 1 / len(head_set)
        
        # Third Case
        for edge in self.hyperedges:
            if edge != hyperedge_id:
                tail_set_prime, head_set_prime = self.hyperedges[edge]
                common_tail_nodes = set(tail_set) & set(head_set_prime)
                if common_tail_nodes:
                    for node in common_tail_nodes:
                        deg_x_in = self.node_degree(node)[0]
                        for nodes in tail_set_prime:
                            if deg_x_in != 0:  
                                mu_A_in[nodes] += 1 / (len(tail_set) * len(tail_set_prime) * deg_x_in)

                common_head_nodes = set(head_set) & set(tail_set_prime)
                if common_head_nodes:
                    for node in common_head_nodes:
                        deg_x_out = self.node_degree(node)[1]
                        for nodes in head_set_prime:
                            if deg_x_out != 0:    
                                mu_B_out[nodes] += 1 / (len(head_set) * len(head_set_prime) * deg_x_out)
              
        total_mass_A = sum(mu_A_in.values())
        total_mass_B = sum(mu_B_out.values())
        
        # Normalize the probability distributions
        if total_mass_A == 0:
            mu_A_in ={node: mass for node, mass in mu_A_in.items()}
        else:
            mu_A_in = {node: mass / total_mass_A for node, mass in mu_A_in.items()}
        
        if total_mass_B == 0:
            mu_B_out = {node: mass for node, mass in mu_B_out.items()}
        else:
            mu_B_out = {node: mass / total_mass_B for node, mass in mu_B_out.items()}
        
        return mu_A_in, mu_B_out
    
    
    def earthmover_distance_gurobi_distance_matrix(self, hyperedge_id, distance_matrix, verbose):
        '''Function to calculate EMD using the distance matrix (Optimized)'''
        #TODO: think of how to combine these for undirected//directed. Most of this is the exact same.
        # seems directed does the combinations in the calculate probability distributions section
        # Get the probability distributions for the specified hyperedge.
        mu_A, mu_B = self.calculate_probability_distributions(hyperedge_id)

        # Convert distributions from dictionary to list format and print for debugging
        nodes_A = sorted(mu_A.keys())
        nodes_B = sorted(mu_B.keys())
        distribution1 = [mu_A[node] for node in nodes_A]
        distribution2 = [mu_B[node] for node in nodes_B]
    
        # Print the distributions to verify correctness
        if verbose:
            print("Nodes in mu_A:", nodes_A)
            print("Nodes in mu_B:", nodes_B)
            print("Distribution mu_A:", distribution1)
            print("Distribution mu_B:", distribution2)

        # Check if distributions sum to the same value
        total_mass_A = sum(distribution1)
        total_mass_B = sum(distribution2)
        if verbose:
            print("Total mass in mu_A:", total_mass_A)
            print("Total mass in mu_B:", total_mass_B)
    
        if abs(total_mass_A - total_mass_B) > 1e-6:
            raise ValueError('The total mass of the distributions mu_A and mu_B are not equal.')
        

        # Create a mapping of nodes to their indices in the distance matrix.
        node_to_index = self.node_index
        
        try:
            model = Model("EarthMoverDistance")

            # Set up the log file
            # log_filename = f"gurobi_log_{hyperedge_id}.log"
            # model.setParam('LogFile', log_filename)

            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)
            
            # Should make it less verbose
            if not verbose:
                model.Params.LogToConsole = 0

            # Update the objective function to use the distance matrix.
            model.setObjective(quicksum(distance_matrix[node_to_index[x]][node_to_index[y]] * variables[x, y]
                                for x in mu_A for y in mu_B), GRB.MINIMIZE)

            # Add constraints
            for x in mu_A:
                model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

            for y in mu_B:
                model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")

            # start_time = time.time()
            model.optimize()
            # end_time = time.time()

            # time_taken = end_time - start_time

            if model.status == GRB.OPTIMAL:
                total_cost = model.getObjective().getValue()
                print("Total EMD Cost:", total_cost)
                # print("Time taken to find the optimal solution: {:.4f} seconds".format(time_taken))

                for x in mu_A:
                    for y in mu_B:
                        amount_moved = variables[x, y].X
                        if amount_moved > 0:
                            print(f"Move {amount_moved} from {x} to {y}")
                return total_cost
            else:
                print(f"No optimal solution found for nodes {nodes_A} and {nodes_B}")
                print(f"The probability distributions are A: {mu_A} \nAnd B: {mu_B}")
                print('Model Status', model.status)
                print(f"The masses are {total_mass_A} for A and {total_mass_B} for B")
                return None
            
        except Exception as e:
            print(f"Gurobi Error: {e}\n for hyperedge {hyperedge_id}")
            return None    
    
    
def save_matrix_csv(matrix:list[list], filename:str) -> None:
    '''Function to save the matrix as a CSV file

    :param list[list] matrix: matrix to be written
    :param str filename: place to write it
    '''    
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)


def update_orc_and_weights_iter(distance_matrix:list[list], graph:Hypergraph, targ_graph:Hypergraph,  iteration:int, verbose:bool, file_format='csv', op_flag = False):
    '''The main function of this whole sheboodle. Run the whole process for the given itteration

    :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
    :param Hypergraph graph: the source graph we're looking at (or at least its current itteration)
    :param Hypergraph targ_graph: the target graph we're headed to
    :param int iteration: the round we're on
    :param bool verbose: verbose flag
    :param str file_format: defaults to 'csv'
    :param bool op_flag: option to mark the file as 'op' (used in second half of
                         script), defaults to False
    '''
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
                # Normalize the curvature
                normalized_orc = ricci_normalizing(orc)
                # un-normalizing
                # normalized_orc = orc
                # add the value to our graph
                graph.add_ricci_curvature(hyperedge_id, normalized_orc)
                # grab the latest weight the weights
                weight = graph.weights[hyperedge_id][-1]
                if iteration != 0:
                    orc_targ = targ_graph.ricci_curvature[hyperedge_id][-1]
                    
                    if weight != 0:
                        #simple version
                        wtplus1 = weight*(1  - (normalized_orc - orc_targ))
                        normalized_weight = wtplus1
                    else:
                        normalized_weight = 0

                    graph.add_weights(hyperedge_id, normalized_weight)
                
                    writer.writerow([hyperedge_id, normalized_orc, normalized_weight])
                else: 
                    writer.writerow([hyperedge_id, normalized_orc, weight])
 
 
def calculate_target_orc(distance_matrix: list[list], graph:Hypergraph, verbose:bool, file_format='csv', op_flag=False):
    '''The function to calculate the staring infor for the target graph

    :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
    :param Hypergraph graph: the actual source graph
    :param bool verbose: verbose flag
    :param str file_format: defaults to 'csv'
    :param bool op_flag: option to mark the file as 'op' (used in second half of
                         script), defaults to False
    '''
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
                orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix, verbose)
            else: # We're a directed graph
                orc = 1.0
            # normalizing
            normalized_orc = ricci_normalizing(orc)
            # un-normalizing
            # normalized_orc = orc
            graph.add_ricci_curvature(hyperedge_id, normalized_orc)
            weight = graph.weights[hyperedge_id][-1]
            writer.writerow([hyperedge_id, normalized_orc, weight])  
    return


def ricci_normalizing(R: float)->float: 
    '''
    Using the normalization function sigma(R)/sigma(1) 
    Where sigma(x) is the standard sigmoid function 1/(1+\exp(-x))

    :param float R: the ORC value to be normalized 
    :return float: The normalized ORC value
    ''' 
    return ((1 - np.exp(-1))/(1+ np.exp(-R)))


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
    
  
if __name__ == "__main__": 
    # TODO: implement Ricci for Directed
    # Check if the ratio of the weights is more or less tha same 
    # average absolute difference and see if that's small
    # try a network such that the sum of the two weights are the same
    '''
    '''
    directed_flag = False
    verbose = False
    
    clean_output(verbose)
    
    #For now, the nodes have to be labeled the same way. We're going to assume the hyperedges are going to be labeled the same.
    # This section works
    
    '''
    The data needs to come in as a csv with three columns labeled 'source', 'target', and 'weight'
    This will be read as a pandas dataframe. 
    And the nodes must be labeled the same in both graphs for this to work
    
    # TODO: Change the weight of 5 edges, the 100 edges, and 1000 edges for 10 different pairs. For 100 node graphs
    # TODO: Also check out the directed graphs
    '''
    
    target_filename = 'ERgraph50nodesweight1.csv'
    source_filename = 'ERgraph50n5changev4.csv'
    
    data_target = pd.read_csv(f'inputfiles/{target_filename}', dtype ={'source': str, 'target':str}, sep=',')  
    data_source = pd.read_csv(f'inputfiles/{source_filename}', dtype ={'source': str, 'target':str}, sep=',')  

    # Scorecard Writing
    write_scorecard('----- Targeted Ricci Curvature -----')
    write_scorecard(f'target filename: {target_filename}')
    write_scorecard(f'source filename: {source_filename}')
    
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
         
    connected = source_graph.is_weakly_connected()
    max_degree, min_degree, avg_degree = source_graph.calculate_degrees()
    
    if verbose:
        print('type of graph', type(source_graph))
        print("Number of edges:",len(source_graph.hyperedges)) #Printing the number of (hyper)edges in our network.
        print("Number of nodes",len(source_graph.nodes)) #Printing the number of nodes in the network.
        print('The actual nodes:', source_graph.nodes)
        print('The actual edges with weights:', source_graph.weights)
        
        print("The hypergraph is weakly connected." if connected else "The hypergraph is not weakly connected.")
        
        print(f"Max Degree: {max_degree}")
        print(f"Min Degree: {min_degree}")
        print(f"Average Degree: {avg_degree}")
    
    write_scorecard('----- Graph Statistics -----')
    write_scorecard(f'Type of Graph: {type(source_graph)}')
    write_scorecard(f'Number of edges: {len(source_graph.hyperedges)}')
    write_scorecard(f'Number of nodes: {len(source_graph.nodes)}')
    if connected: write_scorecard('The hypergraph is weakly connected.')
    else: write_scorecard('The hypergraph is not weakly connected.')
    write_scorecard(f"Max Degree: {max_degree}")
    write_scorecard(f"Min Degree: {min_degree}")
    write_scorecard(f"Average Degree: {avg_degree}")
    
    #TODO: rewrite the following as a little function we can send things to
    
    print('working on distance matrices')
    distance_matrix = source_graph.floyd_warshall()
    save_matrix_csv(distance_matrix, 'outputfiles/undirected_source_dist_fw.csv')
   
    target_distance_matrix = target_graph.floyd_warshall()
    save_matrix_csv(target_distance_matrix, 'outputfiles/undirected_target_dist_fw.csv')
    print(source_graph.node_index)
    print(target_graph.node_index)
    
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
        if i == 1: # take care of the getting started case
            continue
        for e in source_graph.hyperedges:
            clist = source_graph.ricci_curvature[e]
            old = clist[-2]
            new = clist[-1]
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
            write_scorecard('----- Results -----')
            write_scorecard(f'Source to target distance is {i}')
            break
        
    if not allstable:
        # print(target_graph.weights)
        write_scorecard('Source to target did not stablize.')
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
        if i == 1: # take care of the getting started case
            continue
        for e in source_graph.hyperedges:
            clist = source_graph.ricci_curvature[e]
            old = clist[-2]
            new = clist[-1]
            error = abs((old-new)/old) #error as a percentage of old
            # error = abs(old-new)
            if error > 0.01:
                if verbose:
                    print('unstable for edge ', e, ' with error ', error)
                finustab = e
                allstable = False
                break
        if allstable:
            print('STABILIZED! Target to source distance is ',i)
            write_scorecard(f'Target to source distance is {i}')
            break
            
    if not allstable:
        # print(target_graph.weights)
        write_scorecard('Target to source did not stablize.')
        print(source_graph.weights[finustab])
    