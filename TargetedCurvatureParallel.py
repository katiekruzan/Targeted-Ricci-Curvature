'''Written by Katie Kruzan in March of 2025. this can be run with the following commandline
mpiexec -n 4 python .\katie_parallel.py

    :raises ValueError: _description_
    :raises this: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return _type_: _description_
    
#TODO: make the two scripts into one script? Or structure it better
'''

import pandas as pd
import csv
import numpy as np
from itertools import combinations
from gurobipy import Model, GRB, quicksum, LinExpr, Env
import time
import os
from numbers import Number
from mpi4py import MPI
from pprint import pprint
import ot

COMM = MPI.COMM_WORLD

now = time.time()

RND1 = True

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
       
    def remove_hyperedge(self, hyperedge_id:str) -> None:
        ''' Remove a hyperedge from the graph. Notably, will not remove the nodes

        :param str hyperedge_id: The hyperedge to be deleted
        '''
        del self.hyperedges[hyperedge_id]
        return
    
    def update_node_index(self) -> None:
        '''The goal is to ensure there is a static node index for the graph. This function generates it
        '''
        if len(self.node_index) == len(self.nodes):
            return
        else:
            self.node_index = {node: idx for idx, node in enumerate(list(self.nodes))}
                        
    def is_2_uniform(self) -> bool:
        '''Check if size of each edge is 2

        :return bool: true if graph is 2 uniform
        '''
        for edges in self.hyperedges.items():
            if len(edges) != 2:
                return False
        return True
               
    def is_weakly_connected(self)-> bool:
        '''Check if the underlying graph is weakly connected

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
    
    def is_strongly_connected(self)-> bool:
        '''Check if the underlying graph is strongly connected

        :return bool: True if weakly connected
        '''        
        # I think this is saying an empty graph is weakly connected
        if not self.nodes: 
            return True

        edges = self.hyperedges
        
        visited = set()

        def dfs(node):
            '''Depth First Search'''
            if node in visited:
                # print('got here')
                return visited
            visited.add(node)
            for edge in edges.values():
                # print(edge)
                if isinstance(self, UndirectedHypergraph):
                    if node in edge:
                        for next_node in edge:
                            # print(next_node)
                            if next_node != node:
                                dfs(next_node)
                elif isinstance(self, DirectedHypergraph):
                    # note tail and head
                    tail, head = edge
                    if node in tail:
                        for next_node in head:
                            # print(next_node)
                            if next_node != node:
                                dfs(next_node)

        # Do DFS from each node
        for v in iter(self.nodes):
            visited = set()
            # print('hellloooo')
            dfs(v)
            # write_scorecard(str(visited))
            if visited != self.nodes:
                # write_scorecard(f'not strongly connected based on vertex {v}')
                # write_scorecard(str(visited))
                return False
            else: 
                visited = set()
        return True 
    
    def floyd_warshall(self) -> list[list]:
        '''Use the Floyd-Warshall algorithm to find the shortest distances between
           each pair of vertices. Right now, if you cannot get from one node to 
           another, the distance will be the maximum value + 1. This will be relevant in the 
           case of directed, not strongly connected graphs.

        :return list[list]: a matrix with the shortest distances
        '''        
    
        # Initialize the distance matrix with "infinite" distances
        # Assume self.nodes is a list or set of nodes
        node_list = list(self.nodes) # Convert to list to ensure consistent ordering
        node_count = len(node_list)
        # print(node_count)
        
        # Create a mapping of node to index
        self.update_node_index()
        node_index = self.node_index
        # print(node_index)

        # Initialize a 2D list (matrix) with "infinite" distances
        dist = [[float('inf') for _ in range(node_count)] for _ in range(node_count)]
        
        # Set the diagonal to 0 (distance from each node to itself) 
        for i in range(node_count):
            dist[i][i] = 0
            
        # pprint(dist)
        
        # Set the distance for directly connected nodes based on edge weights
        for hyperedge_id, nodes in self.hyperedges.items():
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
                    dist[node_index[tail]][node_index[head]] = min(dist[node_index[tail]][node_index[head]],
                                                                   self.weights[hyperedge_id][-1])  # Using the last weight in the list
        # print('adding weights')
        # pprint(dist)
        # Floyd-Warshall algorithm to update distances
        for k in self.nodes:
            # print('check in at node', k)
            # pprint(dist)
            for i in self.nodes:
                for j in self.nodes:
                    if dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]] < dist[node_index[i]][node_index[j]]:
                        dist[node_index[i]][node_index[j]] = dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]]
        
        # replacing inf with max + 1
        tmp = np.array(dist)
        tmp[tmp == float('inf')] = -1
        tmp[tmp==-1] = np.max(tmp) + 1
        dist = tmp.tolist()
        # pprint(dist)
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
    
    
    def earthmover_distance_distance_matrix(self, data, distance_matrix, verbose):
        '''Trying to combine the two functions into one

        :param _type_ data: _description_
        :param _type_ distance_matrix: _description_
        :param _type_ verbose: _description_
        :raises this: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return _type_: _description_
        '''
        global RND1
        gurobi = True
        # error handling
        if isinstance(self, UndirectedHypergraph):
            # check data is node A node B
            node_A, node_B, hyperedge_id = data
            if node_A not in self.nodes or node_B not in self.nodes:
                print(f"Node {node_A} or {node_B} does not exist in the Undirected hypergraph.")
                return None  # Return None if either node does not exist
            mu_A = self.node_probability(node_A)
            mu_B = self.node_probability(node_B)
        elif isinstance(self, DirectedHypergraph):
            # check data is a hyperedge_id
            hyperedge_id = data 
            if hyperedge_id not in self.hyperedges:
                print(f"Edge {hyperedge_id} does not exist in the Directed hypergraph.")
                return None  # Return None if edge does not exist
            mu_A, mu_B = self.calculate_probability_distributions(hyperedge_id)
        
        #TODO: check if this works for directed
        if approx_emd:
            weight = self.weights[hyperedge_id][-1]
            Na = self.neighbours(node_A)
            Nb = self.neighbours(node_B)
            da = len(Na)
            db = len(Nb)
            commonNeighbors = Na.intersection(Nb)
            mins =[]
            maxs =[]
            
            for n in commonNeighbors:
                aEdges = self.find_hyperedges_containing_all_nodes(n,node_A)
                bEdges = self.find_hyperedges_containing_all_nodes(n,node_B)
                for na_id in aEdges:
                    for nb_id in bEdges:
                        naw = self.weights[na_id][-1]
                        nbw = self.weights[nb_id][-1]
                        mins.append(min(naw/da, nbw/db))
                        maxs.append(max(naw/da, nbw/db)) 
            
            low = - min((1 - (weight/da) - (weight/db) - sum(maxs)), 0) - min((1 - (weight/da) - (weight/db) - sum(mins)), 0) + (sum(mins))
            high = sum(mins)
            orc = (low + high)/2
            return 1-orc
        
        # Convert distributions from dictionary to list format and print for debugging
        distribution1 = [mu_A[node] for node in self.node_index]
        distribution2 = [mu_B[node] for node in self.node_index]
        
        # Print the distributions to verify correctness
        if verbose:
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

        if RND1:
            clock_time('starting one job')
        
        if gurobi: 
            return self.earthmover_distance_gurobi_distance_matrix((mu_A, mu_B), distance_matrix, verbose)
            
        
    
    def earthmover_distance_gurobi_distance_matrix(self, data, distance_matrix, verbose):
        '''This is now the section that is just gurobi-specific

        '''
        global RND1
        
        mu_A, mu_B = data
            
        # Convert distributions from dictionary to list format and print for debugging
        node_to_index = self.node_index

        if RND1:
            clock_time('starting one job')
        
        env = Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()
        
        try:
            # Create a new model in Gurobi.
            model = Model("EarthMoverDistance", env=env)
            
            # # Make it less verbose
            # model.Params.LogToConsole = 0   
            
            # Set up the log file
            # log_filename = f"gurobi_log_{data}.log"
            # model.setParam('LogFile', log_filename)
            
            # Create variables for the linear program.
            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0) 
            # print('boom')
            expr = LinExpr(3.0)
            expr.clear()
            for x in mu_A:
                for y in mu_B:
                    expr.addTerms(distance_matrix[node_to_index[x]][node_to_index[y]], variables[x,y])
            
            # Set the objective of the linear program to minimize the total cost.
            model.setObjective(expr, GRB.MINIMIZE)
            
            # Add constraints to ensure the conservation of mass.
            for x in mu_A:
                model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

            for y in mu_B:
                model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")
            # print('bang')
            # Start the timer, solve the model, and calculate the time taken.
            model.optimize()
            if RND1:
                clock_time('finished the model')
                RND1 = False
            # print('done with the model')
            
            # Check the model status and process the results.
            if model.status == GRB.OPTIMAL:
                total_cost = model.getObjective().getValue()
                model.dispose()
                env.dispose()
                return total_cost
            else:
                print(f"No optimal solution found for data {data}")
                print(f"The probability distributions are A: {mu_A} \nAnd B: {mu_B}")
                print('Model Status', model.status)
                # print(f"The masses are {total_mass_A} for A and {total_mass_B} for B")
                model.dispose()
                env.dispose()
                return None
            
        except Exception as e:
            print(f"Gurobi Error: {e}\n for data {data}")
            return None 
        
            
    def add_missing_edges_shortest_path(self, other_graph, self_dist_mat, verbose:bool) -> None:
        '''So the idea, will be to add this edge, and then later delete it. Need to think about how to keep track of them
        For now, we're just testing on undirected. So will go forward on that.

        :param Hypergraph other_graph: The other graph we're working with. Should be of the same type as self.
        :param bool verbose: verbose flag
        '''
        for e in set(other_graph.hyperedges) - set(self.hyperedges):
            if isinstance(self, UndirectedHypergraph):
                node1 = other_graph.hyperedges[e][0]
                node2 = other_graph.hyperedges[e][1]
                dist = self_dist_mat[self.node_index[node1]][self.node_index[node2]]
                self.add_hyperedge(e, other_graph.hyperedges[e], [dist], verbose)
            if isinstance(self, DirectedHypergraph):
                (tail, head) = other_graph.hyperedges[e]
                #TODO: implement for hypegraphs
                dist = self_dist_mat[self.node_index[next(iter(tail))]][self.node_index[next(iter(head))]]
                self.add_hyperedge(e, tail, head, [dist], verbose)
            #TODO: Implement for Directed


class UndirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, nodes:list, weight_list = [1], verbose=True)-> None:
        '''Add a hyperedge to the hypergraph. Automatically adds missing nodes.

        :param str hyperedge_id: the name you would like to be used for the hyperedge
        :param list nodes: a list of the adjacent nodes
        :param list weight_list: Should start as a list with a single element (expect for weird cases), defaults to [1]
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
        self.weights[hyperedge_id] = weight_list
        return
        
    def build_from_dataframe(self, df:pd.DataFrame, verbose=True)-> None:
        '''Build hypergraph from a DataFrame

        :param pd.DataFrame df: has the columns 'source', 'target', and 'weight'
        :param bool verbose: verbose flag, defaults to True
        '''        
        # make an edge from each row in the csv
        for _, row in df.iterrows():
            node1 = row['source'].strip() #start
            node2 = row['target'].strip() #end
            weight = float(row['weight'])
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, [node1, node2], [weight], verbose)
            if verbose:
                print(f'Added hyperedge {edgeid} between {node1} and {node2}')
        return
    
    def node_degree(self, node) -> int:
        ''' Calculate the degree of a node. Degree is the numer of hyperedges 
        containing this node.

        :param _type_ node: the node we want to actually capture info for
        :raises ValueError: Raises if the node doesn't exist in the graph
        :return int: The number of hyperedges containing the node
        '''
        if node not in self.nodes:
            raise ValueError("Node does not exist in the graph.")
        return sum(node in hyperedge for hyperedge in self.hyperedges.values())
    
    def find_hyperedges_containing_all_nodes(self, *nodes):
        '''
        Find hyperedges that contain all of the specified nodes.
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
            if nodes_set.issubset(nodes_set.intersection(hyperedge_nodes)):  # Check if intersection contains all nodes we want
                found_hyperedges.append(hyperedge_id)
        
        return found_hyperedges
    
    
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
            emd = self.earthmover_distance_gurobi_distance_matrix((node_A, node_B, hyperedge_id), distance_matrix, verbose=False)
            
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
                # TODO: check to see if this makes sense for weight =0 in a real way. 
                # In theory this should return -inf
                return -np.inf
            else:
                # this is the orc. This is the EMD/dist(u,v) and dist(u,v) will just be the weight of the edge
                return 1 - average_emd/weight
        else:
            print(f"No valid EMD computations were possible. For hyperedge {hyperedge_id} with EMD {emd}")
            return None
    
  
class DirectedHypergraph(Hypergraph):
    def add_hyperedge(self, hyperedge_id:str, tail_set:set, head_set:set, weight_list = [1], verbose=True) -> None:
        '''Function to add a hyperedge to the hypergraph, if the nodes are not 
        there, will add the nodes
        
        :param str hyperedge_id: the name you would like to be used for the hyperedge
        :param set tail_set: a list of the tail nodes (nodes leaving from)
        :param set head_set: a list of the head nodes (nodes going to)
        :param list weight_list: Should start as a list with a single element (expect for weird cases), defaults to [1]
        :param bool verbose: verbose flag, defaults to True
        '''
        # Check if hyperedge already exists
        if hyperedge_id in self.hyperedges:
            print(f"Hyperedge {hyperedge_id} already exists with nodes {self.hyperedges[hyperedge_id]}")
            return
        
        # Add missing nodes to the node set
        for node in tail_set.union(head_set):
            if node not in self.nodes:
                self.add_node(node)
                
        # Add the hyperedge
        if verbose:
            f'Adding hyperedge {hyperedge_id} with tail nodes {tail_set} and head nodes {head_set}'
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]= weight_list
        return
        

    def build_from_dataframe(self, df:pd.DataFrame, verbose=True) -> None:
        '''Build hypergraph from a DataFrame

        :param pd.DataFrame df: has the columns 'source', 'target', and 'weight'
        :param bool verbose: verbose flag, defaults to True
        '''
        # make an edge from each row in the csv
        # TODO: make actually work for hypergraphs
        for _, row in df.iterrows():
            node1 = row['source'].strip() #start
            node2 = row['target'].strip() #end
            weight = float(row['weight'])
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, set([node1]), set([node2]), [weight])
            if verbose:
                print(f'Added hyperedge {edgeid} with head set {node1} and tail set {node2}')
        return
       
    def get_underlying_edges(self) -> set:
        '''Function to get the edges from the hyperedges.
            We're basically going to make it look like an undirected graph

        :return set: a set of edges with just the edgeid and then the nodes in a list
        '''
        edges = dict()
        
        for key in self.hyperedges.keys():
            tail, head = self.hyperedges[key]
            edges[key] = list(set(tail.union(head)))
        return edges
    
    def node_degree(self, node) -> np.array:
        '''Calculate the degree of a node. Degree is the number of hyperedges 
        containing this node.
        
        Will always return a numpy array (in-deg, out-deg)
    
        :param _type_ node: the node we want to actually capture info for
        :raises ValueError: _description_
        :return np.array: array of degrees with (in-deg, out-deg)
        '''
        if node not in self.nodes:
            raise ValueError("Node does not exist in the graph.")
        
        d_in_x = 0
        d_out_x = 0
        for _, (tail_set, head_set) in self.hyperedges.items():
            if node in head_set:
                d_in_x += 1
            if node in tail_set:
                d_out_x += 1  
        
        return [d_in_x, d_out_x]
    
    def calculate_probability_distributions(self, hyperedge_id:str):
        '''Function to calculate the probability distributions over all nodes based on the hyperedge

        :param str hyperedge_id: Name of the edge we're working with
        :return _type_: _description_
        '''
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
    
    
def save_matrix_csv(matrix:list[list], filename:str) -> None:
    '''Function to save the matrix as a CSV file

    :param list[list] matrix: matrix to be written
    :param str filename: place to write it
    '''    
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)
    
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

        
def clock_time(message:str)-> None:
    '''Get the time from the start of the process and write it to the scorecard with some message

    :param str message: The message to put before the time being spent
    '''
    now = time.time()
    rt = now-start
    write_scorecard(f'{message}: {rt}')
    return


def update_orc_and_weights_iter_manager(npr, distance_matrix:list[list], graph:Hypergraph, targ_graph:Hypergraph,  iteration:int, verbose:bool, file_format='csv', op_flag = False):
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
                
            # split the edges. Just do it the simple way. Just have the last one take the rest
            edges = list(graph.hyperedges.keys())
            njobs = len(graph.hyperedges)
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
                    SLICE = (jobstosend, distance_matrix, graph, targ_graph, file_name, verbose, iteration)
                    COMM.send(SLICE, dest = i, tag=44)
                    if verbose:
                        print('-> manager sends job', jobcnt, 'to worker', i, 'number of jobs', len(SLICE[0]))
                clock_time(f'manager has sent all the jobs')
                # receive the jobs // sync the graphs.
                for i in range(1, npr):
                    newgraph, jobs = COMM.recv(source=i, tag=11)
                    if verbose:
                        print('-> manager received data from worker', i, 'number of jobs', len(jobs))
                    for e in jobs: #sync up the graph
                        graph.add_ricci_curvature(e, newgraph.ricci_curvature[e][-1])
                        graph.add_weights(e, newgraph.weights[e][-1])
                        writer.writerow([e, newgraph.ricci_curvature[e][-1], newgraph.weights[e][-1]]) 
                    # clock_time(f'gathered data from processor: {i}')
    return
            
                    
def update_orc_and_weights_iter_worker():
    global RND1
    specs = COMM.recv(source = 0, tag = 44)
    if specs == -1: 
        return False
    jobs, distance_matrix, graph, targ_graph, file_name, verbose, itteration = specs
    RND1 = True

    # with open(file_name, 'a', newline='') as file:
        # writer = csv.writer(file)
    for hyperedge_id in jobs:
        if isinstance(graph, UndirectedHypergraph):
            orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix, verbose)
        elif isinstance(graph, DirectedHypergraph): 
            orc = 1 - graph.earthmover_distance_gurobi_distance_matrix(hyperedge_id, distance_matrix, verbose=False)
        normalized_orc = ricci_normalizing(orc)
        graph.add_ricci_curvature(hyperedge_id, normalized_orc)
        weight = graph.weights[hyperedge_id][-1]
        if itteration != 0:
            orc_targ = targ_graph.ricci_curvature[hyperedge_id][-1]
            if weight != 0:
                #simple version
                step = 1
                wtplus1 = weight*(1  - step*(normalized_orc - orc_targ))
                normalized_weight = wtplus1
            else:
                normalized_weight = 0
            
            graph.add_weights(hyperedge_id, normalized_weight)
            
                # writer.writerow([hyperedge_id, normalized_orc, normalized_weight])
            # else: 
                # writer.writerow([hyperedge_id, normalized_orc, weight])
    COMM.send((graph,jobs) , dest=0, tag=11)
    return True


def calculate_target_orc_manager(npr, distance_matrix: list[list], graph:Hypergraph, verbose:bool, op_flag=False):
    # Works for Directed
    '''The function to calculate the staring infor for the target graph

    :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
    :param Hypergraph graph: the actual source graph
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
            
        # split the edges. Just do it the simple way. Just have the last one take the rest
        edges = list(graph.hyperedges.keys())
        njobs = len(edges)
        chunksize = njobs//(npr-1)
        remainder = njobs % (npr-1)
        jobcnt = 0
        print('Number of jobs ', njobs)
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
            clock_time('manager sent all the jobs')
            # receive the jobs // sync the graphs.
            for i in range(1, npr):
                newgraph, jobs = COMM.recv(source=i, tag=11)
                clock_time(f'gathered data from processor: {i}')
                if True:
                    print('-> manager received data from worker', i, 'number of jobs', len(jobs))
                for e in jobs: #sync up the graph
                    graph.add_ricci_curvature(e, newgraph.ricci_curvature[e][-1])
                    graph.add_weights(e, newgraph.weights[e][-1])
                    writer.writerow([e, newgraph.ricci_curvature[e][-1], newgraph.weights[e][-1]]) 
    return


def calculate_target_orc_worker():
    global RND1
    specs = COMM.recv(source = 0, tag = 33)
    if specs == -1: 
        return
    jobs, distance_matrix, graph, file_name, verbose = specs
    RND1 = True

    # with open(file_name, 'a', newline='') as file:
        # writer = csv.writer(file)
    for hyperedge_id in jobs:
        if isinstance(graph, UndirectedHypergraph):
            orc = graph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix, verbose)
        elif isinstance(graph, DirectedHypergraph): 
            orc = 1 - graph.earthmover_distance_gurobi_distance_matrix(hyperedge_id, distance_matrix, verbose=False)
        if RND1:
            clock_time('finished ORC')
        normalized_orc = ricci_normalizing(orc)
        graph.add_ricci_curvature(hyperedge_id, normalized_orc)
        # weight = graph.weights[hyperedge_id][-1]
            # writer.writerow([hyperedge_id, normalized_orc, weight])  
    clock_time('finished worker now sending over')
    COMM.send((graph,jobs) , dest=0, tag=11)
    return

        
def early_analysis(src_graph:Hypergraph, verbose:bool):
    '''Get the information on the graph we're working on

    :param Hypergraph src_graph: The graph we're analyzing
    :param bool verbose: verbose flag
    '''
    connected = src_graph.is_weakly_connected()
    if isinstance(src_graph, DirectedHypergraph):
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
    if isinstance(src_graph, DirectedHypergraph):
        if strconnect: write_scorecard('The hypergraph is strongly connected.')
        else: write_scorecard('The hypergraph is not strongly connected.')
    write_scorecard(f"Max Degree: {max_degree}")
    write_scorecard(f"Min Degree: {min_degree}") # Quick note: directed can have these be 0
    write_scorecard(f"Average Degree: {avg_degree}") # for directed graphs, this should be equal.
    write_scorecard('----------------------------')
    return


def set_up_one_direction(src_graph:Hypergraph, targ_graph:Hypergraph, op_flag=False):
    '''Setting up the one direction stuff

    :param Hypergraph src_graph: The source graph
    :param Hypergraph targ_graph: The target graph
    :param int tot_its: the maximum number of steps it can take, defaults to 100
    '''
    print('working on distance matrices')
    distance_matrix = src_graph.floyd_warshall()
    matfilename = 'outputfiles/'
    if op_flag: matfilename += 'op_'
    matfilename += 'source_dist_fw.csv'
    save_matrix_csv(distance_matrix, matfilename)
    
    clock_time('Time to make the source distance matrix')

    target_distance_matrix = targ_graph.floyd_warshall()
    matfilename = 'outputfiles/'
    if op_flag: matfilename += 'op_'
    matfilename += 'target_dist_fw.csv'
    save_matrix_csv(target_distance_matrix, matfilename)
    
    clock_time('Time to make the target distance matrix')
    
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
        clock_time('time to add missing edges')
        
        # recalculate the matrices
        distance_matrix = src_graph.floyd_warshall()
        
        target_distance_matrix = targ_graph.floyd_warshall()
        clock_time('time to recalc the distances')
    
    print('len mising from source', len(missing_from_src))
    print('len mising from targ', len(missing_from_targ))
    
    return target_distance_matrix, distance_matrix, missing_from_src, missing_from_targ

        
def one_direction_of_work_manager(npr, src_graph, targ_graph, tot_its = 100, op_flag=False):
    # One Direction of Work
    targ_distance_matrix, distance_matrix, missing_from_src, missing_from_targ = set_up_one_direction(src_graph, targ_graph, op_flag)
    
    clock_time('time to set up')
    print('starting ricci curvature')
    
    calculate_target_orc_manager(npr, targ_distance_matrix, targ_graph, verbose, op_flag=op_flag)

    clock_time('Time to calc target ORC')

    update_orc_and_weights_iter_manager(npr, distance_matrix, src_graph, targ_graph, iteration=0, verbose=verbose, op_flag=op_flag)

    clock_time('Time to calc source ORC')
    
    for i in range(1, tot_its + 1):
        print('Working on itteration', i)
        distance_matrix_i = src_graph.floyd_warshall()
        clock_time(f'finished distance matrix {i}')
        if i>1:
            if len(missing_from_src)> 0 or len(missing_from_targ)> 0:
                # We're gonna to the reset here
                src_graph.add_missing_edges_shortest_path(targ_graph, distance_matrix, verbose)
                targ_graph.add_missing_edges_shortest_path(src_graph, targ_distance_matrix, verbose)
        update_orc_and_weights_iter_manager(npr, distance_matrix_i, src_graph, targ_graph, iteration=i, verbose=verbose, op_flag=op_flag)
        clock_time(f'Time for ORC {i}')
        
        # We will do a "reset" here
        if len(missing_from_src)> 0 or len(missing_from_targ)> 0:
            # We're gonna to the reset here
            #first delete all the edges
            for e in missing_from_src:
                src_graph.remove_hyperedge(e)
            for e in missing_from_targ:
                targ_graph.remove_hyperedge(e)
                
        allstable = True
        finustab = None
        if i == 1: 
            #TODO: fix this weirdness
            # take care of the getting started case
            continue
        errorlist = []
        for e in src_graph.hyperedges:
            clist = src_graph.ricci_curvature[e]
            old = clist[-2]
            new = clist[-1]
            if old != 0:
                if absolute_change: error = abs(old-new)
                else: error = abs((old-new)/old) #relative change
            else: 
                error = abs(old-new)
                if (not absolute_change):
                    error = error / old
            if maximum_error:
                if absolute_change and (error > 0.01):
                    # if verbose:
                    clock_time(f'unstable for edge {e} with error {error}')
                    finustab = e
                    allstable = False
                    break
                if (not absolute_change) and (error > 0.05): # relative change
                    clock_time(f'unstable for edge {e} with error {error}')
                    finustab = e
                    allstable = False
                    break
            else:
                errorlist.append(error)
        #find the average error
        if not maximum_error: # AKA we're in average error zone
            # assumed to be in absolute error zone
            avg_err = np.average(errorlist)
            if avg_err > 0.0001:
                clock_time(f'unstable with average error {avg_err}')
                finustab = e
                allstable = False
        if allstable:
            #turn off all workers.
            for k in range(1,npr):
                COMM.send(-1, dest = k, tag = 44)
            print('STABILIZED! Source to target distance is ',i)
            write_scorecard('\n\n----- Results -----')
            write_scorecard(f'Source to target distance is {i}')
            break
    return 
    

def one_direction_of_work_worker(tot_its = 100):   
    calculate_target_orc_worker()
    update_orc_and_weights_iter_worker() 
    cont=True
    cnt = 1
    while cont and (cnt<=tot_its):
        cont = update_orc_and_weights_iter_worker()
        cnt = cnt + 1
    return 

    
def manager(npr, verbose = True):
    # starting off things
    clean_output(verbose)
    
    # source_filename = os.environ.get('SOURCE_FILENAME')
    # target_filename = os.environ.get('TARGET_FILENAME')
    # source_filename = 'ERgraph500nodep4.csv'
    # source_filename = 'ERgraph100nodep4.csv'
    source_filename = 'petersen/petersengraph.csv'
    target_filename = 'petersen/petersengraph_newbigweights.csv'
    # target_filename = 'rangechanges/ERgraph500n5changenewrange1000to2000v3.csv'
    # target_filename = 'rangechanges/ERgraph100n100changenewrange1000to2000v3.csv'

    data_target = pd.read_csv(f'inputfiles/{target_filename}', dtype ={'source': str, 'target':str}, sep=',')  
    data_source = pd.read_csv(f'inputfiles/{source_filename}', dtype ={'source': str, 'target':str}, sep=',')  

    # Scorecard Writing
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
    clock_time('Time to read the data in seconds')
    
    if directed_flag:
        source_graph = DirectedHypergraph()
        target_graph = DirectedHypergraph()
    else:
        source_graph = UndirectedHypergraph()
        target_graph = UndirectedHypergraph() 
        
    print('building source')          
    source_graph.build_from_dataframe(data_source, verbose)
    # print(source_graph.nodes)
    print('building target')
    target_graph.build_from_dataframe(data_target, verbose)
    
    clock_time('Time to build the graphs')
    
    if not (source_graph.is_2_uniform() and target_graph.is_2_uniform()) :
        print('This has not been fully fleshed out for hypergraphs. Please give a 2-uniform graph')
        quit()
    
    early_analysis(source_graph, verbose)
    clock_time('Time to analyze graphs')
    
    # One Direction of Work
    one_direction_of_work_manager(npr, source_graph, target_graph, tot_its = ITS)
    
    clock_time('Time for source->target')

    write_scorecard('\n')

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
    
    one_direction_of_work_manager(npr, source_graph, target_graph, tot_its=ITS, op_flag = True)
      
    clock_time('Time for final')
    
    # tell the jobs to sleep (at the very end)
    # send the jobs
    for i in range(1, npr):
        SLICE = -33
        COMM.send(SLICE, dest = i, tag=55)
        if verbose:
            print(f'-> manager sends {SLICE} to worker', i)
    return

def worker(w, verbose = True):
    one_direction_of_work_worker(tot_its = ITS)
    one_direction_of_work_worker(tot_its = ITS)
    while True:
        specs = COMM.recv(source = 0, tag = 55)
        if specs == -33: 
            if verbose:
                print(f'Worker {w} goes to sleep')
            break
    return    
  
if __name__ == "__main__": 
    directed_flag = True
    verbose = True
    absolute_change = True # False is relative change
    maximum_error = True # False is average error
    approx_emd = os.environ.get('APPROX')
    if approx_emd is None: # Make the default False
        approx_emd = 'False'
    approx_emd = eval(approx_emd)
    
    start = time.time()
    
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    ITS = 100
    if RANK == 0:
        manager(SIZE, verbose=False)
    else: 
        worker(RANK, verbose=False)
    print(f'node {RANK} made it to the end')
    # main()