'''Definition of a hypergraph object. This will be used in the Ricci Flow Experiments

:raises ValueError: _description_
:return _type_: _description_
'''
from numbers import Number
import numpy as np
from typing import *
import pandas as pd
from abc import ABC, abstractmethod
from itertools import combinations

class Hypergraph(ABC):
    '''This is the main class we use for they hypergraph object that stores 
    Ricci curvatures as well
    '''    
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
        :param Number weight: a number for the weight
        ''' 
        if weight is not None:
            self.weights[hyperedge_id].append(weight)
        return
       
    def remove_hyperedge(self, hyperedge_id:str) -> None:
        ''' Remove a hyperedge from the graph. Notably, will not remove the nodes

        :param str hyperedge_id: The hyperedge to be deleted
        '''
        del self.hyperedges[hyperedge_id]
        return  
    
    def update_node_index(self) -> None:
        '''The goal is to ensure there is a static node index for the graph. 
        This function generates it. It will be an attribute.
        '''
        if len(self.node_index) == len(self.nodes):
            return
        else:
            self.node_index = {node: idx for idx, node in enumerate(list(self.nodes))}
        return
                        
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
        
        edges = self.get_underlying_edges()
        
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
           each pair of vertices. Right now, if you cannot get from one node to 
           another, the distance will be the maximum value + 1. This will be relevant in the 
           case of directed, not strongly connected graphs.

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
        for i in range(node_count):
            dist[i][i] = 0
        
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

        # Floyd-Warshall algorithm to update distances
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]] < dist[node_index[i]][node_index[j]]:
                        dist[node_index[i]][node_index[j]] = dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]]
        
        # replacing inf with max + 1
        tmp = np.array(dist)
        tmp[tmp == float('inf')] = -1
        tmp[tmp==-1] = np.max(tmp) + 1
        dist = tmp.tolist()
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
    
    def earthmover_distance_distance_matrix(self, data, distance_matrix:list[list], verbose:bool) -> float:
        '''Getting the actual EMD calculations. This is combining the work between 
        Undirected and Directed graphs into a single function

        :param _type_ data: _description_
        :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
        :param bool verbose: verbose flag
        :raises ValueError: Will throw if the total masses aren't equal. Comes up if something has gone wrong in the EMD calculation
        :return float: The actual EMD value
        '''    
        global RND1
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

        if RND1 and verbose:
            clock_time('starting one job')
        
        if gurobi_flag: 
            return self.earthmover_distance_gurobi_distance_matrix((mu_A, mu_B), distance_matrix, verbose)
        else: # then we're in POT world
            # ensure the distribution and the distance matrix line up. This should be the case if we've done floyd warshall for this distance matrix. Which I am going to assume we do.
            W = ot.emd2(distribution1, distribution2, distance_matrix)
            return W
            
    def earthmover_distance_gurobi_distance_matrix(self, data, distance_matrix, verbose):
        '''This is now the section that is just gurobi-specific

        '''
        global RND1
        
        mu_A, mu_B = data
            
        # Convert distributions from dictionary to list format and print for debugging
        node_to_index = self.node_index

        if RND1 and verbose:
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
            if RND1 and verbose:
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
        
    # These are implemented in the subclasses
    @abstractmethod
    def is_strongly_connected(self):
        pass
    
    @abstractmethod
    def build_from_dataframe(self):
        pass 
    
    @abstractmethod
    def add_missing_edges_shortest_path(self):
        pass
        
    @abstractmethod
    def node_degree(self):
        pass
    
    @abstractmethod
    def get_underlying_edges(self):
        pass
        
class UndirectedHypergraph(Hypergraph):
    def build_from_dataframe(self, df:pd.DataFrame, verbose=False)-> None:
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
    
    def add_missing_edges_shortest_path(self, other_graph, self_dist_mat, verbose:bool) -> None:
        '''So the idea, will be to add this edge, and then later delete it. Need to think about how to keep track of them
        For now, we're just testing on undirected. So will go forward on that.

        :param Hypergraph other_graph: The other graph we're working with. Should be of the same type as self.
        :param bool verbose: verbose flag
        '''
        #TODO: raise error if other_graph is not Undirected
        for e in set(other_graph.hyperedges) - set(self.hyperedges):
            node1 = other_graph.hyperedges[e][0]
            node2 = other_graph.hyperedges[e][1]
            dist = self_dist_mat[self.node_index[node1]][self.node_index[node2]]
            self.add_hyperedge(e, other_graph.hyperedges[e], [dist], verbose)
        return
    
    def get_underlying_edges(self) -> dict:
        '''Get the underlying edges (undirected)

        :return dict: dict from hyperedge id to lists of nodes in that edge
        '''        
        return self.hyperedges    
            
    def is_strongly_connected(self)-> bool:
        '''Checks if the graph is strongly connected. For Undirected, that's 
        the same as weakly connected

        :return bool: True is the graph is strongly connected
        '''     
        return self.is_weakly_connected()   
    
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
    
    
    def earthmover_distance_hyperedge_combinations(self, hyperedge_id:str, distance_matrix:list[list], verbose:bool) -> float:
        '''This buddy gets the average EMD across the whole edge

        :param str hyperedge_id: The identifier for the hyperedge
        :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
        :param bool verbose: verbose flag
        :return float:  The average EMD for all permutations of node pairs, or None if the hyperedge does not exist or has errors.
        '''       
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
            emd = self.earthmover_distance_distance_matrix((node_A, node_B, hyperedge_id), distance_matrix, verbose=False)
            
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
                return -np.inf
            else:
                # this is the orc. This is the EMD/dist(u,v) and dist(u,v) will just be the weight of the edge
                return 1 - average_emd/weight
        else:
            print(f"No valid EMD computations were possible. For hyperedge {hyperedge_id} with EMD {emd}")
            return None
        
    def node_probability(self, node:any) -> dict:
        '''Calculate the probability distributions of a specific node. This will
        calculate 1/deg(u) spread out around the neighbors. But it's also a lazy 
        distribution.

        :param any node: should pull up a node
        :raises ValueError: Will be raised if there is not a node of the name given in the graph
        :return dict: mu will be {node: dist} dictionary
        '''        
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
    
            
    
class DirectedHypergraph(Hypergraph):
    def build_from_dataframe(self, df:pd.DataFrame, verbose=False) -> None:
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
    
    def add_missing_edges_shortest_path(self, other_graph, self_dist_mat, verbose:bool) -> None:
        '''So the idea, will be to add this edge, and then later delete it. Need to think about how to keep track of them
        For now, we're just testing on undirected. So will go forward on that.

        :param Hypergraph other_graph: The other graph we're working with. Should be of the same type as self.
        :param bool verbose: verbose flag
        '''
        #TODO: raise error if other_graph is not Directed
        for e in set(other_graph.hyperedges) - set(self.hyperedges):
            (tail, head) = other_graph.hyperedges[e]
            #TODO: implement for hypegraphs
            dist = self_dist_mat[self.node_index[next(iter(tail))]][self.node_index[next(iter(head))]]
            self.add_hyperedge(e, tail, head, [dist], verbose)
            
            
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
    
    def is_strongly_connected(self)-> bool:
        '''Check if the underlying graph is strongly connected

        :return bool: True if strongly connected
        '''        
        # I think this is saying an empty graph is strongly connected
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
                tail, head = edge
                if node in tail:
                    for next_node in head:
                        if next_node != node:
                            dfs(next_node)

        # Do DFS from each node
        for v in iter(self.nodes):
            visited = set()
            dfs(v)
            if visited != self.nodes:
                return False
            else: 
                visited = set()
        return True 
    
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