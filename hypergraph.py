'''Definition of a hypergraph object. This will be used in the Ricci Flow Experiments

:raises ValueError: _description_
:return _type_: _description_
'''
from numbers import Number
import numpy as np
from typing import *

class Hypergraph:
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
    
    def get_underlying_edges(self) -> dict:
        '''Get the underlying edges (undirected)

        :return dict: dict from hyperedge id to lists of nodes in that edge
        '''        
        return self.hyperedges    
    
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

        # TODO: make a .get_underlying_edges() function override for DirectedHypergraph
        
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
    
    # def is_strongly_connected(self)-> bool:
    #     #TODO: This is only for directed hypergraphs.
    #     '''Check if the underlying graph is strongly connected

    #     :return bool: True if strongly connected
    #     '''        
    #     # I think this is saying an empty graph is strongly connected
    #     if not self.nodes: 
    #         return True

    #     edges = self.hyperedges
        
    #     visited = set()

    #     def dfs(node):
    #         '''Depth First Search'''
    #         if node in visited:
    #             # print('got here')
    #             return visited
    #         visited.add(node)
    #         for edge in edges.values():
    #             # print(edge)
    #             if isinstance(self, UndirectedHypergraph):
    #                 if node in edge:
    #                     for next_node in edge:
    #                         # print(next_node)
    #                         if next_node != node:
    #                             dfs(next_node)
    #             elif isinstance(self, DirectedHypergraph):
    #                 # note tail and head
    #                 tail, head = edge
    #                 if node in tail:
    #                     for next_node in head:
    #                         # print(next_node)
    #                         if next_node != node:
    #                             dfs(next_node)

    #     # Do DFS from each node
    #     for v in iter(self.nodes):
    #         visited = set()
    #         # print('hellloooo')
    #         dfs(v)
    #         # write_scorecard(str(visited))
    #         if visited != self.nodes:
    #             # write_scorecard(f'not strongly connected based on vertex {v}')
    #             # write_scorecard(str(visited))
    #             return False
    #         else: 
    #             visited = set()
    #     return True 
    
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
            
        # pprint(dist)
        
        # Set the distance for directly connected nodes based on edge weights
        for hyperedge_id, nodes in self.hyperedges.items():
            #TODO: figure this out for how to split it
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
        # gurobi = True
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
        
        
        
class UndirectedHypergraph(Hypergraph):
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
            
    
class DirectedHypergraph(Hypergraph):
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
