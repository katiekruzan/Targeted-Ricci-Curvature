'''Definition of a hypergraph object. This will be used in the Ricci Flow Experiments
'''
from numbers import Number
import numpy as np
from typing import *
import pandas as pd
from abc import ABC, abstractmethod
from itertools import combinations
import ot


class Hypergraph(ABC):
    '''This is the main class we use for they hypergraph object that stores 
    Ricci curvatures as well
    '''

    def __init__(self):
        '''Initializing the hypergraph
        '''
        self.nodes = set()  # arbitrary, not defined type as of now.
        self.hyperedges = {}  # dict from hyperedge id to lists of nodes in that edge
        self.weights = {}  # dict that had hyperedge ids to weights
        # dict with hyperedge id to list of ricci curvatures (floats)
        self.ricci_curvature = {}
        self.node_index = {}

    def add_node(self, node: any) -> None:
        '''Function to add a node to the hypergraph. The type is not set

        :param any node: The node to be added
        '''
        self.nodes.add(node)
        return

    def add_ricci_curvature(self, hyperedge_id: str, orc: float) -> None:
        '''Function to add ollivier ricci curvature for all hyperedges for each iteration.
            It will be appending to a list

        :param str hyperedge_id: The id of the hyperedge of interest
        :param float orc: the curvature to be appended
        '''
        if hyperedge_id not in self.ricci_curvature:
            # Initialize with an empty list if key doesn't exist
            self.ricci_curvature[hyperedge_id] = []
        self.ricci_curvature[hyperedge_id].append(orc)
        return

    def add_weights(self, hyperedge_id: str, weight: Number) -> None:
        '''Function to add weights for all hyperedges for each iteration.
            It will be appending to a list

        :param str hyperedge_id: the id of the hyperedge of interest
        :param Number weight: a number for the weight
        '''
        if weight is not None:
            self.weights[hyperedge_id].append(weight)
        return

    def remove_hyperedge(self, hyperedge_id: str) -> None:
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
            self.node_index = {node: idx for idx,
                               node in enumerate(list(self.nodes))}
        return

    def is_2_uniform(self) -> bool:
        '''Check if size of each edge is 2

        :return bool: true if graph is 2 uniform
        '''
        for edges in self.hyperedges.items():
            if len(edges) != 2:
                return False
        return True

    def is_weakly_connected(self) -> bool:
        '''Check if the underlying graph is weakly connected. Will return True 
         for empty graphs.

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

    def connected_components(self) -> dict:
        '''Taken from the implementation of DirectedHypergraph.py on 
        Prith's Ricci-Flow-on-Hypergraphs

        :return dict: A dictionary of the nodes to some arbitrary label for what community it belongs to
        '''
        # empty graph has no components
        if not self.nodes:
            return []

        visited = set()
        components = {}
        edges = self.get_underlying_edges()

        def dfs(node, component):
            '''Depth First Search'''
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for edge in edges.values():
                if node in edge:
                    for next_node in edge:
                        if next_node != node:
                            dfs(next_node, component)

        ind = 1
        for node in self.nodes:
            if node not in visited:
                # make a new component
                current_component = set()
                dfs(node, current_component)
                for node in current_component:
                    components[node] = ind
                ind = ind+1

        return components

    def floyd_warshall(self) -> list[list]:
        '''Use the Floyd-Warshall algorithm to find the shortest distances between
           each pair of vertices. Right now, if you cannot get from one node to 
           another, the distance will be the maximum value + 1. This will be relevant in the 
           case of directed, not strongly connected graphs.

        :return list[list]: a matrix with the shortest distances, that will be ordered by self.node_index
        '''
        # Assume self.nodes is a list or set of nodes
        # Convert to list to ensure consistent ordering
        node_list = list(self.nodes)
        node_count = len(node_list)

        # Create a mapping of node to index
        self.update_node_index()
        node_index = self.node_index

        # Initialize a 2D list (matrix) with "infinite" distances
        dist = [[float('inf') for _ in range(node_count)]
                for _ in range(node_count)]

        # Set the diagonal to 0 (distance from each node to itself)
        for i in range(node_count):
            dist[i][i] = 0

        # Set the distance for directly connected nodes based on edge weights
        for hyperedge_id, nodes in self.hyperedges.items():
            if isinstance(self, UndirectedHypergraph):
                tail_set, head_set = nodes, nodes
            else:
                tail_set, head_set = nodes

            # TODO: test this when it comes to actual hypergraphs
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
                                                                   # Using the last weight in the list
                                                                   self.weights[hyperedge_id][-1])

        # Floyd-Warshall algorithm to update distances
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]] < dist[node_index[i]][node_index[j]]:
                        dist[node_index[i]][node_index[j]] = dist[node_index[i]
                                                                  ][node_index[k]] + dist[node_index[k]][node_index[j]]

        # replacing inf with max + 1
        tmp = np.array(dist)
        tmp[tmp == float('inf')] = -1
        tmp[tmp == -1] = np.max(tmp) + 1
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

    def earthmover_distance_distance_matrix(self, data, distance_matrix: list[list], approx_emd: bool, gurobi_flag: bool, verbose: bool) -> float:
        '''Getting the actual EMD calculations. This combines the work between 
        Undirected and Directed graphs into a single function

        :param _type_ data: The data will differ based on if self is an Undirected or Directed Hypergraph
        :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
        :param bool approx_emd: True if wanting to calculate approx EMD flag (only works for Undirected)
        :param bool gurobi_emd: True if wanting to calculate EMD using Gurobi solver. Alternatively using pyot, flag
        :param bool verbose: verbose flag
        :raises ValueError: Will throw if the total masses aren't equal. Comes up if something has gone wrong in the EMD calculation
        :return float: The actual EMD value
        '''
        # error handling
        if isinstance(self, UndirectedHypergraph):
            # check data is node A node B
            node_A, node_B, hyperedge_id = data
            if node_A not in self.nodes or node_B not in self.nodes:
                print(
                    f"Node {node_A} or {node_B} does not exist in the Undirected hypergraph.")
                return None  # Return None if either node does not exist
            mu_A = self.node_probability(node_A)
            mu_B = self.node_probability(node_B)
            if approx_emd:  # Note, this will only happen if undirected.
                # This approximates ORC based on (Jost & Liu)
                weight = self.weights[hyperedge_id][-1]
                Na = self.neighbours(node_A)
                Nb = self.neighbours(node_B)
                da = len(Na)
                db = len(Nb)
                commonNeighbors = Na.intersection(Nb)
                mins = []
                maxs = []

                for n in commonNeighbors:
                    aEdges = self.find_hyperedges_containing_all_nodes(
                        n, node_A)
                    bEdges = self.find_hyperedges_containing_all_nodes(
                        n, node_B)
                    for na_id in aEdges:
                        for nb_id in bEdges:
                            naw = self.weights[na_id][-1]
                            nbw = self.weights[nb_id][-1]
                            mins.append(min(naw/da, nbw/db))
                            maxs.append(max(naw/da, nbw/db))

                low = - min((1 - (weight/da) - (weight/db) - sum(maxs)), 0) - \
                    min((1 - (weight/da) - (weight/db) - sum(mins)), 0) + (sum(mins))
                high = sum(mins)
                orc = (low + high)/2
                return 1-orc  # this will return the actual approx EMD
        elif isinstance(self, DirectedHypergraph):
            # check data is a hyperedge_id
            hyperedge_id = data
            if hyperedge_id not in self.hyperedges:
                print(
                    f"Edge {hyperedge_id} does not exist in the Directed hypergraph.")
                return None  # Return None if edge does not exist
            mu_A, mu_B = self.calculate_probability_distributions(hyperedge_id)

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
            raise ValueError(
                'The total mass of the distributions mu_A and mu_B are not equal.')

        # TODO: implement the gurobi_flag
        # if gurobi_flag:
        #     return self.earthmover_distance_gurobi_distance_matrix((mu_A, mu_B), distance_matrix, verbose)
        else:  # then we're in POT world
            # ensure the distribution and the distance matrix line up. This should be the case if we've done floyd warshall for this distance matrix. Which I am going to assume we do.
            W = ot.emd2(distribution1, distribution2, distance_matrix)
            return W

    # def earthmover_distance_gurobi_distance_matrix(self, data, distance_matrix, verbose):
    #     '''This is now the section that is just gurobi-specific

    #     '''

    #     mu_A, mu_B = data

    #     # Convert distributions from dictionary to list format and print for debugging
    #     node_to_index = self.node_index

    #     env = Env(empty=True)
    #     env.setParam("OutputFlag",0)
    #     env.start()

    #     try:
    #         # Create a new model in Gurobi.
    #         model = Model("EarthMoverDistance", env=env)

    #         # # Make it less verbose
    #         # model.Params.LogToConsole = 0

    #         # Set up the log file
    #         # log_filename = f"gurobi_log_{data}.log"
    #         # model.setParam('LogFile', log_filename)

    #         # Create variables for the linear program.
    #         variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)
    #         # print('boom')
    #         expr = LinExpr(3.0)
    #         expr.clear()
    #         for x in mu_A:
    #             for y in mu_B:
    #                 expr.addTerms(distance_matrix[node_to_index[x]][node_to_index[y]], variables[x,y])

    #         # Set the objective of the linear program to minimize the total cost.
    #         model.setObjective(expr, GRB.MINIMIZE)

    #         # Add constraints to ensure the conservation of mass.
    #         for x in mu_A:
    #             model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

    #         for y in mu_B:
    #             model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")
    #         # print('bang')
    #         # Start the timer, solve the model, and calculate the time taken.
    #         model.optimize()
    #         # print('done with the model')

    #         # Check the model status and process the results.
    #         if model.status == GRB.OPTIMAL:
    #             total_cost = model.getObjective().getValue()
    #             model.dispose()
    #             env.dispose()
    #             return total_cost
    #         else:
    #             print(f"No optimal solution found for data {data}")
    #             print(f"The probability distributions are A: {mu_A} \nAnd B: {mu_B}")
    #             print('Model Status', model.status)
    #             # print(f"The masses are {total_mass_A} for A and {total_mass_B} for B")
    #             model.dispose()
    #             env.dispose()
    #             return None

    #     except Exception as e:
    #         print(f"Gurobi Error: {e}\n for data {data}")
    #         return None

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
    def add_hyperedge(self):
        pass

    @abstractmethod
    def node_degree(self):
        pass

    @abstractmethod
    def get_underlying_edges(self):
        pass


class UndirectedHypergraph(Hypergraph):
    def build_from_dataframe(self, df: pd.DataFrame, verbose=False) -> None:
        '''Build hypergraph from a DataFrame

        :param pd.DataFrame df: has the columns 'source', 'target', and 'weight'
        :param bool verbose: verbose flag, defaults to True
        '''
        # make an edge from each row in the csv
        for _, row in df.iterrows():
            node1 = row['source'].strip()  # start
            node2 = row['target'].strip()  # end
            weight = float(row['weight'])
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, [node1, node2], [weight], verbose)
            if verbose:
                print(f'Added hyperedge {edgeid} between {node1} and {node2}')
        return

    def add_hyperedge(self, hyperedge_id: str, nodes: list, weight_list=[1], verbose=True) -> None:
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
            print(
                f"Hyperedge {hyperedge_id} already exists with nodes {self.hyperedges[hyperedge_id]}")
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

    def add_missing_edges_shortest_path(self, other_graph: Hypergraph, self_dist_mat: list[list], verbose: bool) -> None:
        '''So the idea, will be to add this edge, and then later delete it. 
        Need to think about how to keep track of them
        For now, we're just testing on undirected. So will go forward on that.

        :param Hypergraph other_graph: The other graph we're working with. Should be of the same type as self.
        :param list[list] self_dist_mat: matrix of minimal distances from the floyd_warshall function
        :param bool verbose: verbose flag
        '''
        # Ensure nodes is a list
        if not isinstance(other_graph, UndirectedHypergraph):
            raise ValueError(
                "Trying to compare and Undirected graph with something else")
        for e in set(other_graph.hyperedges) - set(self.hyperedges):
            node1 = other_graph.hyperedges[e][0]
            node2 = other_graph.hyperedges[e][1]
            dist = self_dist_mat[self.node_index[node1]
                                 ][self.node_index[node2]]
            self.add_hyperedge(e, other_graph.hyperedges[e], [dist], verbose)
        return

    def get_underlying_edges(self) -> dict:
        '''Get the underlying edges (undirected)

        :return dict: dict from hyperedge id to lists of nodes in that edge
        '''
        return self.hyperedges

    def is_strongly_connected(self) -> bool:
        '''Checks if the graph is strongly connected. For Undirected, that's 
        the same as weakly connected

        :return bool: True is the graph is strongly connected
        '''
        return self.is_weakly_connected()

    def node_degree(self, node) -> int:
        ''' Calculate the degree of a node. Degree is the number of hyperedges 
        containing this node.

        :param node: the node we want to actually capture info for
        :raises ValueError: Raises if the node doesn't exist in the graph
        :return int: The number of hyperedges containing the node
        '''
        if node not in self.nodes:
            raise ValueError("Node does not exist in the graph.")
        return sum(node in hyperedge for hyperedge in self.hyperedges.values())

    def earthmover_distance_hyperedge_combinations(self, hyperedge_id: str, distance_matrix: list[list], approx_emd: bool, gurobi_flag: bool, verbose: bool) -> float:
        '''This buddy gets the average EMD across the whole edge

        :param str hyperedge_id: The identifier for the hyperedge
        :param list[list] distance_matrix: matrix of minimal distances from the floyd_warshall function
        :param bool approx_emd: flag to be passed to the emd calculation
        :param bool gurobi_emd: flag to be passed to the emd calculation
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
            emd = self.earthmover_distance_distance_matrix(
                (node_A, node_B, hyperedge_id), distance_matrix, approx_emd, gurobi_flag, verbose=verbose)

            if emd is not None:
                sum_emd += emd
                pair_count += 1
            else:
                print('emd is none on nodes ', node_A, 'and', node_B)

        if pair_count > 0:
            # Compute the average EMD
            average_emd = sum_emd / pair_count
            weight = self.weights[hyperedge_id][-1]
            if weight == 0:
                # this is the orc
                # TODO: check to see if this makes sense for weight =0 in a real way.
                return -np.inf
            else:
                # this is the orc. This is the EMD/dist(u,v) and dist(u,v) will just be the weight of the edge
                return 1 - average_emd/weight
        else:
            print(
                f"No valid EMD computations were possible. For hyperedge {hyperedge_id} with EMD {emd}")
            return None

    def find_hyperedges_containing_all_nodes(self, *nodes) -> list:
        '''Find hyperedges that contain all of the specified nodes. This is used in approx emd calc

        :return list: list of hyperedges that contain all the specified nodes
        '''
        nodes_set = set(
            nodes)  # Convert list to set for efficient intersection checks

        # Handle different types of inputs
        for node in nodes:
            # If the input is any kind of collection
            if isinstance(node, (list, set, tuple)):
                nodes_set.update(node)  # Add all elements to the set
            else:
                nodes_set.add(node)  # Add the single element to the set

        found_hyperedges = []
        # Ensure all nodes in the set are in our nodes list
        if not nodes_set.issubset(self.nodes):
            print("Some nodes are not in the hypergraph.")

        # Iterate through all hyperedges
        for hyperedge_id, hyperedge_nodes in self.hyperedges.items():
            # Check if intersection contains all nodes we want
            if nodes_set.issubset(nodes_set.intersection(hyperedge_nodes)):
                found_hyperedges.append(hyperedge_id)

        return found_hyperedges

    def find_hyperedges_containing_nodes(self, *nodes) -> list:
        '''Find hyperedges that contain any of the specified nodes.

        :return list: list of hyperedges that contain any the specified nodes

        '''
        nodes_set = set(
            nodes)  # Convert list to set for efficient intersection checks

        # Handle different types of inputs
        for node in nodes:
            # If the input is any kind of collection
            if isinstance(node, (list, set, tuple)):
                nodes_set.update(node)  # Add all elements to the set
            else:
                nodes_set.add(node)  # Add the single element to the set

        found_hyperedges = []
        # Ensure all nodes in the set are in our nodes list
        if not nodes_set.issubset(self.nodes):
            print("Some nodes are not in the hypergraph.")

        # Iterate through all hyperedges
        for hyperedge_id, hyperedge_nodes in self.hyperedges.items():
            # Check if intersection is not empty
            if nodes_set.intersection(hyperedge_nodes):
                found_hyperedges.append(hyperedge_id)

        return found_hyperedges

    def neighbours(self, node) -> set:
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

        # Remove the node itself from the set of neighbours
        neighbours.discard(node)
        return neighbours

    def node_probability(self, node) -> dict:
        '''Calculate the probability distributions of a specific node. This will
        calculate 1/deg(u) spread out around the neighbors. But it's also a lazy 
        distribution.

        :param node: should pull up a node
        :raises ValueError: Will be raised if there is not a node of the name given in the graph
        :return dict: mu will be {node: dist} dictionary
        '''
        alpha = 0.1  # Self-transition probability factor
        # Initialize probabilities
        probability_distribution = {n: 0.0 for n in self.nodes}

        if node not in self.nodes:
            raise ValueError("Node does not exist in the hypergraph.")

        # Calculate the denominator: sum of (|f| - 1) for all f containing node
        denominator = 0
        hyperedges_containing_node = self.find_hyperedges_containing_nodes(
            node)  # the part that's no good for directed
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
            hyperedges_containing_both = self.find_hyperedges_containing_nodes(
                node, neighbour)
            numerator = len(hyperedges_containing_both)
            '''
            for hyperedge_id in hyperedges_containing_both:
                hyperedge = self.hyperedges[hyperedge_id]
                numerator += (len(hyperedge))
            '''
            # Update the probability of transitioning to the neighbor
            probability_distribution[neighbour] = (
                1 - alpha) * numerator / denominator

        # Assign the self-loop probability
        probability_distribution[node] = alpha

        # Normalization step
        total_probability = sum(probability_distribution.values())
        for n in probability_distribution:
            probability_distribution[n] /= total_probability

        return probability_distribution


class DirectedHypergraph(Hypergraph):
    def build_from_dataframe(self, df: pd.DataFrame, verbose=False) -> None:
        '''Build hypergraph from a DataFrame

        :param pd.DataFrame df: has the columns 'source', 'target', and 'weight'
        :param bool verbose: verbose flag, defaults to True
        '''
        # make an edge from each row in the csv
        # TODO: make actually work for hypergraphs
        for _, row in df.iterrows():
            node1 = row['source'].strip()  # start
            node2 = row['target'].strip()  # end
            weight = float(row['weight'])
            edgeid = node1 + '_to_' + node2
            self.add_hyperedge(edgeid, set([node1]), set([node2]), [weight])
            if verbose:
                print(
                    f'Added hyperedge {edgeid} with head set {node1} and tail set {node2}')
        return

    def add_hyperedge(self, hyperedge_id: str, tail_set: set, head_set: set, weight=[1], verbose=False) -> None:
        '''Function to add a hyperedge to the hypergraph, if the nodes are not 
        there, will add the nodes

        :param str hyperedge_id: the name you would like to be used for the hyperedge
        :param set tail_set: a list of the tail nodes (nodes leaving from)
        :param set head_set: a list of the head nodes (nodes going to)
        :param numeric weight: Is going to be the most recent weight
        :param bool verbose: verbose flag, defaults to True
        '''
        # Check if hyperedge already exists
        if hyperedge_id in self.hyperedges:
            print(
                f"Hyperedge {hyperedge_id} already exists with nodes {self.hyperedges[hyperedge_id]}")
            return

        # Add missing nodes to the node set
        for node in tail_set.union(head_set):
            if node not in self.nodes:
                self.add_node(node)

        # Add the hyperedge
        if verbose:
            f'Adding hyperedge {hyperedge_id} with tail nodes {tail_set} and head nodes {head_set}'
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id] = weight
        return

    def add_missing_edges_shortest_path(self, other_graph: Hypergraph, self_dist_mat: list[list], verbose: bool) -> None:
        '''So the idea, will be to add this edge, and then later delete it. Need to think about how to keep track of them
        For now, we're just testing on undirected. So will go forward on that.

        :param Hypergraph other_graph: The other graph we're working with. Should be of the same type as self.
        :param list[list] self_dist_mat: matrix of minimal distances from the floyd_warshall function
        :param bool verbose: verbose flag
        '''
        if not isinstance(other_graph, DirectedHypergraph):
            raise ValueError(
                "Trying to compare and Directed graph with something else")
        for e in set(other_graph.hyperedges) - set(self.hyperedges):
            (tail, head) = other_graph.hyperedges[e]
            # TODO: implement for hypegraphs
            dist = self_dist_mat[self.node_index[next(
                iter(tail))]][self.node_index[next(iter(head))]]
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

    def is_strongly_connected(self) -> bool:
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

        :param  node: the node we want to actually capture info for
        :raises ValueError: raises if the node doesn't exist in the graph
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

    def calculate_probability_distributions(self, hyperedge_id: str) -> dict:
        '''Function to calculate the probability distributions over all nodes based on the hyperedge

        :param str hyperedge_id: Name of the edge we're working with
        :return dict, dict: muA in and muB out
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
                                mu_A_in[nodes] += 1 / \
                                    (len(tail_set) * len(tail_set_prime) * deg_x_in)

                common_head_nodes = set(head_set) & set(tail_set_prime)
                if common_head_nodes:
                    for node in common_head_nodes:
                        deg_x_out = self.node_degree(node)[1]
                        for nodes in head_set_prime:
                            if deg_x_out != 0:
                                mu_B_out[nodes] += 1 / \
                                    (len(head_set) * len(head_set_prime) * deg_x_out)

        total_mass_A = sum(mu_A_in.values())
        total_mass_B = sum(mu_B_out.values())

        # Normalize the probability distributions
        if total_mass_A == 0:
            mu_A_in = {node: mass for node, mass in mu_A_in.items()}
        else:
            mu_A_in = {node: mass / total_mass_A for node,
                       mass in mu_A_in.items()}

        if total_mass_B == 0:
            mu_B_out = {node: mass for node, mass in mu_B_out.items()}
        else:
            mu_B_out = {node: mass / total_mass_B for node,
                        mass in mu_B_out.items()}

        return mu_A_in, mu_B_out
