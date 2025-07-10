import pandas as pd
import csv
import numpy as np
from itertools import combinations
from gurobipy import Model, GRB, quicksum, LinExpr,Env
import time
import os
from numbers import Number
import networkx as nx
import pprint

# now = time.time()

class Hypergraph:
    # NOTE: Should just be able to use networkx native set up?
    def __init__(self):       
        self.nodes = set()
        self.hyperedges = {}
        self.weights = {}
        self.ricci_curvature = {}
        self.node_index = {}

    def add_node(self, node: any) -> None:
        self.nodes.add(node)
        
    def get_total_weight(self):
        return sum(self.weights.values())

    def add_ricci_curvature(self, hyperedge_id: str, orc: float) -> None:
        if hyperedge_id not in self.ricci_curvature:
            self.ricci_curvature[hyperedge_id] = []
        self.ricci_curvature[hyperedge_id].append(orc)
        
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
    
    def earthmover_distance_gurobi_distance_matrix(self, data, distance_matrix, verbose=False):
        '''Trying to combine the two functions into one
        '''
        # error handling
        # check data is node A node B
        node_A, node_B = data
        if node_A not in self.nodes or node_B not in self.nodes:
            print(f"Node {node_A} or {node_B} does not exist in the Undirected hypergraph.")
            return None  # Return None if either node does not exist
        mu_A = self.node_probability(node_A)
        mu_B = self.node_probability(node_B)
        
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
        
        env = Env()
        env.setParam("OutputFlag",0)
        env.start()
        
        try:
            # Create a new model in Gurobi.
            model = Model("EarthMoverDistance", env=env)
            
            # Create variables for the linear program.
            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0) 
            
            # # Make it less verbose
            # model.Params.LogToConsole = 0    
            
            expr = LinExpr(3.0)
            expr.clear()
            
            for x in mu_A:
                for y in mu_B:
                    expr.addTerms(distance_matrix[x][y], variables[x,y])
                    
            
            # Set the objective of the linear program to minimize the total cost.
            model.setObjective(expr, GRB.MINIMIZE)
                        
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
                print(f"No optimal solution found for data {data}")
                print(f"The probability distributions are A: {mu_A} \nAnd B: {mu_B}")
                print('Model Status', model.status)
                print(f"The masses are {total_mass_A} for A and {total_mass_B} for B")
                return None
            
        except Exception as e:
            print(f"Gurobi Error: {e}\n for data {data}")
            return None 


def update_orc_and_weights_iter(hypergraph:Hypergraph, dist_matrix, iteration, graphname, verbose=False):
    file_name = f'outputfiles/dataset_targeted_curvature_{graphname}_iteration_{iteration}.csv'
    # max_dist = max([dist for node_dists in dist_matrix.values() for dist in node_dists.values() if dist < float('inf')])
    updated_weights = {}
    # NOTE: for right now am going to make this 1
    # totweight = hypergraph.get_total_weight() 
    totweight = 1.0
    
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Hyperedge ID', 'ORC: (based on t-1 weights)', 'Weight:t'])

        for hyperedge_id, nodes in hypergraph.hyperedges.items():
            if len(nodes) < 2: #hyper edges with less than 2 edges
                continue

            u, v = nodes[0], nodes[1] # Notably will only work for graphs, not hyper graphs
            if u not in dist_matrix or v not in dist_matrix[u]:
                continue

            # d_uv = dist_matrix[u][v] # Is dist_matrix symmetric?
            #TODO: This is in need of a major fix
            # need 1- EMD
            orc = 1 - hypergraph.earthmover_distance_gurobi_distance_matrix((u, v), dist_matrix) # This is not correct..
            # Normalize the curvature
            normalized_orc = ricci_normalizing(orc)
            # print(f'Edge {hyperedge_id} with ORC {orc}\nNormalized ORC: {normalized_orc}')
            
            hypergraph.add_ricci_curvature(hyperedge_id, normalized_orc)

            weight = hypergraph.weights[hyperedge_id]
            if iteration != 0:
                if weight != 0:
                    step = 1
                    wtplus1 = weight * (1 - step * normalized_orc) 
                else:
                    wtplus1 = weight
                    
                # NOTE:The only thing here, is I *think* weights are allowed to be 0
                updated_weights[hyperedge_id] = max(wtplus1, 1e-4) 
            else: updated_weights[hyperedge_id] = weight
        
        for hyperedge_id, new_weight in updated_weights.items():
            # Will also normalize the weights
            newtotwt = sum(updated_weights.values())
            normfactor = totweight/newtotwt
            # normalized_wt = normfactor * new_weight
            normalized_wt = new_weight
            writer.writerow([hyperedge_id, hypergraph.ricci_curvature[hyperedge_id][-1], normalized_wt])
            # NOTE: Deleting old weights. which probably makes sense. Other than to check convergence maybe
            hypergraph.weights[hyperedge_id] = normalized_wt
    


def one_direction_of_work(source_file, graphname, verbose=False):
    """
    Run Ricci flow a bunch of times and return the final dist matrix and graph
    """
    source_graph = Hypergraph()
    build_graph_from_csv(source_file, source_graph)
    
    source_graph.node_index = {node: idx for idx, node in enumerate(list(source_graph.nodes))}

    for i in range(ITTS):
        dist_matrix = compute_distance_dict(source_graph)
        update_orc_and_weights_iter(source_graph, dist_matrix, i, graphname, verbose)
        clock_time(f'Time for ORC {i}')
        # if i == 2:
        #     quit()
        
        allstable = True
        if i <2: 
            # Take care of the getting started case
            continue
        errorlist = []
        for e in source_graph.hyperedges:
            clist = source_graph.ricci_curvature[e]
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
                    allstable = False
                    break
                if (not absolute_change) and (error > 0.05): # relative change
                    clock_time(f'unstable for edge {e} with error {error}')
                    allstable = False
                    break
            else:
                errorlist.append(error)
        if not maximum_error: # AKA we're in average error zone
            # assumed to be in absolute error zone
            avg_err = np.average(errorlist)
            if avg_err > 0.0001:
                clock_time(f'unstable with average error {avg_err}')
                allstable = False
        if allstable:
            print('STABILIZED! Source to target distance is ',i)
            write_scorecard('\n\n----- Results -----')
            write_scorecard(f'Source to target distance is {i}')
            break
        # quit()
    
    #TODO: This has no convergence check. Just running it 100 times. Will need to check for convergence

    final_dist_matrix = compute_distance_dict(source_graph)
    return final_dist_matrix, source_graph


def build_graph_from_csv(path, hypergraph):
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        u, v, weight = str(row[0]), str(row[1]), float(row[2])
        edge_id = f"{u}_to_{v}"
        hypergraph.nodes.update([u, v])
        hypergraph.hyperedges[edge_id] = [u, v]
        hypergraph.weights[edge_id] = weight


def compute_distance_dict(hypergraph):
    G = nx.Graph()
    for edge_id, (u, v) in hypergraph.hyperedges.items():
        G.add_edge(u, v, weight=hypergraph.weights[edge_id])
    fw = nx.floyd_warshall(G, weight='weight')
    return {a:dict(b) for a, b in fw.items()}


def build_distance_vectors(dist_dict, nodes_order):
    vectors = []
    for src in nodes_order:
        vec = [dist_dict[src].get(tgt, float('inf')) for tgt in nodes_order]
        vectors.append(vec)
    return np.array(vectors)


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

def ricci_normalizing(R: float)->float: 
    '''
    Using the normalization function sigma(R)/sigma(1) 
    Where sigma(x) is the standard sigmoid function 1/(1+\exp(-x))

    :param float R: the ORC value to be normalized 
    :return float: The normalized ORC value
    ''' 
    return ((1 - np.exp(-1))/(1+ np.exp(-R)))


def main():
    
    clean_output(False)
    
    path1 = "inputfiles/ERgraph100nodep4.csv"
    path2 = "inputfiles/ERgraph100n100changenewrange1000to2000v3.csv"
    
    # Scorecard Writing
    write_scorecard('----- Targeted Ricci Curvature -----')
    write_scorecard(f'Graph 1 filename: {path1}')
    write_scorecard(f'Graph 2 filename: {path2}')
    
    if absolute_change:
        write_scorecard('Absolute vs Relative change: absolute change')
    else:
        write_scorecard('Absolute vs Relative change: relative change')
    if maximum_error:
        write_scorecard('Max vs Avg error: Maximum')
    else: 
        write_scorecard('Max vs Avg error: Avg')
    
    clock_time('Time to read the data in seconds')

    dist_G1, G1 = one_direction_of_work(path1, "Graph1")
    write_scorecard('Finished the first graph')
    dist_G2, G2 = one_direction_of_work(path2, "Graph2")
    write_scorecard('Finished the second graph')

    order_G1 = sorted(G1.nodes) # Might need to think through if there are different nodes? Otherwise, should just be able to use order_G1 for both?
    # order_G2 = sorted(G2.nodes)

    D1 = build_distance_vectors(dist_G1, order_G1)
    D2 = build_distance_vectors(dist_G2, order_G1)
    #TODO: Persist the ending vectors
    np.savetxt("outputfiles/Graph1FinalDistance.txt", D1, delimiter=" ", fmt="%f")  
    np.savetxt("outputfiles/Graph2FinalDistance.txt", D2, delimiter=" ", fmt="%f")  
    # print(D1)
    # print(D2)
    
    write_scorecard('The final distance:')
    write_scorecard(str(np.linalg.norm(D1 - D2)))
    write_scorecard("New Distance")
    dist = 0
    for v in range(len(D1)):
        top = np.dot(D1[v], D2[v])
        bot = np.linalg.norm(D1) * np.linalg.norm(D2)
        # print(top/bot)
        dist = dist + (top/bot)
    write_scorecard(str(dist))
    # print(np.linalg.norm(D1 - D2))

    # I don't think we need to match them here. Since we have known node correspondence.
    # mapping = solve_assignment(D1, D2)

    #NOTE: The problem is we need to find a distance. So the mapping section should be moot. 
    # In theory we're just finding the distance between the 2 vectors (which should be quick to compute w/ numpy or similar.)
    # print("Alignment between G1 and G2:")
    # for i, j in mapping:
    #     print(f"Node {order_G1[i]} in G1 matched to Node {order_G2[j]} in G2")


if __name__ == "__main__":
    ITTS = 100
    start = time.time()
    
    absolute_change = False # False is relative change
    maximum_error = True # False is average error
    
    main()
