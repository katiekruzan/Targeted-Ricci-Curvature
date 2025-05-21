import pandas as pd
import csv
import numpy as np
from itertools import combinations
from gurobipy import Model, GRB, quicksum, LinExpr
import time
import os
from numbers import Number
import networkx as nx

now = time.time()

class Hypergraph:
    def __init__(self):       
        self.nodes = set()
        self.hyperedges = {}
        self.weights = {}
        self.ricci_curvature = {}
        self.node_index = {}

    def add_node(self, node: any) -> None:
        self.nodes.add(node)

    def add_ricci_curvature(self, hyperedge_id: str, orc: float) -> None:
        if hyperedge_id not in self.ricci_curvature:
            self.ricci_curvature[hyperedge_id] = []
        self.ricci_curvature[hyperedge_id].append(orc)


def update_orc_and_weights_iter(hypergraph, dist_matrix, verbose=False):
    max_dist = max([dist for node_dists in dist_matrix.values() for dist in node_dists.values() if dist < float('inf')])
    updated_weights = {}

    for hyperedge_id, nodes in hypergraph.hyperedges.items():
        if len(nodes) < 2:
            continue

        u, v = nodes[0], nodes[1]
        if u not in dist_matrix or v not in dist_matrix[u]:
            continue

        d_uv = dist_matrix[u][v]
        orc = 1 - d_uv / max_dist if max_dist > 0 else 0
        hypergraph.add_ricci_curvature(hyperedge_id, orc)

        normalized_orc = orc
        weight = hypergraph.weights[hyperedge_id]
        if weight != 0:
            step = 1
            wtplus1 = weight * (1 - step * normalized_orc)
        else:
            wtplus1 = weight

        updated_weights[hyperedge_id] = max(wtplus1, 1e-4)

    for hyperedge_id, new_weight in updated_weights.items():
        hypergraph.weights[hyperedge_id] = new_weight


def one_direction_of_work(source_file, verbose=False):
    source_graph = Hypergraph()
    build_graph_from_csv(source_file, source_graph)

    for _ in range(100):
        dist_matrix = compute_distance_matrix(source_graph)
        update_orc_and_weights_iter(source_graph, dist_matrix, verbose)

    final_dist_matrix = compute_distance_matrix(source_graph)
    return final_dist_matrix, source_graph


def build_graph_from_csv(path, hypergraph):
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        u, v = int(row[0]), int(row[1])
        edge_id = f"{u}-{v}"
        hypergraph.nodes.update([u, v])
        hypergraph.hyperedges[edge_id] = [u, v]
        hypergraph.weights[edge_id] = 1.0


def compute_distance_matrix(hypergraph):
    G = nx.Graph()
    for edge_id, (u, v) in hypergraph.hyperedges.items():
        G.add_edge(u, v, weight=hypergraph.weights[edge_id])
    return dict(nx.floyd_warshall(G, weight='weight'))


def build_distance_vectors(dist_dict, nodes_order):
    vectors = []
    for src in nodes_order:
        vec = [dist_dict[src].get(tgt, float('inf')) for tgt in nodes_order]
        vectors.append(vec)
    return np.array(vectors)


def earthmover_distance_gurobi_distance_matrix(a, b):
    assert len(a) == len(b), "Vectors must be the same length"
    n = len(a)

    model = Model("emd")
    model.setParam('OutputFlag', 0)

    f = {}
    for i in range(n):
        for j in range(n):
            f[i, j] = model.addVar(lb=0, name=f"f_{i}_{j}")

    model.update()

    for i in range(n):
        model.addConstr(quicksum(f[i, j] for j in range(n)) == a[i])
    for j in range(n):
        model.addConstr(quicksum(f[i, j] for i in range(n)) == b[j])

    cost_expr = quicksum(abs(i - j) * f[i, j] for i in range(n) for j in range(n))
    model.setObjective(cost_expr, GRB.MINIMIZE)
    model.optimize()

    return model.ObjVal


def solve_assignment(D1, D2):
    n, m = len(D1), len(D2)
    model = Model("emd_alignment")
    model.setParam('OutputFlag', 0)

    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(m)) == 1)
    for j in range(m):
        model.addConstr(quicksum(x[i, j] for i in range(n)) == 1)

    cost_expr = LinExpr()
    for i in range(n):
        for j in range(m):
            cost = earthmover_distance_gurobi_distance_matrix(D1[i], D2[j])
            cost_expr += cost * x[i, j]

    model.setObjective(cost_expr, GRB.MINIMIZE)
    model.optimize()

    matching = [(i, j) for i in range(n) for j in range(m) if x[i, j].X > 0.5]
    return matching


def main():
    path1 = "G1.csv"
    path2 = "G2.csv"

    dist_G1, G1 = one_direction_of_work(path1)
    dist_G2, G2 = one_direction_of_work(path2)

    order_G1 = sorted(G1.nodes)
    order_G2 = sorted(G2.nodes)

    D1 = build_distance_vectors(dist_G1, order_G1)
    D2 = build_distance_vectors(dist_G2, order_G2)

    mapping = solve_assignment(D1, D2)

    print("Alignment between G1 and G2:")
    for i, j in mapping:
        print(f"Node {order_G1[i]} in G1 matched to Node {order_G2[j]} in G2")


if __name__ == "__main__":
    main()
