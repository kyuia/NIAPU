import os
import numpy as np
import pandas as pd
import networkx as nx
from time import perf_counter
from sklearn.preprocessing import RobustScaler
from Paths import PATH_TO_DATASETS, PATH_TO_GRAPHS

def create_graph_from_PPI(path_to_PPI, disease_id, graph_name, scale=False):
    """基于PPI数据创建带NeDBIT特征的图结构"""
    t_start = perf_counter()
    
    # ----------------- 1.图结构构建阶段 -----------------
    print('[+] Reading PPI...', end='')
    biogrid = pd.read_csv(path_to_PPI, sep='\t', low_memory=False)
    biogrid = biogrid[(biogrid['Organism ID Interactor A'] == 9606) 
                    & (biogrid['Organism ID Interactor B'] == 9606)]
    print('ok')

    print('[+] Creating the graph...', end='')
    G = nx.Graph()
    for _, row in biogrid.iterrows():
        p1 = row['Official Symbol Interactor A'].replace('-', '_').replace('.', '_')
        p2 = row['Official Symbol Interactor B'].replace('-', '_').replace('.', '_')
        G.add_edge(p1, p2)
    print(f'ok (Nodes: {len(G.nodes)}, Edges: {len(G.edges)})')

    # ----------------- 2.图结构清理阶段 -----------------
    print('[+] Removing self loops...', end='')
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'ok (Edges: {len(G.edges)})')

    print('[+] Taking the LCC...', end='')
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    print(f'ok (Nodes: {len(G.nodes)}, Edges: {len(G.edges)})')

    # ----------------- 3.特征映射阶段 -----------------
    print('[+] Loading NeDBIT features...', end='')
    FEATURE_COLS = ['degree', 'ring', 'NetRank', 'NetShort', 'HeatDiff', 'InfoDiff']
    nedbit_path = os.path.join(PATH_TO_DATASETS, f"{disease_id}_features")
    nedbit_df = pd.read_csv(nedbit_path).set_index('name')
    print('ok')

    print('[+] Mapping node features...')
    missing_nodes = []
    for node in G:
        try:
            features = nedbit_df.loc[node]
            for col in FEATURE_COLS:
                G.nodes[node][col] = features[col]
        except KeyError:
            missing_nodes.append(node)
    
    if missing_nodes:
        print(f'[WARN] {len(missing_nodes)} nodes missing features: {missing_nodes[:3]}...')
        G.remove_nodes_from(missing_nodes)
        print(f'Removed {len(missing_nodes)} nodes, final graph: {len(G.nodes)} nodes')

    # ----------------- 特征标准化阶段 -----------------
    if scale:
        print('[+] Normalizing features...', end='')
        feature_matrix = np.array([[
            G.nodes[node][col] for col in FEATURE_COLS
        ] for node in G.nodes])
        
        scaler = RobustScaler().fit(feature_matrix)
        scaled_features = scaler.transform(feature_matrix)
        
        for i, node in enumerate(G.nodes):
            for j, col in enumerate(FEATURE_COLS):
                G.nodes[node][col] = scaled_features[i][j]
        print('ok')

    # ----------------- 结果保存阶段 -----------------
    graph_path = os.path.join(PATH_TO_GRAPHS, f"{graph_name}.gml")
    print(f'[+] Saving graph to {graph_path}')
    nx.write_gml(G, graph_path)
    
    print(f'[i] Total time: {perf_counter()-t_start:.2f}s')
    return graph_path