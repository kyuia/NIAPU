from CreateDatasetv2 import get_dataset_from_graph
import torch
import networkx as nx
import os

# 输入参数
disease_id = "C0006142"        # 疾病ID
mode = "binary"                # 可选：binary/multiclass
quartile = False               # 是否用四分位数划分非种子基因
verbose = True                 # 打印进度信息

# 动态生成路径 ---------------------------------------------------
# 输入图路径
path_to_graph = f"Graphs/grafo_nedbit_test_{disease_id}.gml" 

# 输出路径（自动创建目录）
output_dir = f"Datasets/{disease_id}"
os.makedirs(output_dir, exist_ok=True)

# PyG数据集保存路径
dataset_save_path = f"{output_dir}/{disease_id}_{mode}_dataset.pt" 

# 原始图保存路径（可选）
original_graph_save_path = f"Graphs/{disease_id}_original.gml" 

# 调用函数生成数据集 --------------------------------------------
dataset, G = get_dataset_from_graph(
    path_to_graph=path_to_graph,
    disease_id=disease_id,
    mode=mode,
    quartile=quartile,
    verbose=verbose
)

# 保存结果 ------------------------------------------------------
# 保存PyG数据集
torch.save(dataset, dataset_save_path)  
print(f"[+] 数据集已保存至: {dataset_save_path}")

# 保存原始图（可选）
nx.write_gml(G, original_graph_save_path)  
print(f"[+] 原始图已备份至: {original_graph_save_path}")