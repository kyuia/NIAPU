from CreateGraph import create_graph_from_PPI  # 假设文档3保存为`文档3_module.py`

# 输入参数
path_to_ppi = "Datasets/BIOGRID-ALL-4.4.244.tab3.txt"  # PPI文件路径
disease_id = "C3714756"                         # 疾病ID（需与文件名匹配）
graph_name = f"grafo_nedbit_test_{disease_id}"        # 生成的图文件名
scale_features = True                           # 是否对特征标准化

# 执行生成
graph_path = create_graph_from_PPI(
    path_to_ppi, 
    disease_id, 
    graph_name, 
    scale=scale_features
)
print(f"图文件已生成：{graph_path}")