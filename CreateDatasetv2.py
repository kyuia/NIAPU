### 模块文件
# 导入必要的库
import torch
import numpy as np
import pandas as pd
import networkx as nx
from time import perf_counter  # 用于精确计时

# PyTorch Geometric相关导入
from torch_geometric.utils import from_networkx  # 将networkx图转换为PyG数据格式
from torch_geometric.data import InMemoryDataset  # 用于创建内存数据集

# 数据划分工具
from sklearn.model_selection import train_test_split

# 自定义路径配置
from Paths import PATH_TO_DATASETS  # 从自定义路径文件导入数据集路径

# 设置随机种子保证可重复性
torch.manual_seed(42)

class MyDataset(InMemoryDataset):
    """自定义PyG数据集类，用于处理图神经网络数据"""
    def __init__(self, G, labels, attributes, num_classes=2):
        """
        初始化数据集
        Args:
            G (networkx.Graph): 输入的网络图
            labels (np.array): 节点标签数组
            attributes (list): 要保留的节点属性列表
            num_classes (int): 类别数量，默认为2分类
        """
        super(MyDataset, self).__init__('.', None, None, None)  # 初始化父类

        # 从networkx图转换数据，保留指定属性
        try:
            # attributes 参数指定了要从 NetworkX 节点属性中提取哪些作为节点特征
            data = from_networkx(G, attributes)
        except KeyError as e:
            # 如果 G.nodes[node] 中缺少 attributes 列表中的某个属性，这里会报错
            print(f"\n[错误] 从NetworkX图创建PyG数据时发生 KeyError: {e}")
            print(f"请确保图 G 中的节点确实包含所有指定的属性: {attributes}")
            print(f"这些属性应该由 CreateGraph.py 脚本添加并保存在 GML 文件中。")
            raise  # 重新抛出异常，因为无法继续
        except Exception as e:
            print(f"\n[错误] 从NetworkX图创建PyG数据时发生未知错误: {e}")
            raise
            
        # 将标签转换为torch长整型张量
        y = torch.from_numpy(labels).type(torch.long)

        # 设置数据属性
        data.x = data.x.float()         # 特征矩阵转换为浮点型
        data.y = y.clone().detach()     # 标签张量
        data.num_classes = num_classes  # 类别数

        # 使用分层抽样划分训练集、测试集、验证集
        indices = range(G.number_of_nodes())  # 生成节点索引
        try:
            # 第一次划分：训练集（70%）和临时测试集（30%）
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                data.x, data.y, indices, 
                test_size=0.3, 
                stratify=labels,  # 保持类别分布
                random_state=42
            )
        
            # 第二次划分：将临时测试集分为验证集（15%）和最终测试集（15%）
            X_test, X_val, y_test, y_val, test_idx, val_idx = train_test_split(
                X_test, y_test, test_idx,
                test_size=0.5,
                stratify=y_test,
                random_state=42
            )
            print("[+]数据集划分使用分层抽样")
        except ValueError as e:
            # 如果分层抽样失败 (通常因为某个类别样本太少)
            print(f"[警告] 分层抽样失败: {e}。将回退到随机抽样。")
            # 执行非分层抽样
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                data.x, data.y, indices, test_size=0.3, stratify=None,random_state=42
            )
            X_val, X_test, y_val, y_test, val_idx, test_idx = train_test_split(
                X_test, y_test, test_idx, test_size=0.5, stratify=None,random_state=42
            )
            
        # 初始化掩码张量
        n_nodes = G.number_of_nodes()
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        # 填充训练集掩码
        for idx in train_idx:
            train_mask[idx] = True

        # 填充测试集掩码
        for idx in test_idx:
            test_mask[idx] = True
        
        # 填充验证集掩码
        for idx in val_idx:
            val_mask[idx] = True

        # 将掩码添加到数据对象
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        data['val_mask'] = val_mask

        # 处理数据为PyG需要的格式
        self.data, self.slices = self.collate([data])


def get_dataset_from_graph(path_to_graph, disease_id, verbose=True, quartile=True):
    """
    从图文件创建数据集
    Args:
        path_to_graph (str): 图文件路径
        disease_id (str): 疾病ID
        verbose (bool): 是否打印过程信息
        quartile (bool): 是否使用四分位数划分标签
    Returns:
        tuple: (PyG数据对象, 原始networkx图)
    """
    t_start = perf_counter()  # 开始计时

    # 1. 读取图数据 ------------------------------------------------------------
    if verbose: print('[+] Reading graph...', end='')
    G = nx.read_gml(path_to_graph)
    if verbose: print('ok')

    # 2. 准备数据集 ------------------------------------------------------------
    if verbose: print('[+] Creating dataset...', end='')
    
    # 设置文件路径
    path = PATH_TO_DATASETS
    path_to_seed_genes = path + disease_id + '_seed_genes.txt'

    # 读取种子基因数据
    seed_genes = pd.read_csv(path_to_seed_genes, header=None, sep=' ')

    # 处理列名
    seed_genes.columns = ["name", "GDA Score"]
    seeds_list = seed_genes["name"].values.tolist()  # 转换为列表

    # 3. 读取NedBit分数 --------------------------------------------------------
    nedbit_scores = pd.read_csv(path + disease_id + '_ranking', sep=' ', header=None)
    nedbit_scores.columns = ["name", "out", "label"]
    
    # 数据清洗：统一基因名称格
    # 替换各种格式变体为统一格式
    nedbit_scores['name'] = nedbit_scores['name'].str.replace("ORF", 'orf')
    nedbit_scores['name'] = nedbit_scores['name'].str.replace("Morf", 'MORF')
    nedbit_scores['name'] = nedbit_scores['name'].str.replace("^orf1$", 'ORF1', regex=True)
    nedbit_scores['name'] = nedbit_scores['name'].str.replace("SERF2_C15orf63", 'SERF2_C15ORF63')
    nedbit_scores['name'] = nedbit_scores['name'].str.replace("LOC100499484_C9orf174", 'LOC100499484_C9ORF174')

    # 4. 标签处理 --------------------------------------------------------------
    if not quartile:
        # 直接映射现有标签
        nedbit_scores["label"].replace(to_replace=1, value="P", inplace=True)   # Positive
        nedbit_scores["label"].replace(to_replace=2, value="LP", inplace=True)  # Likely Positive
        nedbit_scores["label"].replace(to_replace=3, value="WN", inplace=True)  # Weakly Negative
        nedbit_scores["label"].replace(to_replace=4, value="LN", inplace=True)  # Likely Negative
        nedbit_scores["label"].replace(to_replace=5, value="RN", inplace=True)  # Robustly Negative

        # 创建节点到标签的映射字典
        nodes_labels = dict(zip(nedbit_scores['name'], nedbit_scores['label']))

        # 定义标签编码字典
        labels_dict = {'P':0, 'LP': 1, 'WN': 2, 'LN': 3, 'RN': 4}
        # 在标签生成循环中添加异常捕获：
        labels = []
        for node in G:
            try:
                labels.append(labels_dict[nodes_labels[node]])
            except KeyError:
                print(f"[错误] 节点 {node} 在标签字典中不存在！")
                raise

    else:
        # 使用四分位数划分标签（排除种子基因）
        # 过滤掉种子基因
        nedbit_scores_not_seed = nedbit_scores[~nedbit_scores['name'].isin(seeds_list)]
        
        # 按分数降序排序
        nedbit_scores_not_seed = nedbit_scores_not_seed.sort_values(by="out", ascending=False)
        
        # 使用分位数划分（4分位数创建5个类别）
        pseudo_labels = pd.qcut(x=nedbit_scores_not_seed["out"], q=4, labels=["RN", "LN", "WN", "LP"])
        nedbit_scores_not_seed['label'] = pseudo_labels

        # 处理种子基因标签
        nedbit_scores_seed = nedbit_scores[nedbit_scores['name'].isin(seeds_list)]
        nedbit_scores_seed = nedbit_scores_seed.assign(label='P')  # 种子基因标记为P

        # 合并标签字典
        not_seed_labels = dict(zip(nedbit_scores_not_seed['name'], nedbit_scores_not_seed['label']))
        seed_labels = dict(zip(nedbit_scores_seed['name'], nedbit_scores_seed['label']))

        # 生成最终标签列表
        labels_dict = {'P':0, 'LP':1, 'WN':2, 'LN':3, 'RN':4}
        labels = []
        for node in G:
            if node in not_seed_labels:
                labels.append(labels_dict[not_seed_labels[node]])
            else:  # 处理种子基因
                labels.append(labels_dict[seed_labels[node]])

    # 转换为numpy数组
    labels = np.asarray(labels)

    # 5. 定义使用的节点属性 -----------------------------------------------------
    attributes = ['degree', 'ring', 'NetRank', 'NetShort', 'HeatDiff', 'InfoDiff']

    # 6. 创建PyG数据集 ---------------------------------------------------------
    dataset_with_nedbit = MyDataset(G, labels, attributes, num_classes=5)
    if verbose: print('ok')

    data_with_nedbit = dataset_with_nedbit[0]  # 获取数据对象
    
    # 7. 计时结束 --------------------------------------------------------------
    t_end = perf_counter()
    if verbose: print('[i] Elapsed time:', round(t_end - t_start, 3))

    return data_with_nedbit, G  # 返回数据和原始图对象