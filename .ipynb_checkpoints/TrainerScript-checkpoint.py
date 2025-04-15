# 导入必要的模块和类
from GNNTrain import train, predict_from_saved_model          # 导入GNN训练和预测函数
import CreateDatasetv2_binary
from CreateDatasetv2 import get_dataset_from_graph           # 导入多分类数据集创建函数
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS           # 导入路径配置
from GDARanking import (                                     # 导入基因关联分析相关函数
    get_ranking, predict_candidate_genes, 
    validate_with_extended_dataset, 
    get_ranking_no_LP_intersection, 
    validate_with_extended_dataset_no_LP
)
from GraphSageModel import GNN7L_Sage                         # 导入7层GraphSAGE模型定义

def trainGNN(disease_Id, mode='binary'):
    """训练GNN模型的主函数
    Args:
        disease_Id (str): 疾病标识符(如C0009402)
        mode (str): 训练模式，可选 'binary'(二分类) 或 'multiclass'(多分类)，默认为二分类
    """
    # 初始化配置参数
    classes = ['P', 'LP', 'WN', 'LN', 'RN']                   # 多分类标签类别
    model_name = f'GraphSAGE_{disease_Id}_{mode}'       # 基础模型名称格式
    graph_path = PATH_TO_GRAPHS + 'grafo_nedbit_test_' + disease_Id + '.gml'  # 图数据路径

    # 根据模式选择数据集生成方式
    dataset = None  # 数据集对象
    G = None        # NetworkX图对象
    
    if mode == 'binary':
        # 二分类模式配置
        classes = ['P', 'U']                                  # 二分类标签(Positive/Unlabeled)
        #model_name += '_binary'                               # 模型名称添加后缀
        # 生成二分类数据集(使用特殊处理的数据集模块)
        dataset, G = CreateDatasetv2_binary.get_dataset_from_graph(
            graph_path, disease_Id, quartile=False)
    else:
        # 多分类模式配置
        # 生成多分类数据集(使用标准数据集创建函数)
        dataset, G = get_dataset_from_graph(
            graph_path, disease_Id, 
            quartile=False,     # 不使用四分位数划分标签
        )

    # 设置训练超参数
    lr = 0.001                # 学习率
    epochs = 40000            # 训练轮次
    weight_decay = 0.0005     # L2正则化系数

    # 初始化模型
    model = GNN7L_Sage(dataset)  # 创建7层GraphSAGE模型实例

    # 执行训练流程
    preds = train(
        model, dataset, 
        epochs, lr, weight_decay,
        classes,              # 类别标签列表
        model_name,            # 模型保存名称
    )

if __name__ == '__main__':
    """主程序入口"""
    # 待训练的疾病标识列表 (示例列表，实际可根据需要修改)
    disease_Ids = ['C0376358','C0860207','C3714756']  # 原始列表
    #disease_Ids = ['C0001973','C0005586','C0006142',]  # 训练过的测试列表
    
    # 遍历所有疾病标识进行训练
    for disease_Id in disease_Ids:
        print('[+] Training', disease_Id)
        
        # 二分类训练
        trainGNN(disease_Id, mode='binary')
        
        # 多分类训练
        trainGNN(disease_Id, mode='multiclass')