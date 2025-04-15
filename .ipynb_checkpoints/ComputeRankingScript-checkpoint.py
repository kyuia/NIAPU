# 导入必要的模块和类
from GNNTrain import predict_from_saved_model  # 从训练模块导入预测函数
from CreateDatasetv2 import get_dataset_from_graph  # 多分类数据集处理函数
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS, PATH_TO_MODELS  # 路径配置文件
from GDARanking import predict_candidate_genes  # 基因关联预测函数
import CreateDatasetv2_binary  # 二分类数据集处理模块

import os  # 系统路径操作
import sys  # 命令行参数处理
from time import perf_counter  # 精确计时

# 可处理的疾病ID列表（测试用简化列表）
disease_Ids = ['C0001973']
#已处理[, 'C0006142','C0009402'],'C0011581','C0023893''C0023893','C0036341',,'C0860207','C3714756'

# 支持的基因关联分析方法列表
methods = ['gnnexplainer',  'gnnexplainer_only']
#          'graphsvx',      'graphsvx_only',              'subgraphx',     'subgraphx_only'
def check_args(args):
    """验证命令行参数有效性
    Args:
        args (list): 命令行参数列表
    Returns:
        tuple|int: 验证通过返回(疾病ID, 方法名, 核心数)，失败返回-1
    """
    # 参数数量校验
    if len(args) < 3:
        if len(args) == 2 and (args[1] in ('-h', '--help')):  # 帮助信息
            print('\n\n[Usage]: python ComputeRankingScript.py disease_id explainability_method num_cores\n')
            print('可用疾病ID:\n', disease_Ids, '\n输入"all"处理所有疾病\n')
            print('可用方法:\n', methods, '\n输入"all"使用所有方法\n')
            print('num_cores: 指定并行计算使用的CPU核心数\n\n')
        else:
            print('\n\n[错误] 参数错误: 使用 -h 或 --help 查看帮助\n\n')
        return -1

    # 参数解析
    disease_Id = args[1]    # 疾病标识符
    METHOD = args[2]        # 解释方法名称
    num_cpus = int(args[3]) # CPU核心数

    # 疾病ID有效性检查
    if disease_Id not in disease_Ids and disease_Id.lower() != 'all':
        print(f'\n[错误] 疾病ID {disease_Id} 不存在\n')
        return -1

    # 方法名称有效性检查
    if METHOD not in methods and METHOD.lower() != 'all':
        print(f'\n[错误] 方法 {METHOD} 不支持\n')
        return -1

    # CPU核心数有效性检查
    if num_cpus < 1:
        print(f'\n[错误] 无效的核心数 {num_cpus}\n')
        return -1
    
    return disease_Id, METHOD, num_cpus

def ranking(disease_Id, METHOD, num_cpus, filename, modality='multiclass'):
    """执行基因关联排名计算流程
    Args:
        disease_Id (str): 疾病标识符
        METHOD (str): 解释方法名称
        num_cpus (int): 并行计算核心数
        filename (str): 结果保存路径
        modality (str): 模式类型，'binary'或'multiclass'
    """
    # 模型配置
    model_name = f'GraphSAGE_{disease_Id}'  # 基础模型名称
    graph_path = f'{PATH_TO_GRAPHS}grafo_nedbit_{disease_Id}.gml'  # 图数据路径
    classes = ['P', 'LP', 'WN', 'LN', 'RN']  # 多分类标签

    # 根据模式调整配置
    if modality == 'binary':
        model_name += '_binary'  # 二分类模型后缀
        classes = ['P', 'U']     # 二分类标签(Positive/Unlabeled)
        # 加载二分类数据集
        dataset, G = CreateDatasetv2_binary.get_dataset_from_graph(
            graph_path, disease_Id, quartile=False)
    else:
        # 加载多分类数据集
        dataset, G = get_dataset_from_graph(
            graph_path, disease_Id, 
            quartile=False, 
        )
        print(f"[DEBUG] 数据集标签是否存在: {'y' in dir(dataset)}")  # 应为 True
        print(f"[DEBUG] 测试集掩码是否存在: {'test_mask' in dir(dataset)}")  # 应为 True
    # 模型名称追加训练参数和模式信息
    epochs = 40000
    weight_decay = 0.0005
    model_name += f'_{modality}_{epochs}_{weight_decay:.4f}'.replace('.', '_')
    
    # 打印当前加载的模型名称
    print(f"[DEBUG] 正在加载模型: {model_name}")
    print(f"[DEBUG] 模型预期路径: {os.path.join(PATH_TO_MODELS, model_name)}")

    # 加载训练好的模型进行预测
    preds, probs, model = predict_from_saved_model(
        disease_id=disease_Id,        # 显式赋值给disease_id参数
        mode=modality,               # 显式赋值给mode参数  
        data=dataset,                # 显式赋值给data参数
        classes=classes,              # 显式赋值给classes参数
        save_to_file=True
    )
    # 生成基因关联排名
    ranking = predict_candidate_genes(
        model=model,
        dataset=dataset,
        predictions=preds,
        explainability_method=METHOD,
        disease_Id=disease_Id,
        explanation_nodes_ratio=1,  # 解释节点比例
        masks_for_seed=10,          # 种子掩码数量
        num_hops=1,                 # 网络跳数
        G=G,                        # NetworkX图对象
        num_pos="all",              # 考虑所有正样本
        num_workers=num_cpus        # 并行工作进程数
    )

    # 保存结果到文件
    print(f'[+] 保存排名结果到 {filename}', end='...')
    with open(filename, 'w') as f:
        for line in ranking:
            f.write(line + '\n')
    print('完成')

def sanitized_input(prompt, accepted_values):
    """安全获取用户输入
    Args:
        prompt (str): 提示信息
        accepted_values (list): 可接受的值列表
    Returns:
        str: 用户输入的标准结果
    """
    while True:
        res = input(prompt).strip().lower()
        if res in accepted_values:
            return res
        print(f"无效输入，请选择: {accepted_values}")

if __name__ == '__main__':
    # 程序入口
    t_start = perf_counter()  # 启动计时

    # 参数校验
    args = check_args(sys.argv)
    if args == -1:
        sys.exit()  # 参数错误时退出
    
    # 解析有效参数
    disease_Id, METHOD, num_cpus = args

    # 配置处理范围
    if disease_Id != 'all':
        disease_Ids = [disease_Id]  # 指定单个疾病
    
    if METHOD != 'all':
        methods = [METHOD]          # 指定单个方法

    print(f'[i] 开始为 {len(disease_Ids)} 个疾病计算排名')

    # 主处理循环
    for disease_Id in disease_Ids:
        for METHOD in methods:
            print(f'[i] 启动疾病 {disease_Id} 的方法 {METHOD}')

            # 生成结果文件名
            filename = f'{PATH_TO_RANKINGS}{disease_Id}_all_positives_'
            
            # 根据方法名确定模式
            modality = 'multiclass'
            if '_only' in METHOD:
                modality = 'binary'
                filename += METHOD.lower().replace("_only", "") + '.txt'
            else:
                filename += f'xgdag_{METHOD.lower()}.txt'

            # 检查文件是否已存在
            if os.path.exists(filename):
                # 用户交互确认覆盖
                res = sanitized_input(
                    f'[+] 疾病 {disease_Id} 的 {METHOD} 排名已存在，是否覆盖? (y/n) ',
                    ['y', 'n']
                )
                if res == 'n':
                    print(f'[i] 跳过 {disease_Id} 的 {METHOD}')
                    continue

            # 执行排名计算
            ranking(disease_Id, METHOD, num_cpus, filename, modality)

    # 输出总耗时
    t_end = perf_counter()
    print(f'[i] 总耗时: {round(t_end - t_start, 3)} 秒')