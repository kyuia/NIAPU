
"""
疾病基因关联预测评估系统
功能：对比不同排序算法的性能，生成评估指标和可视化图表
"""
import pandas as pd
from scipy.stats import wilcoxon  # 用于非参数统计检验
import matplotlib.pyplot as plt  # 可视化库
from Paths import PATH_TO_RANKINGS
import os


# 问题2：路径拼接错误（假设PATH_TO_RANKINGS是目录路径）
METHOD_CONFIG = {
    # 修正路径拼接方式（添加路径分隔符）
    "GNNExplainer": os.path.join(PATH_TO_RANKINGS, "C0001973_all_positives_new_ranking_gnnexplainer.txt"),
}

def evaluate_method(method_path):
    # 问题3：分隔符不匹配（文件使用单列无分隔符）
    rank_df = pd.read_csv(method_path, 
                         header=None,
                         names=["gene_id"],
                         sep=None)  # 改为自动检测分隔符
# 真实关联基因文件路径
TRUE_GENES_PATH = "Datasets/curated_gene_disease_associations.tsv"

# 方法配置字典（格式：{"方法显示名称": "排序文件路径"}）
METHOD_CONFIG = {
    "XGDAG": f"{PATH_TO_RANKINGS}_xgdag_gnnexplainer.txt",
    # 添加新方法示例：
    # "NewMethod": "data/methods/new_method.csv"
}

# 评估的Top K值列表
K_VALUES = [25, 50, 100, 200, 500, 
           750, 1000, 1500, 2000, 3000]

# 可视化样式配置
VISUAL_STYLE = {
    "figure_size": (12, 7),          # 图表尺寸
    "colors": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"],  # 颜色方案
    "line_styles": ["-", "--", "-.", ":"],    # 线型样式
    "markers": ["o", "s", "^", "v"],          # 数据点标记
    "font_size": 12,                 # 字体大小
    "save_path": "results/compare.png"  # 图表保存路径
}

# ====================== 核心功能函数 =======================

def calculate_metrics(top_k_genes, true_genes_set, k):
    """
    计算分类评估指标
    
    参数：
        top_k_genes (list): 算法预测的前k个基因ID列表
        true_genes_set (set): 已知真实关联基因集合
        k (int): 当前评估的Top K值
    
    返回：
        dict: 包含TP/FP/Precision/Recall/F1的字典
    """
    top_k_set = set(top_k_genes[:k])
    tp = len(top_k_set & true_genes_set)
    fp = k - tp
    fn = len(true_genes_set) - tp

    precision = tp / k if k else 0
    recall = tp / len(true_genes_set) if true_genes_set else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) else 0

    return {
        "K": k,
        "TP": tp,
        "FP": fp,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    }

def load_true_genes():
    """
    加载真实关联基因集合
    
    返回：
        set: 基因ID集合
    """
    with open(TRUE_GENES_PATH, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def evaluate_method(method_path):
    """
    评估单个方法的性能
    
    参数：
        method_path (str): 方法结果文件路径
    
    返回：
        DataFrame: 包含各K值指标的结果表
    """
    # 加载排序结果
    rank_df = pd.read_csv(method_path, 
                         header=None,          # 无列标题
                         names=["gene_id"],     # 指定列名
                         sep='\t')             # 使用制表符分隔
    # 加载真实基因
    true_genes = load_true_genes()
    
    # 计算所有K值指标
    results = []
    for k in K_VALUES:
        metrics = calculate_metrics(
            rank_df["gene_id"].tolist(),
            true_genes,
            k
        )
        results.append(metrics)
    
    return pd.DataFrame(results)

def compare_methods():
    """
    对比所有配置方法的性能
    
    返回：
        DataFrame: 合并后的评估结果，包含"Method"列
    """
    all_results = []
    
    # 遍历每个配置方法
    for method_name, file_path in METHOD_CONFIG.items():
        # 执行评估
        method_df = evaluate_method(file_path)
        # 添加方法标签
        method_df["Method"] = method_name
        all_results.append(method_df)
    
    return pd.concat(all_results, ignore_index=True)

def visualize_comparison(combined_df):
    """
    生成方法对比可视化图表
    
    参数：
        combined_df (DataFrame): 合并后的评估结果
    """
    plt.figure(figsize=VISUAL_STYLE["figure_size"])
    
    # 为每个方法绘制曲线
    for idx, (method_name, group_df) in enumerate(combined_df.groupby("Method")):
        plt.plot(
            group_df["K"],
            group_df["F1"],
            label=method_name,
            color=VISUAL_STYLE["colors"][idx % len(VISUAL_STYLE["colors"])],
            linestyle=VISUAL_STYLE["line_styles"][idx % len(VISUAL_STYLE["line_styles"])],
            marker=VISUAL_STYLE["markers"][idx % len(VISUAL_STYLE["markers"])],
            markersize=8,
            linewidth=2
        )
    
    # 设置图表样式
    plt.title("疾病基因关联预测方法对比", fontsize=14, pad=20)
    plt.xlabel("Top K 基因数", fontsize=VISUAL_STYLE["font_size"], labelpad=10)
    plt.ylabel("F1 Score", fontsize=VISUAL_STYLE["font_size"], labelpad=10)
    plt.xticks(K_VALUES, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(VISUAL_STYLE["save_path"], dpi=300)
    plt.close()

def perform_statistical_test(combined_df, method_a, method_b):
    """
    执行Wilcoxon符号秩检验
    
    参数：
        combined_df (DataFrame): 合并后的评估结果
        method_a (str): 方法A名称
        method_b (str): 方法B名称
    """
    # 提取两种方法的F1分数
    f1_a = combined_df[combined_df["Method"] == method_a]["F1"]
    f1_b = combined_df[combined_df["Method"] == method_b]["F1"]
    
    # 执行检验
    stat, p_value = wilcoxon(f1_a, f1_b)
    print(f"\n统计检验结果（{method_a} vs {method_b}）:")
    print(f"Wilcoxon统计量: {stat:.4f}")
    print(f"P值: {p_value:.4e}")

# ====================== 主执行流程 ======================= 
if __name__ == "__main__":
    for method_name, path in METHOD_CONFIG.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"排序文件 {path} 不存在")
    # 阶段1：执行性能对比
    print("正在执行方法对比...")
    result_df = compare_methods()
    
    # 保存CSV结果
    result_df.to_csv("results/evaluation_results.csv", index=False)
    print("评估结果已保存至 results/evaluation_results.csv")
    
    # 阶段2：生成可视化
    print("生成对比图表...")
    visualize_comparison(result_df)
    print(f"图表已保存至 {VISUAL_STYLE['save_path']}")
    
    # 阶段3：执行统计检验（对比前两种方法）
    methods = list(METHOD_CONFIG.keys())
    if len(methods) >= 2:
        perform_statistical_test(result_df, methods[0], methods[1])