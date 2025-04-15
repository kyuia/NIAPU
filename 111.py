# 临时调试代码：打印前5行数据的列数和首行数据
import pandas as pd
df = pd.read_csv("Datasets/BIOGRID-ALL-4.4.244.tab2.txt", sep='\t', nrows=5)
print("Columns:", df.columns.tolist())
print("First row data:", df.iloc[0].tolist())