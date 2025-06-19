import os
import pandas as pd

# 设置你的目标文件夹路径
folder_path = '/data/home/yekai/github/DiffAdvPerturbationBench/multi-dreambooth/eval_result_multi'  # 替换为你自己的路径
output_file = 'merged_output.csv'  # 合并后的输出文件名

# 获取所有 CSV 文件路径（按文件名排序）
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

# 初始化一个列表，用于存储 DataFrame
dataframes = []

for idx, file_name in enumerate(csv_files):
    file_path = os.path.join(folder_path, file_name)
    if idx == 0:
        # 第一个文件：读取包括列名
        df = pd.read_csv(file_path)
    else:
        # 后续文件：不读取列名
        df = pd.read_csv(file_path, header=0)
    dataframes.append(df)

# 合并所有 DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# 保存合并后的结果
merged_df.to_csv(os.path.join(folder_path, output_file), index=False)

print(f"成功合并 {len(csv_files)} 个文件，结果保存为: {output_file}")