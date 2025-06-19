import os
import pandas as pd

# 设置包含 CSV 文件的目录路径
input_dir = '/data/home/yekai/github/DiffAdvPerturbationBench/eval/eval_result'  # ←←← 修改为你的目录路径
output_file = '/data/home/yekai/github/DiffAdvPerturbationBench/eval/merged_output.csv'      # 输出文件名

# 获取所有 CSV 文件
csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
csv_files.sort()  # 可选：按照文件名排序合并

# 初始化一个列表用于保存数据框
df_list = []

# 遍历文件列表
for i, file in enumerate(csv_files):
    if i == 0:
        # 第一个文件：读取包括标题
        df = pd.read_csv(file)
    else:
        # 其余文件：跳过第一行标题
        df = pd.read_csv(file, skiprows=1, header=None)
        df.columns = df_list[0].columns  # 设置列名与第一个文件一致
    df_list.append(df)

# 合并数据
merged_df = pd.concat(df_list, ignore_index=True)

# 保存合并结果
merged_df.to_csv(os.path.join(input_dir, output_file), index=False)

print(f"合并完成：{os.path.join(input_dir, output_file)}")