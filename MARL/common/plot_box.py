import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义算法和种子文件的路径
# algorithms = ["MAPPO", 'INT-SAFT-MAPPO']
# algorithms = ["hetro_mappo", 'hetro_int']
algorithms = ["MAA2C", 'MAPPO-PIS',"MAACKTR"]
# seeds = ['0', '1000', '2024']
seeds = ['0']
file_template = '../results/{}_hard_{}/eval_logs/PET.npy'

# 初始化一个空的DataFrame
df = pd.DataFrame()

# 加载每个文件的数据并合并到一个DataFrame中
for algorithm in algorithms:
    for seed in seeds:
        file_path = file_template.format(algorithm, seed)
        data = np.load(file_path)
        data = np.log(data)
        episode = np.arange(len(data))
        # mean_reward = data.squeeze()  # 将数据从二维变为一维
        temp_df = pd.DataFrame({'episode': episode, 'PET': data})
        temp_df['algorithm'] = algorithm
        # temp_df['seed'] = seed
        df = pd.concat([df, temp_df], ignore_index=True)


sns.set()
# 创建绘图
plt.figure(figsize=(10, 6))
# lower_limit = -10
# upper_limit = 50
# filtered_pet = df[(df['PET'] >= lower_limit) & (df['PET'] <= upper_limit)]
# df["algorithm"] = df["algorithm"].isin(["MAPPO", "INT-SAFT-MAPPO"])
ax = sns.boxplot(x="algorithm", y="PET", data=df, hue='algorithm', dodge=False)
# ax.xaxis.set_visible(False)
plt.title('Post-Encroachment Time (PET) in hard mode', fontsize=24)
plt.ylabel('PET (s)', fontsize=22)
plt.xticks(fontsize=22)
plt.legend(title='Algorithm', loc='upper right', fontsize=18)

plt.savefig(r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\plot\ab_box_2_log.png', dpi=1000)
plt.show()