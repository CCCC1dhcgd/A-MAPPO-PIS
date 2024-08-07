import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义算法和种子文件的路径
# algorithms = ["MAPPO", 'MAPPO-PIS']
# algorithms = ["MAA2C", 'MAPPO-PIS',"MAACKTR"]
# seeds = ['0', '1000', '2024']
algorithms = ["No Curriculum Learning", 'Curriculum Learning']
seeds = ['0']
file_template = '../results/{}_hard_{}/average_speed.npy'
# file_template = '../results/{}_hard_{}/episode_rewards.npy'


# 初始化一个空的DataFrame
df = pd.DataFrame()

# 加载每个文件的数据并合并到一个DataFrame中
for algorithm in algorithms:
    for seed in seeds:
        file_path = file_template.format(algorithm, seed)
        data = np.load(file_path)
        episode = np.arange(len(data))
        mean_reward = data.squeeze()  # 将数据从二维变为一维
        temp_df = pd.DataFrame({'episode': episode, 'mean_reward': mean_reward})
        temp_df['algorithm'] = algorithm
        temp_df['seed'] = seed
        df = pd.concat([df, temp_df], ignore_index=True)


df['smoothed_mean_reward'] = df.groupby(['algorithm', 'seed'])['mean_reward'].transform(lambda x: x.rolling(window=300, min_periods=1).mean())
# sns.set(style="whitegrid")
sns.set()
# 创建绘图
plt.figure(figsize=(10, 6))

# 使用Seaborn绘制折线图
sns.lineplot(data=df, x='episode', y='smoothed_mean_reward', hue='algorithm', ci='sd', estimator='mean', err_style="band", linewidth=1.8)

# 设置y轴范围
# plt.ylim([-60, 60])
plt.ylim([0, 30])
# 设置标签和标题
plt.xlabel('Episode', fontsize=22)
# plt.ylabel('Agent Reward', fontsize=22)
plt.ylabel('Average Speed', fontsize=22)
plt.legend(title='Algorithm', loc='lower right', fontsize=18)
plt.title('Hard Mode', fontsize=24)

# 保存图片
plt.savefig('../plot/Speed_curriculum.png', dpi=1000)
plt.show()