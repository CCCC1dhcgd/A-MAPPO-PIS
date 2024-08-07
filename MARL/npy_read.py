import numpy as np

file = np.load(r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\results\MAPPO_easy_0\eval_logs\vehicle_position.npy', allow_pickle=True)
print(file)
# np.savetxt('.../a/timestamps.txt',file)
