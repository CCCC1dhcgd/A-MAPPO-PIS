import gym
import highway_env
import numpy as np
from MAPPO import MAPPO
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import configparser
import os
from datetime import datetime


def parse_args():
    """
    Description for this experiment:
        + hard: 7-steps, curriculum
        + seed = 0
    """
    default_base_dir = "./eval/"
    default_config_dir = 'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\configs\configs_ppo.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using MA2C'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experi base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='evaluate', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default=r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\results\MAPPO-PIS_hard_0', help="pretrained model path")
                        # intent_hard_0_5\\MAPPO_hard_0
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args

def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '/configs/configs_ppo.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '/eval_videos'
    eval_logs = args.model_dir + '/eval_logs'

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('new-merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    # assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    mappo = MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density
                  )

    # load the model if exist
    mappo.load(model_dir, train_mode=False)

    state, action_mask = env.reset(is_training=False, testing_seeds=0)
    n_agents = len(env.controlled_vehicles)
    # 定义线圈位置（例如在主干道的不同位置）
    coil_positions = [325, 350, 375, 400, 425, 450]  # 根据实际情况设置位置
    # 初始化数据存储
    vehicle_data = {pos: [] for pos in coil_positions}

    average_speeds = []
    # 运行仿真并收集数据
    for step in range(150):  # 假设运行1000步
        action = mappo.action(state, n_agents)
        state, reward, done, info = env.step(action)
        vehicles = env.vehicle.road.vehicles
        for vehicle in vehicles:
            position = vehicle.position[0]  # 车辆的纵向位置
            speed = vehicle.speed
            # 记录在每个线圈位置的速度
            for coil_pos in coil_positions:
                if abs(position - coil_pos) < 5:  # 设定一个范围，例如5个单位
                    vehicle_data[coil_pos].append(speed)

        if done:
            obs = env.reset()

        step_speeds = calculate_average_speed(vehicle_data, coil_positions)
        average_speeds.append(step_speeds)
        env.render()

    # 转换为numpy数组以便绘图
    average_speeds = average_speeds[55:]
    average_speeds = np.array(average_speeds).T

    time_steps = np.arange(1,average_speeds.shape[1]+1)
    coil_labels=[f'WX{pos}' for pos in coil_positions]

    # 创建网格
    X, Y = np.meshgrid(time_steps, coil_positions)

    # 绘制等高线图
    plt.figure(figsize=(12, 6))
    levels = np.arange(12, 30, 2)
    contour = plt.contourf(X, Y, average_speeds, levels, cmap="RdYlGn")
    # contour = plt.contourf(X, Y, average_speeds, cmap="RdYlGn")

    # 设置坐标轴和标题
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Coil', fontsize=24)
    plt.title('Contour of MAPPO-PIS', fontsize=26)

    # 调整刻度
    plt.xticks(rotation=0)
    plt.yticks(coil_positions, coil_labels)

    # 添加颜色条
    cbar = plt.colorbar(contour)
    # cbar.set_label('v', rotation=270, labelpad=15,)
    plt.savefig(r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\plot\contour_ours.png',
                dpi=1000)
    plt.show()


# 绘制热力图
def plot_heatmap(average_speeds, coil_positions, vehicle_data):
    data_matrix = []
    for pos in range(len(coil_positions)):
        data_matrix.append(average_speeds[pos])

    plt.figure(figsize=(12, 8))
    sns.heatmap(data_matrix, cmap="RdYlGn", xticklabels=range(44), yticklabels=coil_positions)
    plt.xlabel('Time Step')
    plt.ylabel('Coil Position')
    plt.title('Average Speed Heatmap')
    plt.show()


# 计算每个断面的平均速度
def calculate_average_speed(vehicle_data, coil_positions):
    average_speeds = []
    for coil_pos in coil_positions:
        speeds_at_coil = vehicle_data[coil_pos]
        if speeds_at_coil:
            average_speed = np.mean(speeds_at_coil)
        else:
            average_speed = 0
        average_speeds.append(average_speed)
    return average_speeds


if __name__ == "__main__":
    args = parse_args()
    # eval
    evaluate(args)

