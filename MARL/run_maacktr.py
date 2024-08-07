from MAACKTR import JointACKTR as MAACKTR
from common.utils import agg_double_list, copy_file_akctr, init_dir

import os
import sys
sys.path.append("../highway-env")
import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env
from datetime import datetime
import torch as th
import seaborn as sns
import pandas as pd


import argparse
import configparser


def parse_args():
    """
    Description for this experiment:
        + easy: maacktr
        + seed = 0
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_acktr.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using MA2C'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='evaluate', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default=r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\results\MAACKTR_hard_0', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = datetime.utcnow().strftime("%b-%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file_akctr(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

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

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-multi-agent-v0')
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
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval = gym.make('merge-multi-agent-v0')
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env_eval.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env_eval.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    best_eval_reward = -100
    maacktr = MAACKTR(env=env, memory_capacity=MEMORY_CAPACITY,
                      state_dim=state_dim, action_dim=action_dim,
                      batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                      actor_lr=actor_lr, critic_lr=critic_lr,
                      reward_gamma=reward_gamma, reward_scale=reward_scale,
                      actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                      roll_out_n_steps=ROLL_OUT_N_STEPS, test_seeds=test_seeds,
                      max_grad_norm=MAX_GRAD_NORM, reward_type=reward_type,
                      episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density)

    # load the model if exist
    maacktr.load(model_dir, train_mode=True)
    env.seed = env.config['seed']
    episodes = []
    eval_rewards = []
    while maacktr.n_episodes < MAX_EPISODES:
        maacktr.interact()
        if maacktr.n_episodes >= EPISODES_BEFORE_TRAIN:
            maacktr.train()
        if maacktr.episode_done and ((maacktr.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards, _, _, _ = maacktr.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (maacktr.n_episodes + 1, rewards_mu))
            episodes.append(maacktr.n_episodes + 1)
            eval_rewards.append(rewards_mu)
            np.save(output_dir + '/{}'.format('episode_rewards'), np.array(maacktr.episode_rewards))
            np.save(output_dir + '/{}'.format('eval_rewards'), np.array(eval_rewards))
            np.save(output_dir + '/{}'.format('average_speed'), np.array(maacktr.average_speed))
            # save the model
            if rewards_mu > best_eval_reward:
                maacktr.save(dirs['models'], 100000)
                maacktr.save(dirs['models'], maacktr.n_episodes + 1)
                best_eval_reward = rewards_mu
            else:
                maacktr.save(dirs['models'], maacktr.n_episodes + 1)

    # save the model
    maacktr.save(dirs['models'], MAX_EPISODES + 2)
    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MAACKTR"])
    plt.show()


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '/configs/configs_acktr.ini'
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

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-multi-agent-v0')
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

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    maacktr = MAACKTR(env=env, memory_capacity=MEMORY_CAPACITY,
                      state_dim=state_dim, action_dim=action_dim,
                      batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                      actor_lr=actor_lr, critic_lr=critic_lr,
                      reward_gamma=reward_gamma, reward_scale=reward_scale,
                      actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                      roll_out_n_steps=ROLL_OUT_N_STEPS, test_seeds=test_seeds,
                      max_grad_norm=MAX_GRAD_NORM, reward_type=reward_type,
                      episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density)

    # load the model if exist
    maacktr.load(model_dir, train_mode=False, map_location=th.device('cpu'))
    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, PETs, collision_rates = maacktr.evaluation(env, video_dir, len(seeds),
                                                                                       is_train=False)
    rewards_mu, rewards_std = agg_double_list(rewards)
    success_rate = sum(np.array(steps) == 100) / len(steps)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)
    pet = pd.DataFrame(PETs, columns=['PET'])

    # 绘制箱型图
    plt.figure(figsize=(10, 6))
    sns.set()
    lower_limit = 0
    upper_limit = 40
    filtered_pet = pet[(pet['PET'] >= lower_limit) & (pet['PET'] <= upper_limit)]
    sns.boxplot(data=filtered_pet, y='PET')
    plt.title('Post-Encroachment Time (PET)')
    plt.ylabel('PET Value')
    plt.xlabel('INT-SAFT-MAPPO')

    # plt.savefig('D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\plot\mappo_box_1.png', dpi=1200)
    plt.show()

    collision_rate = sum(collision_rates) / len(collision_rates)

    print("Evaluation Reward and std %.2f, %.2f " % (rewards_mu, rewards_std))
    print("Collision Rate %.2f" % (collision_rate))
    print("Average Speed and std %.2f , %.2f " % (avg_speeds_mu, avg_speeds_std))

    np.save(eval_logs + '/{}'.format('eval_rewards'), np.array(rewards))
    np.save(eval_logs + '/{}'.format('eval_steps'), np.array(steps))
    np.save(eval_logs + '/{}'.format('eval_avg_speeds'), np.array(avg_speeds))
    np.save(eval_logs + '/{}'.format('PET'), np.array(PETs))
    np.save(eval_logs + '/{}'.format('vehicle_speed'), np.array(vehicle_speed))
    np.save(eval_logs + '/{}'.format('vehicle_position'), np.array(vehicle_position))


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
