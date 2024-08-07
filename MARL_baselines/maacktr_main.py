from __future__ import print_function, division
import argparse
import configparser
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import torch as th
from datetime import datetime

from MAACKTR import JointACKTR as MAACKTR
from common.utils import agg_double_list, copy_file_akctr, init_dir

sys.path.append("../highway-env")
import highway_env


def parse_args():
    """Parse command-line arguments."""
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_acktr.ini'
    parser = argparse.ArgumentParser(description='Train or evaluate policy using MAACKTR')
    parser.add_argument('--base-dir', type=str, default=default_base_dir, help="Experiment base directory")
    parser.add_argument('--option', type=str, default='evaluate', help="Choose between 'train' or 'evaluate'")
    parser.add_argument('--config-dir', type=str, default=default_config_dir, help="Path to the config file")
    parser.add_argument('--model-dir', type=str,
                        default=r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\results\MAACKTR_hard_0',
                        help="Path to the pretrained model")
    parser.add_argument('--evaluation-seeds', type=str, default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="Comma-separated random seeds for evaluation")
    return parser.parse_args()


def load_config(config_dir):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(config_dir)
    return config


def init_env(config, seed_offset=0):
    """Initialize the environment."""
    env = gym.make('merge-multi-agent-v0')
    for key in config['ENV_CONFIG']:
        env.config[key] = config.get('ENV_CONFIG', key)
    env.config['seed'] += seed_offset
    return env


def init_maacktr(env, config, test_seeds):
    """Initialize the MAACKTR model."""
    state_dim = env.n_s
    action_dim = env.n_a
    return MAACKTR(env=env, memory_capacity=config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY'),
                   state_dim=state_dim, action_dim=action_dim,
                   batch_size=config.getint('MODEL_CONFIG', 'BATCH_SIZE'),
                   entropy_reg=config.getfloat('MODEL_CONFIG', 'ENTROPY_REG'),
                   actor_lr=config.getfloat('TRAIN_CONFIG', 'actor_lr'),
                   critic_lr=config.getfloat('TRAIN_CONFIG', 'critic_lr'),
                   reward_gamma=config.getfloat('MODEL_CONFIG', 'reward_gamma'),
                   reward_scale=config.getfloat('TRAIN_CONFIG', 'reward_scale'),
                   actor_hidden_size=config.getint('MODEL_CONFIG', 'actor_hidden_size'),
                   critic_hidden_size=config.getint('MODEL_CONFIG', 'critic_hidden_size'),
                   roll_out_n_steps=config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS'),
                   test_seeds=test_seeds,
                   max_grad_norm=config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM'),
                   reward_type=config.get('MODEL_CONFIG', 'reward_type'),
                   episodes_before_train=config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN'),
                   traffic_density=config.getint('ENV_CONFIG', 'traffic_density'))


def train(args):
    """Train the MAACKTR model."""
    config = load_config(args.config_dir)

    now = datetime.utcnow().strftime("%b-%d_%H_%M_%S")
    output_dir = os.path.join(args.base_dir, now)
    dirs = init_dir(output_dir)
    copy_file_akctr(dirs['configs'])

    model_dir = args.model_dir if os.path.exists(args.model_dir) else dirs['models']

    env = init_env(config)
    env_eval = init_env(config, seed_offset=1)
    maacktr = init_maacktr(env, config, args.evaluation_seeds)

    maacktr.load(model_dir, train_mode=True)
    env.seed = env.config['seed']

    episodes, eval_rewards = [], []
    best_eval_reward = -100

    while maacktr.n_episodes < config.getint('TRAIN_CONFIG', 'MAX_EPISODES'):
        maacktr.interact()
        if maacktr.n_episodes >= config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN'):
            maacktr.train()

        if maacktr.episode_done and (maacktr.n_episodes + 1) % config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL') == 0:
            rewards, _, _, _ = maacktr.evaluation(env_eval, dirs['train_videos'],
                                                  config.getint('TRAIN_CONFIG', 'EVAL_EPISODES'))
            rewards_mu, rewards_std = agg_double_list(rewards)
            print(f"Episode {maacktr.n_episodes + 1}, Average Reward {rewards_mu:.2f}")

            episodes.append(maacktr.n_episodes + 1)
            eval_rewards.append(rewards_mu)

            np.save(os.path.join(output_dir, 'episode_rewards'), np.array(maacktr.episode_rewards))
            np.save(os.path.join(output_dir, 'eval_rewards'), np.array(eval_rewards))
            np.save(os.path.join(output_dir, 'average_speed'), np.array(maacktr.average_speed))

            if rewards_mu > best_eval_reward:
                maacktr.save(dirs['models'], 100000)
                maacktr.save(dirs['models'], maacktr.n_episodes + 1)
                best_eval_reward = rewards_mu
            else:
                maacktr.save(dirs['models'], maacktr.n_episodes + 1)

    maacktr.save(dirs['models'], config.getint('TRAIN_CONFIG', 'MAX_EPISODES') + 2)

    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MAACKTR"])
    plt.show()


def evaluate(args):
    """Evaluate the MAACKTR model."""
    if os.path.exists(args.model_dir):
        model_dir = os.path.join(args.model_dir, 'models')
    else:
        raise Exception("Sorry, no pretrained models")

    config = load_config(os.path.join(args.model_dir, 'configs/configs_acktr.ini'))
    video_dir = os.path.join(args.model_dir, 'eval_videos')
    eval_logs = os.path.join(args.model_dir, 'eval_logs')

    env = init_env(config)
    maacktr = init_maacktr(env, config, args.evaluation_seeds)

    maacktr.load(model_dir, train_mode=False, map_location=th.device('cpu'))
    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, PETs, collision_rates = maacktr.evaluation(env,
                                                                                                              video_dir,
                                                                                                              len(args.evaluation_seeds.split(
                                                                                                                  ',')),
                                                                                                              is_train=False)

    rewards_mu, rewards_std = agg_double_list(rewards)
    success_rate = sum(np.array(steps) == 100) / len(steps)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

    pet_df = pd.DataFrame(PETs, columns=['PET'])
    plt.figure(figsize=(10, 6))
    sns.set()
    sns.boxplot(data=pet_df[(pet_df['PET'] >= 0) & (pet_df['PET'] <= 40)], y='PET')
    plt.title('Post-Encroachment Time (PET)')
    plt.ylabel('PET Value')
    plt.xlabel('INT-SAFT-MAPPO')
    plt.show()

    collision_rate = sum(collision_rates) / len(collision_rates)

    print(f"Evaluation Reward and std {rewards_mu:.2f}, {rewards_std:.2f}")
    print(f"Collision Rate {collision_rate:.2f}")
    print(f"Average Speed and std {avg_speeds_mu:.2f}, {avg_speeds_std:.2f}")

    np.save(os.path.join(eval_logs, 'eval_rewards'), np.array(rewards))
    np.save(os.path.join(eval_logs, 'eval_steps'), np.array(steps))
    np.save(os.path.join(eval_logs, 'eval_avg_speeds'), np.array(avg_speeds))
    np.save(os.path.join(eval_logs, 'PET'), np.array(PETs))
    np.save(os.path.join(eval_logs, 'vehicle_speed'), np.array(vehicle_speed))
    np.save(os.path.join(eval_logs, 'vehicle_position'), np.array(vehicle_position))


if __name__ == "__main__":
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
