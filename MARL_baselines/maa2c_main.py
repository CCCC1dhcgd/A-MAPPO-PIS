from __future__ import print_function, division
import argparse
import configparser
import gym
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from MAA2C import MAA2C
from common.utils import agg_double_list, copy_file, init_dir

sys.path.append("../highway-env")
import highway_env


def parse_args():
    """Parse command-line arguments."""
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs.ini'
    parser = argparse.ArgumentParser(description='Train or evaluate policy on RL environment using MA2C')
    parser.add_argument('--base-dir', type=str, default=default_base_dir, help="Experiment base dir")
    parser.add_argument('--option', type=str, default='evaluate', help="Train or evaluate")
    parser.add_argument('--config-dir', type=str, default=default_config_dir, help="Experiment config path")
    parser.add_argument('--model-dir', type=str,
                        default=r'D:\py_learning\marl_cav_new\MARL_AD_U-20240629T131712Z-001\MARL_AD_U\MARL\results\MAA2C_hard_0',
                        help="Pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="Random seeds for evaluation, split by ,")
    return parser.parse_args()


def load_config(config_dir):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(config_dir)
    return config


def init_env(config, env_type='train'):
    """Initialize the environment."""
    env = gym.make('merge-multi-agent-v0')
    for key in config['ENV_CONFIG']:
        env.config[key] = config.get('ENV_CONFIG', key)
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    assert env.T % config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS') == 0
    return env


def init_ma2c(env, config, test_seeds):
    """Initialize the MAA2C model."""
    state_dim = env.n_s
    action_dim = env.n_a
    return MAA2C(env, state_dim=state_dim, action_dim=action_dim,
                 memory_capacity=config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY'),
                 roll_out_n_steps=config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS'),
                 reward_gamma=config.getfloat('MODEL_CONFIG', 'reward_gamma'),
                 reward_scale=config.getfloat('TRAIN_CONFIG', 'reward_scale'),
                 actor_hidden_size=config.getint('MODEL_CONFIG', 'actor_hidden_size'),
                 critic_hidden_size=config.getint('MODEL_CONFIG', 'critic_hidden_size'),
                 actor_lr=config.getfloat('TRAIN_CONFIG', 'actor_lr'),
                 critic_lr=config.getfloat('TRAIN_CONFIG', 'critic_lr'),
                 optimizer_type="rmsprop", entropy_reg=config.getfloat('MODEL_CONFIG', 'ENTROPY_REG'),
                 max_grad_norm=config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM'),
                 batch_size=config.getint('MODEL_CONFIG', 'BATCH_SIZE'),
                 episodes_before_train=config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN'),
                 use_cuda=False, training_strategy=config.get('MODEL_CONFIG', 'training_strategy'),
                 epsilon=config.getfloat('MODEL_CONFIG', 'epsilon'),
                 alpha=config.getfloat('MODEL_CONFIG', 'alpha'),
                 traffic_density=config.getint('ENV_CONFIG', 'traffic_density'),
                 test_seeds=test_seeds,
                 state_split=config.getboolean('MODEL_CONFIG', 'state_split'),
                 shared_network=config.getboolean('MODEL_CONFIG', 'shared_network'),
                 reward_type=config.get('MODEL_CONFIG', 'reward_type'))


def train(args):
    """Train the model."""
    config = load_config(args.config_dir)

    now = datetime.now().strftime("%b-%d_%H_%M_%S")
    output_dir = os.path.join(args.base_dir, now)
    dirs = init_dir(output_dir)
    copy_file(dirs['configs'])

    model_dir = args.model_dir if os.path.exists(args.model_dir) else dirs['models']
    env = init_env(config)
    env_eval = init_env(config, 'eval')
    ma2c = init_ma2c(env, config, args.evaluation_seeds)

    ma2c.load(model_dir, train_mode=True)
    env.seed = env.config['seed']
    env.unwrapped.seed = env.config['seed']

    episodes, eval_rewards = [], []
    best_eval_reward = -100

    while ma2c.n_episodes < config.getint('TRAIN_CONFIG', 'MAX_EPISODES'):
        ma2c.explore()
        if ma2c.n_episodes >= config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN'):
            ma2c.train()

        if ma2c.episode_done and ((ma2c.n_episodes + 1) % config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL') == 0):
            rewards, _, _, _ = ma2c.evaluation(env_eval, dirs['train_videos'],
                                               config.getint('TRAIN_CONFIG', 'EVAL_EPISODES'))
            rewards_mu, rewards_std = agg_double_list(rewards)
            print(f"Episode {ma2c.n_episodes + 1}, Average Reward {rewards_mu:.2f}")
            episodes.append(ma2c.n_episodes + 1)
            eval_rewards.append(rewards_mu)

            if rewards_mu > best_eval_reward:
                ma2c.save(dirs['models'], 100000)
                ma2c.save(dirs['models'], ma2c.n_episodes + 1)
                best_eval_reward = rewards_mu
            else:
                ma2c.save(dirs['models'], ma2c.n_episodes + 1)

        np.save(os.path.join(output_dir, 'eval_rewards'), np.array(eval_rewards))
        np.save(os.path.join(output_dir, 'episode_rewards'), np.array(ma2c.episode_rewards))
        np.save(os.path.join(output_dir, 'epoch_steps'), np.array(ma2c.epoch_steps))
        np.save(os.path.join(output_dir, 'average_speed'), np.array(ma2c.average_speed))

    ma2c.save(dirs['models'], config.getint('TRAIN_CONFIG', 'MAX_EPISODES') + 2)

    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Training Reward")
    plt.legend(["MAA2C"])
    plt.savefig(os.path.join(output_dir, "maa2c_train.png"))
    plt.show()


def evaluate(args):
    """Evaluate the model."""
    if os.path.exists(args.model_dir):
        model_dir = os.path.join(args.model_dir, 'models')
    else:
        raise Exception("Sorry, no pretrained models")

    config = load_config(os.path.join(args.model_dir, 'configs/configs.ini'))
    video_dir = os.path.join(args.model_dir, 'eval_videos')
    eval_logs = os.path.join(args.model_dir, 'eval_logs')

    env = init_env(config)
    ma2c = init_ma2c(env, config, args.evaluation_seeds)

    ma2c.load(model_dir, train_mode=False)
    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, PETs, collision_rates = ma2c.evaluation(env,
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
