from MAPPO import MAPPO
from common.utils import agg_double_list, copy_file_ppo, init_dir

import sys

sys.path.append("../highway-env")

import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env
import argparse
import configparser
import os
import pandas as pd
from datetime import datetime
import seaborn as sns


def parse_args():
    """
    Parses command line arguments for the experiment.
    """
    default_base_dir = "./results/"
    default_config_dir = 'D:/py_learning/marl_cav_new/MARL_AD_U-20240629T131712Z-001/MARL_AD_U/MARL/configs/configs_ppo.ini'
    parser = argparse.ArgumentParser(description='Train or evaluate policy on RL environment using MA2C')
    parser.add_argument('--base-dir', type=str, default=default_base_dir, help="Base directory for experiments")
    parser.add_argument('--option', type=str, default='evaluate', help="Specify whether to train or evaluate")
    parser.add_argument('--config-dir', type=str, default=default_config_dir, help="Path to experiment configuration")
    parser.add_argument('--model-dir', type=str,
                        default='D:/py_learning/marl_cav_new/MARL_AD_U-20240629T131712Z-001/MARL_AD_U/MARL_baselines/results/MAPPO-PIS_hard_0',
                        help="Path to pretrained model")
    parser.add_argument('--evaluation-seeds', type=str, default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="Comma-separated list of random seeds for evaluation")
    return parser.parse_args()


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # Create experiment directory
    now = datetime.utcnow().strftime("%b-%d_%H_%M_%S")
    output_dir = os.path.join(base_dir, now)
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs['configs'])

    model_dir = args.model_dir if os.path.exists(args.model_dir) else dirs['models']

    # Model configurations
    model_config = {
        'BATCH_SIZE': config.getint('MODEL_CONFIG', 'BATCH_SIZE'),
        'MEMORY_CAPACITY': config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY'),
        'ROLL_OUT_N_STEPS': config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS'),
        'reward_gamma': config.getfloat('MODEL_CONFIG', 'reward_gamma'),
        'actor_hidden_size': config.getint('MODEL_CONFIG', 'actor_hidden_size'),
        'critic_hidden_size': config.getint('MODEL_CONFIG', 'critic_hidden_size'),
        'MAX_GRAD_NORM': config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM'),
        'ENTROPY_REG': config.getfloat('MODEL_CONFIG', 'ENTROPY_REG'),
        'reward_type': config.get('MODEL_CONFIG', 'reward_type'),
        'TARGET_UPDATE_STEPS': config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS'),
        'TARGET_TAU': config.getfloat('MODEL_CONFIG', 'TARGET_TAU'),
        'action_masking': config.getboolean('MODEL_CONFIG', 'action_masking')
    }

    # Training configurations
    train_config = {
        'actor_lr': config.getfloat('TRAIN_CONFIG', 'actor_lr'),
        'critic_lr': config.getfloat('TRAIN_CONFIG', 'critic_lr'),
        'MAX_EPISODES': config.getint('TRAIN_CONFIG', 'MAX_EPISODES'),
        'EPISODES_BEFORE_TRAIN': config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN'),
        'EVAL_INTERVAL': config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL'),
        'EVAL_EPISODES': config.getint('TRAIN_CONFIG', 'EVAL_EPISODES'),
        'reward_scale': config.getfloat('TRAIN_CONFIG', 'reward_scale')
    }

    # Environment configurations
    env_config = {
        'seed': config.getint('ENV_CONFIG', 'seed'),
        'simulation_frequency': config.getint('ENV_CONFIG', 'simulation_frequency'),
        'duration': config.getint('ENV_CONFIG', 'duration'),
        'policy_frequency': config.getint('ENV_CONFIG', 'policy_frequency'),
        'COLLISION_REWARD': config.getint('ENV_CONFIG', 'COLLISION_REWARD'),
        'HIGH_SPEED_REWARD': config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD'),
        'HEADWAY_COST': config.getint('ENV_CONFIG', 'HEADWAY_COST'),
        'HEADWAY_TIME': config.getfloat('ENV_CONFIG', 'HEADWAY_TIME'),
        'MERGING_LANE_COST': config.getint('ENV_CONFIG', 'MERGING_LANE_COST'),
        'traffic_density': config.getint('ENV_CONFIG', 'traffic_density'),
        'safety_guarantee': config.getboolean('ENV_CONFIG', 'safety_guarantee'),
        'n_step': config.getint('ENV_CONFIG', 'n_step')
    }

    # Initialize environments
    env = gym.make('merge-multi-agent-v0')
    env.config.update(env_config)
    env_eval = gym.make('merge-multi-agent-v0')
    env_eval.config.update({**env_config, 'seed': env_config['seed'] + 1})

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds

    # Initialize MAPPO
    mappo = MAPPO(
        env=env,
        memory_capacity=model_config['MEMORY_CAPACITY'],
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=model_config['BATCH_SIZE'],
        entropy_reg=model_config['ENTROPY_REG'],
        roll_out_n_steps=model_config['ROLL_OUT_N_STEPS'],
        actor_hidden_size=model_config['actor_hidden_size'],
        critic_hidden_size=model_config['critic_hidden_size'],
        actor_lr=train_config['actor_lr'],
        critic_lr=train_config['critic_lr'],
        reward_scale=train_config['reward_scale'],
        target_update_steps=model_config['TARGET_UPDATE_STEPS'],
        target_tau=model_config['TARGET_TAU'],
        reward_gamma=model_config['reward_gamma'],
        reward_type=model_config['reward_type'],
        max_grad_norm=model_config['MAX_GRAD_NORM'],
        test_seeds=test_seeds,
        episodes_before_train=train_config['EPISODES_BEFORE_TRAIN'],
        traffic_density=env_config['traffic_density']
    )

    # Load model
    mappo.load(model_dir, train_mode=True)
    env.seed = env.config['seed']

    episodes = []
    eval_rewards = []
    best_eval_reward = -100

    while mappo.n_episodes < train_config['MAX_EPISODES']:
        mappo.interact()
        if mappo.n_episodes >= train_config['EPISODES_BEFORE_TRAIN']:
            mappo.train()
        if mappo.episode_done and ((mappo.n_episodes + 1) % train_config['EVAL_INTERVAL'] == 0):
            rewards, _, _, _ = mappo.evaluation(env_eval, dirs['train_videos'], train_config['EVAL_EPISODES'])
            rewards_mu, rewards_std = agg_double_list(rewards)
            print(f"Episode {mappo.n_episodes + 1}, Average Reward {rewards_mu:.2f}")
            episodes.append(mappo.n_episodes + 1)
            eval_rewards.append(rewards_mu)
            # Save the model
            if rewards_mu > best_eval_reward:
                mappo.save(dirs['models'], 100000)
                mappo.save(dirs['models'], mappo.n_episodes + 1)
                best_eval_reward = rewards_mu
            else:
                mappo.save(dirs['models'], mappo.n_episodes + 1)
            np.save(os.path.join(output_dir, 'episode_rewards.npy'), np.array(mappo.episode_rewards))
            np.save(os.path.join(output_dir, 'eval_rewards.npy'), np.array(eval_rewards))
            np.save(os.path.join(output_dir, 'average_speed.npy'), np.array(mappo.average_speed))

    # Save final model
    mappo.save(dirs['models'], train_config['MAX_EPISODES'] + 2)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MAPPO"])
    plt.show()


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = os.path.join(args.model_dir, 'models')
    else:
        raise Exception("No pretrained models found")

    config_dir = os.path.join(args.model_dir, 'configs', 'configs_ppo.ini')
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = os.path.join(args.model_dir, 'eval_videos')
    eval_logs = os.path.join(args.model_dir, 'eval_logs')

    # Model configurations
    model_config = {
        'BATCH_SIZE': config.getint('MODEL_CONFIG', 'BATCH_SIZE'),
        'MEMORY_CAPACITY': config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY'),
        'ROLL_OUT_N_STEPS': config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS'),
        'reward_gamma': config.getfloat('MODEL_CONFIG', 'reward_gamma'),
        'actor_hidden_size': config.getint('MODEL_CONFIG', 'actor_hidden_size'),
        'critic_hidden_size': config.getint('MODEL_CONFIG', 'critic_hidden_size'),
        'MAX_GRAD_NORM': config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM'),
        'ENTROPY_REG': config.getfloat('MODEL_CONFIG', 'ENTROPY_REG'),
        'reward_type': config.get('MODEL_CONFIG', 'reward_type'),
        'TARGET_UPDATE_STEPS': config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS'),
        'TARGET_TAU': config.getfloat('MODEL_CONFIG', 'TARGET_TAU'),
        'action_masking': config.getboolean('MODEL_CONFIG', 'action_masking')
    }

    # Training configurations
    train_config = {
        'actor_lr': config.getfloat('TRAIN_CONFIG', 'actor_lr'),
        'critic_lr': config.getfloat('TRAIN_CONFIG', 'critic_lr'),
        'EPISODES_BEFORE_TRAIN': config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN'),
        'reward_scale': config.getfloat('TRAIN_CONFIG', 'reward_scale')
    }

    # Environment configurations
    env_config = {
        'seed': config.getint('ENV_CONFIG', 'seed'),
        'simulation_frequency': config.getint('ENV_CONFIG', 'simulation_frequency'),
        'duration': config.getint('ENV_CONFIG', 'duration'),
        'policy_frequency': config.getint('ENV_CONFIG', 'policy_frequency'),
        'COLLISION_REWARD': config.getint('ENV_CONFIG', 'COLLISION_REWARD'),
        'HIGH_SPEED_REWARD': config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD'),
        'HEADWAY_COST': config.getint('ENV_CONFIG', 'HEADWAY_COST'),
        'HEADWAY_TIME': config.getfloat('ENV_CONFIG', 'HEADWAY_TIME'),
        'MERGING_LANE_COST': config.getint('ENV_CONFIG', 'MERGING_LANE_COST'),
        'traffic_density': config.getint('ENV_CONFIG', 'traffic_density'),
        'action_masking': config.getboolean('MODEL_CONFIG', 'action_masking')
    }

    env = gym.make('merge-multi-agent-v0')
    env.config.update(env_config)

    assert env.T % model_config['ROLL_OUT_N_STEPS'] == 0

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    # Initialize MAPPO
    mappo = MAPPO(
        env=env,
        memory_capacity=model_config['MEMORY_CAPACITY'],
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=model_config['BATCH_SIZE'],
        entropy_reg=model_config['ENTROPY_REG'],
        roll_out_n_steps=model_config['ROLL_OUT_N_STEPS'],
        actor_hidden_size=model_config['actor_hidden_size'],
        critic_hidden_size=model_config['critic_hidden_size'],
        actor_lr=train_config['actor_lr'],
        critic_lr=train_config['critic_lr'],
        reward_scale=train_config['reward_scale'],
        target_update_steps=model_config['TARGET_UPDATE_STEPS'],
        target_tau=model_config['TARGET_TAU'],
        reward_gamma=model_config['reward_gamma'],
        reward_type=model_config['reward_type'],
        max_grad_norm=model_config['MAX_GRAD_NORM'],
        test_seeds=test_seeds,
        episodes_before_train=train_config['EPISODES_BEFORE_TRAIN'],
        traffic_density=env_config['traffic_density']
    )

    # Load model
    mappo.load(model_dir, train_mode=False)
    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, PETs, collision_rates = mappo.evaluation(
        env, video_dir, len(seeds), is_train=False
    )
    rewards_mu, rewards_std = agg_double_list(rewards)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)
    pet_df = pd.DataFrame(PETs, columns=['PET'])

    # Plot Post-Encroachment Time (PET) box plot
    plt.figure(figsize=(10, 6))
    sns.set()
    lower_limit = 0
    upper_limit = 40
    filtered_pet = pet_df[(pet_df['PET'] >= lower_limit) & (pet_df['PET'] <= upper_limit)]
    sns.boxplot(data=filtered_pet, y='PET')
    plt.title('Post-Encroachment Time (PET)')
    plt.ylabel('PET Value')
    plt.xlabel('INT-SAFT-MAPPO')
    plt.savefig('D:/py_learning/marl_cav_new/MARL_AD_U-20240629T131712Z-001/MARL_AD_U/MARL_baselines/plot/mappo_box_1.png',
                dpi=1200)
    plt.show()

    collision_rate = sum(collision_rates) / len(collision_rates)
    vehicle_speed_flattened = [arr.flatten() for arr in vehicle_speed]
    vehicle_position_flattened = [arr.flatten() for arr in vehicle_position]

    print(f"Evaluation Reward and std: {rewards_mu:.2f}, {rewards_std:.2f}")
    print(f"Collision Rate: {collision_rate:.2f}")
    print(f"Average Speed and std: {avg_speeds_mu:.2f}, {avg_speeds_std:.2f}")

    np.save(os.path.join(eval_logs, 'eval_rewards.npy'), np.array(rewards))
    np.save(os.path.join(eval_logs, 'eval_steps.npy'), np.array(steps))
    np.save(os.path.join(eval_logs, 'eval_avg_speeds.npy'), np.array(avg_speeds))
    np.save(os.path.join(eval_logs, 'PET.npy'), np.array(PETs))
    np.save(os.path.join(eval_logs, 'vehicle_speed.npy'), np.array(vehicle_speed_flattened))
    np.save(os.path.join(eval_logs, 'vehicle_position.npy'), np.array(vehicle_position_flattened))


if __name__ == "__main__":
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
