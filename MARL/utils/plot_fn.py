import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')


def smooth(x, timestamps=9):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot_reward():
    reward_hard_bs = np.load('/home/dong/PycharmProjects/MARL_AD_U/MARL/results/Oct-24_23:35:34/eval_rewards.npy')
    reward_hard_safe = np.load('/home/dong/PycharmProjects/MARL_AD_U/MARL/results/Oct-25_14:43:29/eval_rewards.npy')
    # reward_lstm = np.load(
    #     '/home/dong/PycharmProjects/MARL_AD_U_v0/MARL/results/Mar-20_00:38:36/episode_rewards.npy')
    # reward_lstm1 = np.load(
    #     '/home/dong/PycharmProjects/MARL_AD_U_v1/MARL/results/Mar-20_03:40:28/episode_rewards.npy')
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("Reward")
    plt.title("Epoch Reward")
    plt.plot(smooth(reward_hard_bs), label='bs')
    plt.plot(smooth(reward_hard_safe), label='safe')
    # plt.plot(smooth(reward_lstm1), label='lstm1')
    plt.xlim([0, 70])
    plt.ylim([-20, 40])
    plt.legend(loc="lower right", ncol=2)
    plt.show()


if __name__ == '__main__':
    plot_reward()