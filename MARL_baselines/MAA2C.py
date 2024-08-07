import os
import logging
import configparser
import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork, ActorCriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var, VideoRecorder

# Load configuration
config_dir = 'configs/configs.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')

# Set random seed for reproducibility
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(torch_seed)


class MAA2C(Agent):
    """
    Multi-agent Advantage Actor-Critic (MAA2C) algorithm.
    Reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10, reward_gamma=0.99, reward_scale=1.0,
                 done_penalty=None, actor_hidden_size=32, critic_hidden_size=32,
                 critic_loss="mse", actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01, max_grad_norm=0.5,
                 batch_size=100, episodes_before_train=100, use_cuda=True,
                 training_strategy="concurrent", epsilon=1e-5, alpha=0.99,
                 traffic_density=1, test_seeds=0, state_split=False,
                 shared_network=False, reward_type='regionalR'):
        super(MAA2C, self).__init__(env, state_dim, action_dim,
                                    memory_capacity, max_steps,
                                    reward_gamma, reward_scale, done_penalty,
                                    actor_hidden_size, critic_hidden_size, critic_loss,
                                    actor_lr, critic_lr, optimizer_type, entropy_reg,
                                    max_grad_norm, batch_size, episodes_before_train,
                                    use_cuda)

        assert training_strategy in ["concurrent", "centralized"], \
            "Invalid training strategy: choose 'concurrent' or 'centralized'"
        assert traffic_density in [1, 2, 3], "Invalid traffic density: choose 1, 2, or 3"
        assert reward_type in ["greedy", "regionalR", "global_R"], \
            "Invalid reward type: choose 'greedy', 'regionalR', or 'global_R'"

        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.test_seeds = test_seeds
        self.traffic_density = traffic_density
        self.shared_network = shared_network
        self.reward_type = reward_type

        # Determine maximum number of vehicles based on traffic density
        max_num_vehicle = {1: 3, 2: 4, 3: 6}[self.traffic_density]

        if not self.shared_network:
            self.actors = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, state_split)
            critic_state_dim = max_num_vehicle * self.state_dim if self.training_strategy == "centralized" else self.state_dim
            self.critics = CriticNetwork(critic_state_dim, self.critic_hidden_size, 1, state_split)
            self.actor_optimizers, self.critic_optimizers = self._initialize_optimizers(optimizer_type, epsilon, alpha)
        else:
            self.policy = ActorCriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1, state_split)
            self.policy_optimizers = self._initialize_policy_optimizer(optimizer_type, epsilon, alpha)

        if self.use_cuda:
            self._move_models_to_cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    def _initialize_optimizers(self, optimizer_type, epsilon, alpha):
        if optimizer_type == "adam":
            return Adam(self.actors.parameters(), lr=self.actor_lr), Adam(self.critics.parameters(), lr=self.critic_lr)
        elif optimizer_type == "rmsprop":
            return (RMSprop(self.actors.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha),
                    RMSprop(self.critics.parameters(), lr=self.critic_lr, eps=epsilon, alpha=alpha))
        else:
            raise ValueError("Invalid optimizer type: choose 'adam' or 'rmsprop'")

    def _initialize_policy_optimizer(self, optimizer_type, epsilon, alpha):
        if optimizer_type == "adam":
            return Adam(self.policy.parameters(), lr=self.actor_lr)
        elif optimizer_type == "rmsprop":
            return RMSprop(self.policy.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)
        else:
            raise ValueError("Invalid optimizer type: choose 'adam' or 'rmsprop'")

    def _move_models_to_cuda(self):
        self.actors.cuda()
        self.critics.cuda() if not self.shared_network else self.policy.cuda()

    def explore(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, self.action_mask = self.env.reset()
            self.n_steps = 0

        states, actions, rewards, policies, action_masks = [], [], [], [], []
        average_speed = 0
        done = True

        self.n_agents = len(self.env.controlled_vehicles)

        for _ in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action_masks.append(self.action_mask)
            action, policy = self.exploration_action(self.env_state, self.action_mask)
            next_state, global_reward, done, info = self.env.step(tuple(action))

            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            reward = self._get_reward(info)
            average_speed += info["average_speed"]

            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            policies.append(policy)
            final_state = next_state

            self.env_state = next_state
            self.action_mask = info["action_mask"]

            self.n_steps += 1
            if done:
                self.env_state, self.action_mask = self.env.reset()
                break

        final_r = self._compute_final_rewards(done, final_state)
        rewards = self._scale_and_discount_rewards(rewards, final_r)
        self.memory.push(states, actions, rewards, policies, action_masks)

    def _get_reward(self, info):
        if self.reward_type == "greedy":
            return info["agents_rewards"]
        elif self.reward_type == "regionalR":
            return info["regional_rewards"]
        elif self.reward_type == "global_R":
            return [info["global_reward"]] * self.n_agents

    def _compute_final_rewards(self, done, final_state):
        if done:
            self.n_episodes += 1
            self.episode_done = True
            self.average_speed[-1] = self.average_speed[-1] / self.epoch_steps[-1]
            self.episode_rewards.append(0)
            self.epoch_steps.append(0)
            self.average_speed.append(0)
            return [0.0] * self.n_agents
        else:
            final_action = self.action(final_state, self.n_agents, self.action_mask)
            one_hot_action = [index_to_one_hot(a, self.action_dim) for a in final_action]
            return self.value(final_state, one_hot_action)

    def _scale_and_discount_rewards(self, rewards, final_r):
        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale
        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_r[agent_id])
        return rewards.tolist()

    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        action_masks_var = to_tensor_var(batch.action_masks, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)

        for agent_id in range(self.n_agents):
            if not self.shared_network:
                self._train_actor_critic(agent_id, states_var, action_masks_var, actions_var, rewards_var,
                                         whole_states_var)
            else:
                self._train_policy(agent_id, states_var, action_masks_var, actions_var, rewards_var, whole_states_var)

    def _train_actor_critic(self, agent_id, states_var, action_masks_var, actions_var, rewards_var, whole_states_var):
        self.actor_optimizers.zero_grad()
        action_log_probs = self.actors(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        policy_loss = -th.mean(action_log_probs * (
                    self._get_target_v(agent_id, whole_states_var) - self._get_current_v(agent_id,
                                                                                         whole_states_var).detach()))
        (policy_loss + self.entropy_reg * entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.actors.parameters(), self.max_grad_norm)
        self.actor_optimizers.step()

        self.critic_optimizers.zero_grad()
        value_loss = nn.MSELoss()(self._get_current_v(agent_id, whole_states_var),
                                  self._get_target_v(agent_id, whole_states_var).detach())
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critic_optimizers.step()

    def _train_policy(self, agent_id, states_var, action_masks_var, actions_var, rewards_var, whole_states_var):
        self.policy_optimizers.zero_grad()
        action_log_probs = self.policy(states_var, action_masks_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        target_v = self._get_target_v(agent_id, whole_states_var)
        current_v = self._get_current_v(agent_id, whole_states_var)
        policy_loss = -th.mean(action_log_probs * (target_v - current_v.detach()))
        (policy_loss + self.entropy_reg * entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizers.step()

    def _get_current_v(self, agent_id, whole_states_var):
        return self.critics(whole_states_var) if not self.shared_network else self.policy.critic(whole_states_var)

    def _get_target_v(self, agent_id, whole_states_var):
        return self.policy(whole_states_var) if self.shared_network else self.critics(whole_states_var)

    def test(self, n_episodes=1, save_video=False, video_name="test.mp4"):
        self.env_state, self.action_mask = self.env.reset()
        if save_video:
            self.video_recorder = VideoRecorder(video_name)
            self.video_recorder.init(self.env, video_name)

        for _ in range(n_episodes):
            done = False
            while not done:
                action = self.action(self.env_state, self.n_agents, self.action_mask)
                self.env_state, _, done, info = self.env.step(tuple(action))
                if save_video:
                    self.video_recorder.capture_frame()

        if save_video:
            self.video_recorder.close()
