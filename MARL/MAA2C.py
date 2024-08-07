import torch as th
import os, logging
import configparser


config_dir = 'configs/configs.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(torch_seed)

from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np
from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork, ActorCriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var, VideoRecorder


class MAA2C(Agent):
    """
    An multi-agent learned with Advantage Actor-Critic
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, training_strategy="concurrent", epsilon=1e-5, alpha=0.99,
                 traffic_density=1, test_seeds=0, state_split=False, shared_network=False, reward_type='regionalR'):
        super(MAA2C, self).__init__(env, state_dim, action_dim,
                                    memory_capacity, max_steps,
                                    reward_gamma, reward_scale, done_penalty,
                                    actor_hidden_size, critic_hidden_size, critic_loss,
                                    actor_lr, critic_lr,
                                    optimizer_type, entropy_reg,
                                    max_grad_norm, batch_size, episodes_before_train,
                                    use_cuda)

        assert training_strategy in ["concurrent", "centralized"]
        assert traffic_density in [1, 2, 3]
        assert reward_type in ["greedy", "regionalR", "global_R"]

        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.test_seeds = test_seeds
        self.traffic_density = traffic_density
        self.shared_network = shared_network
        self.reward_type = reward_type

        # maximum number of CAVs in each mode
        if self.traffic_density == 1:
            max_num_vehicle = 3
        elif self.traffic_density == 2:
            max_num_vehicle = 4
        elif self.traffic_density == 3:
            max_num_vehicle = 6

        if not self.shared_network:
            """separate actor and critic network"""
            self.actors = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, state_split)

            if self.training_strategy == "concurrent":
                self.critics = CriticNetwork(self.state_dim, self.critic_hidden_size, 1, state_split)
            elif self.training_strategy == "centralized":
                critic_state_dim = max_num_vehicle * self.state_dim
                self.critics = CriticNetwork(critic_state_dim, self.critic_hidden_size, 1, state_split)

            if optimizer_type == "adam":
                self.actor_optimizers = Adam(self.actors.parameters(), lr=self.actor_lr)
                self.critic_optimizers = Adam(self.critics.parameters(), lr=self.critic_lr)
            elif optimizer_type == "rmsprop":
                self.actor_optimizers = RMSprop(self.actors.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)
                self.critic_optimizers = RMSprop(self.critics.parameters(), lr=self.critic_lr, eps=epsilon, alpha=alpha)
            if self.use_cuda:
                self.actors.cuda()
                self.critics.cuda()
        else:
            """An actor-critic network that sharing lower-layer representations but
            have distinct output layers"""
            self.policy = ActorCriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1, state_split)
            if optimizer_type == "adam":
                self.policy_optimizers = Adam(self.policy.parameters(), lr=self.actor_lr)
            elif optimizer_type == "rmsprop":
                self.policy_optimizers = RMSprop(self.policy.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)

            if self.use_cuda:
                self.policy.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    # agent interact with the environment to collect experience
    def explore(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, self.action_mask = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        policies = []
        action_masks = []
        done = True
        average_speed = 0

        self.n_agents = len(self.env.controlled_vehicles)
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action_masks.append(self.action_mask)
            action, policy = self.exploration_action(self.env_state, self.action_mask)
            next_state, global_reward, done, info = self.env.step(tuple(action))
            # self.env.render()
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            if self.reward_type == "greedy":
                reward = info["agents_rewards"]
            elif self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            average_speed += info["average_speed"]
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            policies.append(policy)
            final_state = next_state

            # next state and corresponding action mask
            self.env_state = next_state
            self.action_mask = info["action_mask"]

            self.n_steps += 1
            if done:
                self.env_state, self.action_mask = self.env.reset()
                break

        # discount reward
        if done:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.episode_rewards.append(0)
            self.epoch_steps.append(0)
            self.average_speed.append(0)
        else:
            self.episode_done = False
            final_action = self.action(final_state, self.n_agents, self.action_mask)
            one_hot_action = [index_to_one_hot(a, self.action_dim) for a in final_action]
            final_r = self.value(final_state, one_hot_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale
        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_r[agent_id])
        rewards = rewards.tolist()

        self.memory.push(states, actions, rewards, policies, action_masks)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass
        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        action_masks_var = to_tensor_var(batch.action_masks, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)

        for agent_id in range(self.n_agents):
            if not self.shared_network:
                # update actor network
                self.actor_optimizers.zero_grad()
                action_log_probs = self.actors(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
                entropy_loss = th.mean(entropy(th.exp(action_log_probs) + 1e-8))
                action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)

                if self.training_strategy == "concurrent":
                    values = self.critics(states_var[:, agent_id, :])
                elif self.training_strategy == "centralized":
                    values = self.critics(whole_states_var)

                advantages = rewards_var[:, agent_id, :] - values.detach()
                pg_loss = -th.mean(action_log_probs * advantages)
                actor_loss = pg_loss - entropy_loss * self.entropy_reg
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actors.parameters(), self.max_grad_norm)
                self.actor_optimizers.step()

                # update critic network
                self.critic_optimizers.zero_grad()
                target_values = rewards_var[:, agent_id, :]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)
                critic_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
                self.critic_optimizers.step()
            else:
                # update actor-critic network
                self.policy_optimizers.zero_grad()
                action_log_probs = self.policy(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
                entropy_loss = th.mean(entropy(th.exp(action_log_probs) + 1e-8))
                action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
                values = self.policy(states_var[:, agent_id, :], out_type='v')

                target_values = rewards_var[:, agent_id, :]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)

                advantages = rewards_var[:, agent_id, :] - values.detach()
                pg_loss = -th.mean(action_log_probs * advantages)
                loss = pg_loss - entropy_loss * self.entropy_reg + critic_loss
                loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizers.step()

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # predict softmax actions based on state
    def _softmax_action(self, state, n_agents, action_mask):
        state_var = to_tensor_var([state], self.use_cuda)
        action_mask_var = to_tensor_var([action_mask], self.use_cuda)
        softmax_action = []
        for agent_id in range(n_agents):
            if not self.shared_network:
                softmax_action_var = th.exp(self.actors(state_var[:, agent_id, :], action_mask_var[:, agent_id, :]))
            else:
                softmax_action_var = th.exp(self.policy(state_var[:, agent_id, :], action_mask_var[:, agent_id, :]))
            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # predict actions based on state, added random noise for exploration in training
    def exploration_action(self, state, action_mask):
        # print(self.n_steps)
        if self.n_steps == 100:
            print('')
        softmax_actions = self._softmax_action(state, self.n_agents, action_mask)
        policy = []
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
            policy.append(pi)
        return actions, policy

    # predict actions based on state for execution
    def action(self, state, n_agents, action_mask):
        softmax_actions = self._softmax_action(state, n_agents, action_mask)
        # actions = np.argmax(softmax_actions, axis=1)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents * self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents * self.action_dim)
        values = [0] * self.n_agents

        for agent_id in range(self.n_agents):
            if not self.shared_network:
                """conditions for different action types"""
                if self.training_strategy == "concurrent":
                    value_var = self.critics(state_var[:, agent_id, :])
                elif self.training_strategy == "centralized":
                    value_var = self.critics(whole_state_var)
            else:
                """conditions for different action types"""
                if self.training_strategy == "concurrent":
                    value_var = self.policy(state_var[:, agent_id, :], out_type='v')
                elif self.training_strategy == "centralized":
                    value_var = self.policy(whole_state_var, out_type='v')

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        rewards = []
        infos = []
        avg_speeds = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]
        pets = []
        collision_rate = []
        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 4)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

            for vehicle in env.road.vehicles:
                vehicle.crossed_t = None

            total_collisions = 0
            total_steps = 0
            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                          "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                          '.mp4')
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      5))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not done:
                step += 1
                action = self.action(state, n_agents, action_mask)
                state, reward, done, info = env.step(action)
                action_mask = info["action_mask"]
                avg_speed += info["average_speed"]
                rendered_frame = env.render(mode="rgb_array")
                # Check for collisions
                for vehicle in env.road.vehicles:
                    if vehicle.crashed:  # Check if the vehicle has crashed
                        total_collisions += 1
                # Check for vehicle crossing events
                for i, vehicle_1 in enumerate(env.road.vehicles):
                    for j, vehicle_2 in enumerate(env.road.vehicles):
                        if i != j:
                            # Check if vehicle_2 has just crossed the path of vehicle_1
                            if vehicle_2.crossed_t is None and np.linalg.norm(
                                    vehicle_1.position - vehicle_2.position) < (
                                    vehicle_1.LENGTH + vehicle_2.LENGTH) / 2:
                                vehicle_2.crossed_t = step
                            elif vehicle_2.crossed_t is not None:
                                pet = self.calculate_pet(vehicle_1, vehicle_2, step)
                                if pet is not None:
                                    pets.append(pet)

                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            # Calculate the collision rate
            total_steps = step
            col_rate = total_collisions / total_steps
            collision_rate.append(col_rate)

        if video_recorder is not None:
            video_recorder.release()
        env.close()
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, pets, collision_rate

    def calculate_pet(self, vehicle_1, vehicle_2, t_step):
        distance = np.linalg.norm(vehicle_1.position - vehicle_2.position)
        if distance < (vehicle_1.LENGTH + vehicle_2.LENGTH) / 2:
            pet = t_step - vehicle_2.crossed_t
            return pet
        return None

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            # logging.info('Checkpoint loaded: {}'.format(file_path))
            if not self.shared_network:
                self.actors.load_state_dict(checkpoint['model_state_dict'])
                if train_mode:
                    self.actor_optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.actors.train()
                else:
                    self.actors.eval()
            else:
                self.policy.load_state_dict(checkpoint['model_state_dict'])
                if train_mode:
                    self.policy_optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.policy.train()
                else:
                    self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        if not self.shared_network:
            th.save({'global_step': global_step,
                     'model_state_dict': self.actors.state_dict(),
                     'optimizer_state_dict': self.actor_optimizers.state_dict()},
                    file_path)
        else:
            th.save({'global_step': global_step,
                     'model_state_dict': self.policy.state_dict(),
                     'optimizer_state_dict': self.policy_optimizers.state_dict()},
                    file_path)