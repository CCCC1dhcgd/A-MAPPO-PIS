import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, hidden_size, output_size, state_split=False):
        super(ActorNetwork, self).__init__()
        self.state_split = state_split
        if self.state_split:
            self.fc11 = nn.Linear(5, hidden_size // 4)
            self.fc12 = nn.Linear(10, hidden_size // 2)
            self.fc13 = nn.Linear(10, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 4 + hidden_size // 2 + hidden_size // 2, hidden_size)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action_mask):
        if self.state_split:
            state1 = th.cat([state[:, 0:1], state[:, 5:6], state[:, 10:11], state[:, 15:16], state[:, 20:21]], 1)
            state2 = th.cat([state[:, 1:3], state[:, 6:8], state[:, 11:13], state[:, 16:18], state[:, 21:23]], 1)
            state3 = th.cat([state[:, 3:5], state[:, 8:10], state[:, 13:15], state[:, 18:20], state[:, 23:25]], 1)
            out1 = F.relu(self.fc11(state1))
            out2 = F.relu(self.fc12(state2))
            out3 = F.relu(self.fc13(state3))
            out = th.cat([out1, out2, out3], 1)
        else:
            out = F.relu(self.fc1(state))

        out = F.relu(self.fc2(out))
        """invalid action masking"""
        logits = self.fc3(out)
        logits[action_mask == 0] = th.tensor([-1e8])
        return F.log_softmax(logits + 1e-8, dim=1)


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, hidden_size, output_size=1, state_split=False):
        super(CriticNetwork, self).__init__()
        self.state_split = state_split
        if self.state_split:
            self.fc11 = nn.Linear(5, hidden_size // 4)
            self.fc12 = nn.Linear(10, hidden_size // 2)
            self.fc13 = nn.Linear(10, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 4 + hidden_size // 2 + hidden_size // 2, hidden_size)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state):
        if self.state_split:
            state1 = th.cat([state[:, 0:1], state[:, 5:6], state[:, 10:11], state[:, 15:16], state[:, 20:21]], 1)
            state2 = th.cat([state[:, 1:3], state[:, 6:8], state[:, 11:13], state[:, 16:18], state[:, 21:23]], 1)
            state3 = th.cat([state[:, 3:5], state[:, 8:10], state[:, 13:15], state[:, 18:20], state[:, 23:25]], 1)
            out1 = F.relu(self.fc11(state1))
            out2 = F.relu(self.fc12(state2))
            out3 = F.relu(self.fc13(state3))
            out = th.cat([out1, out2, out3], 1)
        else:
            out = F.relu(self.fc1(state))

        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that sharing lower-layer representations but
    have distinct output layers
    """

    def __init__(self, state_dim, action_dim, hidden_size, critic_output_size=1, state_split=False):
        super(ActorCriticNetwork, self).__init__()
        self.state_split = state_split
        if self.state_split:
            self.fc11 = nn.Linear(5, hidden_size // 4)
            self.fc12 = nn.Linear(10, hidden_size // 2)
            self.fc13 = nn.Linear(10, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 4 + hidden_size // 2 + hidden_size // 2, hidden_size)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)

    def __call__(self, state, action_mask=None, out_type='p'):
        if self.state_split:
            state1 = th.cat([state[:, 0:1], state[:, 5:6], state[:, 10:11], state[:, 15:16], state[:, 20:21]], 1)
            state2 = th.cat([state[:, 1:3], state[:, 6:8], state[:, 11:13], state[:, 16:18], state[:, 21:23]], 1)
            state3 = th.cat([state[:, 3:5], state[:, 8:10], state[:, 13:15], state[:, 18:20], state[:, 23:25]], 1)
            out1 = F.relu(self.fc11(state1))
            out2 = F.relu(self.fc12(state2))
            out3 = F.relu(self.fc13(state3))
            out = th.cat([out1, out2, out3], 1)
        else:
            out = F.relu(self.fc1(state))

        out = F.relu(self.fc2(out))
        if out_type == 'p':
            """invalid action masking"""
            logits = self.actor_linear(out)
            # logits[action_mask == 0] = th.tensor([-1e8]).cuda()
            logits[action_mask == 0] = th.tensor([-1e8])
            return F.log_softmax(logits + 1e-8, dim=1)
        else:
            return self.critic_linear(out)
