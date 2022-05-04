from minesweeper_env import *

import numpy as np
import torch
import torch.nn as nn


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        self.network = nn.Sequential()
        # 일단 수업시간에 다룬 DQN figure대로
        self.network.add_module('ConvolutionLayer1',
                                nn.Conv1d(in_channels=self.state_shape[0], out_channels=16, kernel_size=3))
        self.network.add_module('ReLU1', nn.ReLU())
        self.network.add_module('ConvolutionLayer2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3))
        self.network.add_module('ReLU2', nn.ReLU())
        self.network.add_module('Flatten', nn.Flatten())
        # in feature 계산 못하겠어서 일단 state_dim 좀 보려고 대충 256 넣어봄
        self.network.add_module('LinearLayer1', nn.Linear(in_features=256, out_features=128))
        self.network.add_module('LinearLayer2', nn.Linear(in_features=128, out_features=self.n_actions))

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.network(state_t)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        #assert len(qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

env = MinesweeperEnv(9, 9, 10)
observation, reward, is_done = env.step(0)
print("옵져베이션", observation)
print("리워드", reward)
print("이즈던", is_done)
print(env.get_board())
"""
for i in range(몇개의 게임으로 학습할 것인가):
    done = false
    while not done:
        여기서 에피소드 진행하며 학습
    
"""