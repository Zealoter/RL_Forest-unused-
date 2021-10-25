import gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = func.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Agent(object):
    def __init__(self, **kwargs):
        self.gamma = 0.9
        self.batch_size = 128
        self.capacity = 1000
        self.decay = 200
        self.epsilon_high = 0.9
        self.epsilon_low = 0.05
        self.lr = 0.0002
        self.action_space_dim = 0
        self.state_space_dim = 0
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsilon:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float).view(1, -1)
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(np.array(s0), dtype=torch.float)
        a0 = torch.tensor(np.array(a0), dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor(np.array(r1), dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(np.array(s1), dtype=torch.float)

        y_true = r1 + self.gamma * torch.max(self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)

        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    params = {
        'gamma': 0.8,
        'epsilon_high': 0.9,
        'epsilon_low': 0.05,
        'decay': 200,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    agent = Agent(**params)
    score = []
    mean = []

    for episode in range(1000):
        obs = env.reset()
        total_reward = 1
        while True:
            # env.render()
            act = agent.act(obs)
            new_obs, reward, done, _ = env.step(act)
            if done:
                reward = -1
            agent.put(obs, act, reward, new_obs)

            if done:
                break

            total_reward += reward
            obs = new_obs
            agent.learn()

        print(total_reward)
        score.append(total_reward)
        mean.append(sum(score[-100:]) / 100)
