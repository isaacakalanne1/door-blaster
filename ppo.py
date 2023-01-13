import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gameenv import GameEnv


def get_action(state, agent):
    action_probs = agent(state)
    action_probs = torch.softmax(action_probs, dim=-1)
    return action_probs


def select_action(state, agent):
    state = torch.FloatTensor(state).unsqueeze(0)
    action_probs = agent(state)
    action_probs = torch.softmax(action_probs, dim=-1)
    action = action_probs.multinomial(num_samples=1).item()
    return action


class PPOAgent(nn.Module):
    def __init__(self, state_sz, action_sz):
        super(PPOAgent, self).__init__()
        self.fc1 = nn.Linear(state_sz, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_sz)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:
    def __init__(self, state_sz, action_sz, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64, learning_rate=3e-4):
        self.state_size = state_sz
        self.action_size = action_sz
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.agent1 = PPOAgent(state_sz, action_sz)
        self.agent2 = PPOAgent(state_sz, action_sz)
        self.optimizer1 = optim.Adam(self.agent1.parameters(), lr=learning_rate)
        self.optimizer2 = optim.Adam(self.agent2.parameters(), lr=learning_rate)
        self.game_env = GameEnv()

    def update(self, agent, optimizer, states, actions, old_action_probs, rewards, dones):
        for _ in range(self.ppo_epochs):
            for state, action, old_action_prob, reward, done in zip(states, actions, old_action_probs, rewards, dones):
                action_probs = get_action(state, agent)
                prob = action_probs.squeeze(0)[action]
                ratio = prob / old_action_prob
                advantage = reward - (1 - done)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                loss = -torch.min(surr1, surr2)
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

    def train(self, n_episodes):
        scoresCount = []
        for episode in range(n_episodes):
            state = self.game_env.reset()
            score = 0
            while True:
                action1 = select_action(state, self.agent1)
                action2 = select_action(state, self.agent2)
                actions = [action1, action2]
                old_action_probs1 = get_action(torch.FloatTensor(state).unsqueeze(0), self.agent1).detach()
                old_action_probs2 = get_action(torch.FloatTensor(state).unsqueeze(0), self.agent2).detach()
                old_action_probs = [old_action_probs1, old_action_probs2]
                next_state, reward, done, = self.game_env.step(actions)
                score += reward
                states = [torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(state).unsqueeze(0)]
                self.update(self.agent1, self.optimizer1, states, actions, old_action_probs, reward, done)
                self.update(self.agent2, self.optimizer2, states, actions, old_action_probs, reward, done)
                state = next_state
                if done:
                    scoresCount.append(score)
                    print("Episode: {}, Score: {}".format(episode, score))
                    break
        return scoresCount


if __name__ == 'main':
    state_size = len(GameEnv().reset())
    action_size = 8
    ppo = PPO(state_size, action_size)
    scores = ppo.train(n_episodes=1000)
