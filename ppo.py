import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gameenv import GameEnv

class PPO_Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):

    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module)

    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.
        )


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
        scores_list = []
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
                    scores_list.append(score)
                    print("Episode: {}, Score: {}".format(episode, score))
                    break
        return scores_list


if __name__ == 'main':
    state_size = len(GameEnv().reset())
    action_size = 8
    ppo = PPO(state_size, action_size)
    scores = ppo.train(n_episodes=1000)
