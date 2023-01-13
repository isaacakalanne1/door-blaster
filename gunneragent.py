import torch
import torch.optim as optim


class PPOAgent:
    def __init__(self, state_size, action_size, clip_epsilon=0.2, gae_lambda=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda

        # Define the policy and value networks
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)

        # Define the optimizers for the networks
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    def collect_data(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []

        # Run the policy network to collect data
        state = env.reset()
        for _ in range(num_steps_per_iteration):
            state_vec = state_to_vec(state)  # assuming that the function state_to_vec has been implemented
            state_vec = torch.from_numpy(state_vec).float()
            state_vec = state_vec.to(device)
            action_probs = self.policy_network(state_vec)
            action = torch.multinomial(action_probs, 1)
            next_state, reward, terminal, _ = env.step(action.item())
            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)
            next_states.append(state_to_vec(next_state))
            done.append(terminal)
            state = next_state
        return states, actions, rewards, next_states, done

    def compute_advantages(self, rewards, next_states, done):
        advantages = []
        next_values = self.value_network(next_states).detach()
        next_values = next_values.squeeze()
        for i in range(len(rewards) - 1, -1, -1):
            if done[i]:
                next_value = 0
            else:
                next_value = next_values[i]
                advantages.append(rewards[i] + self.gae_lambda * next_value - self.value_network(states[i]))
                advantages = torch.stack(advantages[::-1])
                return advantages

    def compute_policy_loss(self, states, actions, advantages):
        old_action_probs = self.policy_network(states).gather(1, actions)
        new_action_probs = self.policy_network(states)
        ratios = new_action_probs / old_action_probs
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def compute_value_loss(self, states, rewards, next_states, done):
        values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).detach().squeeze()
        targets = rewards + (1 - done) * next_values
        return (values - targets).pow(2).mean()

    def train(self, num_iterations):
        for iteration in range(num_iterations):
            states, actions, rewards, next_states, done = self.collect_data()
            advantages = self.compute_advantages(rewards, next_states, done)

            # Update the policy network
            policy_loss = self.compute_policy_loss(states, actions, advantages)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update the value network
            value_loss = self.compute_value_loss(states, rewards, next_states, done)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

