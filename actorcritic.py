import torch
import torch.nn as nn
import torch.optim as optim

# Define the actor and critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the actor and critic networks
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

# Define the PPO loss function
def ppo_loss(old_policy, new_policy, advantage, epsilon=0.2):
    ratio = new_policy / old_policy
    surrogate_loss = torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
    return -torch.mean(surrogate_loss)

# Define the optimizers for the actor and critic networks
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# Define the training loop
for episode in range(num_episodes):
    # Collect data from the environment
    state = env.reset()
    done = False
    while not done:
        # Get the action probabilities from the actor network
        action_probs = actor(state)

        # Sample an action from the action probabilities
        action = torch.multinomial(action_probs, 1)

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Calculate the advantage
        advantage = reward + gamma * critic(next_state) - critic(state)

        # Calculate the old policy
        old_policy = action_probs[:, action]

        # Calculate the new policy
        new_policy = actor(state)[:, action]

        # Calculate the PPO loss
        ppo_loss = ppo_loss(old_policy, new_policy, advantage)

        # Optimize the actor and critic networks
        actor_optimizer.zero_grad()
        ppo_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        advantage.backward()
        advantage.backward()
        critic_optimizer.step()
        state = next_state

        # Use the state values from the environment to include the relative positions of the enemies and ammo packs to the player
        state = torch.tensor([player1.x, player1.y, player2.x, player2.y, *[e.x for e in enemies], *[e.y for e in enemies],
                              *[a.x for a in ammo_packs], *[a.y for a in ammo_packs]])
        state_dim = 4 + 2 * 10 + 2 * 2
        action_dim = 8
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]  # 0-3 for player1 moving direction, 4-7 for player1 shooting direction



