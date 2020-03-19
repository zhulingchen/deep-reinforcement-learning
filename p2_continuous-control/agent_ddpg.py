import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model_ddpg import Actor, Critic
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
START_SIZE = 1024       # when to start training
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
TRAIN_EVERY = 5         # how often to train a batch
TRAIN_STEPS = 3         # how many training steps when a batch is trained

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, state_size, action_size, random_seed, use_per=False):
        """Initialize an Agent object.
        
        Params
        ======
            num_agents (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            use_per (bool): whether to use prioritized replay buffer
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.use_per = use_per

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        if use_per:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step
        self.t_step = 0

    def get_critic_Q(self, states, actions, rewards, next_states, dones, gamma, is_train=True):
        # Get max predicted Q values (for next states) from target model
        if is_train:
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_targets_next)
            Q_expected = self.critic_local(states, actions)
        else:
            self.actor_local.eval()
            self.actor_target.eval()
            self.critic_local.eval()
            self.critic_target.eval()
            with torch.no_grad():
                actions_next = self.actor_target(next_states)
                Q_targets_next = self.critic_target(next_states, actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (gamma * (1 - dones) * Q_targets_next)
                Q_expected = self.critic_local(states, actions)
            self.actor_local.train()
            self.actor_target.train()
            self.critic_local.train()
            self.critic_target.train()
        return Q_expected, Q_targets

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if self.use_per:
            # Convert numpy array to torch tensor
            states = torch.from_numpy(states).float().to(device)
            actions = torch.from_numpy(actions).float().to(device)
            rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(device)
            next_states = torch.from_numpy(next_states).float().to(device)
            dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().unsqueeze(1).to(device)
            # Get max predicted Q values (for next states) from target model
            Q_expected, Q_targets = self.get_critic_Q(states, actions, rewards, next_states, dones, GAMMA, is_train=False)
            # Convert torch tensor to numpy array
            states = states.cpu().data.numpy()
            actions = actions.cpu().data.numpy()
            rewards = rewards.cpu().data.numpy().squeeze().tolist()
            next_states = next_states.cpu().data.numpy()
            dones = dones.cpu().data.numpy().squeeze().astype(np.bool).tolist()
            # Calculate error
            errors = Q_expected - Q_targets
            errors = errors.cpu().data.numpy().squeeze()
            for i in range(self.num_agents):
                self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i], errors[i])
        else:
            for i in range(self.num_agents):
                self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Update time step
        self.t_step += 1

        # If enough samples are available in memory,
        if len(self.memory) >= START_SIZE:
            # Get random subset and learn every TRAIN_EVERY time steps,
            if self.t_step % TRAIN_EVERY == 0:
                for _ in range(TRAIN_STEPS):
                    if self.use_per:
                        experiences, idx_tree, is_weight = self.memory.sample()
                        self.learn(experiences, GAMMA, idx_tree, is_weight)
                    else:
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns epsilon-greedy actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += np.concatenate([np.expand_dims(self.noise.sample(), axis=0) for _ in range(self.num_agents)], axis=0)
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, idx_tree=None, is_weight=None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_expected, Q_targets = self.get_critic_Q(states, actions, rewards, next_states, dones, gamma, is_train=True)
        # Compute critic loss
        if self.use_per:
            assert ((is_weight is not None) and (is_weight.size > 0))
            is_weight = torch.from_numpy(is_weight).float().to(device)
            critic_loss = (is_weight * F.smooth_l1_loss(Q_expected, Q_targets, reduction='none').squeeze()).mean()
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # use gradient norm clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # update priority
        if self.use_per:
            assert((idx_tree is not None) and (len(idx_tree) > 0))
            errors = Q_expected - Q_targets
            errors = errors.cpu().data.numpy().squeeze()
            for i in range(self.memory.batch_size):
                self.memory.update(idx_tree[i], errors[i])
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state