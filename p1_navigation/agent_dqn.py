import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from model_dqn import QNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
TRAIN_EVERY = 4         # how often to train a batch
TRAIN_STEPS = 2         # how many training steps when a batch is trained

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, use_double=False, use_dueling=False, use_per=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            use_double (bool): whether to use double deep Q-learning
            use_dueling (bool): whether to use the dueling network architecture
            use_per (bool): whether to use prioritized replay buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_double = use_double
        self.use_per = use_per

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, use_dueling=use_dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, use_dueling=use_dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if use_per:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step
        self.t_step = 0

    def get_Q(self, state, action, reward, next_state, done, gamma, is_train=True):
        # Get max predicted Q values (for next states) from target model
        if is_train:
            if self.use_double:
                _, idx = self.qnetwork_local(next_state).detach().max(1)
                Q_target_next = self.qnetwork_target(next_state).detach().gather(1, idx.unsqueeze(1))
            else:
                Q_target_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            Q_target = reward + (gamma * (1 - done) * Q_target_next)
            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(state).gather(1, action)
        else:
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            with torch.no_grad():
                if self.use_double:
                    _, idx = self.qnetwork_local(next_state).squeeze().max(0)
                    Q_target_next = self.qnetwork_target(next_state).squeeze()[idx]
                else:
                    Q_target_next = self.qnetwork_target(next_state).squeeze().max()
                # Compute Q targets for current states
                Q_target = reward + (gamma * (1 - done) * Q_target_next)
                # Get expected Q values from local model
                Q_expected = self.qnetwork_local(state).squeeze()[action]
            self.qnetwork_local.train()
            self.qnetwork_target.train()
        return Q_expected, Q_target

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.use_per:
            # Convert numpy array to torch tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            done = int(done)
            # Get max predicted Q values (for next states) from target model
            Q_expected, Q_target = self.get_Q(state, action, reward, next_state, done, GAMMA, is_train=False)
            # Convert torch tensor to numpy array
            state = state.cpu().data.numpy().squeeze()
            next_state = next_state.cpu().data.numpy().squeeze()
            done = bool(done)
            # Calculate error
            error = Q_expected - Q_target
            error = error.cpu().data.numpy().item()
            self.memory.add(state, action, reward, next_state, done, error)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Update time step
        self.t_step += 1

        # If enough samples are available in memory,
        if len(self.memory) >= BATCH_SIZE:
            # Get random subset and learn every TRAIN_EVERY time steps,
            if self.t_step % TRAIN_EVERY == 0:
                for _ in range(TRAIN_STEPS):
                    if self.use_per:
                        experiences, idx_tree, is_weight = self.memory.sample()
                    else:
                        experiences = self.memory.sample()
                    self.learn(experiences, idx_tree, is_weight, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).squeeze()
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action.item()

    def learn(self, experiences, idx_tree, is_weight, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            idx_tree
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_expected, Q_targets = self.get_Q(states, actions, rewards, next_states, dones, gamma, is_train=True)

        # Calculate error
        errors = Q_expected - Q_targets
        errors = errors.cpu().data.numpy().squeeze()

        # update priority
        for i in range(self.memory.batch_size):
            self.memory.update(idx_tree[i], errors[i])

        # Compute loss
        if self.use_per:
            is_weight = torch.from_numpy(is_weight).float().to(device)
            loss = (is_weight * (Q_expected - Q_targets) ** 2).mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network  
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)