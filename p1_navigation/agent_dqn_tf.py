import numpy as np
import random
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  # disable eager execution (enabled by TF2 by default)

from model_dqn_tf import QNetwork
from replay_buffer_tf import ReplayBuffer, PrioritizedReplayBuffer

# constant values
BUFFER_SIZE = int(1e5)  # replay buffer size
START_SIZE = int(1e3)  # when to start training
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
TRAIN_EVERY = 4  # how often to train a batch
TRAIN_STEPS = 2  # how many training steps when a batch is trained



class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, seed, use_double=False, use_dueling=False, use_per=False):
        """Initialize an Agent object.

        Params
        ======
            action_size (int): dimension of each action
            seed (int): random seed
            use_double (bool): whether to use double deep Q-learning
            use_dueling (bool): whether to use the dueling network architecture
            use_per (bool): whether to use prioritized replay buffer
        """
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_double = use_double
        self.use_per = use_per

        # Q-Network
        self.qnetwork_local = QNetwork(action_size, seed, use_dueling=use_dueling)
        self.qnetwork_target = QNetwork(action_size, seed, use_dueling=use_dueling)
        self.optimizer = tf.keras.optimizers.Adam(lr=LR)

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
                idx = tf.argmax(self.qnetwork_local(next_state), axis=-1, output_type=tf.int32)
                Q_target_next = tf.expand_dims(tf.gather_nd(self.qnetwork_target(next_state), tf.stack([tf.range(len(idx)), idx], axis=-1)), axis=-1)
            else:
                Q_target_next = tf.expand_dims(tf.reduce_max(self.qnetwork_target(next_state), axis=-1), axis=-1)
            # Compute Q targets for current states
            Q_target = reward + (gamma * (1 - done) * Q_target_next)
            # Get expected Q values from local model
            Q_expected = tf.expand_dims(tf.gather_nd(self.qnetwork_local(state), tf.stack([tf.range(len(action)), action[:, 0]], axis=-1)), axis=-1)
        else:
            if self.use_double:
                idx = np.argmax(self.qnetwork_local.predict(next_state), axis=-1).astype('int')
                Q_target_next = np.take_along_axis(self.qnetwork_target.predict(next_state), idx[:, np.newaxis], axis=-1)
            else:
                Q_target_next = self.qnetwork_target.predict(next_state).max(-1)[:, np.newaxis]
            # Compute Q targets for current states
            Q_target = reward + (gamma * (1 - done) * Q_target_next)
            # Get expected Q values from local model
            Q_expected = np.take_along_axis(self.qnetwork_local.predict(state), action, axis=-1)
        return Q_expected, Q_target

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.use_per:
            state = state[np.newaxis]
            action = np.array([action])[np.newaxis]
            next_state = next_state[np.newaxis]
            done = int(done)
            # Get max predicted Q values (for next states) from target model
            Q_expected, Q_target = self.get_Q(state, action, reward, next_state, done, GAMMA, is_train=False)
            state = state.squeeze()
            action = action.item()
            next_state = next_state.squeeze()
            done = bool(done)
            # Calculate error
            error = Q_expected - Q_target
            error = error.item()
            self.memory.add(state, action, reward, next_state, done, error)
        else:
            self.memory.add(state, action, reward, next_state, done)

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

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_values = self.qnetwork_local.predict(np.atleast_2d(state))

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.squeeze())
        else:
            action = random.choice(np.arange(self.action_size))
        return action.item()

    def learn(self, experiences, gamma, idx_tree=None, is_weight=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            idx_tree
            experiences (Tuple[numpy.ndarray]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute loss
        with tf.GradientTape() as tape:
            # Get max predicted Q values (for next states) from target model
            Q_expected, Q_targets = self.get_Q(states, actions, rewards, next_states, dones, gamma, is_train=True)
            if self.use_per:
                assert (is_weight is not None) and (is_weight.size > 0)
                huber_loss = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
                loss = tf.reduce_mean(is_weight * tf.squeeze(huber_loss(Q_expected, Q_targets)))
            else:
                loss = tf.reduce_mean(tf.square(Q_targets - Q_expected))

        # One-step gradient descent for training network weights
        variables = self.qnetwork_local.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # update priority
        if self.use_per:
            assert (idx_tree is not None) and (len(idx_tree) > 0)
            errors = Q_expected - Q_targets
            if tf.executing_eagerly():
                errors = errors.numpy().squeeze()
            else:
                errors = tf.keras.backend.eval(errors).squeeze()
            for i in range(self.memory.batch_size):
                self.memory.update(idx_tree[i], errors[i])

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (TensorFlow model): weights will be copied from
            target_model (TensorFlow model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.trainable_variables, local_model.trainable_variables):
            target_param.assign(tau * local_param + (1.0 - tau) * target_param)