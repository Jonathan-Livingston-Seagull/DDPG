import numpy as np
import random
import copy
from collections import namedtuple, deque
import tensorflow as tf

from model_tf import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(action_size)
        self.actor_target = Actor(action_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic()
        self.critic_target = Critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        action = self.actor_local(tf.convert_to_tensor(state))
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def critic_loss(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        inter_1 = tf.multiply(Q_targets_next, (tf.subtract(tf.convert_to_tensor(1.0, dtype=tf.float64), dones)))
        Q_targets = tf.add(rewards, tf.multiply(tf.convert_to_tensor(gamma, dtype=tf.float64), inter_1))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        return tf.reduce_mean(tf.square(Q_expected - Q_targets))

    def actor_loss(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        return -tf.reduce_mean(self.critic_local(states, actions_pred))

    def learn(self, experiences, gamma):
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
        trainable_variables_critic = self.critic_local.trainable_variables + self.actor_target.trainable_variables +  \
                              self.critic_target.trainable_variables
        trainable_variables_actor = self.critic_local.trainable_variables + self.actor_local.trainable_variables

        self.critic_optimizer.minimize(lambda: self.critic_loss(experiences, gamma), trainable_variables_critic)
        self.actor_optimizer.minimize(lambda: self.actor_loss(experiences), trainable_variables_actor)
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        weights = local_model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau* weights[i] + (1 - tau) * target_weights[i]
        target_model.set_weights(target_weights)


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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]))
        actions = tf.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]))
        rewards = tf.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]))
        next_states = tf.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = tf.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float64)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)