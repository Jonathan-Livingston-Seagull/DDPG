
import tensorflow as tf


class Actor(tf.keras.Model):
    """Actor (Policy) Model."""

    def __init__(self, action_size, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=fc1_units, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=fc2_units, activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(action_size, activation=tf.nn.tanh)

    def call(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return self.fc3(self.fc2(self.fc1(tf.convert_to_tensor(state))))


class Critic(tf.keras.Model):
    """Critic (Value) Model."""

    def __init__(self, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======

            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fcs1 = tf.keras.layers.Dense(fcs1_units, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(fc2_units, activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        return self.fc3(self.fc2(tf.concat([self.fcs1(tf.convert_to_tensor(state)), tf.convert_to_tensor(action)], axis=1)))
