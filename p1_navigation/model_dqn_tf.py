import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  # disable eager execution (enabled by TF2 by default)



class Feature(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Feature, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(128)
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.actv_1 = tf.keras.layers.ReLU()
        self.dense_2 = tf.keras.layers.Dense(64)
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.actv_2 = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.bn_1(x)
        x = self.actv_1(x)
        x = self.dense_2(x)
        x = self.bn_2(x)
        return self.actv_2(x)


class StateValue(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StateValue, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(32)
        self.actv_1 = tf.keras.layers.ReLU()
        self.value = tf.keras.layers.Dense(1)  # linear activation

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.actv_1(x)
        return self.value(x)


class ActionValue(tf.keras.layers.Layer):
    def __init__(self, action_size, **kwargs):
        super(ActionValue, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(32)
        self.actv_1 = tf.keras.layers.ReLU()
        self.value = tf.keras.layers.Dense(action_size)  # linear activation

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.actv_1(x)
        return self.value(x)


class QNetwork(tf.keras.Model):
    def __init__(self, action_size, seed, use_dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
            use_dueling (bool): whether to use the dueling network architecture
        """
        super(QNetwork, self).__init__()
        tf.random.set_seed(seed)
        self.use_dueling = use_dueling
        self.feature_net = Feature()
        if use_dueling:
            # state value function V(s)
            self.v_net = StateValue()
            # advantage value function A(s, a)
            self.a_net = ActionValue(action_size)
        else:
            # state-action value function Q(s, a)
            self.q_net = ActionValue(action_size)

    def call(self, state):
        """Build a network that maps state -> action values."""
        features = self.feature_net(state)
        if self.use_dueling:
            v_val = self.v_net(features)
            a_val = self.a_net(features)
            return v_val + (a_val - tf.reduce_mean(a_val))
        else:
            return self.q_net(features)