import gym
import numpy as np
import tensorflow as tf

def mulitlayer_perceptron(input_layer, layer_sizes, activation, output_activation):
    # Creates a feed forward network and returns the output layer.
    previous_layer = input_layer
    for layer_size in layer_sizes[:-1]:
        previous_layer = tf.layers.dense(previous_layer, units=layer_size, activation=activation)
    return tf.layers.dense(previous_layer, units=layer_size, activation=activation)


class Episodes:

    def __init__(self):

        self.observations = []
        self.actions = []
        self.weights = []
        self.returns = []
        self.lengths = []



class Policy:

    def __init__(self, environment):

        # Initialize the policy network.
        self.observation_placeholder = tf.placeholder(shape=(None, environment.observation_space.shape[0]), dtype=tf.float32)
        self.logits = mulitlayer_perceptron(self.observation_placeholder, [32, environment.action_space.n], tf.tanh, None)

        # Initialize the action selection operation.
        self.action = tf.squeeze(tf.multinomial(logits=self.logits, num_samples=1), axis=1)

        weights_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
        action_placeholder = tf.placeholder(shape(None,), dtype=tf.int32)

        action_masks = tf.one_hot(action_placeholder, environment.action_space.n)

        log_probabilities = tf.reduce_sum(actions_masks * tf.nn.log_softmax(logits), axis=1)

        # Initialize the pseudo loss function used to obtain the policy gradient.
        psuedo_loss = -tf.reduce_mean(weights_placeholder * log_probabilities)

        # Initialize the training operation.
        self.train = tf.train.AdamOptimizer(learning_rate=0.95).minimize(loss)

        # Initialize the tensorflow session.
        # TODO: check out the other types of sessions.
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def sample(self, observation):
        return self.session.run(self.action, {observation_placeholder: observation.reshape(1, -1)})[0]

    def update(self, episodes):
        pass


environment_name = 'CartPole-v0'
num_epochs = 1
episodes_per_epoch = 100

# Initialize the environment.
environment = gym.make(environment_name)

# Initialize a policy for the environment.
policy = Policy(environment)

for epoch in range(num_epochs):

    # Retrieve the initial observation corresponding to the initial state.
    observation = environment.reset()

    episode_batch = Episodes()

    for step in range(episodes_per_epoch):

        # Optionally Rrender the environment for debugging via visual
        # inspection.
        environment.render()

        # Generate the next actions given the current observation.
        action = policy.sample(observation)

        # Record the action taken for the given observation.
        episode_batch.add_observation(observation)
        episode_batch.add_action(action)

        # Step the environment for the
        observation, reward, done, info = environment.step(actions)

        # Record the reward recieved ...
        episode_batch.add_reward(reward)

        if done:




