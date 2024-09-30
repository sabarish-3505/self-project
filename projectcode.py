import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, optimizers

# Environment setup
env = gym.make("Pendulum-v1", render_mode="human")

num_states = env.observation_space.shape[0]
num_actions = 11  # Discretize the action space to 11 actions (-2 to 2 with step 0.4)
action_space = np.linspace(-2, 2, num_actions)

print("Size of State Space ->  {}".format(num_states))
print("Size of Action Space ->  {}".format(num_actions))

# Define the Q-network
def build_q_network():
    model = models.Sequential()
    model.add(layers.Input(shape=(num_states,)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(num_actions, activation="linear"))
    return model

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_counter = 0

    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.buffer_counter % self.buffer_size] = (state, action, reward, next_state, done)
        self.buffer_counter += 1

    def sample(self):
        indices = np.random.choice(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

# Hyperparameters
total_episodes = 600
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
batch_size = 64
buffer_size = 50000

# Create Q-networks
q_network = build_q_network()
target_q_network = build_q_network()
target_q_network.set_weights(q_network.get_weights())
optimizer = optimizers.Adam(learning_rate)

# Initialize replay buffer
replay_buffer = ReplayBuffer(buffer_size, batch_size)

# Training loop
reward_history = []

for episode in range(total_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action_idx = np.random.choice(num_actions)
        else:
            q_values = q_network.predict(state[np.newaxis])
            action_idx = np.argmax(q_values[0])

        action = action_space[action_idx]
        next_state, reward, done, truncated, _ = env.step([action])
        done = done or truncated
        episode_reward += reward

        replay_buffer.store(state, action_idx, reward, next_state, done)

        state = next_state

        # Sample a batch and train the network
        if len(replay_buffer.buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample()

            # Predict Q-values for the next states using the target network
            next_q_values = target_q_network.predict(next_states)
            max_next_q_values = np.max(next_q_values, axis=1)

            target_qs = rewards + (1 - dones) * gamma * max_next_q_values

            with tf.GradientTape() as tape:
                q_values = q_network(states)
                q_values = tf.gather_nd(q_values, np.array([[i, actions[i]] for i in range(batch_size)]))
                loss = tf.reduce_mean(tf.square(target_qs - q_values))

            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

        # Update target network
        if done:
            target_q_network.set_weights(q_network.get_weights())

    reward_history.append(episode_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}, Reward: {episode_reward}, Epsilon: {epsilon}")

# Plotting the rewards
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Save the model
q_network.save_weights("pendulum_dqn.h5")
