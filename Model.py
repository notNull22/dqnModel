from collections import deque
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
from CarlaEnv import CarlaEnv

REPLAY_MEMORY_SIZE = 5000
MIN_RM_SIZE = 1000
MINIBATCH = 4
BATCH_SIZE = 1
TRAINING_BATCH_SIZE = 1
MEMORY_FRACTION = 0.8
EPISODES = 100
DISCOUNT = 0.99

MIN_EPSILON = 0.01
AGGREGATE = 10
IMG_WIDTH = 640
IMG_HEIGHT = 360
UPDATE_TARGET_EVERY = 10
MIN_REWARD = -200
LEARNING_RATE = 0.001


class Model:
    def __init__(self):
        self.model_network = self.create_model()
        self.model_target_network = self.create_model()
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.step_counter = 0
        self.epsilon_decay = 0.001
        self.epsilon = 0.99

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=["accuracy"])
        return model

    def replay_buffer_store(self, current_state, action, step_reward, new_state, is_done):
        # print("added")
        self.replay_buffer.append(
            (current_state, action, step_reward, new_state, is_done))

    def get_random(self):
        return random.sample(self.replay_buffer, MINIBATCH)

    def get_greedy_policy_action(self, state):
        if np.random.random() < self.epsilon:
            action = random.randrange(5)
        else:
            state = np.array(state)
            action = tf.math.argmax(
                self.model_network(state), axis=1).numpy()[0]
        return action

    def train(self):
        if len(self.replay_buffer) < MINIBATCH:
            return
        if self.step_counter == UPDATE_TARGET_EVERY:
            self.model_target_network.set_weights(
                self.model_network.get_weights())
            self.step_counter = 0
        random_tuples = self.get_random()
        # print("tuples shape ", random_tuples.shape)
        for random_tuple in random_tuples:
            mini_state, mini_action, mini_reward, mini_new_state, mini_done = random_tuple

        predicted_Q = self.model_network(mini_state)
        next_state_Q = self.model_target_network(mini_new_state)
        next_state_max_Q = tf.math.reduce_max(
            next_state_Q, axis=1, keepdims=1).numpy()
        tarqet_Q = np.copy(predicted_Q)
        for i in range(mini_done.shape[0]):
            target_q_value = mini_reward[i]
            if not mini_done[i]:
                target_q_value += DISCOUNT * next_state_max_Q[i]
            tarqet_Q[i, mini_action[i]] = target_q_value
        self.model_network.train_on_batch(mini_state, tarqet_Q)

        if self.epsilon > MIN_EPSILON:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = MIN_EPSILON
        self.step_counter += 1

    def training(self, env):
        rewards, epsioides, rewards_avg, obj = [], [], [], []
        goal = 0
        f = 0
        for i in range(EPISODES):
            done = False
            ep_reward = 0
            current_state = env.reset()
            while not done:
                action = self.get_greedy_policy_action(current_state)
                new_state, reward, is_done = env.step(action)
                self.replay_buffer_store(
                    current_state, action, reward, new_state, is_done)
                ep_reward += reward
                current_state = new_state
                self.train()
            rewards.append(ep_reward)
            obj.append(goal)
            epsioides.append(i)
            reward_avg = np.mean(rewards[-100:])
            rewards_avg.append(reward_avg)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, EPISODES, ep_reward, self.epsilon,
                                                                             rewards_avg))
            if reward_avg >= -110 and ep_reward >= -108:
                self.model_network.save(
                    ("saved_networks/dqn_model{0}".format(f)))
                self.model_target_network.save_weights(
                    ("saved_networks/dqn_model{0}/net_weights{0}.h5".format(f)))
            f = f + 1


print("hi")
if __name__ == "__main__":
    print("hi")
    carlaEnv = CarlaEnv()
    model = Model()
    model.training(carlaEnv)
