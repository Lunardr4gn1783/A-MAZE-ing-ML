import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    """A Deep Q-Network Agent with a multi-input CNN architecture."""
    def __init__(self, vision_shape, ray_shape, direction_shape, action_size):
        self.vision_shape = vision_shape
        self.ray_shape = ray_shape
        self.direction_shape = direction_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Input branch for the 5x5 vision grid
        vision_input = Input(shape=self.vision_shape, name='vision_input')
        conv = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(vision_input)
        vision_flat = Flatten()(conv)

        # Input branch for the 4 raycast distances
        ray_input = Input(shape=self.ray_shape, name='ray_input')

        # Input branch for the goal direction vector
        direction_input = Input(shape=self.direction_shape, name='direction_input')

        # Combine the outputs of all three branches
        combined = concatenate([vision_flat, ray_input, direction_input])

        # Dense layers for final Q-value prediction
        dense1 = Dense(64, activation='relu')(combined)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(self.action_size, activation='linear')(dense2)

        model = Model(inputs=[vision_input, ray_input, direction_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        # Unpack the three-part states for the batch
        vision_states = np.array([t[0][0][0] for t in minibatch])
        ray_states = np.array([t[0][1][0] for t in minibatch])
        direction_states = np.array([t[0][2][0] for t in minibatch])
        
        vision_next_states = np.array([t[3][0][0] for t in minibatch])
        ray_next_states = np.array([t[3][1][0] for t in minibatch])
        direction_next_states = np.array([t[3][2][0] for t in minibatch])

        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        current_states = [vision_states, ray_states, direction_states]
        next_states = [vision_next_states, ray_next_states, direction_next_states]

        target = self.model.predict(current_states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        
        for i in range(len(minibatch)):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * (np.amax(target_next[i]))
        
        self.model.fit(current_states, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
