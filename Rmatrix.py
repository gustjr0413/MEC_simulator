import numpy as np

class Rmatrix():
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.M = np.zeros(action_dim)

    def update(self, action, reward):
        self.M[action] = reward

    def select_action(self):
        return np.argmax(self.M)

    def reset(self):
        self.M = np.zeros(self.action_dim)
