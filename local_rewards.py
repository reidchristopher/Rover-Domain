import numpy as np
from scipy.stats import norm


class VectorDotRewardEvaluator:

    def __init__(self, vector):

        self.vector = np.array(vector) / np.linalg.norm(vector)

    def get_reward(self, rover_state):

        return np.dot(self.vector, rover_state)


class StandardBasisRewardEvaluator(VectorDotRewardEvaluator):

    def __init__(self, index, n_dim, positive=True):

        vector = np.array([1.0 if i == index else 0.0 for i in range(n_dim)])

        if not positive:
            vector *= -1

        super().__init__(vector)


class OneDGaussianReward:

    def __init__(self, index, mean, std):

        self.mean = mean
        self.std = std
        self.index = index

    def get_reward(self, rover_state):

        return norm(self.mean, self.std).pdf(rover_state[self.index])

