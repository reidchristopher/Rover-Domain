import numpy as np


class VectorDotRewardEvaluator:

    def __init__(self, vector):

        self.vector = np.array(vector) / np.linalg.norm(vector)

    def get_reward(self, rover_state):

        return np.dot(self.vector, rover_state)


class StandardBasisRewardEvaluator(VectorDotRewardEvaluator):

    def __init__(self, index, n_dim):

        vector = np.array([1.0 if i == index else 0.0 for i in range(n_dim)])

        super().__init__(vector)
