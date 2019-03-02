import numpy as np
import random
cimport cython

cdef class Ccea:
    cdef public double mut_prob
    cdef public double epsilon
    cdef public int n_populations
    cdef public int population_size
    cdef public int policy_size
    cdef public double[:, :, :] pops
    cdef public double[:, :] fitness
    cdef public int[:, :] team_selection

    def __init__(self, num_agents, mutation_rate=0.1, epsilon=0.1, population_size=50):
        self.mut_prob = mutation_rate
        self.epsilon = epsilon
        self.n_populations = num_agents  # One population for each rover
        self.population_size  = population_size  # Number of policies in each pop
        # TODO these are hard-coded at the moment.
        n_inputs = 8
        n_outputs = 2
        n_nodes = 15 # Number of nodes in hidden layer
        self.policy_size = (n_inputs + 1)*n_nodes + (n_nodes + 1)*n_outputs  # Number of weights for NN
        self.pops = np.zeros((self.n_populations, self.population_size, self.policy_size))
        self.fitness = np.zeros((self.n_populations, self.population_size))
        self.team_selection = np.zeros((self.n_populations, self.population_size), dtype = np.int32)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef reset_populations(self):  # Re-initializes CCEA populations for new run
        cdef int pop_index, policy_index, w
        for pop_index in range(self.n_populations):
            for policy_index in range(self.population_size):
                for w in range(self.policy_size):
                    self.pops[pop_index, policy_index, w] = random.uniform(-1, 1)
                self.team_selection[pop_index, policy_index] = -1

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef select_policy_teams(self):  # Create policy teams for testing
        cdef int pop_id, policy_id, j, k, rpol
        for pop_id in range(self.n_populations):
            for policy_id in range(self.population_size):
                self.team_selection[pop_id, policy_id] = -1

        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                rpol = random.randint(0, (self.population_size - 1))  # Select a random policy from pop
                k = 0
                while k < j:  # Check for duplicates
                    if rpol == self.team_selection[pop_id, k]:
                        rpol = random.randint(0, (self.population_size - 1))
                        k = -1
                    k += 1
                self.team_selection[pop_id, j] = rpol  # Assign policy to team

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef mutate(self):
        cdef int pop_index, policy_index, target
        cdef int half_pop_length = self.population_size/2
        cdef double rnum
        for pop_index in range(self.n_populations):
            policy_index = half_pop_length
            # print(policy_index)
            while policy_index < self.population_size:
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_prob:
                    # print('mutate')
                    target = random.randint(0, (self.policy_size - 1))  # Select random weight to mutate
                    self.pops[pop_index, policy_index, target] = random.uniform(-1, 1)
                policy_index += 1

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef epsilon_greedy_select(self):  # Replace the bottom half with parents from top half
        cdef int pop_id, policy_id, k, parent
        cdef double rnum
        cdef int half_pop_length = self.population_size/2
        for pop_id in range(self.n_populations):
            policy_id = half_pop_length
            # print(policy_id)
            while policy_id < self.population_size:
                rnum = random.uniform(0, 1)
                # print(rnum)
                if rnum >= self.epsilon:  # Choose best policy
                    # print('Keep Best')
                    for k in range(self.policy_size):
                        self.pops[pop_id, policy_id, k] = self.pops[pop_id, 0, k]  # Best policy
                else:
                    parent = random.randint(0, half_pop_length)  # Choose a random parent
                    # print(parent)
                    for k in range(self.policy_size):
                        self.pops[pop_id, policy_id, k] = self.pops[pop_id, parent, k]  # Random policy
                policy_id += 1

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef down_select(self):  # Create a new offspring population using parents from top 50% of policies
        cdef int pop_id, j, k
        # Reorder populations in terms of fitness (top half = best policies)
        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                k = j + 1
                while k < self.population_size:
                    if self.fitness[pop_id, j] < self.fitness[pop_id, k]:
                        self.fitness[pop_id, j], self.fitness[pop_id, k] = self.fitness[pop_id, k], self.fitness[pop_id, j]
                        self.pops[pop_id, j], self.pops[pop_id, k] = self.pops[pop_id, k], self.pops[pop_id, j]
                    k += 1

        self.epsilon_greedy_select()  # Select parents for offspring population
        self.mutate()  # Mutate offspring population