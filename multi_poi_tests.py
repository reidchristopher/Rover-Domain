import multi_poi_rover_domain
import numpy as np
import pandas as pd
import learning.MLP as MLP
import multiprocessing
import torch
import random
import qlearner
import pickle


def manual_poi_optimization(state):
    """
    Manually goes toward the strongest POI sensor in the sensor range specified
    :param state: iterable, length 4, of the sensor readings for a single type of POI
    :return: List, the [dx, dy] action to take to advance on the POI
    """
    try:
        direction = int(np.argmax(state))
    except ValueError:
        return [0.0, 0.0]
    if direction == 0:
        return [-0.7, -0.7]
    elif direction == 1:
        return [0.7, -0.7]
    elif direction == 2:
        return [0.7, 0.7]
    elif direction == 3:
        return [-0.7, 0.7]


def evaluate_policy(policies, poi_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence, **kwargs):
    """
    Creates a new world, evaluates set of policies p on the world.
    :param policies: the set of policies (agents) to try the world
    :param poi_positions: np array, double, the positions of the POI's
    :param num_rovers: integer, number of rovers in the problem
    :param num_steps: integer, number of timesteps in the simulation
    :param num_poi: integer, number of POI in the problem
    :param poi_types: list, the type of EVERY POI in the problem
    :param poi_sequence: dict, the graph sequence of POI parents
    :return: Rewards for each agent
    """
    rd = multi_poi_rover_domain.SequentialPOIRD(num_rovers, num_poi, num_steps, poi_types, poi_sequence, **kwargs)
    rd.poi_positions = poi_positions
    done = False
    state = rd.rover_observations
    while not done:
        actions = []
        for i, p in enumerate(policies):
            s = np.array(state[i])
            s = s.flatten()
            visited = []
            for key in sorted(rd.poi_visited):
                visited.append(int(rd.poi_visited[key]))
            s = np.concatenate((s, np.array(visited)))
            s = torch.tensor(s.flatten())
            with torch.set_grad_enabled(False):
                actions.append(np.array(p.forward(s.float())))
        actions = np.array(actions, dtype="double")
        state, reward, done, _ = rd.step(actions)
        # Updates the sequence map
        rd.update_sequence_visits()
    with open("G-paths.npy", 'wb') as f:
        np.save(f, np.array(rd.rover_position_histories))
    return [rd.sequential_score()]*len(policies)


def evaluate_policy_hierarchy(policies, poi_positions, num_rovers, num_steps, num_poi,
                              poi_types, poi_sequence, **kwargs):
    """
    Creates a new world, evaluates set of policies p on the world.
    :param policies: the set of policies (agents) to try the world
    :param poi_positions: np array, double, the positions of the POI's
    :param num_rovers: integer, number of rovers in the problem
    :param num_steps: integer, number of timesteps in the simulation
    :param num_poi: integer, number of POI in the problem
    :param poi_types: list, the type of EVERY POI in the problem
    :param poi_sequence: dict, the graph sequence of POI parents
    :return: Rewards for each agent
    """
    rd = multi_poi_rover_domain.SequentialPOIRD(num_rovers, num_poi, num_steps, poi_types, poi_sequence, **kwargs)
    rd.poi_positions = poi_positions
    done = False
    state = rd.rover_observations

    while not done:
        actions = []
        for i, p in enumerate(policies):
            s = np.array(state[i])
            s = s.flatten()
            visited = []

            for key in sorted(rd.poi_visited):
                visited.append(int(rd.poi_visited[key]))

            s = np.concatenate((s, np.array(visited)))
            s = torch.tensor(s.flatten())
            with torch.set_grad_enabled(False):
                output = np.array(p.forward(s.float()))
                selection = np.argmax(output)
                if selection == 0:
                    a = manual_poi_optimization(s.flatten()[4:8])
                elif selection == 1:
                    a = manual_poi_optimization(s.flatten()[8:12])
                elif selection == 2:
                    a = manual_poi_optimization(s.flatten()[12:16])
                elif selection == 3:
                    a = manual_poi_optimization(s.flatten()[16:20])
                elif selection == 4:
                    a = manual_poi_optimization(s.flatten()[20:24])
                elif selection == 5:
                    a = manual_poi_optimization(s.flatten()[24:28])
                actions.append(a)
        actions = np.array(actions, dtype="double")
        # print(actions)
        state, reward, done, _ = rd.step(actions)
        # Updates the sequence map
        rd.update_sequence_visits()
    return [rd.sequential_score()]*len(policies)


class HierarchyAgent:
    def __init__(self, num_actions):
        self.learner = qlearner.QLearner(num_actions=num_actions)
        self.state_hist = []
        self.action_hist = []

    def select_action(self, state, eps=0.0):
        """
        Given state, return the best action to take
        Selects epsilon greedy if eps is non-zero

        :param state:
        :param eps:
        :return: Selected action
        """
        self.state_hist.append(state)
        self.learner.epsilon = eps
        self.learner.checkState(state)
        action = self.learner.selectAction(state)
        self.action_hist.append(action)
        return action

    def update_policy(self, reward):
        """
        Updates the Q learner parameters based on the
        :param reward:
        :return: None
        """
        # Assemble SASR tuples
        SASR = []
        for i, s in enumerate(self.state_hist[:-1]):
            s_prime = self.state_hist[i+1]
            a = self.action_hist[i]
            # Use a linear decay in the reward signal
            r = reward*(i/len(self.state_hist))
            SASR.append((s, a, s_prime, r))
        # Learn based on the tuples, starting from highest reward
        for s, a, s_prime, r in reversed(SASR):
            self.learner.updateQValue(s, a, s_prime, r)
        # Reset history
        self.state_hist = []
        self.action_hist = []


class Agent:
    def __init__(self, input_l, middle_l, out_l, pool_size=10):
        self.policy_pool = []
        self.cum_rewards = [0]*pool_size
        for _ in range(pool_size):
            self.policy_pool.append(MLP.Policy(input_l, middle_l, out_l))

    def reset(self):
        self.cum_rewards = [0]*len(self.cum_rewards)


def test_G(poi_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence, **kwargs):
    """
    Tests a set of agents learning direct control actions only on G

    :param poi_positions: np array, double, the positions of the POI's
    :param num_rovers: integer, number of rovers in the problem
    :param num_steps: integer, number of timesteps in the simulation
    :param num_poi: integer, number of POI in the problem
    :param poi_types: list, the type of EVERY POI in the problem
    :param poi_sequence: dict, the graph sequence of POI parents
    :return: List, The score of the best performing team
    """
    pool = multiprocessing.Pool()
    agents = []
    best_performance = []
    num_agents = num_rovers
    agent_pool_size = 5
    for _ in range(num_agents):
        agents.append(Agent(19, 32, 2, agent_pool_size))

    with torch.set_grad_enabled(False):
        for gen in range(10000):
            teams = []
            for _ in agents:
                teams.append(list(range(agent_pool_size)))
                random.shuffle(teams[-1])
            teams = np.array(teams)
            teams = teams.transpose()
            policy_teams = []
            for t in teams:
                p = []
                for i in range(num_agents):
                    p.append(agents[i].policy_pool[t[i]])
                policy_teams.append((p, poi_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence))

            team_performances = pool.starmap(evaluate_policy, policy_teams)
            # Update the cumulative performance of each policy
            for i, t in enumerate(teams):
                for a in range(len(agents)):
                    agents[a].cum_rewards[t[a]] += team_performances[i][a]
            print("Gen {} top 5 avg team: {} best team: {}".format(gen,
                                                                   np.average(team_performances[:5],axis=0),
                                                                   max(team_performances)))
            best_performance.append(max(team_performances)[0])

            # Rank and update each agent population every 10 generations
            if gen % 10 == 0:
                for a in agents:
                    # Gets the keys for policy sorted by highest cumulative score first
                    results = sorted(zip(list(range(len(a.policy_pool))), a.cum_rewards),
                                     key=lambda x: x[1], reverse=True)
                    results = results[0]
                    for r in results[10:20]:
                        copy_state_dict = a.policy_pool[random.choice(results[:10])].state_dict()
                        a.policy_pool[r].load_state_dict(copy_state_dict)
                        a.policy_pool[r].mutate()
                    for r in results[18:]:
                        # Inject random policies into a.policy_pool
                        a.policy_pool[r] = MLP.Policy(19, 32, 2)
                    # zero out the score again
                    a.reset()
    return best_performance


def test_hierarchy(poi_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence, **kwargs):
    """
    Tests the hierarchy on the sequential learning problem. Uses a-priori low-level policies.

    :param poi_positions: np array, double, the positions of the POI's
    :param num_rovers: integer, number of rovers in the problem
    :param num_steps: integer, number of timesteps in the simulation
    :param num_poi: integer, number of POI in the problem
    :param poi_types: list, the type of EVERY POI in the problem
    :param poi_sequence: dict, the graph sequence of POI parents
    :return: List, The score of the best performing team
    """
    pool = multiprocessing.Pool()
    agents = []
    best_performance = []
    num_agents = num_rovers
    agent_pool_size = 5
    actions = 6
    state_size = 19
    for _ in range(num_agents):
        agents.append(Agent(state_size, 32, actions, agent_pool_size))

    with torch.set_grad_enabled(False):
        for gen in range(10000):
            teams = []
            for _ in agents:
                teams.append(list(range(agent_pool_size)))
                random.shuffle(teams[-1])
            teams = np.array(teams)
            teams = teams.transpose()
            policy_teams = []
            for t in teams:
                p = []
                for i in range(num_agents):
                    p.append(agents[i].policy_pool[t[i]])
                policy_teams.append((p, poi_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence))

            team_performances = pool.starmap(evaluate_policy_hierarchy, policy_teams)
            # Update the cumulative performance of each policy
            for i, t in enumerate(teams):
                for a in range(len(agents)):
                    agents[a].cum_rewards[t[a]] += team_performances[i][a]
            print("Gen {} top 5 avg team: {} best team: {}".format(gen,
                                                                   np.average(team_performances[:5], axis=0),
                                                                   max(team_performances)))
            best_performance.append(max(team_performances)[0])

            # Rank and update each agent population
            if gen % 10 == 0:
                for a in agents:
                    # Gets the keys for policy sorted by highest cumulative score first
                    results = sorted(zip(list(range(len(a.policy_pool))), a.cum_rewards),
                                     key=lambda x: x[1], reverse=True)
                    results = results[0]
                    for r in results[5:10]:
                        copy_state_dict = a.policy_pool[random.choice(results[:5])].state_dict()
                        a.policy_pool[r].load_state_dict(copy_state_dict)
                        a.policy_pool[r].mutate()
                    for r in results[10:]:
                        # Inject random policies into a.policy_pool
                        a.policy_pool[r] = MLP.Policy(state_size, 32, actions)
                    # zero out the score again
                    a.reset()
    return best_performance


def test_q_learn_hierarchy(poi_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence, **kwargs):
    agents = []
    best_performance = []
    agent_pool_size = 5
    actions = 6
    state_size = 19
    eps = 0.9

    for _ in range(num_rovers):
        agents.append(HierarchyAgent(len(poi_sequence)))

    print("Begining test...")
    for iteration in range(10000):
        rd = multi_poi_rover_domain.SequentialPOIRD(num_rovers, num_poi, num_steps, poi_types, poi_sequence, **kwargs)
        eps = eps*0.999
        rd.poi_positions = poi_positions
        done = False
        state = rd.rover_observations
        while not done:
            actions = []
            for i, a in enumerate(agents):
                s = np.array(state[i])
                s = s.flatten()
                visited = []
                for key in sorted(rd.poi_visited):
                    visited.append(int(rd.poi_visited[key]))

                selection = a.select_action(str(visited), eps)
                if selection == 0:
                    move = manual_poi_optimization(s.flatten()[4:8])
                elif selection == 1:
                    move = manual_poi_optimization(s.flatten()[8:12])
                elif selection == 2:
                    move = manual_poi_optimization(s.flatten()[12:16])
                elif selection == 3:
                    move = manual_poi_optimization(s.flatten()[16:20])
                elif selection == 4:
                    move = manual_poi_optimization(s.flatten()[20:24])
                elif selection == 5:
                    move = manual_poi_optimization(s.flatten()[24:28])
                actions.append(move)
            actions = np.array(actions, dtype="double")
            state, reward, done, _ = rd.step(actions)

            # Updates the sequence map
            rd.update_sequence_visits()
        # Update Q tables
        rewards = [rd.sequential_score()]*len(agents)
        best_performance.append(rewards[0])
        if iteration%100 == 0:
            print("Iteration: {}, Score: {}".format(iteration, rewards))
            with open("Q-paths.npy", 'wb') as f:
                np.save(f, np.array(rd.rover_position_histories))

        for i, a in enumerate(agents):
            a.update_policy(rewards[i])
    return best_performance


if __name__ == '__main__':
    poi_positions = np.array([[-10, 20], [20, 20], [20, -10], [-10, -10]], dtype="double")
    num_poi = len(poi_positions)
    num_agents = 2
    num_steps = 100
    poi_types = [0, 0, 1, 2]
    poi_sequence = {0: None, 1: [0], 2: [1]}

    key = "two_agent/square_poi_arrangement_test"
    trials = 10
    performance = []
    for i in range(trials):
        performance.append(test_G(poi_positions, num_agents, num_steps, num_poi, poi_types, poi_sequence))
    best_performance = pd.DataFrame(performance)
    best_performance.to_hdf("./hierarchy-multi-reward_best.h5", key="G/"+key)

    # Can perform these tests in parallel
    performance = []
    args = [(poi_positions, num_agents, num_steps, num_poi, poi_types, poi_sequence)]*trials
    pool = multiprocessing.Pool()
    performance = pool.starmap(test_q_learn_hierarchy, args)
    best_performance = pd.DataFrame(performance)
    best_performance.to_hdf("./hierarchy-multi-reward_best.h5", key="q/"+key)
    # best_performance.to_hdf("./q-results-multi-reward_best.h5", key="q/"+key)

    # test_G()
#
    """
    performance = []
    for i in range(5):
        performance.append(test_hierarchy(poi_positions, num_agents, num_steps, num_poi, poi_types, poi_sequence))
    performance = pd.DataFrame(performance)
    performance.to_hdf("./hierarchy-multi-reward_best.h5", key="hierarchy/"+key)
    """
