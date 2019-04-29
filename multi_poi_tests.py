import multi_poi_rover_domain
import numpy as np
import pandas as pd
import learning.MLP as MLP
import multiprocessing
import torch
import random
import qlearner
import sys
import yaml
import pickle

POOL_LIMIT = 10

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


def evaluate_policy(policies, poi_positions, agent_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence, **kwargs):
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
    rd.rover_positions = agent_positions
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


def test_G(poi_positions, agent_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence, **kwargs):
    """
    Tests a set of agents learning direct control actions only on G

    :param agent_positions:
    :param poi_positions: np array, double, the positions of the POI's
    :param num_rovers: integer, number of rovers in the problem
    :param num_steps: integer, number of timesteps in the simulation
    :param num_poi: integer, number of POI in the problem
    :param poi_types: list, the type of EVERY POI in the problem
    :param poi_sequence: dict, the graph sequence of POI parents
    :return: List, The score of the best performing team
    """
    pool = multiprocessing.Pool(POOL_LIMIT)
    agents = []
    best_performance = []
    num_agents = num_rovers
    agent_pool_size = 5
    state_size = 4 + (4*len(poi_sequence)) + len(poi_sequence)
    for _ in range(num_agents):
        agents.append(Agent(state_size, 32, 2, agent_pool_size))

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
                policy_teams.append((p, poi_positions, agent_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence))

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
                        a.policy_pool[r] = MLP.Policy(state_size, 32, 2)
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
    pool = multiprocessing.Pool(POOL_LIMIT)
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

def evaluate_q_hierarchy_performance(rover_domain, agents):
    """
    Tests the Q hierarchcy with no exploratory action
    :param rover_domain: rover domain instance to test the agents in
    :param agents: list of agent policies in the world
    :return: Float, Score of the team on the world problem
    """
    done = False
    state = rover_domain.rover_observations
    while not done:
        actions = []
        for i, a in enumerate(agents):
            s = np.array(state[i])
            s = s.flatten()
            visited = []
            for key in sorted(rover_domain.poi_visited):
                visited.append(int(rover_domain.poi_visited[key]))

            # Doing a test, ie not taking exploratory action and only evaluating the performance based on
            # the policy, select the action with epsilon=0
            selection = a.select_action(str(visited))
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
        state, reward, done, _ = rover_domain.step(actions)

        # Updates the sequence map
        rover_domain.update_sequence_visits()
    # Get score out
    return rover_domain.sequential_score()


def test_q_learn_hierarchy(poi_positions, agent_positions, num_rovers, num_steps, num_poi, poi_types, poi_sequence,
                           **kwargs):
    agents = []
    best_performance = []
    eps = 0.9

    for _ in range(num_rovers):
        agents.append(HierarchyAgent(len(poi_sequence)))

    print("Begining test...")
    for iteration in range(10000):
        rd = multi_poi_rover_domain.SequentialPOIRD(num_rovers, num_poi, num_steps, poi_types, poi_sequence, **kwargs)
        eps = eps*0.999
        rd.poi_positions = poi_positions
        rd.rover_positions = agent_positions
        done = False
        state = rd.rover_observations
        if iteration == 9999:
            choices = []
        while not done:
            actions = []
            for i, a in enumerate(agents):
                s = np.array(state[i])
                s = s.flatten()
                visited = []
                for poi_id in sorted(rd.poi_visited):
                    visited.append(int(rd.poi_visited[poi_id]))

                selection = a.select_action(str(visited), eps)
                if iteration == 9999:
                    choices.append(selection)
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
        # Log information
        rewards = [rd.sequential_score()]*len(agents)
        test_rd = multi_poi_rover_domain.SequentialPOIRD(num_rovers, num_poi, num_steps, poi_types, poi_sequence, **kwargs)
        best_performance.append(evaluate_q_hierarchy_performance(test_rd, agents))
        if iteration%100 == 0:
            print("Iteration: {}, Score: {}".format(iteration, rewards))
            with open("Q-paths.npy", 'wb') as f:
                np.save(f, np.array(rd.rover_position_histories))

        # Update Q tables
        for i, a in enumerate(agents):
            a.update_policy(rewards[i])
        # Log information for the last path
        if iteration == 9999:
            print('Gathering and logging agent and POI positions...')
            # Combine the poi positions and poi types into a single tuple
            data = [list(a[0])+[a[1]] for a in zip(poi_positions, poi_types)]
            pois = pd.DataFrame(columns=["X", 'Y', 'Type'], data=data)
            index = pd.MultiIndex.from_product((range(num_steps+1), ["Agent {}".format(i+1) for i in range(num_rovers)]),
                                               names=["Timestep", "ID"])
            data = []
            positions = np.asarray(rd.rover_position_histories)
            for t in positions:
                for a in t:
                    data.append(a)
            agent_paths = pd.DataFrame(index=index, data=data, columns=["X", 'Y'])
            # Add a NaN to the end of the choices, since the agent actions are n+1, and it's fucky at the moment...
            choices.extend([np.nan]*num_rovers)
            agent_paths["Choices"] = choices

            pois.to_hdf(config_data["H5 Output File"], key=key+"/q/POI_Positions")
            agent_paths.to_hdf(config_data["H5 Output File"], key=key+"/q/Agent_Positions")
    return best_performance


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as stream:
        try:
            config_data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    poi_positions = np.array(config_data["POI Positions"], dtype="double")
    agent_positions = np.array(config_data["Agent Positions"], dtype="double")
    num_poi = len(poi_positions)
    num_agents = config_data["Number of Agents"]
    num_steps = config_data["Number of Timesteps"]
    poi_types = config_data["POI Types"]
    if config_data["Shuffle POI"]:
        random.shuffle(poi_types)
    poi_sequence = config_data["POI Sequence"]

    key = config_data["Experiment Name"] + "/" + "agents_" + str(num_agents)
    trials = config_data["Trials"]


    performance = []
    for i in range(trials):
        performance.append(test_G(poi_positions, agent_positions, num_agents, num_steps, num_poi, poi_types, poi_sequence))
    best_performance = pd.DataFrame(performance)
    best_performance.to_hdf(config_data["H5 Output File"], key=key+"/G")

    # Can perform these tests in parallel
    performance = []
    args = [(poi_positions, agent_positions, num_agents, num_steps, num_poi, poi_types, poi_sequence)]*trials

    pool = multiprocessing.Pool(POOL_LIMIT)
    performance = pool.starmap(test_q_learn_hierarchy, args)
    best_performance = pd.DataFrame(performance)
    best_performance.to_hdf(config_data["H5 Output File"], key=key+"/q/Performance")

