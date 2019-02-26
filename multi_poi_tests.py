import multi_poi_rover_domain
import numpy as np
import pandas as pd
import learning.MLP as MLP
import multiprocessing
import torch
import random


def policy_A(state):
    try:
        direction = int(np.argmax(state[4:8]))
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


def policy_B(state):
    try:
        direction = np.argmax(state[8:12])
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


def policy_C(state):
    try:
        direction = np.argmax(state[12:16])
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

def policy_D(state):
    try:
        direction = np.argmax(state[16:20])
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

def policy_E(state):
    try:
        direction = np.argmax(state[20:24])
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


def policy_F(state):
    try:
        direction = np.argmax(state[24:28])
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

def evaluate_policy(policies):
    """
    Creates a new world, evaluates set of policies p on the world.
    :param policies: the set of policies (agents) to try the world
    :return: Rewards for each agent
    """
    rd = multi_poi_rover_domain.SequentialPOIRD()
    rd.poi_positions = np.array([[0, 20], [18, 20], [10, 20]], dtype="double")
    done = False
    state = rd.rover_observations
    if not done:
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
    return [rd.sequential_score()]*len(policies)


def evaluate_policy_heirarchy(policies):
    """
    Creates a new world, evaluates the heirarchy policies in it
    :param policies: team of policies to be used
    :return: Score of the team
    """
    rd = multi_poi_rover_domain.SequentialPOIRD()
    # rd.poi_positions = np.array([[0, 20], [18, 20], [10, 20], [24, 20]], dtype="double")
    rd.poi_positions = np.array([[0, 20], [18, 20], [10, 20], [24, 20], [32, 20], [40, 20]], dtype="double")
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
                    a = policy_A(s.flatten())
                elif selection == 1:
                    a = policy_B(s.flatten())
                elif selection == 2:
                    a = policy_C(s.flatten())
                elif selection == 3:
                    a = policy_D(s.flatten())
                elif selection == 4:
                    a = policy_E(s.flatten())
                elif selection == 5:
                    a = policy_F(s.flatten())
                actions.append(a)
        actions = np.array(actions, dtype="double")
        # print(actions)
        state, reward, done, _ = rd.step(actions)
        # Updates the sequence map
        rd.update_sequence_visits()
    return [rd.easy_sequential_score()]*len(policies)


class Agent:
    def __init__(self, input_l, middle_l, out_l, pool_size=10):
        self.policy_pool = []
        self.cum_rewards = [0]*pool_size
        for _ in range(pool_size):
            self.policy_pool.append(MLP.Policy(input_l, middle_l, out_l))

    def reset(self):
        self.cum_rewards = [0]*len(self.cum_rewards)


def test_G():
    """
    Tests a set of agents learning direct control actions only on G
    :return: None
    """
    pool = multiprocessing.Pool()
    agents = []
    best_performance = []
    num_agents = 3
    agent_pool_size = 5
    for _ in range(num_agents):
        agents.append(Agent(19, 32, 2, agent_pool_size))

    with torch.set_grad_enabled(False):
        for gen in range(1000):
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
                policy_teams.append(p)

            team_performances = pool.map(evaluate_policy, policy_teams)
            # Update the cumulative performance of each policy
            for i, t in enumerate(teams):
                for a in range(len(agents)):
                    agents[a].cum_rewards[t[a]] += team_performances[i][a]
            print("Gen {} top 5 avg team: {} best team: {}".format(gen, np.average(team_performances[:5], axis=0), max(team_performances)))
            best_performance.append(max(team_performances)[0])

            # Rank and update each agent population every 10 generations
            if gen % 10 == 0:
                for a in agents:
                    # Gets the keys for policy sorted by highest cumulative score first
                    results = sorted(zip(list(range(len(a.policy_pool))), a.cum_rewards), key=lambda x:x[1], reverse=True)
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
    # best_performance = pd.DataFrame(best_performance)
    # best_performance.to_pickle("./G_multi-reward_best.pkl")


def test_heirarchy():
    """
    Tests the heirarchy on the sequential learning problem. Uses a-priori low-level policies.
    :return: None
    """
    pool = multiprocessing.Pool()
    agents = []
    best_performance = []
    num_agents = 7
    agent_pool_size = 5
    actions = 6
    state_size = 34
    for _ in range(num_agents):
        agents.append(Agent(state_size, 32, actions, agent_pool_size))

    with torch.set_grad_enabled(False):
        for gen in range(1000):
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
                policy_teams.append(p)

            team_performances = pool.map(evaluate_policy_heirarchy, policy_teams)
            # Update the cumulative performance of each policy
            for i, t in enumerate(teams):
                for a in range(len(agents)):
                    agents[a].cum_rewards[t[a]] += team_performances[i][a]
            print("Gen {} top 5 avg team: {} best team: {}".format(gen, np.average(team_performances[:5], axis=0), max(team_performances)))
            best_performance.append(max(team_performances)[0])

            # Rank and update each agent population
            if gen % 10 == 0:
                for a in agents:
                    # Gets the keys for policy sorted by highest cumulative score first
                    results = sorted(zip(list(range(len(a.policy_pool))), a.cum_rewards), key=lambda x:x[1], reverse=True)
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
    # best_performance = pd.DataFrame(best_performance)
    return best_performance


if __name__ == '__main__':
    # test_G()
    key = "hierarchy/looping_six_poi/seven_agent"
#     performance = []
#     for i in range(10):
#         performance.append(test_G())
#     best_performance = pd.DataFrame(performance)
#     best_performance.to_hdf("./hierarchy-multi-reward_best.h5", key="G/"+key)
#
    performance = []
    for i in range(5):
        performance.append(test_heirarchy())
    best_performance = pd.DataFrame(performance)
    best_performance.to_hdf("./hierarchy-multi-reward_best.h5", key="hierarchy/"+key)
