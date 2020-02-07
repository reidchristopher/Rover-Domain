# For Python Code
# import Python_Code.ccea as ccea
# import Python_Code.neural_net as neural_network
# from Python_Code.homogeneous_rewards import calc_global, calc_difference, calc_dpp

# For Cython Code
import pyximport; pyximport.install(language_level=3)
from Cython_Code.ccea import Ccea
from Cython_Code.neural_network import NeuralNetwork
from Cython_Code.homogeneous_rewards import calc_global, calc_difference, calc_dpp

from AADI_RoverDomain.parameters import Parameters
from AADI_RoverDomain.rover_domain import RoverDomain
from AADI_RoverDomain.rover_setup import init_rover_pos_fixed_middle
import csv; import os; import sys
import numpy as np

from local_rewards import StandardBasisRewardEvaluator
import random
import pickle


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_rover_path(p, rover_path):  # Save path rovers take using best policy found
    dir_name = 'Output_Data/'  # Intended directory for output files
    nrovers = p.num_rovers

    rpath_name = os.path.join(dir_name, 'Rover_Paths.txt')

    rpath = open(rpath_name, 'a')
    for rov_id in range(nrovers):
        for t in range(p.num_steps+1):
            rpath.write('%f' % rover_path[t, rov_id, 0])
            rpath.write('\t')
            rpath.write('%f' % rover_path[t, rov_id, 1])
            rpath.write('\t')
        rpath.write('\n')
    rpath.write('\n')
    rpath.close()


def save_nn_weights(weights, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, "wb") as file:
        pickle.dump(weights, file)

def load_nn_weights(file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    load_file_name = os.path.join(dir_name, file_name)

    with open(load_file_name, "rb") as file:
        weights = pickle.load(file)

    return weights

def find_global_single_reward(reward_save_file, nn_save_file, global_p=None):

    # For Cython Code
    if global_p is None:
        global_p = Parameters()

    cc = Ccea(global_p)
    global_nn = NeuralNetwork(global_p)

    rd = RoverDomain(global_p)

    rd.initial_world_setup()
    rd.rover_initial_pos = init_rover_pos_fixed_middle(global_p.num_rovers, global_p.x_dim, global_p.y_dim)

    for srun in range(global_p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        cc.reset_populations()  # Randomly initialize ccea populations

        global_nn.reset_nn()  # Initialize NN architecture

        reward_history = []

        for gen in range(global_p.generations):
            print("Gen: %i" % gen)

            cc.select_policy_teams()
            for team_number in range(cc.total_pop_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration
                done = False
                rd.istep = 0
                joint_state = rd.get_joint_state()

                while not done:
                    for rover_id in range(rd.nrovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        global_nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)

                    joint_state, done = rd.step(global_nn.out_layer)

                # Update fitness of policies using reward information

                reward = calc_global(global_p, rd.rover_path, rd.poi_values, rd.poi_pos)

                for rover_id in range(rd.nrovers):
                    policy_id = int(cc.team_selection[rover_id, team_number])
                    cc.fitness[rover_id, policy_id] = reward

            # Testing Phase (test best policies found so far)
            rd.reset_to_init()  # Reset rovers to initial positions
            done = False
            rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                for rover_id in range(rd.nrovers):
                    policy_id = int(cc.team_selection[rover_id, team_number])
                    global_nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)

                joint_state, done = rd.step(global_nn.out_layer)

            reward = calc_global(global_p, rd.rover_path, rd.poi_values, rd.poi_pos)
            reward_history.append(reward)
            print(reward)

            if gen == (global_p.generations - 1):  # Save path at end of final generation
                save_rover_path(global_p, rd.rover_path)

            cc.down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, reward_save_file)


def find_global_policy(local_reward_policy_weights, reward_save_file, nn_save_file, global_p=None):
    print(len(local_reward_policy_weights))

    # For Cython Code
    if global_p is None:
        global_p = Parameters()

    global_p.num_outputs = len(local_reward_policy_weights)

    cc = Ccea(global_p)
    global_nn = NeuralNetwork(global_p)

    local_p = Parameters()

    local_p.num_rovers = 1
    local_nn = NeuralNetwork(local_p)

    rd = RoverDomain(global_p)

    rd.initial_world_setup()
    rd.rover_initial_pos = init_rover_pos_fixed_middle(global_p.num_rovers, global_p.x_dim, global_p.y_dim)

    for srun in range(global_p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        cc.reset_populations()  # Randomly initialize ccea populations

        global_nn.reset_nn()  # Initialize NN architecture

        reward_history = []

        for gen in range(global_p.generations):
            print("Gen: %i" % gen)

            cc.select_policy_teams()
            for team_number in range(cc.total_pop_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration
                done = False
                rd.istep = 0
                joint_state = rd.get_joint_state()

                while not done:
                    joint_action = np.zeros((rd.nrovers, 2))
                    for rover_id in range(rd.nrovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        global_nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)

                        local_policy_choice = np.argmax(global_nn.out_layer[rover_id])

                        local_nn.run_neural_network(joint_state[rover_id],
                                                    local_reward_policy_weights[local_policy_choice], 0)
                        joint_action[rover_id] = np.array(local_nn.out_layer)[0]

                    joint_state, done = rd.step(joint_action)

                # Update fitness of policies using reward information

                reward = calc_global(global_p, rd.rover_path, rd.poi_values, rd.poi_pos)

                for rover_id in range(rd.nrovers):
                    policy_id = int(cc.team_selection[rover_id, team_number])
                    cc.fitness[rover_id, policy_id] = reward

            # Testing Phase (test best policies found so far)
            rd.reset_to_init()  # Reset rovers to initial positions
            done = False
            rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                joint_action = np.zeros((rd.nrovers, 2))
                for rover_id in range(rd.nrovers):
                    policy_id = int(cc.team_selection[rover_id, team_number])
                    global_nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)

                    local_policy_choice = np.argmax(global_nn.out_layer[rover_id])

                    local_nn.run_neural_network(joint_state[rover_id],
                                                local_reward_policy_weights[local_policy_choice], 0)
                    joint_action[rover_id] = np.array(local_nn.out_layer)[0]

                joint_state, done = rd.step(joint_action)

            reward = calc_global(global_p, rd.rover_path, rd.poi_values, rd.poi_pos)
            reward_history.append(reward)
            print(reward)

            if gen == (global_p.generations - 1):  # Save path at end of final generation
                save_rover_path(global_p, rd.rover_path)

            cc.down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, reward_save_file)


def find_local_policy(evaluator, reward_save_file, nn_save_file, p=None):
    # For Python code
    # p = Parameters()
    # cc = ccea.Ccea(p)
    # nn = neural_network.NeuralNetwork(p)
    # rd = RoverDomain(p)

    # For Cython Code
    if p is None:
        p = Parameters()

    cc = Ccea(p)
    nn = NeuralNetwork(p)
    rd = RoverDomain(p)

    # Checks to make sure gen switch and step switch are not both engaged
    if p.gen_suggestion_switch and p.step_suggestion_switch:
        sys.exit('Gen Switch and Step Switch are both True')

    rd.initial_world_setup()
    rd.rover_initial_pos = init_rover_pos_fixed_middle(p.num_rovers, p.x_dim, p.y_dim)
    rover_init_init_pos = rd.rover_initial_pos


    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        cc.reset_populations()  # Randomly initialize ccea populations

        nn.reset_nn()  # Initialize NN architecture

        reward_history = []

        for gen in range(p.generations):
            print("Gen: %i" % gen)

            cc.select_policy_teams()
            for team_number in range(cc.total_pop_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration
                if team_number == 0:
                    for rover_id in range(p.num_rovers):
                        rd.rover_initial_pos[rover_id, 0] += random.random() * 0.1 - 0.05
                        rd.rover_initial_pos[rover_id, 1] += random.random() * 0.1 - 0.05
                        rd.rover_initial_pos[rover_id, 2] = (rd.rover_initial_pos[rover_id, 2]
                                                             + random.random() * 2 - 1) % 360
                done = False
                rd.istep = 0
                joint_state = rd.get_joint_state()

                while not done:
                    for rover_id in range(rd.nrovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done = rd.step(nn.out_layer)

                # Update fitness of policies using reward information

                reward = [evaluator.get_reward(rd.get_joint_state()[i]) for i in range(p.num_rovers)]

                for rover_id in range(rd.nrovers):
                    policy_id = int(cc.team_selection[rover_id, team_number])
                    cc.fitness[rover_id, policy_id] = reward[rover_id]

            # Testing Phase (test best policies found so far)
            rd.rover_initial_pos = rover_init_init_pos
            rd.reset_to_init()  # Reset rovers to initial positions
            done = False
            rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                for rover_id in range(rd.nrovers):
                    pol_index = np.argmax(cc.fitness[rover_id])
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, pol_index], rover_id)
                joint_state, done = rd.step(nn.out_layer)

            reward = [evaluator.get_reward(rd.get_joint_state()[i]) for i in range(p.num_rovers)]
            reward_history.append(max(reward))
            print(max(reward))

            if gen == (p.generations - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            cc.down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, reward_save_file)

    test_rewards = []
    for i in range(10 * p.num_rovers):
        # Testing Phase (test best policies found so far)
        rd.rover_initial_pos = rover_init_init_pos
        rd.reset_to_init()  # Reset rovers to initial positions

        for rover_id in range(p.num_rovers):
            rd.rover_initial_pos[rover_id, 0] += random.random() * 1 - 0.5
            rd.rover_initial_pos[rover_id, 1] += random.random() * 1 - 0.5
            rd.rover_initial_pos[rover_id, 2] = (rd.rover_initial_pos[rover_id, 2] + random.random() * 20 - 10) % 360
        done = False
        rd.istep = 0
        joint_state = rd.get_joint_state()
        while not done:
            for rover_id in range(rd.nrovers):
                pol_index = np.argmax(cc.fitness[rover_id])
                nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, pol_index], rover_id)
            joint_state, done = rd.step(nn.out_layer)

        reward = [evaluator.get_reward(rd.get_joint_state()[i]) for i in range(p.num_rovers)]
        test_rewards.append(reward)

    rd.rover_initial_pos = rover_init_init_pos
    best_choice = np.argmax(np.average(test_rewards, axis=0))

    best_weights = np.array(nn.weights)[best_choice]
    save_nn_weights(best_weights, nn_save_file)

    return best_weights


def main():

    params = Parameters()


    import argparse
    parser = argparse.ArgumentParser(description='Run CCEA with multiple rewards')
    parser.add_argument('--from_file', action='store_true')
    parser.add_argument('--num_runs', default=1, type=int)
    args = parser.parse_args()

    for run in range(args.num_runs):
        local_policy_weights = []
        params.load_yaml("local_params.yaml")
        if args.from_file:
            for i in range(params.num_inputs):
                local_policy_weights.append(load_nn_weights("local_weights_%d.pickle"))
        else:

            for i in range(params.num_inputs):

                evaluator = StandardBasisRewardEvaluator(i, params.num_inputs)

                local_policy_weights.append(find_local_policy(evaluator, "local_rewards_%d.csv" % i,
                                                              "local_weights_%d_%d.pickle" % (i, run),
                                                              p=params))
        test_types = ["40x40", "100x100"] #, "200x200"]
        for test in test_types:
            reward_file = "global_rewards_%s.csv" % test
            weights_file = "global_weights_%s.pickle" % test
            params.load_yaml("global_params_%s.yaml" % test)
            find_global_policy(local_policy_weights, reward_file, weights_file, global_p=params)


if __name__ == "__main__":
    main()  # Run the program
