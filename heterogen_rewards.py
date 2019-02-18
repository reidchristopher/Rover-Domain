import numpy as np
from parameters import Parameters as p
import math
from supervisor import one_of_each_type

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
def calc_hetero_global(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    g_reward = 0.0

    # For all POIs
    for poi_id in range(p.num_pois):
        current_poi_reward = 0.0  # Tracks highest reward received from a POI across all timesteps

        for step_id in range(num_steps):  # Calculate rewards at each time step
            observer_count = 0  # Count observers at given time step
            observer_distances = np.zeros((p.num_types, p.num_rovers))  # Track distances between rovers and POI
            summed_distances = 0.0
            types_in_range = []

            # Calculate distance between poi and agent
            for rtype in range(p.num_types):
                for agent_id in range(p.num_rovers):
                    rov_id = int(p.num_rovers*rtype + agent_id) # Converts identifier to be compatible with base code
                    rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rov_id, 0]
                    rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rov_id, 1]
                    distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))

                    if distance < p.min_distance:  # Clip distance to avoid excessively large rewards
                        distance = p.min_distance
                    observer_distances[rtype, agent_id] = distance

                    # Check if agent observes poi
                    if distance <= p.activation_dist: # Rover is in observation range
                        types_in_range.append(rtype)
                        if p.num_types == 1:
                            observer_count += 1

            if p.num_types > 1:
                for t in range(p.num_types):
                    if t in types_in_range:  # If a rover of a given type is in range, count increases
                        observer_count += 1

            # Update closest distance only if poi is observed
            if observer_count >= p.coupling:
                for rv in range(p.coupling):  # Coupling requirement is one of each type
                    assert(min(observer_distances[rv, :]) <= p.activation_dist)
                    summed_distances += min(observer_distances[rv, :])  # Take distance from closest observer
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward  # Track best observation from given POI

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
def calc_hetero_difference(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    difference_reward = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_hetero_global(rover_path, poi_values, poi_positions)  # Get true global reward

    # CALCULATE DIFFERENCE REWARD
    for rtype in range(p.num_types):
        for agent_id in range(p.num_rovers):
            g_without_self = 0.0

            for poi_id in range(p.num_pois):
                current_poi_reward = 0.0

                for step_id in range(num_steps):
                    observer_count = 0
                    observer_distances = np.zeros((p.num_types, p.num_rovers))  # Track distances between POI and rovers
                    summed_distances = 0.0
                    types_in_range = []

                    # Calculate distance between poi and other agents
                    for other_type in range(p.num_types):
                        for other_agent_id in range(p.num_rovers):
                            rov_id = int(p.num_rovers*other_type + other_agent_id)  # Convert rover id to AADI base format
                            if agent_id != other_agent_id or rtype != other_type:
                                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rov_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rov_id, 1]
                                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))

                                if distance < p.min_distance:  # Clip distance to avoid excessively large rewards
                                    distance = p.min_distance
                                observer_distances[other_type, other_agent_id] = distance

                                if distance <= p.activation_dist:  # Track what rover types are observing
                                    types_in_range.append(other_type)
                                    if p.num_types == 1:
                                        observer_count += 1
                            else:
                                observer_distances[rtype, agent_id] = inf  # Ignore self

                    if p.num_types > 1:
                        for t in range(p.num_types):
                            if t in types_in_range:  # If a rover of a given type is in range, count increases
                                observer_count += 1

                    # update closest distance only if poi is observed
                    if observer_count >= p.coupling:
                        for rv in range(p.coupling):  # Coupling requirement is one of each type
                            assert(min(observer_distances[rv, :]) <= p.activation_dist)
                            summed_distances += min(observer_distances[rv, :])
                        temp_reward = poi_values[poi_id]/summed_distances
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward  # Track current best reward for given POI

                g_without_self += current_poi_reward

            rov_id = int(p.num_rovers*rtype + agent_id)  # Convert to AADI compatible rover identifier
            difference_reward[rov_id] = g_reward - g_without_self

    return difference_reward


# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_hetero_dpp(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    dplusplus_reward = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_hetero_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_hetero_difference(rover_path, poi_values, poi_positions)

    # CALCULATE DPP REWARD
    for c_count in range(p.coupling):  # Increment counterfactuals up to coupling requirement

        # Calculate Difference with Extra Me Reward
        for rtype in range(p.num_types):
            for agent_id in range(p.num_rovers):
                g_with_counterfactuals = 0.0; self_dist = 0.0

                for poi_id in range(p.num_pois):
                    current_poi_reward = 0.0

                    for step_id in range(num_steps):
                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0; summed_distances = 0.0
                        observer_distances = [[] for _ in range(p.num_types)]
                        types_in_range = []

                        # Calculate distance between poi and agent
                        for other_type in range(p.num_types):
                            for other_agent_id in range(p.num_rovers):
                                rov_id = int(p.num_rovers*other_type + other_agent_id)  # Make rover ID AADI compatible
                                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rov_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rov_id, 1]
                                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))

                                if distance < p.min_distance:  # Clip distance to avoid excessively large rewards
                                    distance = p.min_distance
                                observer_distances[other_type].append(distance)

                                if other_agent_id == agent_id and other_type == rtype:
                                    self_dist = distance  # Track distance from self for counterfactuals

                                if distance <= p.activation_dist:  # Track rover types currently observing
                                    types_in_range.append(other_type)
                                    if p.num_types == 1:
                                        observer_count += 1

                        if self_dist <= p.activation_dist:  # Add counterfactual partners if in range (more of me)
                            for c in range(c_count):
                                observer_distances[rtype].append(self_dist)
                                types_in_range.append(rtype)

                        if p.num_types > 1:
                            for t in range(p.num_types):
                                if t in types_in_range:  # If a rover of a given type is in range, count increases
                                    observer_count += 1

                        # update closest distance only if poi is observed
                        if observer_count >= p.coupling:
                            for rv in range(p.coupling):  # Coupling is one of each type
                                assert(min(observer_distances[rv][:]) <= p.activation_dist)
                                summed_distances += min(observer_distances[rv][:])
                            temp_reward = poi_values[poi_id]/summed_distances
                        else:
                            temp_reward = 0.0

                        if temp_reward > current_poi_reward:  # Track current best from POI
                            current_poi_reward = temp_reward

                    g_with_counterfactuals += current_poi_reward

                temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
                rov_id = int(p.num_rovers*rtype + agent_id)  # Convert rover identifier to AADI format
                if temp_dpp_reward > dplusplus_reward[rov_id]:
                    dplusplus_reward[rov_id] = temp_dpp_reward

    for rov_id in range(p.num_rovers * p.num_types):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward

# S-D++ REWARD --------------------------------------------------------------------------------------------------------
def calc_sdpp(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    dplusplus_reward = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_hetero_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_hetero_difference(rover_path, poi_values, poi_positions)

    # CALCULATE S-DPP REWARD
    for c_count in range(p.coupling-1):

        # Calculate reward with suggested counterfacual partners
        for rtype in range(p.num_types):
            for agent_id in range(p.num_rovers):
                g_with_counterfactuals = 0.0

                for poi_id in range(p.num_pois):
                    current_poi_reward = 0.0


                    for step_id in range(num_steps):
                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0; summed_distances = 0.0; self_dist = 0.0
                        observer_distances = [[] for _ in range(p.num_types)]
                        types_in_range = []

                        # Calculate distance between poi and agent
                        for other_type in range(p.num_types):
                            for other_agent_id in range(p.num_rovers):
                                rov_id = int(p.num_rovers*other_type + other_agent_id)
                                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rov_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rov_id, 1]
                                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))

                                if distance < p.min_distance:  # Clip distance to avoid excessively large rewards
                                    distance = p.min_distance
                                observer_distances[other_type].append(distance)

                                if other_type == rtype and other_agent_id == agent_id:
                                    self_dist = distance

                                # Check if agent observes poi, update closest step distance
                                if distance <= p.activation_dist:
                                    types_in_range.append(other_type)
                                    if p.num_types == 1:
                                        observer_count += 1

                        #  Add counterfactuals
                        if self_dist <= p.activation_dist:  # Don't add partners unless self in range
                            rov_partners = one_of_each_type(c_count, rtype)

                            for rv in range(c_count):
                                observer_distances[rv].append(rov_partners[rv, 0])
                                if rov_partners[rv, 1] != rtype:
                                    observer_count += 1

                        if observer_count >= p.coupling:
                            for rv in range(p.coupling):  # Coupling is one of each type
                                assert(min(observer_distances[rv][:]) <= p.activation_dist)
                                summed_distances += min(observer_distances[rv][:])
                            temp_reward = poi_values[poi_id]/summed_distances
                        else:
                            temp_reward = 0.0

                        if temp_reward > current_poi_reward:
                            current_poi_reward = temp_reward

                    g_with_counterfactuals += current_poi_reward

                temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
                rov_id = int(p.num_rovers*rtype + agent_id)
                if temp_dpp_reward > dplusplus_reward[rov_id]:
                    dplusplus_reward[rov_id] = temp_dpp_reward

    for rov_id in range(p.num_rovers * p.num_types):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward