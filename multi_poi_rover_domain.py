import rover_domain
import numpy as np


class SequentialPOIRD(rover_domain.RoverDomain):
    """
    Extends the rover domain class to include functions for sequential POI observation tasks,
    where the reward is not handed out until all types of a POI are observed, and there is a specific
    sequential relationship for how each POI can be observed.
    """

    def __init__(self):
        super(SequentialPOIRD, self).__init__()
        self.poi_types = [2, 1, 0]
        self.n_pois = len(self.poi_types)

        # store the sequence as a graph, with each type pointing to the immediate parent(s) which must be satisfied
        # for it to be counted as a score.
        self.sequence = {0: None, 1: [0], 2: [1]}

        # POI visited is a dictionary which keeps track of which types have been visited
        self.poi_visited = {}
        for t in self.sequence:
            self.poi_visited[t] = False

        # Override the size of the observations matrix
        self.rover_observations = np.zeros((self.n_rovers, 1 + len(self.poi_visited), self.n_obs_sections))

    def reset(self):
        """
        Resets the "poi_visited" dictionary to entirely false, in addition to the parent reset behavior.
        :return: None
        """
        super(SequentialPOIRD, self).reset()
        for t in self.poi_types:
            self.poi_visited[t] = False

    def mark_POI_observed(self, poi_type):
        """
        Updates the self.poi_visited[poi_type] if the dependencies are properly met
        Only does the update for the type being queried.

        :param poi_type: String, the type identifier of the poi being observed
        :return: None
        """
        parents = self.sequence[poi_type]
        if parents is not None:
            observed = True
            for p in parents:
                if not self.poi_visited[p]:
                    observed = False
                    break
            if observed:
                self.poi_visited[poi_type] = True
        else:
            # If no parents
            self.poi_visited[poi_type] = True

    def update_sequence_visits(self):
        """
        Uses current rover positions to update observations.

        :return: None
        """
        # TODO implement tight coupling in this scenario (can use parent class functionality?)
        for rover in self.rover_positions:
            for i, poi in enumerate(self.poi_positions):
                # POI is within observation distance
                # print("Distance to {} : {}".format(np.array(poi), np.linalg.norm(np.array(rover) - np.array(poi))))
                if np.linalg.norm(np.array(rover) - np.array(poi)) < self.interaction_dist:
                    self.mark_POI_observed(self.poi_types[i])

    def sequential_score(self):
        """
        Checks if all the POI in poi_visited have been observed. Returns a fixed score at the moment.

        :return: Global reward score for the agents based on the sequential observation task
        """
        # Update to see if the requirements are now set
        # (done until no more updates are found, to account for cascading updates)
        # Now with the updated graph, we check if all POI are observed
        if all(self.poi_visited.values()):
            return 1
        else:
            return 0

    # Needs to observe the different types of poi, and also make the observed POI dictionary into the state as well
    def update_observations(self):
        """
        Updates the internally held observations of each agent.
        State is represented as:
        <agents, poi type A, poi type B, ... , poi type N, poi A observed, poi B observed, ... , poi N observed>
        Where "agents" and each "poi type" are split into the quadrants centered around the rover.
        "poi X observed" is a 0-1 flag on whether or not a POI of type A has been successfully observed.

        Accommodates the reorienting/fixed frame differences.

        :return: None
        """
        # Reset the observation matrix
        for rover_id in range(self.n_rovers):
            for type_id in range(1 + len(self.poi_visited)):
                for section_id in range(self.n_obs_sections):
                    self.rover_observations[rover_id, type_id, section_id] = 0.

        for rover_id in range(self.n_rovers):
            # Update rover type observations
            for other_rover_id in range(self.n_rovers):
                # agents do not sense self (ergo skip self comparison)
                if rover_id == other_rover_id:
                    continue
                self.add_to_sensor(
                    rover_id,
                    0,
                    self.rover_positions[other_rover_id, 0],
                    self.rover_positions[other_rover_id, 1],
                    1.
                )
            # Update POI type observations
            for poi_id in range(self.n_pois):
                poi_type = self.poi_types[poi_id]
                # The index in this call is the poi type +1 (to offset the rovers first)
                # and then is assigned to it's respective section
                self.add_to_sensor(
                    rover_id,
                    poi_type + 1,
                    self.poi_positions[poi_id, 0],
                    self.poi_positions[poi_id, 1],
                    1.
                )

    def score_single_type(self, poi_type):
        """
        Calculates the step score for a single type of POI, ignoring all other POI

        :param poi_type: The type of the point of interest being scored
        :return: score, the global score for observing the specified POI type
        """
        score = 0
        for poi_id in range(self.n_pois):
            if self.poi_types[poi_id] == poi_type:
                score += self.calc_step_eval_from_poi(poi_id)
        return score


if __name__ == '__main__':
    # Fix random seeds
    import random

    random.seed(0)
    np.random.seed(0)

    rd = SequentialPOIRD()
    # rd.reset()
    # print(rd.poi_visited)
    # print("Score: ", rd.sequential_score())
    # print(np.array(rd.rover_positions))
    # print(np.array(rd.poi_positions))
    # print()
    # rd.interaction_dist = 1
    # rd.update_sequence_visits()
    # print(rd.poi_visited)
    # print("Score: ", rd.sequential_score())
    # print()
    # rd.interaction_dist = 2
    # rd.update_sequence_visits()
    # print(rd.poi_visited)
    # print("Score: ", rd.sequential_score())
    # print()
    # rd.interaction_dist = 3
    # rd.update_sequence_visits()
    # print(rd.poi_visited)
    # print("Score: ", rd.sequential_score())
#
    # print("Initial observations:", np.array(rd.rover_observations))
    # rd.update_observations()
    # print("Updated observations:", np.array(rd.rover_observations)[0])

    rd.poi_positions[2, 0] = 3.0
    rd.poi_positions[2, 1] = 3.0
    done = False
    print(np.array(rd.rover_positions))
    while not done:
        actions = np.random.random((10, 2))
        obs, reward, done, _ = rd.step(actions)
        print("Type 0: {} Type 1: {} Type 2: {}".format(rd.score_single_type(0), rd.score_single_type(1), rd.score_single_type(2)))
