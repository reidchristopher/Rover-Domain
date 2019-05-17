import random

from experiments.experiment_gen import YamlGenerator


def randint_list(minimum, maximum, size):
    l = []
    for _ in range(size):
        l.append(random.randint(minimum, maximum))
    return l


yg = YamlGenerator()
yg.add_parameter("Number of POI", int)
# set consistent values
yg.options["Trials"] = 1
yg.options["H5 Output File"] = "big_parameter_run_.h5"
yg.options["Number of Timesteps"] = 50

count = 0
for agent_number in [2, 5, 10, 20, 50]:
    for poi_num in [10, 30, 50, 100]:
        types = []
        l = randint_list(0, int(poi_num * 0.25), poi_num)
        types.append(l)
        l = randint_list(0, int(poi_num * 0.25), poi_num)
        types.append(l)
        l = randint_list(0, int(poi_num * 0.25), poi_num)
        types.append(l)
        # construct sequence
        for t in types:
            # Make copy of t
            yg.options["POI Types"] = list(t)
            sequence = {}
            t = sorted(set(t))
            sequence[0] = None
            for i in range(1, len(t)):
                sequence[t[i]] = [t[i - 1]]
            # With sequence set, set rest of options
            yg.options["Number of POI"] = poi_num
            yg.options["POI Sequence"] = sequence
            yg.options["Number of Agents"] = agent_number
            yg.options["Experiment Name"] = "agents-{}_poi-{}_ratio-{}".format(agent_number, poi_num, len(t) / poi_num)
            yg.write_to_file("experiments/batch_1_{}.yaml".format(count))
            count += 1
