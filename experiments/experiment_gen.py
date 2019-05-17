"""
Generates experiment yaml files.
Implements a class which has type-safe capabilities to safely generate new configuration files for
the multi-reward rover domain. Allows creation of parameters in raw Python in a scripted environment,
but still using the nice configuration file behavior.
"""
import yaml


class YamlGenerator:
    """
    Generates YAML files for experiment configuration.
    Options to guarantee type-safe configuration files
    Will not overwrite existing config files
    """

    def __init__(self):
        """
        Creates the default parameter list, default values, and default types for each parameter
        """
        self.correct_types = []
        self.add_parameter("POI Types", list)
        self.add_parameter("Shuffle POI", bool)
        self.add_parameter("POI Sequence", dict)
        self.add_parameter("Number of Agents", int)
        # self.add_parameter("Agent Positions", list)
        self.add_parameter("Number of Timesteps", int)
        self.add_parameter("Trials", int)
        self.add_parameter("H5 Output File", str)
        self.add_parameter("Experiment Name", str)
        # self.add_parameter("POI Positions", list)

        self.options = {"POI Types": [0, 1],
                        "Shuffle POI": False,
                        "POI Sequence": {0: None, 1: [0]},
                        "Number of Agents": 2,
                        # "Agent Positions": [[15, 5], [25, 0]],
                        "Number of Timesteps": 15,
                        "Trials": 1,
                        "H5 Output File": "test.h5",
                        "Experiment Name": "write_file_test",
                        # "POI Positions": [[5, 10], [20, 10]]
                        }

    def add_parameter(self, name, default_type):
        """
        Adds a parameter to be specified in the output file, and makes sure that the output is type-safe
        :param name: :str: The key to be used in the dictionary of options
        :param default_type: A type specifying which type the option must be
        :return: None
        :raises: TypeError if default type is not a type
        """
        if type(default_type) is not type:
            raise TypeError("Default type must specify a type")
        self.correct_types.append((name, default_type))

    def write_to_file(self, filename):
        """
        Checks the options fields to be correct types, and then writes them to the specified file as yaml
        :param filename: The filename to write to
        :return: None
        :raises: TypeError if option type is not correct
        """
        for (key, default_type) in self.correct_types:
            if type(self.options[key]) is not default_type:
                raise TypeError("Value in key [{}] is not type {}".format(key, default_type))

        # x flag means create only, fail if the file already exists. Implicitly has 'w' as well
        with open(filename, 'x') as f:
            yaml.safe_dump(self.options, f)


if __name__ == '__main__':
    yg = YamlGenerator()
    yg.write_to_file("test_file.yaml")
