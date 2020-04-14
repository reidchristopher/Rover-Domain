import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph results')
    parser.add_argument('--file_name', default="", type=str)
    args = parser.parse_args()

    file = args.file_name
    file = "Output_Data/global_rewards_40x40_with_all_v2_round3.csv"
    data = []
    with open(file) as f:

        reader = csv.reader(f, delimiter=',')

        for row in reader:

            numbers = row[1:]

            for i, num in enumerate(numbers):

                numbers[i] = float(num)

            data.append(numbers)
    num_to_average = 100
    data = np.array(data)


    plot_data = np.zeros(data.shape[1] - num_to_average)

    for i in range(len(plot_data)):
        plot_data[i] = np.average(data[:, i:i+num_to_average])

    print(plot_data.shape)

    plt.plot(range(len(plot_data)), plot_data)
    plt.xlim([0, len(plot_data)])
    plt.ylim([0, 1])

    plt.show()