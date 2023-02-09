import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Parser',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--mode',
    type=str,
    default='helicopter',
    choices=['gravity', 'helicopter', 'thrust'],
)

args = parser.parse_args()

if __name__ == '__main__':
    info_file = open(args.mode + "_info.txt", "r")

    f_max = []
    f_min = []
    f_avg = []

    generations = 0
    for line in info_file.readlines():
        f_values = line.split("-")
        f_max.append(float(f_values[0])) 
        f_min.append(float(f_values[1])) 
        f_avg.append(float(f_values[2])) 
        generations += 1

    gn = np.arange(0, generations, step=1)

    plt.plot(gn, f_max, label="Maximum Fitness")
    plt.plot(gn, f_min, label="Minimum Fitness")
    plt.plot(gn, f_avg, label="Average Fitness")
    plt.legend()
    plt.show()