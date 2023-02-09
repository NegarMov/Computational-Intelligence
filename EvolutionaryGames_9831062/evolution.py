from operator import attrgetter
from player import Player
import numpy as np
import copy

class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        # child: an object of class `Player`
        child.nn.W1 += np.random.normal(0, 0.5, np.shape(child.nn.W1))
        child.nn.b1+= np.random.normal(0, 0.5, np.shape(child.nn.b1))

        child.nn.W2 += np.random.normal(0, 0.5, np.shape(child.nn.W2))
        child.nn.b2 += np.random.normal(0, 0.5, np.shape(child.nn.b2))

        return child


    def generate_new_population(self, num_players, prev_players=None):
        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            # num_players example: 150
            # prev_players: an array of `Player` objects

            parents = []
            children = []
            Q = 5

            for _ in range (num_players):
                competing_players = np.random.choice(prev_players, Q)
                winner_player = max(competing_players, key=attrgetter('fitness'))
                parents.append(winner_player)

            for i in range(0, len(parents), 2):
                child1 = copy.deepcopy(parents[i])
                child2 = copy.deepcopy(parents[i + 1])

                if np.random.uniform() < 0.2:
                    child1.nn.b1, child2.nn.b1 = child2.nn.b1, child1.nn.b1
                
                if np.random.uniform() < 0.2:
                    child1.nn.b2, child2.nn.b2 = child2.nn.b2, child1.nn.b2

                if np.random.uniform() < 0.2:
                    child1.nn.W1, child2.nn.W1 = child2.nn.W1, child1.nn.W1

                if np.random.uniform() < 0.2:
                    child1.nn.W2, child2.nn.W2 = child2.nn.W2, child1.nn.W2

                if np.random.uniform() < 0.5:
                    children.append(self.mutate(child1))
                else:
                    children.append(child1)

                if np.random.uniform() < 0.5:
                    children.append(self.mutate(child2))
                else:
                    children.append(child2)

            return children

    def next_population_selection(self, players, num_players):
        # num_players example: 100
        # players: an array of `Player` objects

        f_sum = 0
        for player in players:
            f_sum += player.fitness

        f_max = max(players, key=attrgetter('fitness')).fitness
        f_min = min(players, key=attrgetter('fitness')).fitness
        f_avg = f_sum / len(players)

        info_file = open(self.mode + "_info.txt", "a")
        info_file.write(str(f_max) + "-" + str(f_min) + "-" + str(f_avg) + "\n")
        info_file.close()

        p = []
        for player in players:
            p.append(player.fitness / f_sum)

        return list(np.random.choice(players, num_players, p=p))
