import random
import numpy as np

# region Constants
ELITISM = 1
SURVIVAL_THRESHOLD = 0.5
POP_SIZE = 2
MAX_WEIGHT = 30
MIN_WEIGHT = -30
CONNECTION_TYPE = 'Conv'  # could be conv, pool, FC
# MUTATION_CHANCE = 0.8
# endregion


class Net:
    def __init__(self, identification, inputs, f=3, k=1):
        self.identification = identification
        self.fitness = np.random.randint(10)
        self.inputs = inputs  # say this is a 5x5 image
        # self.weights = np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, f**2)
        self.weights = np.random.randint(-1, 1, f**2)
        self.weights = np.reshape(self.weights, (f, f))
        self.f = f
        self.k = k
        self.outputs = self.calc_output()


    def __str__(self):
        id_str = f'--Net--\nNet id: {self.identification}\n'
        fitness_str = f'Fitness: {self.fitness}\n'
        input_str = f'Input matrix:\n{self.inputs}\n'
        weight_str = f'Weight kernel:\n{self.weights}\n'
        output_str = f'Output:\n{self.outputs}\n'
        return id_str + fitness_str + input_str + weight_str + output_str

    def calc_output(self):
        in_rows, in_cols = self.inputs.shape
        f = self.f
        out_rows = in_rows - f + 1
        out_cols = in_cols - f + 1
        outputs = np.zeros(out_rows*out_cols)
        outputs = np.reshape(outputs, (out_rows, out_cols))
        kernel = self.weights

        for row in range(out_rows):
            for col in range(out_cols):
                filtered = self.inputs[row:row+f, col:col+f] * kernel
                outputs[row, col] = np.sum(filtered)
        return outputs

class EvolutionEngine:

    def __init__(self, network_list):
        self.nets = []
        self.next_gen = []
        self.set_nets(network_list)

    def set_nets(self, new_net_list):
        sorted_list = sorted(new_net_list, key=lambda x: x.fitness)
        self.nets = sorted_list
        self.next_gen = []

    def evolve(self):
        self.elitism()

        num_survivors = int(SURVIVAL_THRESHOLD * len(self.nets))
        survivor_list = self.nets[0:num_survivors]

        while len(self.next_gen) < POP_SIZE:
            index = random.randint(num_survivors)
            mutated_offspring = self.mutate(survivor_list[index])
            self.next_gen.append(mutated_offspring)

        return self.next_gen

    def elitism(self):
        for ind in range(ELITISM):
            self.next_gen.append(self.nets[ind])


# class EvolutionEngine: this is for genetic algorithms, not CNNs and back propagation
#
#     def __init__(self, network_list):
#         self.nets = []
#         self.next_gen = []
#         self.set_nets(network_list)
#
#     def set_nets(self, new_net_list):
#         sorted_list = sorted(new_net_list, key=lambda x: x.fitness)
#         self.nets = sorted_list
#         self.next_gen = []
#
#     def evolve(self):
#         self.elitism()
#
#         num_survivors = int(SURVIVAL_THRESHOLD * len(self.nets))
#         survivor_list = self.nets[0:num_survivors]
#
#         while len(self.next_gen) < POP_SIZE:
#             index = random.randint(num_survivors)
#             mutated_offspring = self.mutate(survivor_list[index])
#             self.next_gen.append(mutated_offspring)
#
#         return self.next_gen
#
#     def elitism(self):
#         for ind in range(ELITISM):
#             self.next_gen.append(self.nets[ind])
#
#     def mutate(self, survivor):
#         old_weights = survivor.weights  # TODO: get the weights
#         coin_toss = random.random
#         if coin_toss >= MUTATION_CHANCE:  # mutate the weights
#             coin_toss2 = random.random
#             if coin_toss2 >= 0.1:  # uniformly perturb
#
#                 uniform_vec = np.random.uniform(-1, 1, len(old_weights))
#
#                 new_weights = old_weights + uniform_vec
#                 new_weights = [w if w <= MAX_WEIGHT else MAX_WEIGHT for w in new_weights]
#                 new_weights = [w if w >= MIN_WEIGHT else MIN_WEIGHT for w in new_weights]
#             else:  # give a new random value to weights
#                 new_weights = np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, len(old_weights))
#         else:  # keep the old weights
#             new_weights = old_weights
#
#         # TODO : create a net with these new weights and return the net
#         return new_weights


if __name__ == '__main__':

    # inputs = np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, 25)
    inputs = np.random.randint(-1, 1, 25)
    inputs = np.reshape(inputs, (5, 5))
    mynet = Net(1, inputs)

    print(mynet)
