import random
import numpy as np

# region Constants
ELITISM = 1
SURVIVAL_THRESHOLD = 0.5
POP_SIZE = 2
MAX_WEIGHT = 30
MIN_WEIGHT = -30
CONNECTION_TYPE = "Convolution"  # could be convolution, pool, FC
RECEPTIVE_FIELD_DEFAULT = 2
# MUTATION_CHANCE = 0.8
# endregion


class Net:
    # Todo: add compatibility with varying number of image channels
    # Todo: implement the full net backpropagation

    def __init__(self, identification, inputs=np.array([])):
        self.identification = identification
        self.fitness = np.random.randint(10)

        self.inputs = np.array([])  # say this is a 5x5 image
        self.outputs = np.array([])
        self.layer_dict = {}

    def __str__(self):
        id_str = f'--Net--\nNet id: {self.identification}\n'
        fitness_str = f'Fitness: {self.fitness}\n'

        return id_str + fitness_str

    def add_layer(self, layer_num):
        my_conv_layer = ConvolutionLayer(layer_num)
        self.layer_dict[my_conv_layer.layer_number] = my_conv_layer

    def activate(self, net_input):
        layers_dict = self.layer_dict.values()
        prev_input = net_input

        for layer in layers_dict:
            layer.set_inputs(prev_input)
            layer.calc_output()  # update the layer output

            prev_input = layer.outputs  # update the input for the next layer

        final_layer = list(layers_dict)[-1]
        output = final_layer.outputs

        return output

    def backpropagate(self):
        # dX, dW = layer.backpropagate

        pass


class ConvolutionLayer:
    def __init__(self, layer_number, f=RECEPTIVE_FIELD_DEFAULT, k=1):
        self.layer_number = layer_number
        self.weights = np.random.randint(-1, 1, (f, f))
        self.f = f
        self.k = k

        self.input = np.array([])
        self.outputs = np.array([])

    def set_inputs(self, input_vec):
        self.input = input_vec

    def calc_output(self):
        # Calculating the convolution layer output given the input, kernel and kernel number
        in_rows, in_cols = self.input.shape
        f = self.f
        kernel = self.weights

        out_rows = in_rows - f + 1
        out_cols = in_cols - f + 1

        outputs = np.zeros((out_rows, out_cols))

        for row in range(out_rows):
            for col in range(out_cols):
                convolved = self.input[row:row + f, col:col + f] * kernel
                outputs[row, col] = np.sum(convolved)

        self.outputs = outputs

    def calc_gradients(self, dY):
        # Calculating the gradients of the layer given the loss gradient wrt output (dY)

        # Retrieving the input (X) and the kernel weights (W)
        X = self.input
        W = self.weights

        # Retrieving dimensions from W's shape
        (f, f) = W.shape

        # Retrieving dimensions from dY's shape
        (rows_Y, cols_Y) = dY.shape

        # Initializing dX, dW with the correct shapes
        dX = np.zeros(X.shape)
        dW = np.zeros(W.shape)

        # Looping over vertical(h) and horizontal(w) axis of the output
        for h in range(rows_Y):
            for w in range(cols_Y):
                dX[h:h + f, w:w + f] += dY[h, w] * W
                dW += dY[h, w] * X[h:h + f, w:w + f]

        return dX, dW

    def backpropagate(self):
        # Calculating the layer gradients:
        # from loss function to input (dX)
        # and from loss function to weights (dW)

        in_rows, in_cols = self.input.shape
        out_rows = in_rows - self.f + 1

        dY = np.ones((out_rows, out_rows))  # error vector wrt output loss (Y)

        dX, dW = self.calc_gradients(dY)

        return dX, dW

    def __str__(self):
        id_str = f'--Convolution layer--\nLayer number: {self.layer_number}\n'
        input_str = f'Input matrix:\n'
        for elem in self.input:
            input_str += f'\t{elem}\n'
        weight_str = f'Weight kernel:\n'
        for elem in self.weights:
            weight_str += f'\t{elem}\n'
        output_str = f'Output:\n'
        for elem in self.outputs:
            output_str += f'\t{elem}\n'
        return id_str + input_str + weight_str + output_str


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
            mutated_offspring = survivor_list[index]
            self.next_gen.append(mutated_offspring)

        return self.next_gen

    def elitism(self):
        for ind in range(ELITISM):
            self.next_gen.append(self.nets[ind])


if __name__ == '__main__':


    # print(mynet)
    # print(mynet.activate())
    mynet = Net(1)
    mynet.add_layer(1) # adds a convolution layer as the first layer of the net

    in_vec = np.random.randint(-1, 1, (3, 3))
    net_out = mynet.activate(in_vec)
    print(net_out)