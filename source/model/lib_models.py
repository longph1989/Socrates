import autograd.numpy as np
# import torch

class Model:
    def __init__(self, shape, lower, upper, layers, path):
        self.shape = shape
        self.lower = lower
        self.upper = upper
        self.layers = layers

        if layers == None and path != None:
            self.ptmodel = torch.load(path)


    def copy(self):
        new_model = Model(None, None, None, None, None)
        
        new_model.shape = self.shape.copy()
        new_model.lower = self.lower.copy()
        new_model.upper = self.upper.copy()

        if self.layers is None:
            new_model.layers = None
            new_model.ptmodel = self.ptmodel.copy()
        else:
            new_model.layers = []
            for layer in self.layers:
                new_model.layers.append(layer.copy())
            new_model.ptmodel = None

        return new_model


    def __apply_ptmodel(self, x):
        x = torch.from_numpy(x).view(self.shape.tolist())

        with torch.no_grad():
            output = self.ptmodel(x)

        output = output.numpy()

        return output


    def apply(self, x, y=None):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            for layer in self.layers:
                output = layer.apply(output)

        for layer in self.layers:
            layer.reset()

        if y is None:
            return output
        else:
            return output[0, y]


    def apply_to(self, x, idx):
        if self.layers == None:
            # only work when layers is not None
            raise NameError('Not support yet!')

        output = x # no need for recurrent yet

        for i in range(len(self.layers)):
            if i < idx:
                layer = self.layers[i]
                output = layer.apply(output)

        return output


    def forward(self, x_poly, idx, lst_poly):
        if self.layers == None:
            # only work when layers is not None
            raise NameError('Not support yet!')

        output = x_poly # no need for recurrent yet

        for i in range(len(self.layers)):
            if i == idx:
                layer = self.layers[i]
                output = layer.apply_poly(output, lst_poly)
                break

        return output


    def apply_intermediate(self, x, layer_idx=0):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        # only handle single input
        if len != 1:
            return None, None

        layer_output = []

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            j = 0
            for layer in self.layers:
                output = layer.apply(output)
                #sunbing
                #if j == 2:
                #    output[0][20] = 0.0*output[0][20]
                #if j == 0:
                #    output[0][15] =  0.0*output[0][15]
                    #output[0][13] =  0.03*output[0][13]
                if j == layer_idx:
                    layer_output = output[0]
                j = j + 1

        for layer in self.layers:
            layer.reset()

        return output, layer_output

    def apply_repair(self, x, repair_neuron, repair_w, repair_layer):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        length = int(x.size / size_i)

        # only handle single input
        if length != 1:
            return None, None

        for i in range(length):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            j = 0

            # input
            for l_idx in range(0, len(repair_layer)):
                if repair_layer[l_idx] == j:
                    n_idxs = repair_neuron[l_idx]
                    output[0][n_idxs] = (1 + repair_w[l_idx]) * output[0][n_idxs]

            for layer in self.layers:
                output = layer.apply(output)

                for l_idx in range (0, len(repair_layer)):
                    if repair_layer[l_idx] == j:
                        n_idxs = repair_neuron[l_idx]
                        output[0][n_idxs] = (1 + repair_w[l_idx]) * output[0][n_idxs]

                j = j + 1

        for layer in self.layers:
            layer.reset()

        return output


    def apply_lstm_inter(self, x, neuron=0):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        length = int(x.size / size_i)

        hidden_state = [] # should contain xlen elements

        #lstm_cell_mean = 0

        for i in range(length):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i

            layer_index = 0
            for layer in self.layers:
                output = layer.apply(output)
                if layer_index == 0:
                    # lstm layer
                    # test first cell: output[0][0]
                    #lstm_cell_mean = lstm_cell_mean + output[0][0]
                    #output[0][0] = 1.397 * output[0][0];
                    hidden_state.append(output[0][neuron])

                layer_index = layer_index + 1

        #lstm_cell_mean = lstm_cell_mean / length

        for layer in self.layers:
            layer.reset()

        return output, hidden_state


    def apply_lstm_repair(self, x, repair_neuron=0, repair_w=0.0, repair_layer=0):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        length = int(x.size / size_i)

        for i in range(length):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i

            layer_index = 0
            for layer in self.layers:
                output = layer.apply(output)

                for l_idx in range(0, len(repair_layer)):
                    if repair_layer[l_idx] == layer_index:
                        n_idxs = repair_neuron[l_idx]
                        output[0][n_idxs] = (1 + repair_w[l_idx]) * output[0][n_idxs]

                layer_index = layer_index + 1

        for layer in self.layers:
            layer.reset()

        return output

#####################################################################################################################################
    def apply_repair_fixed(self, x, repair_neuron, repair_w, repair_layer):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        length = int(x.size / size_i)

        # only handle single input
        if length != 1:
            return None, None

        for i in range(length):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            j = 0

            # input
            for l_idx in range(0, len(repair_layer)):
                if repair_layer[l_idx] == -1:
                    n_idxs = repair_neuron[l_idx]
                    output[0][n_idxs] = (1 + repair_w[l_idx]) * output[0][n_idxs]

            for layer in self.layers:
                output = layer.apply(output)

                for l_idx in range (0, len(repair_layer)):
                    if repair_layer[l_idx] == j:
                        n_idxs = repair_neuron[l_idx]
                        output[0][n_idxs] = (1 + repair_w[l_idx]) * output[0][n_idxs]

                j = j + 1

        for layer in self.layers:
            layer.reset()

        return output

    def apply_intervention(self, x, do_layer, do_neuron, do_value):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        length = int(x.size / size_i)

        # only handle single input
        if length != 1:
            return None, None

        for i in range(length):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            j = 0

            for layer in self.layers:
                output = layer.apply(output)

                if do_layer == j:
                    output[0][do_neuron] = do_value

                j = j + 1

        for layer in self.layers:
            layer.reset()

        return output


    def apply_get_h(self, x, do_layer, do_neuron):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, *self.shape[1:]]
        size_i = np.prod(shape_i)

        length = int(x.size / size_i)

        hidden = 0.0

        # only handle single input
        if length != 1:
            return None, None

        for i in range(length):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            j = 0

            for layer in self.layers:
                output = layer.apply(output)

                if do_layer == j:
                    hidden = output[0][do_neuron]

                j = j + 1

        for layer in self.layers:
            layer.reset()

        return output, hidden
#####################################################################################################################################