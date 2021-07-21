import numpy as np
import ast
import math
from sklearn.cluster import MiniBatchKMeans

from autograd import grad
import os.path
from os import path
from utils import *
import time
import pyswarms as ps

class DTMCImpl():
    def __init__(self):
        self.model = None
        self.assertion = None
        self.display = None
        self.s = []  # states
        self.s_idx = []  # index of states
        self.s0 = 0
        self.delta = 0.001
        self.error = 0.01
        self.n_ij = []
        self.n_i = []
        self.A = []
        self.m = 0
        self.offset = []
        self.bitshift = []
        self.num_of_path = 0
        #self.gen_path = '../debug/gen_path.txt'  # debug purpose
        self.step = 50  # get_new_sample step
        self.sensitive = []
        self.sens_cluster = 2
        self.feature = []  # additional feature to analyze
        self.feature_cluster = None
        self.intermediate_layers = []  # intermediate layers to analyze
        self.intermediate_layer = None   # current intermediate layer to analyze
        self.neurons = []  # neuron index at intermediate layer to analyze
        self.neuron = None  # current neuron index at intermediate layer analyzing
        self.hidden_cluster = 2
        self.under_analyze = []
        self.final = []  # final state
        self.timeout = 20  # timeout value set to 20min
        self.starttime = time.time()  # start time
        self.label_diff = 0
        self.criteria = 0.1
        self.sens_analysis = False  # sensitivity analysis
        self.dbgmsg = False
        self.output_path = None
        self.datapath = None
        self.repair = False
        self.repair_layer = None
        self.repair_neuron = None
        self.repair_w = None
        self.repair_num = 0
        self.most_sens_cell = None

    def __generate_x(self, shape, lower, upper):
        size = np.prod(shape)
        x = np.random.rand(size)

        x = (upper - lower) * x + lower
        out = np.array([0.0] * x.size)
        i = 0
        for x_i in x:
            x_i = round(x_i)
            out[i] = x_i
            i = i + 1
        return out

    def solve(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display
        self.dbgmsg = True

        spec = assertion

        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
        if 'sens_cluster' in spec:
            self.sens_cluster = ast.literal_eval(read(spec['sens_cluster']))

        if 'timeout' in spec:
            self.timeout = read(spec['timeout']) * 60

        if 'criteria' in spec:
            self.criteria = spec['criteria']

        if 'error' in spec:
            self.error = (spec['error']) / 2

        if 'confidence' in spec:
            self.delta = 1 - math.sqrt(1 - (spec['confidence']))

        if 'outpath' in spec:
            self.output_path = spec['outpath']

        if 'datapath' in spec:
            self.datapath = spec['datapath']

        # analyze sensitive feature only
        self.under_analyze = np.concatenate((self.feature, self.sensitive), 0)

        preparation_time = time.time() - overall_starttime

        print('Sensitive features: {}'.format(self.sensitive))

        print('Error tolerance: {:.5f}'.format(self.error * 2))
        print('Accuracy: {:.5f}'.format(self.delta * 2))
        print('Fairness Criteria: {}'.format(self.criteria))
        print('Timeout: {}s'.format(self.timeout))

        # array = self.aequitas_test()

        # fairness checking:
        # calculate offset to keep each state identical
        self.calc_offset()

        # calculate bitshift for state coding
        self.calc_bitshift()

        # print('Learning DTMC model...')
        self.learn_dtmc_model()

        # analyze fairness
        # print('Analyzing fairness...')
        res, is_fair, prob_diff, _ = self.analyze_fairness()
        if is_fair:
            print('Network is fair!')
        else:
            print('Network is unfair!')
        print('Maximum probability difference: {:.5f}\n'.format(prob_diff))

        self.export_prism_model()
        print('PRISM model generated!\n')

        analyze_time = time.time() - preparation_time - overall_starttime

        print('Analyze time is:{}s'.format(analyze_time))

        if is_fair:
            return

        self.sens_analysis = True
        self.dbgmsg = False
        # adjust accuracy for sensitivity analysis
        ori_delta = self.delta
        ori_error = self.error
        ori_timeout = self.timeout

        self.delta = 0.15
        self.error = 0.025
        self.timeout = 300

        # perform sensitivity analysis
        print("Perform sensitivity analysis:")
        # other features
        print('\nOther Feature Analysis:')
        diff_matrix = []
        if 'feature' in spec:
            other_features = np.array(ast.literal_eval(read(spec['feature'])))
            feature_clusters = np.array([])
            if 'feature_cluster' in spec:
                feature_clusters = np.array(ast.literal_eval(read(spec['feature_cluster'])))
            max_diff = 0.0
            max_feature = other_features[0]

            idx = 0
            for item in other_features:
                self.feature.append(item)
                if len(feature_clusters) < idx:
                    self.feature_cluster = 2
                else:
                    self.feature_cluster = feature_clusters[idx]

                #self.feature = np.array(ast.literal_eval(read(spec['feature'])))
                self.starttime = time.time()
                print('Other feature: {}'.format(self.feature))

                # learn model with other features
                # print matrix

                # calculate offset to keep each state identical
                self.calc_offset()

                # calculate bitshift for state coding
                self.calc_bitshift()

                # print('Learning DTMC model...')
                self.learn_dtmc_model()

                # analyze fairness
                _, _, _, this_diff = self.analyze_fairness()

                if (max_diff < this_diff):
                    max_diff = this_diff
                    max_feature = item
                new_diff = []
                new_diff.append(this_diff)
                new_diff.append(0)  # input layer
                new_diff.append(item)
                diff_matrix.append(new_diff)

                self.feature = []

                idx = idx + 1

            #print('\nMost sensitive other feature is: {}'.format(max_feature))
            #print('Sensitivity is: {:.5f}'.format(max_diff))
            print('Sensitivity of each input:')
            diff_matrix.sort()
            for item in diff_matrix:
                print(item)

        # other neurons
        if 'intermediate' in spec:
            self.intermediate_layers = np.array(ast.literal_eval(read(spec['intermediate'])))

        if 'neurons' in spec:
            self.neurons = np.array(ast.literal_eval(read(spec['neurons'])))

        print('\nHidden Neuron Analysis:')
        print('Intermediate layers: {}'.format(self.intermediate_layers))
        #print('Intermediate neuron index: {}'.format(self.neurons))

        diff_matrix_h = []
        if len(self.intermediate_layers) != 0:
            if 'hidden_cluster' in spec:
                self.hidden_cluster = ast.literal_eval(read(spec['hidden_cluster']))
            # iterate for each hidden layer under analysis
            for layer_idx in range (0, len(self.intermediate_layers)):
                self.intermediate_layer = self.intermediate_layers[layer_idx]
                print('Intermediate layer: {}'.format(self.intermediate_layer))
                # iterate for each neuron at this layer
                for i in range(0, len(self.neurons[layer_idx])):
                    self.neuron = self.neurons[layer_idx][i]
                    print('Neuron: {}'.format(self.neuron))
                    # calculate offset to keep each state identical
                    self.calc_offset()

                    # calculate bitshift for state coding
                    self.calc_bitshift()

                    self.starttime = time.time()

                    # print('Learning DTMC model...')
                    self.learn_dtmc_model()

                    # analyze fairness
                    _, _, _, this_diff = self.analyze_fairness()

                    new_diff = []
                    new_diff.append(this_diff)
                    new_diff.append(self.intermediate_layers[layer_idx])
                    new_diff.append(i)
                    diff_matrix_h.append(new_diff)

            print('\nSensitivity of each hidden neuron:')
            diff_matrix_h.sort()
            for item in diff_matrix_h:
                print(item)

        self.neuron = None
        self.intermediate_layer = None
        self.sens_analysis = False

        sensitivity_time = time.time() - analyze_time - preparation_time - overall_starttime

        # repair
        self.repair = True
        if 'repair_number' in spec:
            self.repair_num = spec['repair_number']

        # find highest self.repair_num item in diff_matrix_h and diff_matrix
        overall_diff = []
        if len(diff_matrix) == 0:
            if len(diff_matrix_h) == 0:
                return
            else:
                overall_diff = diff_matrix_h
        else:
            if len(diff_matrix_h) == 0:
                overall_diff = diff_matrix
            else:
                overall_diff = diff_matrix + diff_matrix_h

        overall_diff.sort()
        overall_diff = overall_diff[::-1]

        self.repair_neuron = []
        self.repair_layer = []
        for i in range (0, self.repair_num):
            self.repair_layer.append(int(overall_diff[i][1]))
            self.repair_neuron.append(int(overall_diff[i][2]))

        print('\nRepair layer: {}'.format(self.repair_layer))
        print('Repair neuron: {}'.format(self.repair_neuron))

        print('Sensitivity of each neuron:')
        for item in overall_diff:
            print(item)


        if self.repair == True:
            # repair
            print('Start reparing...')
            options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}

            optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.repair_num, options=options,
                                                bounds=([[-1.0] * self.repair_num, [1.0] * self.repair_num]),
                                                init_pos=np.zeros((20, self.repair_num), dtype=float), ftol=1e-3, ftol_iter=10)

            # Perform optimization
            best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

            # Obtain the cost history
            print(optimizer.cost_history)
            # Obtain the position history
            print(optimizer.pos_history)
            # Obtain the velocity history
            #print(optimizer.velocity_history)
            print('neuron to repair: {} at layter: {}'.format(self.repair_neuron, self.repair_layer))
            print('best cost: {}'.format(best_cost))
            print('best pos: {}'.format(best_pos))
        self.repair = False

        repair_time = time.time() - sensitivity_time - analyze_time - preparation_time - overall_starttime

        # change back acc req for testing
        self.delta = ori_delta
        self.error = ori_error
        self.timeout = ori_timeout

        self.dbgmsg = True
        # verify prob diff and model accuracy after repair
        r_prob_diff, r_acc = self.test_repaired_net(best_pos)
        print('Probability difference after repair: {}'.format(r_prob_diff))
        print('Network Accuracy after repair: {}'.format(r_acc))

        ori_accuracy = self.net_accuracy_test([], [], [])
        print('Network Accuracy before repair: {}'.format(ori_accuracy))

        valid_time = time.time() - repair_time - sensitivity_time - analyze_time - preparation_time - overall_starttime

        # output repaired network

        r_weights = []
        for i in range (0, len(self.model.layers)):
            if i & 1 != 0:
                continue
            else:
                r_weights.append(self.model.layers[i].get_weight())

        for r_idx in range (0, self.repair_num):
            r_layer = self.repair_layer[r_idx]
            r_neuron = self.repair_neuron[r_idx]
            r_weight = best_pos[r_idx]

            j = int((r_layer + 1) / 2)
            for i in range (0, len(r_weights[j][0])):
                r_weights[j][r_neuron][i] = (1 + r_weight) * r_weights[j][r_neuron][i]

        if path.exists(self.output_path + '/repair') == False:
            os.mkdir(self.output_path + '/repair')
        fout = open(self.output_path + '/repair/weight.txt', 'w+')
        for item in r_weights:
            fout.write(str(item.tolist()))
        fout.close()
        print('Repared network generated!')

        # timing measurement
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))
        print('Model verification time (s): {}'.format(analyze_time))
        print('Sensitivity analysis time (s): {}'.format(sensitivity_time))
        print('Repair time (s): {}'.format(repair_time))
        print('Validation time (s): {}'.format(valid_time))

    def pso_fitness_func(self, weight):

        result = []
        for i in range (0, int(len(weight))):
            self.repair_w =  weight[i]

            accuracy = self.net_accuracy_test(self.repair_neuron, weight[i], self.repair_layer)

            self.starttime = time.time()

            self.learn_dtmc_model()

            _, _, prob_diff, _ = self.analyze_fairness()

            _result = prob_diff + 0.1 * (1 - accuracy)

            self.debug_print('Repaired prob_diff: {}, accuracy: {}'.format(prob_diff, accuracy))

            result.append(_result)
        print(result)

        return result

    def net_accuracy_test(self, r_neuron=0, r_weight=0, r_layer=0):
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        l_pass = 0
        l_fail = 0

        for i in range(100):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))
            if len(r_neuron) != 0:
                y = self.model.apply_repair(x0, r_neuron, r_weight, r_layer)
            else:
                y = self.model.apply(x0)

            lbl_x0 = np.argmax(y, axis=1)[0]

            # accuracy test
            if lbl_x0 == y0s[i]:
                l_pass = l_pass + 1
            else:
                l_fail = l_fail + 1
        acc = l_pass / (l_pass + l_fail)

        #self.debug_print("Accuracy of ori network: %f.\n" % (acc))

        return acc

    def test_repaired_net(self, weight):

        self.repair_w = weight

        self.repair = True
        accuracy = self.net_accuracy_test(self.repair_neuron, weight, self.repair_layer)

        self.starttime = time.time()

        self.learn_dtmc_model()

        _, _, prob_diff, _ = self.analyze_fairness()
        self.repair = False
        return prob_diff, accuracy

    def calc_offset(self):
        lower = self.model.lower
        upper = self.model.upper

        size = upper.size
        self.offset = []
        self.offset.append(1)
        for i in range(1, size):
            self.offset.append(self.offset[i - 1] + upper[i - 1] - lower[i - 1] + 1)

    def _calc_bitshift(self):
        self.bitshift = []
        total_analyze = len(self.intermediate_layer) + 4  # start + input layer (sensitive + other feature) + output layer

        for i in range(0, total_analyze):
            if i == 0:  # start
                self.bitshift.append(0)
            elif i == 1:  # input sensitive
                self.bitshift.append(4)
            elif i == 2:  # input other feature
                # sensitive feature range
                sensitive_range = self.sens_cluster

                # how many bits needed? how many nibbles
                nibbles = int(sensitive_range / 16 + 1)  # + 1 to handle floating point result

                self.bitshift.append(nibbles * 4 + self.bitshift[i - 1])
            elif i != total_analyze - 1:
                if len(self.feature) == 0:
                    self.bitshift.append(self.bitshift[i - 1])
                    continue

                # other feature range
                feature_range = self.feature_cluster

                # how many bits needed? how many nibbles
                nibbles = int(feature_range / 16 + 1)  # + 1 to handle floating point result

                self.bitshift.append(nibbles * 4 + self.bitshift[i - 1])
            elif i == total_analyze - 1:
                self.bitshift.append(1)


    def calc_bitshift(self):
        self.bitshift = []
        total_analyze = 4  # 0: start, 1: input layer (sensitive), 2: input layer (other feature), 3/4: output layer
        if self.neuron != None:
            total_analyze = total_analyze + 1   # 3: hidden neuron

        for i in range(0, total_analyze):
            if i == 0:  # start
                self.bitshift.append(0)
            elif i == 1:  # input sensitive
                self.bitshift.append(4)
            elif i == 2:  # input other feature
                # sensitive feature range
                sensitive_range = self.sens_cluster

                # how many bits needed? how many nibbles
                nibbles = int(sensitive_range / 16 + 1)  # + 1 to handle floating point result

                self.bitshift.append(nibbles * 4 + self.bitshift[i - 1])
            elif i != total_analyze - 1:    # hidden neuron
                if len(self.feature) == 0:
                    self.bitshift.append(self.bitshift[i - 1])
                    continue

                # other feature range
                feature_range = self.feature_cluster

                # how many bits needed? how many nibbles
                nibbles = int(feature_range / 16 + 1)  # + 1 to handle floating point result

                self.bitshift.append(nibbles * 4 + self.bitshift[i - 1])
            elif i == total_analyze - 1:    # output layer
                self.bitshift.append(1)


    '''
    Update model when add a new trace path
    '''
    def update_model(self, path):
        if path[-1] not in self.final:
            self.final.append(path[-1])
        for i in range(0, len(path)):

            # link previous state to n_ij and n_i
            if i == 0:
                continue
            idx_start = self.s.index(path[i - 1])
            idx_end = self.s.index(path[i])
            self.n_ij[idx_start][idx_end] = self.n_ij[idx_start][idx_end] + 1
            self.n_i[idx_start] = self.n_i[idx_start] + 1

        return

    def add_state(self, state, sequence):

        if state not in self.s:
            self.s.append(state)
            self.s_idx.append(sequence)
            self.m = self.m + 1
            # add corresponding A fields
            if len(self.A) == 0:
                self.A.append([0.0])
            else:
                for row in self.A:
                    row.append(0.0)
                self.A.append([0.0] * len(self.A[0]))

            # add corresponding n_ij fields
            if len(self.n_ij) == 0:
                self.n_ij.append([0.0])
            else:
                for row in self.n_ij:
                    row.append(0.0)
                self.n_ij.append([0.0] * len(self.n_ij[0]))

            # add corresponding n_i fields
            self.n_i.append(0.0)

        return

    '''
    init model
    '''

    def init_model(self):
        self.s = []
        self.s_idx = []
        self.m = 0
        self.n_ij = []
        self.n_i = []
        self.A = []
        self.num_of_path = 0

        # initialize states of inputs:

        # start node
        self.add_state(0, 0)

        # input and intermediat layer
        # states: 0: start
        # 1: output 1
        # 2: output 2
        # 3 ... self.sens_group + 3: sens groups
        # self.sens_group + 3 + 1: hidden low
        # self.sens_group + 3 + 1 + 1: hidden high

        if self.sens_analysis == True and self.neuron != None:
            # input layer
            sens_range = self.sens_cluster
            for i in range(0, int(sens_range)):
                state = (int(i) + 1) << self.bitshift[1]
                self.add_state(state, 1)

            # hidden neurons
            for i in range(0, int(self.hidden_cluster)):
                state = (int(i) + 1) << self.bitshift[3]
                self.add_state(state, 2)

            # output layer
            self.add_state(self.offset[-1], 3)
            self.add_state(self.offset[-1] + 1, 3)
        else:
            # other features
            sens_range = self.sens_cluster

            if len(self.feature) != 0:
                other_feature_range = self.feature_cluster
                for i in range (0, int(sens_range)):
                    for j in range (0, int(other_feature_range)):
                        state = (int(i) + 1) << self.bitshift[1] | (int(j) + 1) << self.bitshift[2]
                        self.add_state(state, 1)
            else:
                for i in range (0, int(sens_range)):
                    state = (int(i) + 1) << self.bitshift[1]
                    self.add_state(state, 1)

            # output layer
            self.add_state(self.offset[-1], 2)
            self.add_state(self.offset[-1] + 1, 2)

        self.final.append(self.offset[-1])
        self.final.append(self.offset[-1] + 1)

        return

    '''
    check if more samples needed
    '''

    def is_more_sample_needed(self):
        if (self.m == 0) or (self.m == 1):
            return True

        for i in range(0, self.m):
            if self.s[i] in self.final:
                continue
            max_diff = 0.0
            for j in range(0, self.m):
                if (self.n_i[i] == 0.0):
                    continue
                diff = abs(0.5 - self.n_ij[i][j] / self.n_i[i])
                if diff > max_diff:
                    max_diff = diff
            H = 2.0 / (self.error * self.error) * math.log(2.0 / (self.delta / self.m)) * (
                        0.25 - (max_diff - 2.0 * self.error / 3.0) * (max_diff - 2.0 * self.error / 3.0))
            if self.n_i[i] < H:
                # timeout?
                if ((time.time() - self.starttime) > self.timeout):
                    print('Timeout! States that need more sample:')
                    needs_more = []
                    for k in range(i, self.m):
                        if self.s[k] in self.final:
                            continue
                        max_diff = 0.0
                        for p in range(0, self.m):
                            if self.n_i[k] == 0.0:
                                continue
                            diff = abs(0.5 - self.n_ij[k][p] / self.n_i[k])
                            if diff > max_diff:
                                max_diff = diff
                        H = 2.0 / (self.error * self.error) * math.log(2.0 / (self.delta / self.m)) * (
                                0.25 - (max_diff - 2.0 * self.error / 3.0) * (max_diff - 2.0 * self.error / 3.0))
                        if self.n_i[k] < H:
                            needs_more.append(self.s[k])
                            print("0x%016X: %d" % (int(self.s[k]), int(self.n_i[k])))
                    #print("\n")

                    return False

                return True

        return False

    '''

    '''

    def _get_new_sample(self):
        lower = self.model.lower
        upper = self.model.upper

        generated = self.step
        out = []

        while generated:
            x = self.__generate_x(self.model.shape, lower, upper)
            y, layer_op = self.model.apply_intermediate(x)
            y = np.argmax(y, axis=1)[0]

            intermediate_result = []
            for i in range(0, len(layer_op)):
                if i in self.intermediate_layer:
                    layer_sign = np.sign(layer_op[i])

                    # code into one state: each neuron represendted by 2 bits
                    # TODO: only support positive and non-positive now
                    '''
                    #encode all neurtons in this layer
                    layer_state = 0
                    for j in range (0, len(layer_sign[0])):
                        layer_state = layer_state | (int((layer_sign[0][j] + 1)) << (2 * j))
                    '''
                    '''
                    #count number of activated neuron
                    ayer_activated = np.count_nonzero(layer_sign)
                    layer_state = layer_activated
                    '''

                    # neuron by neuron
                    layer_state = int(layer_sign[0][self.neuron] + 1)
                    self.add_state((1 << self.bitshift[3]), 2)
                    self.add_state((2 << self.bitshift[3]), 2)

                    intermediate_result.append((layer_state << self.bitshift[3]))

            path = [self.s0]

            # input feature under analysis
            # TODO: support only one feature and one sensitive
            to_add = 0
            for i in range(0, len(x)):
                if i in self.feature:
                    new = (int(x[i] - self.model.lower[self.feature[0]]) + 1) << self.bitshift[2]
                    to_add = to_add | new
                if i in self.sensitive:
                    new = (int(x[i] - self.model.lower[self.sensitive[0]]) + 1) << self.bitshift[1]
                    to_add = to_add | new

            path.append(to_add)

            # intermediate layer result under analysis
            # TODO: only support positive and non-positive now
            for i in range(0, len(intermediate_result)):
                path.append(intermediate_result[i])

            path.append(int(y + self.offset[-1]))
            self.num_of_path = self.num_of_path + 1
            out.append(path)
            generated = generated - 1

        return out

    def get_new_sample(self, k_sens, k_feature, k_hidden):
        lower = self.model.lower
        upper = self.model.upper

        generated = self.step

        sens_array = []
        feature_array = []
        hidden_array = []
        y_array = []


        while generated:
            x = self.__generate_x(self.model.shape, lower, upper)
            y = 0
            if self.repair == False:
                if self.neuron != None:
                    y, layer_op = self.model.apply_intermediate(x, self.intermediate_layer)
                else:
                    y = self.model.apply(x)
            else:
                y = self.model.apply_repair(x, self.repair_neuron, self.repair_w, self.repair_layer)

            y = np.argmax(y, axis=1)[0]

            # hidden neuron
            if self.neuron != None:
                neuron_value = layer_op[self.neuron]
                hidden_array.append(neuron_value)

            # input feature under analysis
            # path: s0, sens, feature, output
            to_add_sens = 0
            to_add_feature = 0
            for i in range(0, len(x)):
                if i in self.feature:
                    to_add_feature = x[i]
                if i in self.sensitive:
                    to_add_sens = x[i]

            sens_array.append(to_add_sens)
            feature_array.append(to_add_feature)

            y_array.append(int(y + self.offset[-1]))

            self.num_of_path = self.num_of_path + 1
            generated = generated - 1

        # till now we have all values without clustering
        # perform kmeans clustering on senstive feature
        test = np.array(sens_array).reshape(-1, 1)
        k_sens = k_sens.partial_fit(test)

        # perform kmeans clustering on other feature under analysis
        if len(self.feature) != 0:
            # perform kmeans clustering on other feature
            k_feature = k_feature.partial_fit(np.array(feature_array).reshape(-1, 1))

        # perform kmeans clustering on hidden neuron under analysis
        if self.neuron != None:
            k_hidden = k_hidden.partial_fit(np.array(hidden_array).reshape(-1, 1))

        out_ = []
        for i in range(0, self.step):
            path_ = [self.s0]
            to_add = (int(k_sens.labels_[i]) + 1) << self.bitshift[1]
            if len(self.feature) != 0:
                to_add = to_add | (int(k_feature.labels_[i]) + 1) << self.bitshift[2]
            path_.append(to_add)

            if self.neuron != None:
                path_.append((int(k_hidden.labels_[i]) + 1) << self.bitshift[3])

            path_.append(y_array[i])
            out_.append(path_)

        return out_, k_sens, k_feature, k_hidden

    '''
    learn dtmc model based on given network
    '''

    def learn_dtmc_model(self):
        #file = open(self.gen_path, 'w+')

        kmeans_sens = MiniBatchKMeans(n_clusters=self.sens_cluster, init='k-means++', batch_size=self.step)
        kmeans_other = None
        kmeans_hidden = None

        if len(self.feature) != 0:
            # perform kmeans clustering on other feature
            kmeans_other = MiniBatchKMeans(n_clusters=self.feature_cluster, init='k-means++', batch_size=self.step)

        if len(self.intermediate_layers) != 0:
            # perform kmeans clustering on hidden neuron
            kmeans_hidden = MiniBatchKMeans(n_clusters=self.hidden_cluster, init='k-means++', batch_size=self.step)

        self.init_model()
        while (self.is_more_sample_needed() == True):
            path, kmeans_sens, kmeans_other, kmeans_hidden = self.get_new_sample(kmeans_sens, kmeans_other, kmeans_hidden)

            for i in range(0, self.step):
                self.update_model(path[i])
                #or item in path[i]:
                #    file.write("%f\t" % item)
                #file.write("\n")

        self.finalize_model()

        self.d_detail_print('Number of traces generated: {}'.format(self.num_of_path))
        self.d_detail_print('Number of states: {}'.format(self.m))
        #file.close()
        return

    def finalize_model(self):
        for i in range(0, self.m):
            for j in range (0, self.m):
                if self.n_i[i] == 0.0:
                    self.A[i][j] = 1.0 / self.m
                else:
                    self.A[i][j] = self.n_ij[i][j] / self.n_i[i]
        return

    def analyze_fairness(self):
        # generate weight matrix
        weight = []
        from_symbol = []
        to_symbol = []

        hidden_length = 0
        if self.intermediate_layer != None:
            hidden_length = 1

        for i in range(1, (len(self.sensitive) + hidden_length) + 2):
            w = []
            _from_symbol = []
            _to_symbol = []
            for idx_row in range(0, self.m):
                if self.s_idx[idx_row] == i - 1:
                    _from_symbol.append(idx_row)
                    w_row = []
                    for j in range(0, self.m):
                        if self.s_idx[j] == i:
                            w_row.append(self.A[idx_row][j])
                            if j not in _to_symbol:
                                _to_symbol.append(j)
                    w.append(w_row)

            weight.append(w)
            from_symbol.append(_from_symbol)
            to_symbol.append(_to_symbol)

        # analyze independence fairness
        res = []
        res.append(weight[(len(self.sensitive) + hidden_length)])
        self.detail_print("\nProbabilities:")
        # print('Sensitive feature {}:'.format(self.sensitive[-1]))

        # print index
        self.detail_print("transition from:")
        for item in from_symbol[(len(self.sensitive) + hidden_length)]:
            self.detail_print("0x%016X" % int(self.s[item]))
        self.detail_print("transition to:")
        for item in to_symbol[(len(self.sensitive) + hidden_length)]:
            self.detail_print("0x%016X" % int(self.s[item]))

        # print transformation matrix
        self.detail_print("\n")
        for item in weight[(len(self.sensitive) + hidden_length)]:
            self.detail_print(item)
        # print(np.matrix(weight[len(self.sensitive)]))
        self.detail_print("\n")
        for i in range(0, (len(self.sensitive) + hidden_length)):
            result = np.matmul(weight[(len(self.sensitive) + hidden_length) - i - 1], res[i])
            res.append(result)

            # print index
            self.detail_print("transition from:")
            for item in from_symbol[(len(self.sensitive) + hidden_length) - i - 1]:
                self.detail_print("0x%016X" % int(self.s[item]))
            self.detail_print("transition to:")
            for item in to_symbol[(len(self.sensitive) + hidden_length)]:
                self.detail_print("0x%016X" % int(self.s[item]))

            self.detail_print("\n")
            for item in np.matrix(result):
                self.detail_print(item)

            # print(np.matrix(result))
            self.detail_print("\n")

        # print bitshift
        self.detail_print(
            "State coding bitshift (0: start, 1: sensitive input feature, 2: other input feature, 3+: intermediate state, last: output):")
        for item in self.bitshift:
            self.detail_print(item)

        # check against criteria
        weight_to_check = weight[(len(self.sensitive) + hidden_length)]
        # TODO: to handle more than 2 labels

        weight_to_check.sort()

        prob_diff = weight_to_check[len(weight_to_check) - 1][0] - weight_to_check[0][0]
        max_diff = 0.0  # for sensitivity analysis
        fairness_result = 1
        if self.sens_analysis == False:
            if prob_diff > self.criteria:
                fairness_result = 0
                self.d_detail_print("Failed accurcay criteria!")
            else:
                fairness_result = 1
                self.d_detail_print("Passed accurcay criteria!")

            self.d_detail_print('Probability difference: {:.4f}'.format(prob_diff))

            #print("Total execution time: %fs\n" % (time.time() - self.starttime))
        else:
            # analyse sensitivity

            if self.sens_analysis == True and self.neuron != None:
                # hidden neuron sensitivity analysis

                # calculate P_(h,o)
                w_ho = np.array(weight[2])[:, 0]
                sens_h = []
                # calculate prob diff to hidden states
                hidden_diff = []
                hidden_range = self.hidden_cluster

                w_s0_h = np.matmul(np.array(weight[0]), np.array(weight[1]))

                for i in range (0, hidden_range):
                    w_diff_h = np.array(np.array(weight[1])[:, i])
                    w_diff_h.sort()
                    current_h_diff = w_diff_h[-1] - w_diff_h[0]
                    curretn_sens_h = current_h_diff * w_ho[i]

                    hidden_diff.append(current_h_diff)
                    sens_h.append(curretn_sens_h)

                    # max
                    #new_diff = curretn_sens_h * w_s0_h[i]
                    #if new_diff > max_diff:
                    #    max_diff = new_diff

                # hidden neuron sensitivity @ sum
                sens_n = np.matmul(w_s0_h, np.array(sens_h).reshape(-1, 1))

                max_diff = sens_n[0][0]

            else:
                # other features sensitivity analysis
                sens_range = self.sens_cluster
                other_feature_range = self.feature_cluster
                #p_matrix = []   # probability to feature under analysis
                #w_matrix = []   # weight matrix from feature sate to output state

                for j in range(0, int(other_feature_range)):
                    w_matrix_j = []
                    prob = 0.0
                    for i in range (0, int(sens_range)):
                        #state = (int(i) + 1) << self.bitshift[1] | (int(j) + 1) << self.bitshift[2]
                        state_idx = i * int(other_feature_range) + j
                        prob = prob + weight[0][0][state_idx]
                        w_matrix_j.append(weight[1][state_idx])

                    w_matrix_j_sort = w_matrix_j
                    w_matrix_j_sort.sort()
                    new_diff = abs(w_matrix_j_sort[len(w_matrix_j_sort) - 1][0] - w_matrix_j_sort[0][0]) * prob

                    # sum
                    max_diff = max_diff + new_diff

                    # max
                    #if new_diff > max_diff:
                    #    max_diff = new_diff
                    #p_matrix.append(weight[0][0][state_idx])
                    #w_matrix.append(w_matrix_j)

        # accuracy_diff = self.label_diff / self.num_of_path
        # self.debug_print("Total label difference: %f\n" % (accuracy_diff))

        self.debug_print("Debug message:")

        for i in range(0, len(weight)):
            self.debug_print("weight: %d" % i)
            for item in weight[i]:
                self.debug_print(item)

        return res, fairness_result, prob_diff, max_diff

    def aequitas_test(self):
        num_trials = 400
        samples = 1000

        estimate_array = []
        rolling_average = 0.0

        for i in range(num_trials):
            disc_count = 0
            total_count = 0
            for j in range(samples):
                total_count = total_count + 1

                if (self.aeq_test_new_sample()):
                    disc_count = disc_count + 1

            estimate = float(disc_count) / total_count
            rolling_average = ((rolling_average * i) + estimate) / (i + 1)
            estimate_array.append(estimate)

            print(estimate, rolling_average)

        print("Total execution time: %fs\n" % (time.time() - self.starttime))
        return estimate_array


    def aeq_test_new_sample(self):
        lower = self.model.lower
        upper = self.model.upper

        x = self.__generate_x(self.model.shape, lower, upper)
        y = np.argmax(self.model.apply(x), axis=1)[0]
        x_g = x

        sensitive_feature = self.sensitive[0]
        sens_range = upper[sensitive_feature] - lower[sensitive_feature] + 1

        for val in range(int(lower[sensitive_feature]), int(upper[sensitive_feature]) + 1):
            if val != x[sensitive_feature]:
                x_g[sensitive_feature] = float(val)
                y_g = np.argmax(self.model.apply(x_g), axis=1)[0]

                if y != y_g:
                    return 1
        return 0

    def detail_print(self, x):
        if self.sens_analysis and self.dbgmsg:
            print(x)

    def debug_print(self, x):
        if self.dbgmsg:
            print(x)
        pass

    def d_detail_print(self, x):
        if self.dbgmsg:
            print(x)



    def export_prism_model(self):
        if path.exists(self.output_path + '/output_model') == False:
            os.mkdir(self.output_path + '/output_model')
        fout = open(self.output_path + '/output_model/original_model.pm', 'w+')

        fout.write('dtmc\n\n')

        # start module
        fout.write('module model_learned\n')

        # states
        to_write = 's:[' + str(min(self.s)) + '..' + str(max(self.s)) + '] init ' + str(min(self.s)) + ';\n'
        fout.write(to_write)

        # state transitions
        for i in range (0, len(self.s)):
            to_write = '[]s=' + str(self.s[i]) + ' -> '
            first = True
            is_empty = True
            for j in range (0, len(self.A[i])):
                if self.A[i][j] == 0:
                    continue
                is_empty = False
                if first == True:
                    to_write = to_write + str(self.A[i][j]) + ':(s\'=' + str(self.s[j]) + ')'
                    first = False
                else:
                    to_write = to_write + ' + ' + str(self.A[i][j]) + ':(s\'=' + str(self.s[j]) + ')'
            to_write = to_write + ';\n'
            if is_empty == False:
                fout.write(to_write)

        # end module
        fout.write('\nendmodule')
        fout.flush()


        fout.close()
        return