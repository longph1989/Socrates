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
import matplotlib.pyplot as plt
from solver.dtmc_impl import DTMCImpl
#import lib_models

class CausalImpl():
    def __init__(self):
        self.model = None
        self.assertion = None
        self.display = None
        self.timeout = 5
        self.datapath = None    # path to di data
        self.datalen = 0        # len of di data to sample at each iteration
        self.datalen_tot = 0     # len of total di data
        self.acc_datapath = None    # path to accuracy test data
        self.acc_datalen = 0        # len of existing data
        self.acc_datalen_tot = 0
        self.resultpath = None  # path to output result folder
        self.stepsize = 16      # step size for intervension
        self.do_layer = []
        self.do_neuron = []
        self.r_layer = []
        self.r_neuron = []
        self.best_pos = []
        self.repair_num = 3
        self.class_n = 0
        self.sens_idx = 0
        self.sens_value = []
        self.sens_cluster = 2
        self.sens_threshold = 0
        self.class_n = 0    # test on which class
        self.alpha = 0.3    # importance of accuracy in pso fitness function
        self.criteria = 0.02
        self.error = 0.01    # tolerance
        self.delta = 0.1   # confidence
        self.plot = False   # plot figures?
        self.dbgmsg = False

    def gaussian(self, mu, sigma):
        return np.random.normal(mu, np.sqrt(sigma))

    def gaussian_generate_x(self, shape, mu, std):
        size = np.prod(shape)
        #x = np.random.rand(size)
        x = []
        for i in range (0, size):
            x.append(self.gaussian(mu[i], std[i]))

        out = np.array([0.0] * x.size)
        i = 0
        for x_i in x:
            x_i = round(x_i)
            out[i] = x_i
            i = i + 1

        return out

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

    #
    #   solve for fairness improvement
    #
    def solve(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        if 'timeout' in spec:
            self.timeout = read(spec['timeout']) * 60

        if 'stepsize' in spec:
            self.stepsize = spec['stepsize']

        if 'datalen' in spec:
            self.datalen = spec['datalen']

        if 'datalen_tot' in spec:
            self.datalen_tot = spec['datalen_tot']

        if 'datapath' in spec:
            self.datapath = spec['datapath']

        if 'acc_datalen' in spec:
            self.acc_datalen = spec['acc_datalen']

        if 'acc_datalen_tot' in spec:
            self.acc_datalen_tot = spec['acc_datalen_tot']

        if 'acc_datapath' in spec:
            self.acc_datapath = spec['acc_datapath']

        if 'resultpath' in spec:
            self.resultpath = spec['resultpath']

        if 'do_layer' in spec:
            self.do_layer = np.array(ast.literal_eval(read(spec['do_layer'])))

        if 'do_neuron' in spec:
            self.do_neuron = np.array(ast.literal_eval(read(spec['do_neuron'])))

        if 'error' in spec:
            self.error = (spec['error'])

        if 'confidence' in spec:
            self.delta = (spec['confidence'])

        if 'criteria' in spec:
            self.criteria = spec['criteria']

        if 'acc_alpha' in spec:
            self.alpha = spec['acc_alpha']

        print('Accuracy importance alpha: {}'.format(self.alpha))
        print('Causal analysis step size: {}'.format(self.stepsize))
        print('Accuracy test data percentage: {}'.format(self.acc_datalen / self.acc_datalen_tot))
        print('Discriminitary data sample percentage: {}'.format(self.datalen / self.datalen_tot))

        if 'solve_option' in spec:
            self.solve_option = spec['solve_option']
        else:
            print('Nothing to analyze!')
            return

        # get ace for each hidden neuron

        if len(self.do_layer) == 0 or len(self.do_neuron) == 0:
            print('Nothing to analyze!')
            return

        if self.solve_option == 'solve_fairness':
            self.solve_fairness(model, assertion)
        else:
            self.solve_general(model, assertion)

        return

    def solve_general(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        plt_row = 0
        for i in range (0, len(self.do_layer)):
            if len(self.do_neuron[i]) > plt_row:
                plt_row = len(self.do_neuron[i])
        plt_col = len(self.do_layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(3.5*plt_col, 2.5*plt_row), sharex=False, sharey=True)
        fig.tight_layout()

        row = 0
        col = 0
        ie_ave = []
        for do_layer in self.do_layer:
            row = 0
            print('Analyzing layer {}'.format(col))
            ie_ave_l = []
            for do_neuron in self.do_neuron[col]:
                ie, min, max = self.get_ie_do_h(do_layer, do_neuron, self.stepsize, 0)

                ie_ave_l.append(np.mean(np.array(ie)))

                # plot ACE
                #ax[row, col].set_title('N_' + str(do_layer) + '_' + str(do_neuron))
                ax[row, col].set_xlabel('Intervention Value(alpha)')
                ax[row, col].set_ylabel('Causal Attributions(ACE)')

                # Baseline is np.mean(expectation_do_x)
                ax[row, col].plot(np.linspace(min, max, self.stepsize), np.array(ie) - np.mean(np.array(ie)), label = str(do_layer) + '_' + str(do_neuron), color='b')
                ax[row, col].legend()

                row = row + 1
            if row == len(self.do_neuron[col]):
                for off in range(row, plt_row):
                    ax[off, col].set_axis_off()
            col = col + 1
            ie_ave.append(ie_ave_l)

        plt.savefig(self.resultpath + '/' + 'all' + ".png")
        plt.show()

        # plot average ie
        plt.figure()
        plt.ylabel('Average IE')

        color_tab = ['b','g','r','c','m']

        idx = 0
        for i in range (0, len(ie_ave)):
            for j in range (0, len(ie_ave[i])):
                plt.scatter(idx, ie_ave[i][j], color=color_tab[i])
                idx = idx + 1

        plt.show()

        # save avg ie
        for ie_l in ie_ave:
            print(ie_l)

        # timing measurement
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))

    #
    #   solve for fairness improvement
    #
    def solve_fairness(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        # get sensitive feature details
        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))

        if 'sens_cluster' in spec:
            self.sens_cluster = ast.literal_eval(read(spec['sens_cluster']))

        if 'sens_threshold' in spec:
            self.sens_threshold = ast.literal_eval(read(spec['sens_threshold']))

        if 'class_n' in spec:
            self.class_n = spec['class_n']

        if 'repair_num' in spec:
            self.repair_num = spec['repair_num']

        sens_idx = self.sensitive[0]

        self.sens_idx = sens_idx

        val = self.model.lower[sens_idx]
        while val <= self.model.upper[sens_idx]:
            self.sens_value.append(val)
            val = val + 1.0

        #ori_accuracy = self.net_accuracy_test([], [], [])
        #print('Network Accuracy before repair: {}'.format(ori_accuracy))
        #return
        # analyze fairness
        dtmc = DTMCImpl()
        #dtmc.analyze_fairness_causal(model, self.sensitive, self.sens_cluster, self.timeout, self.criteria, self.error,
        #                             self.delta, self.acc_datapath, self.resultpath, [], [], [], self.sens_threshold)

        #'''#sunbing
        # do causal anaylysis
        ie_ave_matrix = []
        if not self.plot:
            print('Analyze causal attribution of hidden neurons to unfairness:')
            ie_ave = []
            col = 0
            for do_layer in self.do_layer:
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                neuron_idx = 0
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_h_dy(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))
                    new_entry = []
                    new_entry.append(np.mean(np.array(ie)))
                    new_entry.append(do_layer)
                    new_entry.append(do_neuron)
                    ie_ave_matrix.append(new_entry)

                    neuron_idx = neuron_idx + 1

                ie_ave.append(ie_ave_l)
                col = col + 1

        else:
            ie_ave = []
            plt_row = 0
            for i in range (0, len(self.do_layer)):
                if len(self.do_neuron[i]) > plt_row:
                    plt_row = len(self.do_neuron[i])
            plt_col = len(self.do_layer)
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(3.5*plt_col, 2.5*plt_row), sharex=False, sharey=True)
            fig.tight_layout()

            row = 0
            col = 0
            for do_layer in self.do_layer:
                row = 0
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_h_dy(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))

                    # plot ACE
                    #ax[row, col].set_title('N_' + str(do_layer) + '_' + str(do_neuron))
                    ax[row, col].set_xlabel('Intervention Value(alpha)')
                    ax[row, col].set_ylabel('Causal Attributions(ACE)')

                    # Baseline is np.mean(expectation_do_x)
                    ax[row, col].plot(np.linspace(min, max, self.stepsize), np.array(ie) - np.mean(np.array(ie)), label = str(do_layer) + '_' + str(do_neuron), color='b')
                    ax[row, col].legend()

                    row = row + 1
                if row == len(self.do_neuron[col]):
                    for off in range(row, plt_row):
                        ax[off, col].set_axis_off()
                ie_ave.append(ie_ave_l)
                col = col + 1

            plt.show()
            plt.savefig(self.resultpath + '/' + 'all' + ".png")

            # plot average ie
            plt.figure()
            plt.ylabel('Average IE')

            color_tab = ['b','g','r','c','m']

            idx = 0
            for i in range (0, len(ie_ave)):
                for j in range (0, len(ie_ave[i])):
                    plt.scatter(idx, ie_ave[i][j], color=color_tab[i])
                    idx = idx + 1

            plt.show()

            # save avg ie
            for ie_l in ie_ave:
                print(ie_l)

        for item in ie_ave_matrix:
            self.debug_print(item)
        ie_ave_matrix.sort()
        ie_ave_matrix = ie_ave_matrix[::-1]


        self.r_neuron = []
        self.r_layer = []
        for i in range (0, self.repair_num):
            self.r_layer.append(int(ie_ave_matrix[i][1]))
            self.r_neuron.append(int(ie_ave_matrix[i][2]))
        #'''
        fault_loc_time = time.time() - overall_starttime

        # randomly select 10 neurons
        '''#random neuron
        self.r_neuron = []
        self.r_layer = []

        for i in range (0, self.repair_num):
            # layer
            ran_layer_idx = int(np.random.rand() * (len(self.do_neuron)))

            ran_layer = self.do_layer[ran_layer_idx]
            ran_neuron = int(np.random.rand() * len(self.do_neuron[ran_layer_idx]))
            self.r_neuron.append(ran_neuron)
            self.r_layer.append(ran_layer)

        '''

        ''' #all neuron
        self.r_neuron = []
        self.r_layer = []
        i = 0
        for _layer in self.do_layer:
            for _neuron in self.do_neuron[i]:
                self.r_layer.append(_layer)
                self.r_neuron.append(_neuron)
            i = i + 1

        '''

        '''#fix neuron
        self.r_layer = [4, 4, 2, 0, 2, 2, 0, 0, 2, 8, 8, 8, 8]
        self.r_neuron = [4, 1, 28, 28, 17, 15, 30, 8, 0, 3, 2, 1, 0]
        '''

        print('Repair:')

        print('\nRepair layer: {}'.format(self.r_layer))
        print('Repair neuron: {}'.format(self.r_neuron))

        # start repair
        best_pos = self.repair()
        #best_pos =

        self.acc_datalen = self.acc_datalen_tot
        # verify prob diff and model accuracy after repair , weight, layer, neuron, sens_idx, sens_value, class_n
        r_id_rate, r_acc = self.test_repaired_net_fix(best_pos, self.r_layer, self.r_neuron, self.sens_idx, self.sens_value, self.class_n)
        print('Percentage of discriminatory instance after repair: 1.0')
        print('Percentage of discriminatory instance after repair: {}'.format(r_id_rate))
        print('Network Accuracy after repair: {}'.format(r_acc))

        ori_accuracy = self.net_accuracy_test([], [], [])
        self.acc_datalen = 100

        print('Network Accuracy before repair: {}'.format(ori_accuracy))

        #self.timeout = 1800
        # analyze fairness after repair
        #dtmc.analyze_fairness_causal(model, self.sensitive, self.sens_cluster, self.timeout, self.criteria, self.error,
        #                             self.delta, self.acc_datapath, self.resultpath, self.r_layer, self.r_neuron, best_pos, self.sens_threshold)

        # timing measurement
        print('Fault localization time(s): {}'.format(fault_loc_time))
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))

    #
    #   solve for fairness improvement
    #
    def solve_fairness_gradient(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        # get sensitive feature details
        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))

        if 'sens_cluster' in spec:
            self.sens_cluster = ast.literal_eval(read(spec['sens_cluster']))

        if 'sens_threshold' in spec:
            self.sens_threshold = ast.literal_eval(read(spec['sens_threshold']))

        if 'class_n' in spec:
            self.class_n = spec['class_n']

        if 'repair_num' in spec:
            self.repair_num = spec['repair_num']

        sens_idx = self.sensitive[0]

        self.sens_idx = sens_idx

        val = self.model.lower[sens_idx]
        while val <= self.model.upper[sens_idx]:
            self.sens_value.append(val)
            val = val + 1.0


        # analyze fairness
        dtmc = DTMCImpl()
        #dtmc.analyze_fairness_causal(model, self.sensitive, self.sens_cluster, self.timeout, self.criteria, self.error,
        #                             self.delta, self.acc_datapath, self.resultpath, [], [], [], self.sens_threshold)
        #'''#sunbing
        # do causal anaylysis
        ie_ave_matrix = []
        if not self.plot:
            print('Analyze causal attribution of hidden neurons to unfairness:')
            ie_ave = []
            col = 0
            for do_layer in self.do_layer:
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                neuron_idx = 0
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_h_dy_gradient(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))
                    new_entry = []
                    new_entry.append(np.mean(np.array(ie)))
                    new_entry.append(do_layer)
                    new_entry.append(do_neuron)
                    ie_ave_matrix.append(new_entry)

                    neuron_idx = neuron_idx + 1

                ie_ave.append(ie_ave_l)
                col = col + 1


        ie_ave_matrix.sort()
        ie_ave_matrix = ie_ave_matrix[::-1]
        for item in ie_ave_matrix:
            self.debug_print(item)

        self.r_neuron = []
        self.r_layer = []
        for i in range (0, self.repair_num):
            self.r_layer.append(int(ie_ave_matrix[i][1]))
            self.r_neuron.append(int(ie_ave_matrix[i][2]))
        #'''
        fault_loc_time = time.time() - overall_starttime

        # randomly select 10 neurons
        '''#random neuron
        self.r_neuron = []
        self.r_layer = []

        for i in range (0, self.repair_num):
            # layer
            ran_layer_idx = int(np.random.rand() * (len(self.do_neuron)))

            ran_layer = self.do_layer[ran_layer_idx]
            ran_neuron = int(np.random.rand() * len(self.do_neuron[ran_layer_idx]))
            self.r_neuron.append(ran_neuron)
            self.r_layer.append(ran_layer)

        '''

        ''' #all neuron
        self.r_neuron = []
        self.r_layer = []
        i = 0
        for _layer in self.do_layer:
            for _neuron in self.do_neuron[i]:
                self.r_layer.append(_layer)
                self.r_neuron.append(_neuron)
            i = i + 1

        '''

        '''#fix neuron 
        self.r_layer = [4, 4, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2, 2]
        self.r_neuron = [11, 7, 30, 31, 13, 58, 62, 54, 7, 28, 20, 26, 25]
        '''

        print('Repair:')

        print('\nRepair layer: {}'.format(self.r_layer))
        print('Repair neuron: {}'.format(self.r_neuron))

        # start repair
        best_pos = self.repair()


        # verify prob diff and model accuracy after repair , weight, layer, neuron, sens_idx, sens_value, class_n
        self.acc_datalen = 10000
        r_id_rate, r_acc = self.test_repaired_net(best_pos, self.r_layer, self.r_neuron, self.sens_idx, self.sens_value, self.class_n)
        print('Percentage of discriminatory instance after repair: {}'.format(r_id_rate))
        print('Network Accuracy after repair: {}'.format(r_acc))
        self.acc_datalen = 100
        ori_accuracy = self.net_accuracy_test([], [], [])
        print('Network Accuracy before repair: {}'.format(ori_accuracy))

        #self.timeout = 1800
        # analyze fairness after repair
        dtmc.analyze_fairness_causal(model, self.sensitive, self.sens_cluster, self.timeout, self.criteria, self.error,
                                     self.delta, self.acc_datapath, self.resultpath, self.r_layer, self.r_neuron, best_pos, self.sens_threshold)

        # timing measurement
        print('Fault localization time(s): {}'.format(fault_loc_time))
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))

    #
    #   solve for fairness improvement
    #
    def solve_fairness_weight(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        # get sensitive feature details
        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))

        if 'class_n' in spec:
            self.class_n = spec['class_n']

        if 'repair_num' in spec:
            self.repair_num = spec['repair_num']

        sens_idx = self.sensitive[0]

        self.sens_idx = sens_idx

        val = self.model.lower[sens_idx]
        while val <= self.model.upper[sens_idx]:
            self.sens_value.append(val)
            val = val + 1.0

        # do causal anaylysis
        ie_ave_matrix = []
        if not self.plot:
            # analyze fairness
            dtmc = DTMCImpl()
            dtmc.analyze_fairness_causal(model, self.sensitive, len(self.sens_value), self.timeout, self.criteria, self.error,
                                         self.delta, self.acc_datapath, self.resultpath, [], [], [])

            print('Analyze causal attribution of hidden neurons to unfairness:')
            ie_ave = []
            col = 0
            for do_layer in self.do_layer:
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                #neuron_idx = 0
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_w_dy(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))
                    new_entry = []
                    new_entry.append(np.mean(np.array(ie)))
                    #new_entry.append(col + 1)
                    new_entry.append(do_layer)
                    new_entry.append(do_neuron)
                    ie_ave_matrix.append(new_entry)

                    #neuron_idx = neuron_idx + 1

                ie_ave.append(ie_ave_l)
                col = col + 1

            ie_ave_matrix.sort()
            ie_ave_matrix = ie_ave_matrix[::-1]
            for item in ie_ave_matrix:
                self.debug_print(item)

            self.r_neuron = []
            self.r_layer = []
            for i in range(0, self.repair_num):
                self.r_layer.append(int(ie_ave_matrix[i][1]))
                self.r_neuron.append(int(ie_ave_matrix[i][2]))

            print('Repair:')

            print('\nRepair layer: {}'.format(self.r_layer))
            print('Repair neuron: {}'.format(self.r_neuron))

            # start repair
            best_pos = self.repair()

            # verify prob diff and model accuracy after repair , weight, layer, neuron, sens_idx, sens_value, class_n
            r_id_rate, r_acc = self.test_repaired_net(best_pos, self.r_layer, self.r_neuron, self.sens_idx,
                                                      self.sens_value, self.class_n)
            print('Percentage of discriminatory instance after repair: {}'.format(r_id_rate))
            print('Network Accuracy after repair: {}'.format(r_acc))

            ori_accuracy = self.net_accuracy_test([], [], [])
            print('Network Accuracy before repair: {}'.format(ori_accuracy))

            # analyze fairness after repair
            dtmc.analyze_fairness_causal(model, self.sensitive, len(self.sens_value), self.timeout, self.criteria,
                                         self.error,
                                         self.delta, self.acc_datapath, self.resultpath, self.r_layer, self.r_neuron,
                                         best_pos)

        else:
            ie_ave = []
            plt_row = 0
            for i in range (0, len(self.do_layer)):
                if len(self.do_neuron[i]) > plt_row:
                    plt_row = len(self.do_neuron[i])
            plt_col = len(self.do_layer)
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(3.5*plt_col, 2.5*plt_row), sharex=False, sharey=True)
            fig.tight_layout()

            row = 0
            col = 0
            for do_layer in self.do_layer:
                row = 0
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_w_dy(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))

                    # plot ACE
                    ax[row, col].set_title('N_' + str(do_layer) + '_' + str(do_neuron))
                    ax[row, col].set_xlabel('Intervention Value(alpha)')
                    ax[row, col].set_ylabel('Causal Attributions(ACE)')

                    # Baseline is np.mean(expectation_do_x)
                    ax[row, col].plot(np.linspace(min, max, self.stepsize), np.array(ie) - np.mean(np.array(ie)), label = str(do_layer) + '_' + str(do_neuron), color='b')
                    ax[row, col].legend()

                    row = row + 1
                if row == len(self.do_neuron[col]):
                    for off in range(row, plt_row):
                        ax[off, col].set_axis_off()
                ie_ave.append(ie_ave_l)
                col = col + 1

            plt.show()
            plt.savefig(self.resultpath + '/' + 'all' + ".png")

            # plot average ie
            plt.figure()
            plt.ylabel('Average IE')

            color_tab = ['b','g','r','c','m']

            idx = 0
            for i in range (0, len(ie_ave)):
                for j in range (0, len(ie_ave[i])):
                    plt.scatter(idx, ie_ave[i][j], color=color_tab[i])
                    idx = idx + 1

            plt.show()

            # save avg ie
            for ie_l in ie_ave:
                print(ie_l)

        # timing measurement
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))

    def _solve_fairness(self, model, assertion, display=None):
        overall_starttime = time.time()
        self.model = model
        self.assertion = assertion
        self.display = display

        spec = assertion

        # get sensitive feature details
        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))

        if 'class_n' in spec:
            self.class_n = spec['class_n']

        if 'repair_num' in spec:
            self.repair_num = spec['repair_num']

        sens_idx = self.sensitive[0]

        self.sens_idx = sens_idx

        val = self.model.lower[sens_idx]
        while val <= self.model.upper[sens_idx]:
            self.sens_value.append(val)
            val = val + 1.0


        # analyze fairness
        dtmc = DTMCImpl()

        dtmc.analyze_fairness_causal(model, self.sensitive, len(self.sens_value), self.timeout, self.criteria,
                                     self.error, self.delta, self.acc_datapath, self.resultpath, [], [], [])

        # aequitas
        #array, avg = self.aequitas_test()
        #print(array)
        #print(avg)
        '''
        # do causal anaylysis
        ie_ave_matrix = []
        if not self.plot:
            ie_ave = []
            col = 0
            for do_layer in self.do_layer:
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                neuron_idx = 0
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_h_dy(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))
                    new_entry = []
                    new_entry.append(np.mean(np.array(ie)))
                    new_entry.append(col + 1)
                    new_entry.append(neuron_idx)
                    ie_ave_matrix.append(new_entry)

                    neuron_idx = neuron_idx + 1

                ie_ave.append(ie_ave_l)
                col = col + 1

        else:
            ie_ave = []
            plt_row = 0
            for i in range (0, len(self.do_layer)):
                if len(self.do_neuron[i]) > plt_row:
                    plt_row = len(self.do_neuron[i])
            plt_col = len(self.do_layer)
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(3.5*plt_col, 2.5*plt_row), sharex=False, sharey=True)
            fig.tight_layout()

            row = 0
            col = 0
            for do_layer in self.do_layer:
                row = 0
                print('Analyzing layer {}'.format(col))
                ie_ave_l = []
                for do_neuron in self.do_neuron[col]:
                    ie, min, max = self.get_ie_do_h_dy(do_layer, do_neuron, self.sens_idx, self.sens_value, self.stepsize, 0)

                    ie_ave_l.append(np.mean(np.array(ie)))

                    # plot ACE
                    #ax[row, col].set_title('N_' + str(do_layer) + '_' + str(do_neuron))
                    ax[row, col].set_xlabel('Intervention Value(alpha)')
                    ax[row, col].set_ylabel('Causal Attributions(ACE)')

                    # Baseline is np.mean(expectation_do_x)
                    ax[row, col].plot(np.linspace(min, max, self.stepsize), np.array(ie) - np.mean(np.array(ie)), label = str(do_layer) + '_' + str(do_neuron), color='b')
                    ax[row, col].legend()

                    row = row + 1
                if row == len(self.do_neuron[col]):
                    for off in range(row, plt_row):
                        ax[off, col].set_axis_off()
                ie_ave.append(ie_ave_l)
                col = col + 1

            plt.show()
            plt.savefig(self.resultpath + '/' + 'all' + ".png")

            # plot average ie
            plt.figure()
            plt.ylabel('Average IE')

            color_tab = ['b','g','r','c','m']

            idx = 0
            for i in range (0, len(ie_ave)):
                for j in range (0, len(ie_ave[i])):
                    plt.scatter(idx, ie_ave[i][j], color=color_tab[i])
                    idx = idx + 1

            plt.show()

            # save avg ie
            for ie_l in ie_ave:
                print(ie_l)

        ie_ave_matrix.sort()
        ie_ave_matrix = ie_ave_matrix[::-1]
        for item in ie_ave_matrix:
            print(item)

        self.r_neuron = []
        self.r_layer = []
        for i in range (0, self.repair_num):
            self.r_layer.append(int(ie_ave_matrix[i][1]))
            self.r_neuron.append(int(ie_ave_matrix[i][2]))

        print('\nRepair layer: {}'.format(self.r_layer))
        print('Repair neuron: {}'.format(self.r_neuron))

        # start repair
        best_pos = self.repair()


        # verify prob diff and model accuracy after repair , weight, layer, neuron, sens_idx, sens_value, class_n
        r_id_rate, r_acc = self.test_repaired_net(best_pos, self.r_layer, self.r_neuron, self.sens_idx, self.sens_value, self.class_n)
        print('Percentage of discriminatory instance after repair: {}'.format(r_id_rate))
        print('Network Accuracy after repair: {}'.format(r_acc))

        ori_accuracy = self.net_accuracy_test([], [], [])
        print('Network Accuracy before repair: {}'.format(ori_accuracy))

        '''
        self.r_layer = [3, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1]
        self.r_neuron = [11, 56, 31, 18, 7, 10, 27, 13, 12, 2, 5, 22, 1]
        best_pos = [ 0.41925645,  0.09721272,  0.13655903,  0.76384682, -0.18780243,  0.8893769,
  0.10207263,  0.7411251,   0.86598984,  0.92578482,  0.35157176, -0.57699074,
 -0.98120965]
        self.best_pos = best_pos
        # analyze fairness after repair
        dtmc.analyze_fairness_causal(model, self.sensitive, len(self.sens_value), self.timeout, self.criteria, self.error,
                                     self.delta, self.acc_datapath, self.resultpath, self.r_layer, self.r_neuron, best_pos)
        # aequitas
        #print('After repair')
        #array, avg = self.aequitas_test(rep=True)
        #print(array)
        #print(avg)
        # timing measurement
        print('Total execution time(s): {}'.format(time.time() - overall_starttime))

    #
    # get expected value of y with hidden neuron intervention
    #
    def get_y_do_h(self, do_layer, do_neuron, do_value, class_n):
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        #l_pass = 0
        #l_fail = 0

        y_sum = 0.0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y = self.model.apply_intervention(x0, do_layer, do_neuron, do_value)[0][class_n]

            #lbl_x0 = np.argmax(y, axis=1)[0]

            y_sum = y_sum + y

            # accuracy test
            #if lbl_x0 == y0s[i]:
            #    l_pass = l_pass + 1
            #else:
            #    l_fail = l_fail + 1
        #acc = l_pass / (l_pass + l_fail)

        avg = y_sum / self.datalen

        #self.debug_print("Accuracy of network: %f.\n" % (acc))

        return avg

    #
    # get expected value of y with hidden neuron intervention
    #
    def get_dy_do_h(self, do_layer, do_neuron, do_value, class_n, sens_idx, sens_range):
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        #l_pass = 0
        #l_fail = 0

        dy_sum = 0.0

        for i in range(self.datalen):
            # random index
            #i = int(np.random.rand() * self.datalen_tot)
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y = self.model.apply_intervention(x0, do_layer, do_neuron, do_value)
            #lbl_x0 = np.argmax(y, axis=1)[0]

            y = y[0][class_n]

            max_dy = 0.0
            x_n = x0
            for sens_val in self.sens_value:
                if sens_val == x0[sens_idx]:
                    continue
                x_n[sens_idx] = sens_val
                y_n = self.model.apply_intervention(x_n, do_layer, do_neuron, do_value)
                y_n = y_n[0][class_n]
                diff_n = np.abs(y_n - y)
                if max_dy < diff_n:
                    max_dy = diff_n

            dy_sum = dy_sum + max_dy

        avg = dy_sum / self.datalen

        return avg


    #
    # get expected value of dy with weight intervention
    #
    def get_dy_do_w(self, do_layer, do_neuron, do_value, class_n, sens_idx, sens_range):
        pathX = self.datapath + '/'
        #pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        #l_pass = 0
        #l_fail = 0

        dy_sum = 0.0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            #y = self.model.apply_intervention(x0, do_layer, do_neuron, do_value)
            y = self.model.apply_repair_fixed(x0, [do_neuron], [do_value], [do_layer])
            #y = self.model.apply(x0)
            #lbl_x0 = np.argmax(y, axis=1)[0]

            y = y[0][class_n]

            max_dy = 0.0
            x_n = x0
            for sens_val in self.sens_value:
                if sens_val == x0[sens_idx]:
                    continue
                x_n[sens_idx] = sens_val
                y_n = self.model.apply_repair_fixed(x_n, [do_neuron], [do_value], [do_layer])
                #y_n = self.model.apply(x_n)
                y_n = y_n[0][class_n]
                diff_n = np.abs(y_n - y)
                if max_dy < diff_n:
                    max_dy = diff_n

            dy_sum = dy_sum + max_dy

        avg = dy_sum / self.datalen

        return avg

    #
    # get expected value of y with hidden neuron intervention
    #
    def get_y_do_h_single(self, do_layer, do_neuron, do_value, sample):

        x0 = np.array(sample)
        y = self.model.apply_intervention(x0, do_layer, do_neuron, do_value)

        return y

    #
    # given number of steps, get expected ys for each step
    #
    def get_ie_do_h(self, do_layer, do_neuron, num_step=16, class_n=0):
        # get value range of given hidden neuron
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        hidden_max = 0.0
        hidden_min = 0.0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y, hidden = self.model.apply_get_h(x0, do_layer, do_neuron)

            if i == 0:
                hidden_max = hidden
                hidden_min = hidden
            else:
                if hidden > hidden_max:
                    hidden_max = hidden
                if hidden < hidden_min:
                    hidden_min = hidden

        # now we have hidden_min and hidden_max

        # compute interventional expectation for each step
        ie = []
        if hidden_max == hidden_min:
            ie = [hidden_min] * num_step
        else:
            for h_val in np.linspace(hidden_min, hidden_max, num_step):
                y = self.get_y_do_h(do_layer, do_neuron, h_val)
                ie.append(y[0][class_n])

        return ie, hidden_min, hidden_max

    #
    # given number of steps, get expected ys for each step for given sample
    #
    def get_ie_do_h_single(self, sample, do_layer, do_neuron, num_step=16, class_n=0):
        # get value range of given hidden neuron
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        hidden_max = 0.0
        hidden_min = 0.0

        for i in range(self.datalen):
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y, hidden = self.model.apply_get_h(x0, do_layer, do_neuron)

            if i == 0:
                hidden_max = hidden
                hidden_min = hidden
            else:
                if hidden > hidden_max:
                    hidden_max = hidden
                if hidden < hidden_min:
                    hidden_min = hidden

        # now we have hidden_min and hidden_max

        # compute interventional expectation for each step
        ie = []
        if hidden_max == hidden_min:
            ie = [hidden_min] * num_step
        else:
            for h_val in np.linspace(hidden_min, hidden_max, num_step):
                y = self.get_y_do_h_single(do_layer, do_neuron, h_val, sample)
                ie.append(y[0][class_n])

        return ie, hidden_min, hidden_max

    #
    # given number of steps, get expected ys for each step
    #
    def get_ie_do_h_dy(self, do_layer, do_neuron, sens_idx, sens_range, num_step=16, class_n=0):
        # get value range of given hidden neuron
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        hidden_max = 0.0
        hidden_min = 0.0

        for i in range(self.datalen):
            # random index
            #i = int(np.random.rand() * self.datalen_tot)

            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y, hidden = self.model.apply_get_h(x0, do_layer, do_neuron)

            if i == 0:
                hidden_max = hidden
                hidden_min = hidden
            else:
                if hidden > hidden_max:
                    hidden_max = hidden
                if hidden < hidden_min:
                    hidden_min = hidden

        # now we have hidden_min and hidden_max

        # compute interventional expectation for each step
        ie = []
        if hidden_max == hidden_min:
            ie = [hidden_min] * num_step
        else:
            for h_val in np.linspace(hidden_min, hidden_max, num_step):
                dy = self.get_dy_do_h(do_layer, do_neuron, h_val, class_n, sens_idx, sens_range)
                ie.append(dy)

        return ie, hidden_min, hidden_max

    #
    # given number of steps, get expected ys for each step
    #
    def get_ie_do_h_dy_gradient(self, do_layer, do_neuron, sens_idx, sens_range, num_step=16, class_n=0):
        # get value range of given hidden neuron
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        hidden_max = 0.0
        hidden_min = 0.0

        for i in range(self.datalen):
            # random index
            #i = int(np.random.rand() * self.datalen_tot)

            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y, hidden = self.model.apply_get_h(x0, do_layer, do_neuron)

            if i == 0:
                hidden_max = hidden
                hidden_min = hidden
            else:
                if hidden > hidden_max:
                    hidden_max = hidden
                if hidden < hidden_min:
                    hidden_min = hidden

        # now we have hidden_min and hidden_max

        # compute interventional expectation for each step
        ie = []
        die = []
        if hidden_max == hidden_min:
            ie = [hidden_min] * num_step
        else:
            for h_val in np.linspace(hidden_min, hidden_max, num_step):
                dy = self.get_y_do_h(do_layer, do_neuron, h_val, class_n)

                if (len(ie) != 0):
                    die.append(np.abs(dy - ie[-1]))
                ie.append(dy)

        return die, hidden_min, hidden_max

    #
    # given number of steps, get expected ys for each step; weight range (0,2w)
    # for each weight, randomly pick n number of samples to estimate the ie
    #
    def get_ie_do_w_dy(self, do_layer, do_neuron, sens_idx, sens_range, num_step=16, class_n=0):
        # get value range of given hidden neuron
        pathX = self.datapath + '/'
        #pathY = self.datapath + '/labels.txt'

        #y0s = np.array(ast.literal_eval(read(pathY)))

        # compute interventional expectation for each step
        ie = []
        for w_val in np.linspace(-1.0, 1.0, num_step):
            dy = self.get_dy_do_w(do_layer, do_neuron, w_val, class_n, sens_idx, sens_range)
            ie.append(dy)

        return ie, -1.0, 1.0

    def net_accuracy_test(self, r_neuron=0, r_weight=0, r_layer=0):
        pathX = self.acc_datapath + '/'
        pathY = self.acc_datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        l_pass = 0
        l_fail = 0

        for idx in range(self.acc_datalen):
            
            # random index
            i = int(np.random.rand() * self.acc_datalen_tot)
            
            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))
            if len(r_neuron) != 0:
                y = self.model.apply_repair_fixed(x0, r_neuron, r_weight, r_layer)
            else:
                y = self.model.apply(x0)

            lbl_x0 = np.argmax(y, axis=1)[0]
            #print('lbl:{}, predict:{}'.format(y0s[i], lbl_x0))

            # accuracy test
            if lbl_x0 == y0s[i]:
                l_pass = l_pass + 1
            else:
                l_fail = l_fail + 1
        acc = l_pass / (l_pass + l_fail)

        #self.debug_print("Accuracy of network: %f.\n" % (acc))

        return acc

    def net_accuracy_test_fix(self, r_neuron=0, r_weight=0, r_layer=0):
        pathX = self.acc_datapath + '/'
        pathY = self.acc_datapath + '/labels.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        l_pass = 0
        l_fail = 0

        for i in range(self.acc_datalen):

            # random index
            #i = int(np.random.rand() * self.acc_datalen_tot)

            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))
            if len(r_neuron) != 0:
                y = self.model.apply_repair_fixed(x0, r_neuron, r_weight, r_layer)
            else:
                y = self.model.apply(x0)

            lbl_x0 = np.argmax(y, axis=1)[0]
            # print('lbl:{}, predict:{}'.format(y0s[i], lbl_x0))

            # accuracy test
            if lbl_x0 == y0s[i]:
                l_pass = l_pass + 1
            else:
                l_fail = l_fail + 1
        acc = l_pass / (l_pass + l_fail)

        # self.debug_print("Accuracy of network: %f.\n" % (acc))

        return acc
    #
    #   test repair of discriminative instances
    #   @weight: array of fixed weight
    # def get_dy_do_h(self, do_layer, do_neuron, do_value, class_n, sens_idx, sens_range):

    def net_fairness_test(self, weight, layer, neuron, class_n, sens_idx, sens_range):
        pathX = self.datapath + '/'
        pathY = self.datapath + '/labels.txt'

        # y0s = np.array(ast.literal_eval(read(pathY)))

        # l_pass = 0
        # l_fail = 0

        y_same = 0
        y_diff = 0

        for idx in range(self.datalen):

            # random index
            i = int(np.random.rand() * self.datalen_tot)

            x0_file = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(x0_file)))

            y = self.model.apply_repair_fixed(x0, neuron, weight, layer)

            # lbl_x0 = np.argmax(y, axis=1)[0]

            #y = y[0][class_n]
            y = np.argmax(y, axis=1)[0]

            x_n = x0
            for sens_val in self.sens_value:
                if sens_val == x0[sens_idx]:
                    continue
                x_n[sens_idx] = sens_val
                y_n = self.model.apply_repair_fixed(x0, neuron, weight, layer)
                #y_n = y_n[0][class_n]
                y_n = np.argmax(y_n, axis=1)[0]
                if y_n == y:
                    y_same = y_same + 1
                else:
                    y_diff = y_diff + 1
                    #print(i)
                    break

        di_rate = y_diff / self.datalen

        return di_rate

    def test_repaired_net(self, weight, layer, neuron, sens_idx, sens_range, class_n):

        accuracy = self.net_accuracy_test(neuron, weight, layer)

        di_rate = self.net_fairness_test(weight, layer, neuron, class_n, sens_idx, sens_range)

        return di_rate, accuracy

    def test_repaired_net_fix(self, weight, layer, neuron, sens_idx, sens_range, class_n):

        accuracy = self.net_accuracy_test_fix(neuron, weight, layer)

        di_rate = self.net_fairness_test(weight, layer, neuron, class_n, sens_idx, sens_range)

        return di_rate, accuracy

    def pso_fitness_func(self, weight):

        result = []
        for i in range (0, int(len(weight))):
            r_weight =  weight[i]

            di_rate, accuracy = self.test_repaired_net(r_weight, self.r_layer, self.r_neuron, self.sens_idx, self.sens_value, self.class_n)

            _result = (1.0 - self.alpha) * di_rate + self.alpha * (1.0 - accuracy)

            self.debug_print('Repaired di_rate: {}, accuracy: {}'.format(di_rate, accuracy))

            result.append(_result)

        self.debug_print(result)

        return result

    def repair(self):
        # repair
        print('Start reparing...')
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        #'''# original
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.repair_num, options=options,
                                            bounds=([[-1.0] * self.repair_num, [1.0] * self.repair_num]),
                                            init_pos=np.zeros((20, self.repair_num), dtype=float), ftol=1e-3,
                                            ftol_iter=10)
        #'''

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

        # Obtain the cost history
        # print(optimizer.cost_history)
        # Obtain the position history
        # print(optimizer.pos_history)
        # Obtain the velocity history
        # print(optimizer.velocity_history)
        #print('neuron to repair: {} at layter: {}'.format(self.r_neuron, self.r_layer))
        #print('best cost: {}'.format(best_cost))
        #print('best pos: {}'.format(best_pos))
        return best_pos

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

