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


from solver.verify import *

INFO = 3
class VeriFairimpl():
    def __init__(self):
        self.model = None
        self.assertion = None
        self.display = None
        self.sensitive = []
        self.group0 = None
        self.group1 = None
        self.step = 1
        self.mu = []
        self.std = []

    def gaussian(self, mu, sigma):
        return np.random.normal(mu, np.sqrt(sigma))

    def veri_generate_x(self, shape, lower, upper, sens_dix, sens_group):
        size = np.prod(shape)
        #x = np.random.rand(size)
        x = []
        for i in range (0, size):
            x.append(self.gaussian(self.mu[i], self.std[i]))

        tot_sens_cat = upper[sens_dix] - lower[sens_dix] + 1
        threshold = lower[sens_dix] + tot_sens_cat / 2

        x = (upper - lower) * x + lower
        out = np.array([0.0] * x.size)
        i = 0

        for x_i in x:
            x_i = round(x_i)
            out[i] = x_i
            if i == sens_dix:
                if sens_group != 0:
                    if out[i] > int(threshold):
                        return []
                else:
                    if out[i] < int(threshold):
                        return []
            i = i + 1
        '''
        for x_i in x:
            x_i = round(x_i)
            out[i] = x_i
            if i == sens_dix:
                if out[i] != sens_group:
                    return []
            i = i + 1
        '''
        return out


    def solve(self, model, assertion, display=None):
        self.model = model
        spec = assertion
        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))

        if 'sens_group0' in spec:
            self.group0 = spec['sens_group0']

        if 'sens_group1' in spec:
            self.group1 = spec['sens_group1']

        if 'mu' in spec:
            self.mu = spec['mu']

        if 'std' in spec:
            self.std = spec['std']

        self.run_single()

        return

    def get_new_sample_verifair(self, sens_dix, sens_group):
        lower = self.model.lower
        upper = self.model.upper

        generated = self.step

        out = 0
        while generated:
            total = 0
            x = self.veri_generate_x(self.model.shape, lower, upper, sens_dix, sens_group)
            total = total + 1
            while len(x) == 0:
                x = self.veri_generate_x(self.model.shape, lower, upper, sens_dix, sens_group)
                total = total + 1

            y = self.model.apply(x)

            #y = 1 - np.argmax(y, axis=1)[0]
            y = np.argmax(y, axis=1)[0]

            out = out + y

            generated = generated - 1

        return out, total

    def run_single(self):
        # Step 0: Verification parameters
        c = 0.15
        Delta = 0.0
        delta = 0.5 * 1e-10
        n_samples = 1
        n_max = 10000000
        is_causal = False
        log_iters = None

        self.step = n_samples

        # Step 1: Samplers
        #sample_fn = get_model(this_model, dist)

        #model0 = MultiSampler(RejectionSampler(sample_fn, False))
        #model1 = MultiSampler(RejectionSampler(sample_fn, True))

        # Step 2: Verification
        runtime = time.time()
        result, n_total_samples = self.verify(c, Delta, delta, n_samples, n_max, is_causal, log_iters)
        if result is None:
            self.log('Failed to converge!', INFO)
            return

        # Step 3: Post processing
        is_fair, is_ambiguous, n_successful_samples, E = result
        runtime = time.time() - runtime

        self.log('Pr[fair = {}] >= 1.0 - {}'.format(is_fair, 2.0 * delta), INFO)
        self.log('E[ratio] = {}'.format(E), INFO)
        self.log('Is fair: {}'.format(is_fair), INFO)
        self.log('Is ambiguous: {}'.format(is_ambiguous), INFO)
        self.log('Successful samples: {} successful samples, Attempted samples: {}'.format(n_successful_samples,
                                                                                      n_total_samples), INFO)
        self.log('Running time: {} seconds'.format(runtime), INFO)

    def verify(self, c, Delta, delta, n_samples, n_max, is_causal, log_iters):
        # Step 1: Initialization
        nE_A = 0.0
        nE_B = 0.0
        n_samples_0 = 0
        n_samples_1 = 0
        # Step 2: Iteratively sample and check whether fairness holds
        for i in range(n_max):
            # Step 2a: Sample points
            # find positive prediction for each sensitive group

            x, x_tot = self.get_new_sample_verifair(self.sensitive[0], self.group0)
            y, y_tot = self.get_new_sample_verifair(self.sensitive[0], self.group1)

            # Step 2b: Update statistics
            nE_A += x
            nE_B += y

            n_samples_0 = n_samples_0 + x_tot
            n_samples_1 = n_samples_1 + y_tot

            # Step 2c: Normalized statistics
            n = (i + 1) * n_samples
            E_A = nE_A / n
            E_B = nE_B / n

            # logging
            is_log = not log_iters is None and i % log_iters == 0

            # Step 2d: Get type judgement
            t = self.get_fairness_type(c, Delta, n, delta, E_A, E_B, is_causal, is_log)

            # Step 2e: Return if converged
            if not t is None:
                return t + (2 * n, E_A / E_B), (n_samples_0 + n_samples_1)

        # Step 3: Failed to verify after maximum number of samples
        return None

    # Returns the (epsilon, delta) values of the adaptive concentration inequality
    # for the given parameters.
    #
    # n: int (number of samples)
    # delta: float (parameter delta)
    # return: epsilon (parameter epsilon)
    def get_type(self, n, delta):
        n = np.float(n)
        b = -np.log(delta / 24.0) / 1.8
        epsilon = np.sqrt((0.6 * np.log(np.log(n) / np.log(1.1) + 1) + b) / n)
        return epsilon

    # Run type inference to get the type judgement for the fairness criterion.
    #
    # c: float (minimum probability ratio to be fair)
    # Delta: float (threshold on inequalities)
    # n: int (number of samples)
    # delta: float (parameter delta)
    # E_A: float (estimate of p(offer | female))
    # E_B: float (estimate of p(offer | male))
    # is_causal: bool (whether to use the causal specification)
    # is_log: bool (whether to print logging information)
    # return: int * int | None (None - continue sampling, or (fairness, ambiguous) with 0 - not fair, 1 - fair, 0 - not ambiguous, 1 - ambiguous)
    def get_fairness_type(self, c, Delta, n, delta, E_A, E_B, is_causal, is_log):
        # Step 1: Get (epsilon, delta) values from the adaptive concentration inequality for the current number of samples
        epsilon = self.get_type(n, delta)

        # Step 2: Check if |E_B| > epsilon_B
        if np.abs(E_B) <= epsilon:
            return None

        # logging
        if is_log:
            # log('INFO: n = {}, E_A = {}, E_B = {}'.format(n, E_A, E_B), INFO)
            print('INFO: n = {}, E_A = {}, E_B = {}'.format(n, E_A, E_B))

        # Step 3: Compute the type judgement for the fairness property (before the inequality)
        if is_causal:
            E_fair = E_A - E_B + c
            epsilon_fair = 2.0 * epsilon
            delta_fair = 2.0 * delta
        else:
            E_fair = E_A / E_B - (1 - c)
            epsilon_fair = epsilon / np.abs(E_B) + epsilon * (epsilon + np.abs(E_A)) / (
                        np.abs(E_B) * (np.abs(E_B) - epsilon))
            delta_fair = 2.0 * delta

        # logging
        if is_log:
            # log('INFO: n = {}, E_fair = {}, epsilon_fair = {}, delta_fair = {}'.format(n, E_fair, epsilon_fair, delta_fair), INFO)
            print('INFO: n = {}, E_fair = {}, epsilon_fair = {}, delta_fair = {}'.format(n, E_fair, epsilon_fair,
                                                                                         delta_fair))

        # Step 4: Check if fairness holds
        if E_fair - epsilon_fair > 0:
            return 1, 0

        # Step 5: Check if fairness does not hold
        if E_fair + epsilon_fair < 0:
            return 0, 0

        # Step 6: Check if fairness holds (ambiguously)
        if E_fair - epsilon_fair >= -Delta and epsilon_fair <= Delta:
            return 1, 1

        # Step 7: Check if fairness does not hold (ambiguously)
        if E_fair + epsilon_fair <= Delta and epsilon_fair <= Delta:
            return 0, 1

        # Step 6: Continue sampling
        return None

    def log(self, msg, flag):
        print(msg)

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