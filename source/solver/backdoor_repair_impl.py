import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast
import os
import time
import random
import math

import gurobipy as gp
from gurobipy import GRB

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *
from poly_utils import *
from solver.refinement_impl import Poly

import matplotlib.pyplot as plt


class BackDoorRepairImpl():
    def __solve_backdoor_repair(self, model, spec, display):
        target = spec['target']

        rate = spec['rate']
        threshold = spec['threshold']
        
        total_imgs = spec['total_imgs']
        total_imgs = 1000
        num_imgs = spec['num_imgs']
        dataset = spec['dataset']

        repair_num = 10

        valid_x0s = []
        y0s = np.array(ast.literal_eval(read(spec['pathY'])))

        for i in range(total_imgs):
            pathX = spec['pathX'] + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(pathX)))

            output_x0 = model.apply(x0).reshape(-1)
            y0 = np.argmax(output_x0)

            if i < 10:
                print('\n==============\n')
                print('Data {}\n'.format(i))
                print('x0 = {}'.format(x0))
                print('output_x0 = {}'.format(output_x0))
                print('y0 = {}'.format(y0))
                print('y0s[i] = {}\n'.format(y0s[i]))

            if y0 == y0s[i] and y0 != target:
                valid_x0s.append((x0, output_x0))

        if len(valid_x0s) == 0:
            print('No data to analyze target = {}'.format(target))
            return None, None

        print('Number of valid_x0s = {} for target = {}'.format(len(valid_x0s), target))

        print('Lower bound = {} and Upper bound = {}'.format(model.lower[0], model.upper[0]))

        print('\nGenerate reversed trigger with target = {}'.format(target))

        stamp = self.__attack(model, valid_x0s, target, dataset)

        if dataset == 'mnist':
            trigger = stamp[:(1 * 28 * 28)]
            mask1 = stamp[(1 * 28 * 28):]
        elif dataset == 'cifar':
            trigger = stamp[:(3 * 32 * 32)]
            mask1 = stamp[(3 * 32 * 32):]
        mask2 = np.round(mask1)

        # print('trigger = {}'.format(list(trigger)))

        # print('mask1 = {}'.format(list(mask1)))
        # print('sum mask1 = {}'.format(np.sum(mask1)))

        # print('mask2 = {}'.format(list(mask2)))
        # print('sum mask2 = {}'.format(np.sum(mask2)))

        valid_x0s_with_bd1 = self.__get_x0s_with_bd(model, valid_x0s, trigger, mask1, target)
        valid_x0s_with_bd2 = self.__get_x0s_with_bd(model, valid_x0s, trigger, mask2, target)

        # print('len(valid_x0s) =', len(valid_x0s))
        # print('len(valid_x0s_with_bd1) =', len(valid_x0s_with_bd1))
        # print('len(valid_x0s_with_bd2) =', len(valid_x0s_with_bd2))

        mask = mask1
        mask = mask2

        valid_x0s_with_bd = valid_x0s_with_bd1
        valid_x0s_with_bd = valid_x0s_with_bd2

        if len(valid_x0s_with_bd) / len(valid_x0s) < rate:
            # print('\nrate = {}'.format(len(valid_x0s_with_bd) / len(valid_x0s)))
            print('The stamp does not satisfy the success rate = {} with target = {}'.format(rate, target))
            assert False
        else:
            print('The stamp satisfies the success rate = {} with target = {}'.format(rate, target))
            cleansed_model = self.clean_backdoor(model, valid_x0s_with_bd, trigger, mask, target, repair_num)

        return None, None


    def clean_backdoor(self, model, valid_x0s, trigger, mask, target, repair_num):
        print('\nBegin cleansing')
        assert self.__validate(model, valid_x0s, trigger, mask, target, 1.0)

#####################################################################################################################################
        number_of_layers = len(model.layers)

        ie_ave_matrix = []

        for do_layer in range(number_of_layers):

            if model.layers[do_layer].is_linear_layer():
                number_of_neurons = model.layers[do_layer].get_number_neurons()

                for do_neuron in range(number_of_neurons):
                    ie, min_val, max_val = self.get_ie_do_h_dy(model, valid_x0s, trigger, mask, target, do_layer, do_neuron)

                    new_entry = []
                    new_entry.append(np.mean(np.array(ie)))
                    new_entry.append(do_layer)
                    new_entry.append(do_neuron)
                    
                    ie_ave_matrix.append(new_entry)

        ie_ave_matrix.sort(reverse=True)

        repair_layers, repair_neurons = [], []
        for i in range (0, repair_num):
            repair_layers.append(int(ie_ave_matrix[i][1]))
            repair_neurons.append(int(ie_ave_matrix[i][2]))
        
        print('\nRepair layers: {}'.format(repair_layers))
        print('Repair neurons: {}'.format(repair_neurons))


    def get_ie_do_h_dy(self, model, valid_x0s, trigger, mask, target, do_layer, do_neuron):
        # get value range of given hidden neuron

        hidden_max, hidden_min = None, None

        for x0, output_x0 in valid_x0s:
            _, hidden = self.model.apply_get_h(x0, do_layer, do_neuron)

            if hidden_max is None:
                hidden_max = hidden
                hidden_min = hidden
            else:
                if hidden > hidden_max:
                    hidden_max = hidden
                if hidden < hidden_min:
                    hidden_min = hidden

        # now we have hidden_min and hidden_max

        # compute interventional expectation for each step
        ie, num_step = [], 16
        if hidden_max == hidden_min:
            ie = [hidden_min] * num_step
        else:
            for h_val in np.linspace(hidden_min, hidden_max, num_step):
                dy = self.get_dy_do_h(model, valid_x0s, trigger, mask, target, do_layer, do_neuron, h_val)
                ie.append(dy)

        return ie, hidden_min, hidden_max

    #
    # get expected value of y with hidden neuron intervention
    #
    def get_dy_do_h(self, model, valid_x0s, trigger, mask, target, do_layer_idx, do_neuron_idx, do_value):
        dy_sum = 0.0

        for x0, output_x0 in valid_x0s:
            y = self.model.apply_intervention(x0, do_layer, do_neuron, do_value)
            y = y[0, target]

            dy = abs(output_x0[target] - y)

            dy_sum = dy_sum + max_dy

        avg = dy_sum / len(valid_x0s)

        return avg
#####################################################################################################################################

    def __filter_x0s_with_bd(self, model, valid_x0s, trigger, mask, target):
        removed_x0s = []
        for i in range(len(valid_x0s)):
            x0, output_x0 = valid_x0s[i]

            x_bd = (1 - mask) * x0 + mask * trigger

            output_x_bd = model.apply(x_bd).reshape(-1)
            target_x_bd = np.argmax(output_x_bd)

            if target_x_bd != target:
                removed_x0s.insert(0, i)

        for i in removed_x0s:
            valid_x0s.pop(i)


    # def __write_constr_input_layer(self, prob, cnt_imgs, coefs, const, op, backdoor_indexes, prev_var_idx, curr_var_idx):
    #     prob.write('  x{}_{}'.format(curr_var_idx, cnt_imgs))

    #     for i in range(len(coefs)):
    #         coef = coefs[i]
    #         var_idx = prev_var_idx + i

    #         if coef > 0:
    #             if var_idx in backdoor_indexes:
    #                 prob.write(' + {} x{}'.format(coef, var_idx))
    #             else:
    #                 prob.write(' + {} x{}_{}'.format(coef, var_idx, cnt_imgs))
    #         elif coef < 0:
    #             if var_idx in backdoor_indexes:
    #                 prob.write(' - {} x{}'.format(abs(coef), var_idx))
    #             else:
    #                 prob.write(' - {} x{}_{}'.format(abs(coef), var_idx, cnt_imgs))

    #     prob.write(' {} {}\n'.format(op, const))
    #     prob.flush()


    # def __write_constr_hidden_layers(self, prob, cnt_imgs, coefs, const, op, prev_var_idx, curr_var_idx):
    #     prob.write('  x{}_{}'.format(curr_var_idx, cnt_imgs))

    #     for i in range(len(coefs)):
    #         coef = coefs[i]
    #         var_idx = prev_var_idx + i

    #         if coef > 0:
    #             prob.write(' + {} x{}_{}'.format(coef, var_idx, cnt_imgs))
    #         elif coef < 0:
    #             prob.write(' - {} x{}_{}'.format(abs(coef), var_idx, cnt_imgs))

    #     prob.write(' {} {}\n'.format(op, const))
    #     prob.flush()


    # def __write_constr_output_layer(self, prob, cnt_imgs, target, prev_var_idx):
    #     for i in range(10):
    #         if i != target:
    #             prob.write('  x{}_{} - x{}_{} > 0.0\n'.format(prev_var_idx + target, cnt_imgs, prev_var_idx + i, cnt_imgs))

    #     prob.flush()


    # def __write_constr(self, prob, lst_poly_coll, backdoor_indexes, target):
    #     size = len(lst_poly_coll[0][0].lw)
    #     cnt_imgs = 0

    #     for lst_poly in lst_poly_coll:
    #         first_layer = True
    #         prev_var_idx = 0
    #         curr_var_idx = size

    #         for poly in lst_poly[1:]:
    #             if first_layer:
    #                 for i in range(len(poly.lw)):
    #                     coefs = -poly.ge[i][:-1]
    #                     const = poly.ge[i][-1]

    #                     self.__write_constr_input_layer(prob, cnt_imgs, coefs, const, '=', backdoor_indexes, prev_var_idx, curr_var_idx + i)
    #                 first_layer = False
    #             else:
    #                 for i in range(len(poly.lw)):
    #                     ge, le = poly.ge[i], poly.le[i]

    #                     if np.all(ge == le):
    #                         coefs = -poly.ge[i][:-1]
    #                         const = poly.ge[i][-1]

    #                         self.__write_constr_hidden_layers(prob, cnt_imgs, coefs, const, '=', prev_var_idx, curr_var_idx + i)
    #                     else:
    #                         coefs_ge = -poly.ge[i][:-1]
    #                         const_ge = poly.ge[i][-1]

    #                         self.__write_constr_hidden_layers(prob, cnt_imgs, coefs_ge, const_ge, '>=', prev_var_idx, curr_var_idx + i)

    #                         coefs_le = -poly.le[i][:-1]
    #                         const_le = poly.le[i][-1]

    #                         self.__write_constr_hidden_layers(prob, cnt_imgs, coefs_le, const_le, '<=', prev_var_idx, curr_var_idx + i)

    #             prev_var_idx = curr_var_idx
    #             curr_var_idx += len(poly.lw)

    #         self.__write_constr_output_layer(prob, cnt_imgs, target, prev_var_idx)

    #         cnt_imgs += 1


    # def __write_bounds(self, prob, lst_poly_coll, backdoor_indexes):
    #     lw0, up0 = lst_poly_coll[0][0].lw, lst_poly_coll[0][0].up

    #     for var_idx in backdoor_indexes:
    #         prob.write('  {} <= x{} <= {}\n'.format(lw0[var_idx], var_idx, up0[var_idx]))

    #     cnt_imgs = 0

    #     for lst_poly in lst_poly_coll:
    #         var_idx = 0
    #         for poly in lst_poly:
    #             for i in range(len(poly.lw)):
    #                 lw_i = poly.lw[i]
    #                 up_i = poly.up[i]

    #                 if var_idx not in backdoor_indexes:
    #                     if lw_i == up_i:
    #                         prob.write('  x{}_{} = {}\n'.format(var_idx, cnt_imgs, lw_i))
    #                     else:
    #                         prob.write('  {} <= x{}_{} <= {}\n'.format(lw_i, var_idx, cnt_imgs, up_i))

    #                 var_idx += 1
    #         cnt_imgs += 1

    #     prob.flush()


    # def __write_problem(self, lst_poly_coll, backdoor_indexes, target):
    #     filename = 'prob' + str(target) + '.lp'
    #     prob = open(filename, 'w')

    #     prob.write('Minimize\n')
    #     prob.write('  0\n')

    #     prob.write('Subject To\n')

    #     self.__write_constr(prob, lst_poly_coll, backdoor_indexes, target)

    #     prob.write('Bounds\n')

    #     self.__write_bounds(prob, lst_poly_coll, backdoor_indexes)

    #     prob.write('End\n')

    #     prob.flush()
    #     prob.close()


    # def __verifyI(self, model, valid_x0s, valid_bdi, target):
    #     has_unknown = False

    #     for backdoor_indexes in valid_bdi:
    #         has_safe, lst_poly_coll = False, []

    #         for x0, output_x0 in valid_x0s:
    #             lw, up = x0.copy(), x0.copy()

    #             lw[backdoor_indexes] = model.lower[backdoor_indexes]
    #             up[backdoor_indexes] = model.upper[backdoor_indexes]

    #             x0_poly = Poly()
    #             x0_poly.lw, x0_poly.up = lw, up
    #             # just let x0_poly.le and x0_poly.ge is None
    #             x0_poly.shape = model.shape

    #             lst_poly = [x0_poly]
    #             self.__run(model, 0, lst_poly)

    #             output_lw, output_up = lst_poly[-1].lw.copy(), lst_poly[-1].up.copy()
    #             output_lw[target] = output_up[target]

    #             if np.argmax(output_lw) != target:
    #                 has_safe = True
    #                 break
    #             else:
    #                 self.__write_problem([lst_poly], backdoor_indexes, target)

    #                 filename = 'prob' + str(target) + '.lp'
    #                 opt = gp.read(filename)
    #                 opt.setParam(GRB.Param.DualReductions, 0)

    #                 opt.optimize()
    #                 os.remove(filename)

    #                 if opt.status == GRB.INFEASIBLE:
    #                     # print('Infeasible 1 image with target = {}'.format(target))
    #                     has_safe = True
    #                     break

    #             lst_poly_coll.append(lst_poly)

    #         if not has_safe: # unsafe, try solver
    #             self.__write_problem(lst_poly_coll, backdoor_indexes, target)

    #             filename = 'prob' + str(target) + '.lp'
    #             opt = gp.read(filename)
    #             opt.setParam(GRB.Param.DualReductions, 0)

    #             opt.optimize()
    #             os.remove(filename)

    #             if opt.status == GRB.INFEASIBLE:
    #                 # print('Infeasible all images with target = {}'.format(target))
    #                 pass
    #             elif opt.status == GRB.OPTIMAL:
    #                 stamp = self.__get_stamp(opt, backdoor_indexes)

    #                 # print('Solve target = {} with stamp = {} and position = {}'.format(target, stamp, backdoor_indexes))

    #                 if not self.__validate(model, valid_x0s, backdoor_indexes, target, stamp, 1.0):
    #                     # print('The stamp for target = {} is not validate with chosen images I'.format(target))
    #                     stamp = self.__attack(model, valid_x0s, backdoor_indexes, target)

    #                 if stamp is not None:
    #                     return False, (stamp, backdoor_indexes)
    #                 else:
    #                     has_unknown = True
    #             else:
    #                 stamp = self.__attack(model, valid_x0s, backdoor_indexes, target)

    #                 if stamp is not None:
    #                     return False, (stamp, backdoor_indexes)
    #                 else:
    #                     has_unknown = True

    #     if has_unknown:
    #         return False, None
    #     else:
    #         return True, None


    # def __hypothesis_test(self, model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold):
    #     rate_k = pow(rate, num_imgs) # attack num_imgs successfully at the same time

    #     p0 = (1 - rate_k) + threshold # not having the attack
    #     p1 = (1 - rate_k) - threshold

    #     # print('p0 = {}, p1 = {}'.format(p0, p1))

    #     alpha, beta = 0.01, 0.01

    #     h0 = beta / (1 - alpha) # 0.01
    #     h1 = (1 - beta) / alpha # 99.0

    #     pr, no = 1, 0
        
    #     while True:
    #         no = no + 1

    #         if num_imgs > len(valid_x0s):
    #             assert False
    #         else:
    #             chosen_idx = np.random.choice(len(valid_x0s), num_imgs, replace=False)
    #             chosen_x0s = []
    #             for i in chosen_idx:
    #                 chosen_x0s.append(valid_x0s[i])

    #         res, sbi = self.__verifyI(model, chosen_x0s, valid_bdi, target)

    #         if res: # no backdoor
    #             # print('VerifyI with target = {}'.format(target, rate))
    #             pr = pr * p1 / p0 # decrease, favorite H0
    #         elif sbi is not None: # backdoor with stamp
    #             stamp, backdoor_indexes = sbi[0], sbi[1]
    #             if self.__validate(model, valid_x0s, backdoor_indexes, target, stamp, rate): # real stamp
    #                 return False, sbi
    #             else:
    #                 # print('The stamp for target = {} is not validate with all images with rate = {}'.format(target, rate))
    #                 pr = pr * (1 - p1) / (1 - p0) # increase, favorite H1
    #         else: # unknown
    #             # print('Unknown with target = {}'.format(target, rate))
    #             pr = pr * (1 - p1) / (1 - p0) # increase, favorite H1

    #         if pr <= h0:
    #             print('Accept H0 after {} rounds. The probability of not having an attack with target = {} >= {} for K = {}.'.format(no, target, p0, num_imgs))
    #             return True, None
    #         elif pr >= h1:
    #             print('Accept H1 after {} rounds. The probability of not having an attack with target = {} <= {} for K = {}.'.format(no, target, p1, num_imgs))
    #             return False, None


    # def __verify(self, model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold):
    #     if rate == 1.0: # no hypothesis test when rate = 1.0, instead try to verify all images
    #         print('Run verifyI with target = {}'.format(target))
    #         res, sbi = self.__verifyI(model, valid_x0s, valid_bdi, target)
    #     else:
    #         print('Run hypothesis test with target = {}'.format(target))
    #         res, sbi = self.__hypothesis_test(model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold)

    #     if res:
    #         return None, None
    #     elif sbi is not None:
    #         stamp, backdoor_indexes = sbi[0], sbi[1]
    #         print('Real stamp = {} for target = {} at position = {}'.format(stamp, target, backdoor_indexes))
    #         return target, sbi
    #     else:
    #         return target, None


    # def __get_stamp(self, opt, backdoor_indexes):
    #     stamp = []

    #     # opt.write('model.sol')
    #     for idx in backdoor_indexes:
    #         var = opt.getVarByName('x' + str(idx))
    #         stamp.append(var.x)

    #     return np.array(stamp)


    def __validate(self, model, valid_x0s, trigger, mask, target, rate):
        cnt = 0

        for x0, output_x0 in valid_x0s:
            xi = (1 - mask) * x0 + mask * trigger

            output = model.apply(xi).reshape(-1)

            if np.argmax(output) == target: # attack successfully
                cnt += 1

        return (cnt / len(valid_x0s)) >= rate


    def __attack(self, model, valid_x0s, target, dataset):
        def obj_func(x, model, valid_x0s, target, length, half_len):
            res, lam = 0, 1

            for x0, output_x0 in valid_x0s:
                trigger = x[:half_len] # trigger
                mask = x[half_len:] # mask
                
                xi = (1 - mask) * x0 + mask * trigger

                output = model.apply(xi).reshape(-1)
                target_score = output[target]

                output_no_target = output - np.eye(len(output))[target] * 1e9
                max_score = np.max(output_no_target)

                if target_score > max_score:
                    res += 0
                else:
                    res += max_score - target_score + 1e-9

            res += lam * np.sum(mask)

            return res

        if dataset == 'mnist':
            length = 2 * 28 * 28
        elif dataset == 'cifar':
            length = 6 * 32 * 32
        
        half_len = length // 2

        lw = np.zeros(length) # mask and trigger
        up = np.full(length, 1.0) # mask and trigger

        lw[:half_len] = model.lower # lower for trigger
        up[:half_len] = model.upper # upper for trigger

        x = np.zeros(length)

        args = (model, valid_x0s, target, length, half_len)
        jac = grad(obj_func)
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)
        # print('res.fun = {}'.format(res.fun))

        return res.x


    # def __get_backdoor_indexes(self, size, position, dataset):
    #     if position < 0:
    #         return None

    #     if dataset == 'mnist':
    #         num_chans, num_rows, num_cols = 1, 28, 28
    #     elif dataset == 'cifar':
    #         num_chans, num_rows, num_cols = 3, 32, 32

    #     row_idx = int(position / num_cols)
    #     col_idx = position - row_idx * num_cols

    #     if row_idx + size > num_rows or col_idx + size > num_cols:
    #         return None

    #     indexes = []

    #     for i in range(num_chans):
    #         tmp = position + i * num_rows * num_cols
    #         for j in range(size):
    #             for k in range(size):
    #                 indexes.append(tmp + k)
    #             tmp += num_cols

    #     return indexes


    def __run(self, model, idx, lst_poly):
        if idx == len(model.layers):
            return None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        return self.__solve_backdoor_repair(model, assertion, display)
