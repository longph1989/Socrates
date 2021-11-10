import autograd.numpy as np
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


class BackDoorImpl():
    def __solve_backdoor(self, model, spec, display):
        target = spec['target']
        size = spec['size']

        rate = spec['rate']
        threshold = spec['threshold']
        
        alpha = spec['alpha']
        beta = spec['beta']

        atk_only = spec['atk_only']

        total_imgs = spec['total_imgs']
        num_imgs = spec['num_imgs']
        dataset = spec['dataset']

        valid_x0s = []
        y0s = np.array(ast.literal_eval(read(spec['pathY'])))

        for i in range(total_imgs):
            pathX = spec['pathX'] + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(pathX)))

            output_x0 = model.apply(x0).reshape(-1)
            y0 = np.argmax(output_x0)

            if (atk_only or target == 0) and i < 10:
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

        if atk_only or target == 0:
            print('Lower bound = {} and Upper bound = {}'.format(model.lower[0], model.upper[0]))

        if atk_only:
            print('Run attack with target = {}'.format(target))

            position = spec['atk_pos']
            backdoor_indexes = self.__get_backdoor_indexes(size, position, dataset)

            valid_atk_x0s = self.__filter_valid_atk_x0s(model, valid_x0s, backdoor_indexes, target)

            if len(valid_atk_x0s) / len(valid_x0s) >= 0.8:
                stamp = self.__attack(model, valid_atk_x0s, backdoor_indexes, target)

                if stamp is not None:
                    print('Real stamp = {} for target = {} at position = {}'.format(stamp, target, backdoor_indexes))
                    return target, (stamp, backdoor_indexes)
                else:
                    print('No stamp for target = {}'.format(target))
                    return target, None
            else:
                print('Only {} remaining data, not enough for the attack with target = {}'.format(len(valid_atk_x0s), target))
                return None, None

        valid_bdi = []

        if dataset == 'mnist': positions = 784
        elif dataset == 'cifar': positions = 1024

        for position in range(positions):
            backdoor_indexes = self.__get_backdoor_indexes(size, position, dataset)

            if not(backdoor_indexes is None):
                valid_bdi.append(backdoor_indexes)

        if target == 0:
            print('Number of valid positions = {}'.format(len(valid_bdi)))

        return self.__verify(model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold, alpha, beta)
        

    def __filter_valid_atk_x0s(self, model, valid_x0s, backdoor_indexes, target):
        valid_atk_x0s = []

        for i in range(len(valid_x0s)):
            x0, output_x0 = valid_x0s[i]

            x_bd = x0.copy()
            x_bd[backdoor_indexes] = 1.0

            output_x_bd = model.apply(x_bd).reshape(-1)
            target_x_bd = np.argmax(output_x_bd)

            if target_x_bd == target:
                valid_atk_x0s.append((x0, output_x0))

        return valid_atk_x0s


    def __write_constr_input_layer(self, prob, cnt_imgs, coefs, const, op, backdoor_indexes, prev_var_idx, curr_var_idx):
        prob.write('  x{}_{}'.format(curr_var_idx, cnt_imgs))

        for i in range(len(coefs)):
            coef = coefs[i]
            var_idx = prev_var_idx + i

            if coef > 0:
                if var_idx in backdoor_indexes:
                    prob.write(' + {} x{}'.format(coef, var_idx))
                else:
                    prob.write(' + {} x{}_{}'.format(coef, var_idx, cnt_imgs))
            elif coef < 0:
                if var_idx in backdoor_indexes:
                    prob.write(' - {} x{}'.format(abs(coef), var_idx))
                else:
                    prob.write(' - {} x{}_{}'.format(abs(coef), var_idx, cnt_imgs))

        prob.write(' {} {}\n'.format(op, const))
        prob.flush()


    def __write_constr_hidden_layers(self, prob, cnt_imgs, coefs, const, op, prev_var_idx, curr_var_idx):
        prob.write('  x{}_{}'.format(curr_var_idx, cnt_imgs))

        for i in range(len(coefs)):
            coef = coefs[i]
            var_idx = prev_var_idx + i

            if coef > 0:
                prob.write(' + {} x{}_{}'.format(coef, var_idx, cnt_imgs))
            elif coef < 0:
                prob.write(' - {} x{}_{}'.format(abs(coef), var_idx, cnt_imgs))

        prob.write(' {} {}\n'.format(op, const))
        prob.flush()


    def __write_constr_output_layer(self, prob, cnt_imgs, target, prev_var_idx):
        for i in range(10):
            if i != target:
                prob.write('  x{}_{} - x{}_{} > 0.0\n'.format(prev_var_idx + target, cnt_imgs, prev_var_idx + i, cnt_imgs))

        prob.flush()


    def __write_constr(self, prob, lst_poly_coll, backdoor_indexes, target):
        size = len(lst_poly_coll[0][0].lw)
        cnt_imgs = 0

        for lst_poly in lst_poly_coll:
            first_layer = True
            prev_var_idx = 0
            curr_var_idx = size

            for poly in lst_poly[1:]:
                if first_layer:
                    for i in range(len(poly.lw)):
                        coefs = -poly.ge[i][:-1]
                        const = poly.ge[i][-1]

                        self.__write_constr_input_layer(prob, cnt_imgs, coefs, const, '=', backdoor_indexes, prev_var_idx, curr_var_idx + i)
                    first_layer = False
                else:
                    for i in range(len(poly.lw)):
                        ge, le = poly.ge[i], poly.le[i]

                        if np.all(ge == le):
                            coefs = -poly.ge[i][:-1]
                            const = poly.ge[i][-1]

                            self.__write_constr_hidden_layers(prob, cnt_imgs, coefs, const, '=', prev_var_idx, curr_var_idx + i)
                        else:
                            coefs_ge = -poly.ge[i][:-1]
                            const_ge = poly.ge[i][-1]

                            self.__write_constr_hidden_layers(prob, cnt_imgs, coefs_ge, const_ge, '>=', prev_var_idx, curr_var_idx + i)

                            coefs_le = -poly.le[i][:-1]
                            const_le = poly.le[i][-1]

                            self.__write_constr_hidden_layers(prob, cnt_imgs, coefs_le, const_le, '<=', prev_var_idx, curr_var_idx + i)

                prev_var_idx = curr_var_idx
                curr_var_idx += len(poly.lw)

            self.__write_constr_output_layer(prob, cnt_imgs, target, prev_var_idx)

            cnt_imgs += 1


    def __write_bounds(self, prob, lst_poly_coll, backdoor_indexes):
        lw0, up0 = lst_poly_coll[0][0].lw, lst_poly_coll[0][0].up

        for var_idx in backdoor_indexes:
            prob.write('  {} <= x{} <= {}\n'.format(lw0[var_idx], var_idx, up0[var_idx]))

        cnt_imgs = 0

        for lst_poly in lst_poly_coll:
            var_idx = 0
            for poly in lst_poly:
                for i in range(len(poly.lw)):
                    lw_i = poly.lw[i]
                    up_i = poly.up[i]

                    if var_idx not in backdoor_indexes:
                        if lw_i == up_i:
                            prob.write('  x{}_{} = {}\n'.format(var_idx, cnt_imgs, lw_i))
                        else:
                            prob.write('  {} <= x{}_{} <= {}\n'.format(lw_i, var_idx, cnt_imgs, up_i))

                    var_idx += 1
            cnt_imgs += 1

        prob.flush()


    def __write_problem(self, lst_poly_coll, backdoor_indexes, target):
        filename = 'prob' + str(target) + '.lp'
        prob = open(filename, 'w')

        prob.write('Minimize\n')
        prob.write('  0\n')

        prob.write('Subject To\n')

        self.__write_constr(prob, lst_poly_coll, backdoor_indexes, target)

        prob.write('Bounds\n')

        self.__write_bounds(prob, lst_poly_coll, backdoor_indexes)

        prob.write('End\n')

        prob.flush()
        prob.close()


    def __verifyI(self, model, valid_x0s, valid_bdi, target):
        has_unknown = False

        for backdoor_indexes in valid_bdi:
            has_safe, lst_poly_coll = False, []

            for x0, output_x0 in valid_x0s:
                lw, up = x0.copy(), x0.copy()

                lw[backdoor_indexes] = model.lower[backdoor_indexes]
                up[backdoor_indexes] = model.upper[backdoor_indexes]

                x0_poly = Poly()
                x0_poly.lw, x0_poly.up = lw, up
                # just let x0_poly.le and x0_poly.ge is None
                x0_poly.shape = model.shape

                lst_poly = [x0_poly]
                self.__run(model, 0, lst_poly)

                output_lw, output_up = lst_poly[-1].lw.copy(), lst_poly[-1].up.copy()
                output_lw[target] = output_up[target]

                if np.argmax(output_lw) != target:
                    has_safe = True
                    break
                else:
                    self.__write_problem([lst_poly], backdoor_indexes, target)

                    filename = 'prob' + str(target) + '.lp'
                    opt = gp.read(filename)
                    opt.setParam(GRB.Param.DualReductions, 0)

                    opt.optimize()
                    os.remove(filename)

                    if opt.status == GRB.INFEASIBLE:
                        # print('Infeasible 1 image with target = {}'.format(target))
                        has_safe = True
                        break

                lst_poly_coll.append(lst_poly)

            if not has_safe: # unsafe, try solver
                self.__write_problem(lst_poly_coll, backdoor_indexes, target)

                filename = 'prob' + str(target) + '.lp'
                opt = gp.read(filename)
                opt.setParam(GRB.Param.DualReductions, 0)

                opt.optimize()
                os.remove(filename)

                if opt.status == GRB.INFEASIBLE:
                    # print('Infeasible all images with target = {}'.format(target))
                    pass
                elif opt.status == GRB.OPTIMAL:
                    stamp = self.__get_stamp(opt, backdoor_indexes)

                    # print('Solve target = {} with stamp = {} and position = {}'.format(target, stamp, backdoor_indexes))

                    if not self.__validate(model, valid_x0s, backdoor_indexes, target, stamp, 1.0):
                        # print('The stamp for target = {} is not validate with chosen images I'.format(target))
                        stamp = self.__attack(model, valid_x0s, backdoor_indexes, target)

                    if stamp is not None:
                        return False, (stamp, backdoor_indexes)
                    else:
                        has_unknown = True
                else:
                    stamp = self.__attack(model, valid_x0s, backdoor_indexes, target)

                    if stamp is not None:
                        return False, (stamp, backdoor_indexes)
                    else:
                        has_unknown = True

        if has_unknown:
            return False, None
        else:
            return True, None


    def __hypothesis_test(self, model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold, alpha, beta):
        rate_k = pow(rate, num_imgs) # attack num_imgs successfully at the same time

        p0 = (1 - rate_k) + threshold # not having the attack
        p1 = (1 - rate_k) - threshold

        # print('p0 = {}, p1 = {}'.format(p0, p1))

        h0 = beta / (1 - alpha) # 0.01
        h1 = (1 - beta) / alpha # 99.0

        pr, no = 1, 0
        
        while True:
            no = no + 1

            if num_imgs > len(valid_x0s):
                assert False
            else:
                chosen_idx = np.random.choice(len(valid_x0s), num_imgs, replace=False)
                chosen_x0s = []
                for i in chosen_idx:
                    chosen_x0s.append(valid_x0s[i])

            res, sbi = self.__verifyI(model, chosen_x0s, valid_bdi, target)

            if res: # no backdoor
                # print('VerifyI with target = {}'.format(target, rate))
                pr = pr * p1 / p0 # decrease, favorite H0
            elif sbi is not None: # backdoor with stamp
                stamp, backdoor_indexes = sbi[0], sbi[1]
                if self.__validate(model, valid_x0s, backdoor_indexes, target, stamp, rate): # real stamp
                    return False, sbi
                else:
                    # print('The stamp for target = {} is not validate with all images with rate = {}'.format(target, rate))
                    pr = pr * (1 - p1) / (1 - p0) # increase, favorite H1
            else: # unknown
                # print('Unknown with target = {}'.format(target, rate))
                pr = pr * (1 - p1) / (1 - p0) # increase, favorite H1

            if pr <= h0:
                print('Accept H0 after {} rounds. The probability of not having an attack with target = {} >= {} for K = {}.'.format(no, target, p0, num_imgs))
                return True, None
            elif pr >= h1:
                print('Accept H1 after {} rounds. The probability of not having an attack with target = {} <= {} for K = {}.'.format(no, target, p1, num_imgs))
                return False, None


    def __verify(self, model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold, alpha, beta):
        if rate == 1.0: # no hypothesis test when rate = 1.0, instead try to verify all images
            print('Run verifyI with target = {}'.format(target))
            res, sbi = self.__verifyI(model, valid_x0s, valid_bdi, target)
        else:
            print('Run hypothesis test with target = {}'.format(target))
            res, sbi = self.__hypothesis_test(model, valid_x0s, valid_bdi, target, num_imgs, rate, threshold, alpha, beta)

        if res:
            return None, None
        elif sbi is not None:
            stamp, backdoor_indexes = sbi[0], sbi[1]
            print('Real stamp = {} for target = {} at position = {}'.format(stamp, target, backdoor_indexes))
            return target, sbi
        else:
            return target, None


    def __get_stamp(self, opt, backdoor_indexes):
        stamp = []

        # opt.write('model.sol')
        for idx in backdoor_indexes:
            var = opt.getVarByName('x' + str(idx))
            stamp.append(var.x)

        return np.array(stamp)


    def __validate(self, model, valid_x0s, backdoor_indexes, target, stamp, rate):
        cnt = 0

        for x0, output_x0 in valid_x0s:
            xi = x0.copy()
            xi[backdoor_indexes] = stamp

            output = model.apply(xi).reshape(-1)

            if np.argmax(output) == target: # attack successfully
                cnt += 1

        return (cnt / len(valid_x0s)) >= rate


    def __attack(self, model, valid_x0s, backdoor_indexes, target):
        def obj_func(x, model, valid_x0s, backdoor_indexes, target):
            res = 0

            for x0, output_x0 in valid_x0s:
                xi = x0.copy()
                xi[backdoor_indexes] = x

                output = model.apply(xi).reshape(-1)
                target_score = output[target]

                output_no_target = output - np.eye(len(output))[target] * 1e9
                max_score = np.max(output_no_target)

                if target_score > max_score:
                    res += 0
                else:
                    res += max_score - target_score + 1e-9

            return res

        x = np.zeros(len(backdoor_indexes))
        lw = model.lower[backdoor_indexes]
        up = model.upper[backdoor_indexes]

        args = (model, valid_x0s, backdoor_indexes, target)
        # jac = grad(obj_func)
        jac = None
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun <= 0: # an adversarial sample is generated
            # print('Attack target = {} with stamp = {} and position = {}'.format(target, res.x, backdoor_indexes))
            return res.x

        return None


    def __get_backdoor_indexes(self, size, position, dataset):
        if position < 0:
            return None

        if dataset == 'mnist':
            num_chans, num_rows, num_cols = 1, 28, 28
        elif dataset == 'cifar':
            num_chans, num_rows, num_cols = 3, 32, 32

        row_idx = int(position / num_cols)
        col_idx = position - row_idx * num_cols

        if row_idx + size > num_rows or col_idx + size > num_cols:
            return None

        indexes = []

        for i in range(num_chans):
            tmp = position + i * num_rows * num_cols
            for j in range(size):
                for k in range(size):
                    indexes.append(tmp + k)
                tmp += num_cols

        return indexes


    def __run(self, model, idx, lst_poly):
        if idx == len(model.layers):
            return None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        return self.__solve_backdoor(model, assertion, display)
