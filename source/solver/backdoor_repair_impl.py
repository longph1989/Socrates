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


class BackDoorRepairImpl():
    def __solve_backdoor_repair(self, model, spec, display):
        target = spec['target']
        exp_rate = spec['exp_rate']

        total_imgs = spec['total_imgs']
        dataset = spec['dataset']

        known_stamp = spec['known_stamp']
        pathX, pathY = spec['pathX'], spec['pathY']

        y0s = np.array(ast.literal_eval(read(pathY)))
        valid_x0s = self.__get_valid_x0s(model, total_imgs, y0s, pathX, target)

        if len(valid_x0s) == 0:
            print('No data to analyze target = {}'.format(target))
            return None, None

        print('Number of valid_x0s = {} for target = {}'.format(len(valid_x0s), target))

        print('Lower bound = {} and Upper bound = {}'.format(model.lower[0], model.upper[0]))

        if known_stamp:
            position = spec['stamp_pos']
            size = spec['stamp_size']

            print('\nPredefine stamp position = {} with target = {}'.format(position, target))

            backdoor_indexes = self.__get_backdoor_indexes(size, position, dataset)
            print('\nStamp indexes = {}'.format(backdoor_indexes))

            if dataset == 'mnist':
                trigger = np.zeros(1 * 28 * 28)
                mask = np.zeros(1 * 28 * 28)
            elif dataset == 'cifar':
                trigger = np.zeros(3 * 32 * 32)
                mask = np.zeros(3 * 32 * 32)

            trigger[backdoor_indexes] = 1.0
            mask[backdoor_indexes] = 1.0
        else:
            print('\nGenerate reversed trigger with target = {}'.format(target))
            stamp = self.__attack(model, valid_x0s, target, dataset)

            if dataset == 'mnist':
                trigger = stamp[:(1 * 28 * 28)]
                mask = stamp[(1 * 28 * 28):]
            elif dataset == 'cifar':
                trigger = stamp[:(3 * 32 * 32)]
                mask = stamp[(3 * 32 * 32):]

            print('trigger = {}'.format(trigger))
            print('mask = {}'.format(mask))
            print('sum mask = {}\n'.format(np.sum(mask)))

        valid_x0s_with_bd, non_target_cnt, succ_atk_cnt = self.__get_x0s_with_bd(model, valid_x0s, trigger, mask, target)

        assert len(valid_x0s) == len(valid_x0s_with_bd)

        print('len(valid_x0s) =', len(valid_x0s))
        print('len(valid_x0s_with_bd) =', len(valid_x0s_with_bd))
        print('non_target_cnt =', non_target_cnt)
        print('succ_atk_cnt =', succ_atk_cnt)
        
        succ_rate = succ_atk_cnt / non_target_cnt

        if succ_rate < exp_rate:
            print('\nsucc_rate = {}'.format(succ_rate))
            print('The stamp does not satisfy the success rate = {} with target = {}'.format(exp_rate, target))
        else:
            print('The stamp satisfies the success rate = {} with target = {}'.format(exp_rate, target))
            res = self.__clean_backdoor(model, valid_x0s_with_bd, succ_atk_cnt, trigger, mask, spec)
            
            if res: print('\nCleasing finish')
            else: print('\nCannot clean')


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


    def __update_new_weights_and_bias(self, new_model, opt, repair_layer, repair_neuron, num_weights):
        # opt.write('model.sol')

        for idx in range(num_weights):
            var = opt.getVarByName('w' + str(idx))
            new_model.layers[repair_layer + 2].weights[repair_neuron,idx] = var.x


    def __clean_backdoor(self, model, valid_x0s_with_bd, succ_atk_cnt, trigger, mask, spec):
        target = spec['target']

        total_imgs = spec['total_imgs']
        num_imgs = spec['num_imgs']
        num_repair = spec['num_repair']
        
        clean_atk = spec['clean_atk']
        clean_acc = spec['clean_acc']

        time_limit = spec['time_limit']

        pathX, pathY = spec['pathX'], spec['pathY']
        y0s = np.array(ast.literal_eval(read(pathY)))
    
        print('\nBegin cleansing')
        
        number_of_layers = len(model.layers)

        ie_ave_matrix = []

        for do_layer in range(number_of_layers - 1): # not consider the last layer

            if model.layers[do_layer].is_linear_layer():
                number_of_neurons = model.layers[do_layer].get_number_neurons()

                for do_neuron in range(number_of_neurons):
                    start = time.time()
                    ie, min_val, max_val = self.get_ie_do_h_dy(model, valid_x0s_with_bd, trigger, mask, target, do_layer, do_neuron)
                    end = time.time()

                    print('time = {}'.format(end - start))

                    mie = np.mean(np.array(ie))

                    if mie > 0.0:
                        new_entry = []
                        new_entry.append(mie)
                        new_entry.append(do_layer)
                        new_entry.append(do_neuron)
                        
                        ie_ave_matrix.append(new_entry)

        print(ie_ave_matrix)
        ie_ave_matrix.sort(reverse=True)
        print()
        print(ie_ave_matrix)

        repair_layers, repair_neurons = [], []
        for i in range(num_repair):
            if i >= len(ie_ave_matrix): break
            repair_layers.append(int(ie_ave_matrix[i][1]))
            repair_neurons.append(int(ie_ave_matrix[i][2]))
        
        print('\nRepair layers: {}'.format(repair_layers))
        print('Repair neurons: {}'.format(repair_neurons))

        min_weight, max_weight = self.__collect_min_max_value(model)

        print('\nmin_weight = {}, max_weight = {}\n'.format(min_weight, max_weight))

        for repair_layer, repair_neuron in list(zip(repair_layers, repair_neurons)):
            if not model.layers[repair_layer].is_linear_layer(): assert False

            print('\nRepair layer: {}'.format(repair_layer))
            print('Repair neuron: {}'.format(repair_neuron))
            
            for i in range(num_repair):        
                sample_x0s_with_bd = random.sample(valid_x0s_with_bd, num_imgs)
                print('\nSample {} imgs to clean'.format(num_imgs))

                self.__write_problem(model, sample_x0s_with_bd, trigger, mask, target, repair_layer, repair_neuron, min_weight, max_weight)

                filename = 'prob.lp'
                opt = gp.read(filename)
                opt.setParam(GRB.Param.DualReductions, 0)
                opt.setParam(GRB.Param.NonConvex, 2)
                opt.setParam(GRB.Param.TimeLimit, time_limit)

                opt.optimize()
                os.remove(filename)
                
                if opt.status == GRB.TIME_LIMIT:
                    print('Timeout')
                elif opt.status == GRB.INFEASIBLE:
                    print('Infeasible')
                elif opt.status == GRB.OPTIMAL:
                    print('Optimal')
                    
                    new_model = model.copy()
                    num_weights = model.layers[repair_layer + 2].get_number_neurons()
                    
                    self.__update_new_weights_and_bias(new_model, opt, repair_layer, repair_neuron, num_weights)
     
                    new_valid_x0s = self.__get_valid_x0s(new_model, total_imgs, y0s, pathX, target)
                    new_valid_x0s_with_bd, new_non_target_cnt, new_succ_atk_cnt = self.__get_x0s_with_bd(new_model, new_valid_x0s, trigger, mask, target)

                    assert len(new_valid_x0s) == len(new_valid_x0s_with_bd)

                    print('len(new_valid_x0s) =', len(new_valid_x0s))
                    print('len(new_valid_x0s_with_bd) =', len(new_valid_x0s_with_bd))
                    print('new_non_target_cnt =', new_non_target_cnt)
                    print('new_succ_atk_cnt =', new_succ_atk_cnt)

                    if len(new_valid_x0s) / len(valid_x0s_with_bd) >= clean_acc and new_succ_atk_cnt / new_non_target_cnt <= clean_atk:
                        return True
                else:
                    print('Status = {}'.format(opt.status))
                        
        return False


    def __collect_min_max_value(self, model):
        max_weight, max_bias, coef = 0.0, 0.0, 1.0

        for layer in model.layers:
            if layer.is_linear_layer():
                max_weight = max(max_weight, np.max(np.abs(layer.weights)))
                max_bias = max(max_bias, np.max(np.abs(layer.bias)))

        return -coef * max_weight, coef * max_weight


    def __write_bounds(self, prob, lw_coll, up_coll, min_weight, max_weight, num_weights):
        for idx in range(num_weights):
            prob.write('  {} <= w{} <= {}\n'.format(min_weight, idx, max_weight))

        for cnt_imgs in range(len(lw_coll)):
            lw_list = lw_coll[cnt_imgs]
            up_list = up_coll[cnt_imgs]

            for var_idx in range(len(lw_list)):
                lw, up = lw_list[var_idx], up_list[var_idx]
                if lw == up:
                    prob.write('  x{}_{} = {}\n'.format(var_idx, cnt_imgs, lw))
                else:
                    prob.write('  {} <= x{}_{} <= {}\n'.format(lw, var_idx, cnt_imgs, up))


    def __write_binary(self, prob, bins_coll):
        prob.write(' ')
        for cnt_imgs in range(len(bins_coll)):
            for idx in range(bins_coll[cnt_imgs]):
                prob.write(' a{}_{}'.format(idx, cnt_imgs))
        prob.write('\n')


    def __write_objective(self, prob, num_weights, old_weights):
        prob.write('  [ w0 ^ 2')
        for idx in range(1, num_weights):
            prob.write(' + w{} ^ 2 '.format(idx))
        prob.write(' ]')

        for idx in range(num_weights):
            old_weight = old_weights[idx]
            if old_weight > 0.0:
                prob.write(' - {} w{}'.format(2 * old_weight, idx))
            elif old_weight < 0.0:
                prob.write(' + {} w{}'.format(2 * abs(old_weight), idx))
        prob.write('\n')


    def __write_problem(self, model, sample_x0s_with_bd, trigger, mask, target,
            repair_layer, repair_neuron, min_weight, max_weight):
        filename = 'prob.lp'
        prob = open(filename, 'w')

        # fix outgoing weights
        num_weights = model.layers[repair_layer + 2].get_number_neurons()
        old_weights = model.layers[repair_layer + 2].weights[repair_neuron,:].copy()

        prob.write('Minimize\n')
        self.__write_objective(prob, num_weights, old_weights)

        lw_coll, up_coll, bins_coll = [], [], []

        prob.write('Subject To\n')

        cnt_imgs, has_bins = 0, False

        # original input
        for x_0, _, output_x0, _ in sample_x0s_with_bd:
            # compute input up to the next layer
            input_repair = model.apply_to(x_0, repair_layer + 2).reshape(-1)
            y0 = np.argmax(output_x0)

            lw_list, up_list, num_bins = self.__write_constr(prob, model, input_repair, repair_layer, repair_neuron,
                min_weight, max_weight, cnt_imgs, y0)

            if num_bins > 0: has_bins = True
            
            lw_coll.append(lw_list)
            up_coll.append(up_list)
            bins_coll.append(num_bins)
            
            cnt_imgs += 1

        # input with backdoor
        for _, x_bd, output_x0, _ in sample_x0s_with_bd:
            # compute input up to the next layer
            input_repair = model.apply_to(x_bd, repair_layer + 2).reshape(-1)
            y0 = np.argmax(output_x0)

            lw_list, up_list, num_bins = self.__write_constr(prob, model, input_repair, repair_layer, repair_neuron,
                min_weight, max_weight, cnt_imgs, y0)

            if num_bins > 0: has_bins = True
            
            lw_coll.append(lw_list)
            up_coll.append(up_list)
            bins_coll.append(num_bins)
            
            cnt_imgs += 1

        prob.write('Bounds\n')
        self.__write_bounds(prob, lw_coll, up_coll, min_weight, max_weight, num_weights)

        if has_bins:
            prob.write('Binary\n')
            self.__write_binary(prob, bins_coll)

        prob.write('End\n')

        prob.flush()
        prob.close()


    def __write_constr_next_layer(self, prob, repair_neuron, number_of_neurons, lw_prev, up_prev,
            weights, bias, min_weight, max_weight, cnt_imgs, prev_var_idx, curr_var_idx):
        lw_layer, up_layer = [], []

        for neuron_idx in range(number_of_neurons):
            # compute bounds
            lw, up = 0.0, 0.0

            for weight_idx in range(len(weights[neuron_idx])):
                if weight_idx == repair_neuron:
                    lw_val, up_val = lw_prev[weight_idx], up_prev[weight_idx]
                    lw += min(min_weight * lw_val, max_weight * lw_val, min_weight * up_val, max_weight * up_val)
                    up += max(min_weight * lw_val, max_weight * lw_val, min_weight * up_val, max_weight * up_val)
                else:
                    weight_val = weights[neuron_idx][weight_idx]
                    if weight_val > 0:
                        lw += weight_val * lw_prev[weight_idx]
                        up += weight_val * up_prev[weight_idx]
                    elif weight_val < 0:
                        lw += weight_val * up_prev[weight_idx]
                        up += weight_val * lw_prev[weight_idx]

            lw, up = lw + bias[neuron_idx], up + bias[neuron_idx]
            assert lw <= up

            lw_layer.append(lw)
            up_layer.append(up)

            # write constraints
            prob.write('  x{}_{}'.format(curr_var_idx + neuron_idx, cnt_imgs))
            coefs = -weights[neuron_idx]
            for coef_idx in range(len(coefs)):
                coef = coefs[coef_idx]
                if coef_idx == repair_neuron:
                    prob.write(' - [ w{} * x{}_{} ]'.format(neuron_idx, prev_var_idx + coef_idx, cnt_imgs))
                else:
                    if coef > 0.0:
                        prob.write(' + {} x{}_{}'.format(coef, prev_var_idx + coef_idx, cnt_imgs))
                    elif coef < 0.0:
                        prob.write(' - {} x{}_{}'.format(abs(coef), prev_var_idx + coef_idx, cnt_imgs))
            prob.write(' = {}\n'.format(bias[neuron_idx]))

        return lw_layer, up_layer


    def __write_constr_other_layers(self, prob, number_of_neurons, lw_prev, up_prev,
            weights, bias, cnt_imgs, prev_var_idx, curr_var_idx):
        lw_layer, up_layer = [], []

        for neuron_idx in range(number_of_neurons):
            # compute bounds
            lw, up = 0.0, 0.0

            for weight_idx in range(len(weights[neuron_idx])):
                weight_val = weights[neuron_idx][weight_idx]
                if weight_val > 0:
                    lw += weight_val * lw_prev[weight_idx]
                    up += weight_val * up_prev[weight_idx]
                elif weight_val < 0:
                    lw += weight_val * up_prev[weight_idx]
                    up += weight_val * lw_prev[weight_idx]

            lw, up = lw + bias[neuron_idx], up + bias[neuron_idx]
            assert lw <= up

            lw_layer.append(lw)
            up_layer.append(up)

            # write constraints
            prob.write('  x{}_{}'.format(curr_var_idx + neuron_idx, cnt_imgs))
            coefs = -weights[neuron_idx]
            for coef_idx in range(len(coefs)):
                coef = coefs[coef_idx]
                if coef > 0.0:
                    prob.write(' + {} x{}_{}'.format(coef, prev_var_idx + coef_idx, cnt_imgs))
                elif coef < 0.0:
                    prob.write(' - {} x{}_{}'.format(abs(coef), prev_var_idx + coef_idx, cnt_imgs))
            prob.write(' = {}\n'.format(bias[neuron_idx]))

        return lw_layer, up_layer


    def __write_constr_relu_layers(self, prob, number_of_neurons, lw_prev, up_prev,
            cnt_imgs, prev_var_idx, curr_var_idx, num_bins):
        lw_layer, up_layer = [], []

        for neuron_idx in range(number_of_neurons):
            # compute bounds
            lw, up = lw_prev[neuron_idx], up_prev[neuron_idx]
            assert lw <= up

            lw_layer.append(max(lw, 0.0))
            up_layer.append(max(up, 0.0))

            # write constraints
            if lw < 0.0 and up > 0.0:
                cvar_idx = curr_var_idx + neuron_idx
                pvar_idx = prev_var_idx + neuron_idx

                prob.write('  x{}_{} - x{}_{} + {} a{}_{} <= {}\n'.format(cvar_idx, cnt_imgs, pvar_idx, cnt_imgs, -lw, num_bins, cnt_imgs, -lw))
                prob.write('  x{}_{} - x{}_{} >= 0.0\n'.format(cvar_idx, cnt_imgs, pvar_idx, cnt_imgs))
                prob.write('  x{}_{} - {} a{}_{} <= 0.0\n'.format(cvar_idx, cnt_imgs, up, num_bins, cnt_imgs))
                prob.write('  x{}_{} >= 0.0\n'.format(cvar_idx, cnt_imgs))
                num_bins += 1
            elif lw >= 0.0:
                prob.write('  x{}_{} - x{}_{} = 0.0\n'.format(curr_var_idx + neuron_idx, cnt_imgs, prev_var_idx + neuron_idx, cnt_imgs))

        return lw_layer, up_layer, num_bins


    def __write_constr(self, prob, model, input_repair, repair_layer, repair_neuron,
            min_weight, max_weight, cnt_imgs, y0):
        lw_list, up_list = [], []
        lw_input, up_input = [], []
        num_bins = 0

        for input_val in input_repair:
            lw_input.append(input_val)
            up_input.append(input_val)

        lw_list.append(lw_input)
        up_list.append(up_input)

        curr_var_idx = len(input_repair)
        prev_var_idx = 0

        for layer_idx in range(repair_layer + 2, len(model.layers)):
            layer = model.layers[layer_idx]
            lw_layer, up_layer = [], []

            # fully connected layer
            if layer.is_linear_layer():
                weights = layer.weights.transpose(1, 0) # shape: num_neuron X input
                bias = layer.bias.transpose(1, 0).reshape(-1) # shape: num_neuron
                number_of_neurons = layer.get_number_neurons()

                # next linear layer
                if layer_idx == repair_layer + 2:
                    lw_prev, up_prev = lw_list[-1], up_list[-1]
                    lw_layer, up_layer = self.__write_constr_next_layer(prob, repair_neuron, number_of_neurons, lw_prev, up_prev,
                                            weights, bias, min_weight, max_weight, cnt_imgs, prev_var_idx, curr_var_idx)                 
                # other linear layers
                else:
                    lw_prev, up_prev = lw_list[-1], up_list[-1]
                    lw_layer, up_layer = self.__write_constr_other_layers(prob, number_of_neurons, lw_prev, up_prev,
                                            weights, bias, cnt_imgs, prev_var_idx, curr_var_idx)
            # ReLU
            else:
                lw_prev, up_prev = lw_list[-1], up_list[-1]
                number_of_neurons = len(lw_prev)
                lw_layer, up_layer, num_bins = self.__write_constr_relu_layers(prob, number_of_neurons, lw_prev, up_prev,
                                            cnt_imgs, prev_var_idx, curr_var_idx, num_bins)

            lw_list.append(lw_layer)
            up_list.append(up_layer)

            prev_var_idx = curr_var_idx
            curr_var_idx += len(lw_layer)

        # output constraints
        for output_idx in range(len(lw_list[-1])):
            if output_idx != y0:
                # use 0.001 to guarantee the output condition
                prob.write('  x{}_{} - x{}_{} > 0.001\n'.format(prev_var_idx + y0, cnt_imgs, prev_var_idx + output_idx, cnt_imgs))

        flat_lw_list = [item for sublist in lw_list for item in sublist]
        flat_up_list = [item for sublist in up_list for item in sublist]

        prob.flush()

        return flat_lw_list, flat_up_list, num_bins


    def get_ie_do_h_dy(self, model, valid_x0s_with_bd, trigger, mask, target, do_layer, do_neuron):
        # get value range of given hidden neuron

        hidden_max, hidden_min = None, None

        for x0, x_bd, output_x0, output_x_bd in valid_x0s_with_bd:
            _, hidden = model.apply_get_h(x_bd, do_layer, do_neuron)

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
                dy = self.get_dy_do_h(model, valid_x0s_with_bd, trigger, mask, target, do_layer, do_neuron, h_val)
                ie.append(dy)

        return ie, hidden_min, hidden_max

    #
    # get expected value of y with hidden neuron intervention
    #
    def get_dy_do_h(self, model, valid_x0s_with_bd, trigger, mask, target, do_layer, do_neuron, do_value):
        dy_sum = 0.0

        for x0, x_bd, output_x0, output_x_bd in valid_x0s_with_bd:
            output_do = model.apply_intervention(x_bd, do_layer, do_neuron, do_value).reshape(-1)

            dy = abs(output_x_bd[target] - output_do[target])
            dy_sum = dy_sum + dy

        avg = dy_sum / len(valid_x0s_with_bd)

        return avg


    def __get_valid_x0s(self, model, total_imgs, y0s, path, target):
        valid_x0s = []

        for i in range(total_imgs):
            pathX = path + 'data' + str(i) + '.txt'
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

            if y0 == y0s[i]:
                valid_x0s.append((x0, output_x0))

        return valid_x0s


    def __get_x0s_with_bd(self, model, valid_x0s, trigger, mask, target):
        valid_x0s_with_bd, non_target_cnt, succ_atk_cnt = [], 0, 0

        for i in range(len(valid_x0s)):
            x0, output_x0 = valid_x0s[i]
            y0 = np.argmax(output_x0)
            
            x_bd = (1 - mask) * x0 + mask * trigger

            output_x_bd = model.apply(x_bd).reshape(-1)
            y_bd = np.argmax(output_x_bd)

            if y0 != target:
                non_target_cnt += 1
                if y_bd == target: succ_atk_cnt += 1

            valid_x0s_with_bd.append((x0, x_bd, output_x0, output_x_bd))

        return valid_x0s_with_bd, non_target_cnt, succ_atk_cnt


    def __attack(self, model, valid_x0s, target, dataset):
        def obj_func(x, model, valid_x0s, target, length, half_len):
            res, lam = 0.0, 1.0 if len(valid_x0s) >= 100 else 0.1

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


    def __run(self, model, idx, lst_poly):
        if idx == len(model.layers):
            return None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        return self.__solve_backdoor_repair(model, assertion, display)
