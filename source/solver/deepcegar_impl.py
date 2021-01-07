import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast
import os
import time
import random

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *
from poly_utils import *


class Poly():
    def __init__(self):
        self.lw, self.up = None, None
        self.le, self.ge = None, None

    def copy(self):
        new_poly = Poly()

        new_poly.lw = self.lw.copy()
        new_poly.up = self.up.copy()

        new_poly.le = None if self.le is None else self.le.copy()
        new_poly.ge = None if self.ge is None else self.ge.copy()

        return new_poly

    def back_substitute(self, lst_poly, get_ineq=False):
        no_neurons = len(self.lw)

        if no_neurons <= 100 or len(lst_poly) <= 2:
            for i in range(no_neurons):
                args = (i, self.le[i], self.ge[i], lst_poly)
                _, lw_i, up_i, le, ge = back_substitute(args)
                self.lw[i], self.up[i] = lw_i, up_i

                # get_ineq only happens at the last step
                # no_neurons in this case always be 1
                if get_ineq: lst_le, lst_ge = le, ge
        else:
            clones = []

            for i in range(no_neurons):
                clones.append(lst_poly)

            num_cores = os.cpu_count()
            pool = multiprocessing.Pool(num_cores)
            zz = zip(range(no_neurons), self.le, self.ge, clones)
            for i, lw_i, up_i, _, _ in pool.map(back_substitute, zz):
                self.lw[i], self.up[i] = lw_i, up_i
            pool.close()

        if get_ineq: return lst_le, lst_ge


class Task():
    def __init__(self, idx, lst_poly):
        self.idx = idx
        self.lst_poly = lst_poly


class DeepCegarImpl():
    def __init__(self, has_ref, max_ref, ref_typ, max_sus):
        self.has_ref = has_ref
        self.max_ref = max_ref
        self.ref_typ = ref_typ
        self.max_sus = max_sus

        self.cnt_ref = 0
        self.cnt_verified = 0
        self.cnt_removed = 0

        self.tasks = []

        if self.has_ref:
            print('Run with refinement! Max refinement = {}'.format(self.max_ref))
            print('Refinement type: ', end='')
            if self.ref_typ == 0:
                print('Greatest norm abs(coef * (lw,up)) for input/relu nodes.')
            elif self.ref_typ == 1:
                print('Greatest input range (up - lw).')
            elif self.ref_typ == 2:
                print('Random choice.')


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        eps = ast.literal_eval(read(spec['eps']))

        lw = np.maximum(model.lower, x0 - eps)
        up = np.minimum(model.upper, x0 + eps)

        if 'fairness' in spec:
            sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            for index in range(len(x0)):
                if not (index in sensitive):
                    lw[index], up[index] = x0[index], x0[index]

        res, adv = 0, self.__find_adv(model, x0, y0, lw, up)

        if adv is None:
            x0_poly = Poly()

            x0_poly.lw, x0_poly.up = lw, up
            # just let x0_poly.le and x0_poly.ge is None

            lst_poly = [x0_poly]

            res = self.__iterate_verify(model, x0, y0, 0, lst_poly)

            if res == 0:
                print('Unknown!')
            elif res == 1:
                print('The network is robust around x0!')

            print('Refinement: {} times!'.format(self.cnt_ref))
            print('Verified sub tasks: {} tasks!'.format(self.cnt_verified))
            print('Removed sub tasks: {} tasks!'.format(self.cnt_removed))

        return res == 1


    def __find_adv(self, model, x0, y0, lw, up):
        def obj_func(x, model, y0):
            output = model.apply(x).reshape(-1)
            y0_score = output[y0]

            output_no_y0 = output - np.eye(len(output))[y0] * 1e9
            max_score = np.max(output_no_y0)

            return y0_score - max_score

        print('Finding adversarial sample! Try {} times'.format(self.max_sus))

        for i in range(self.max_sus):
            if self.max_sus == 1:
                x = x0.copy()
            else:
                x = generate_x(len(x0), lw, up)

            args = (model, y0)
            jac = grad(obj_func)
            bounds = Bounds(lw, up)

            res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

            if res.fun <= 0: # an adversarial sample is generated
                valid = self.__validate_adv(model, res.x, y0)
                assert valid
                return res.x

        return None


    def __validate_adv(self, model, x, y0):
        output = model.apply(x).reshape(-1)
        y0_score = output[y0]

        output_no_y0 = output - np.eye(len(output))[y0] * 1e9
        max_score = np.max(output_no_y0)

        if y0_score <= max_score:
            y = np.argmax(output_no_y0)

            print('True adversarial sample found!')
            print('x = {}'.format(x))
            print('output_x = {}'.format(output))
            print('y = {}'.format(y))
            return True
        else:
            return False


    def __iterate_verify(self, model, x0, y0, idx, lst_poly):
        task = Task(idx, lst_poly)
        self.tasks.append(task)

        while len(self.tasks) > 0:
            task = self.tasks.pop()
            res = self.__verify(model, x0, y0, task.idx, task.lst_poly)
            if res == 0 or res == 2: return res # False, unknown or with adv

        return 1 # True, robust


    def __verify(self, model, x0, y0, idx, lst_poly):
        res, x, y, lst_ge = self.__run(model, x0, y0, idx, lst_poly)

        if res:
            self.cnt_verified += 1
            return 1 # True, robust
        elif self.__validate_adv(model, x, y0):
            return 2 # False, true adv found
        elif not self.has_ref:
            return 0 # False, unknown
        else:
            ref_layer, ref_index, ref_value = self.__choose_refinement(model, x0, x, y0, y, lst_poly, lst_ge, self.ref_typ)

            if ref_layer != None:
                lst_poly1, lst_poly2 = self.__refine(lst_poly, ref_layer, ref_index, ref_value)

                task1 = Task(ref_layer, lst_poly1)
                task2 = Task(ref_layer, lst_poly2)

                for task in self.tasks:
                    if task.idx > ref_layer:
                        self.cnt_removed += 1
                        self.tasks.remove(task)

                self.tasks.append(task1)
                self.tasks.append(task2)

                return 1 # to continue
            else:
                return 0 # False, unknown


    def __run(self, model, x0, y0, idx, lst_poly):
        # print('\n############################\n')
        # print('idx = {}'.format(idx))
        # print('poly.lw = {}'.format(lst_poly[idx].lw))
        # print('poly.up = {}'.format(lst_poly[idx].up))
        # print('poly.ge = {}'.format(lst_poly[idx].ge))
        # print('poly.le = {}'.format(lst_poly[idx].le))

        if idx == len(model.layers):
            assert len(lst_poly) == len(model.layers) + 1

            poly_out = lst_poly[idx]
            no_neurons = len(poly_out.lw)

            # print('poly_out.lw = {}'.format(poly_out.lw))
            # print('poly_out.up = {}'.format(poly_out.up))

            for y in range(no_neurons):
                if y != y0 and poly_out.lw[y0] <= poly_out.up[y]:
                    poly_res = Poly()

                    poly_res.lw = np.zeros(1)
                    poly_res.up = np.zeros(1)

                    poly_res.le = np.zeros([1, no_neurons + 1])
                    poly_res.ge = np.zeros([1, no_neurons + 1])

                    poly_res.ge[0,y0] = 1
                    poly_res.ge[0,y] = -1

                    lst_le, lst_ge = poly_res.back_substitute(lst_poly, True)

                    assert len(lst_ge) == len(lst_poly)

                    ge_x0 = lst_ge[0]

                    # print('y = {}'.format(y))
                    # print('res.lw = {}'.format(poly_res.lw[0]))

                    if poly_res.lw[0] <= 0:
                        poly0 = lst_poly[0]
                        x = self.__find_sus_adv(ge_x0, x0, poly0.lw, poly0.up)
                        return False, x, y, lst_ge

            return True, None, None, None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, x0, y0, idx + 1, lst_poly)


    def __find_sus_adv(self, ge_x0, x0, lw, up):
        x = cp.Variable(len(x0) + 1)
        lw = np.append(lw, 1)
        up = np.append(up, 1)

        objective = cp.Minimize(cp.sum(cp.multiply(x, ge_x0)))
        constraints = [lw <= x, x <= up]
        problem = cp.Problem(objective, constraints)

        result = problem.solve(solver=cp.CBC)

        if result <= 0:
            return x.value[:-1]
        else:
            assert False


    # choose refinement
    def __choose_refinement(self, model, x0, x, y0, y, lst_poly, lst_ge, ref_typ):
        if ref_typ == 0:
            return self.__norm_impact_refinement(model, x0, x, y0, y, lst_poly, lst_ge)
        elif ref_typ == 1:
            return self.__input_range_refinement(model, x0, x, y0, y, lst_poly, lst_ge)
        elif ref_typ == 2:
            return self.__random_refinement(model, x0, x, y0, y, lst_poly, lst_ge)
        else:
            assert False


    # norm impact refinement
    def __norm_impact_refinement(self, model, x0, x, y0, y, lst_poly, lst_ge):
        best_layer, best_index, ref_value = None, None, None
        if self.cnt_ref >= self.max_ref:
            return best_layer, best_index, ref_value

        self.cnt_ref += 1
        best_value = 0

        for i in range(len(model.layers)):
            layer = model.layers[i]

            if i == 0 or not layer.is_poly_exact():
                poly_i = lst_poly[i]
                ge_i = lst_ge[i]

                func = None if i == 0 else layer.func

                sum_impact = 0

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    cf = ge_i[ref_idx]

                    if ((i == 0 or func == sigmoid or func == tanh) and lw < up) \
                            or (func == relu and lw < 0 and up > 0):
                        impact = max(abs(cf * lw), abs(cf * up))
                        sum_impact = sum_impact + impact

                if sum_impact > 0:
                    for ref_idx in range(len(poly_i.lw)):
                        lw = poly_i.lw[ref_idx]
                        up = poly_i.up[ref_idx]
                        cf = ge_i[ref_idx]

                        if ((i == 0 or func == sigmoid or func == tanh) and lw < up) \
                                or (func == relu and lw < 0 and up > 0):
                            impact = max(abs(cf * lw), abs(cf * up))
                            norm_impact = impact / sum_impact
                            if best_value < norm_impact:
                                best_layer = i
                                best_index = ref_idx
                                best_value = norm_impact
                                ref_value = 0 if func == relu else (lw + up) / 2

        return best_layer, best_index, ref_value


    # input range refinement
    def __input_range_refinement(self, model, x0, x, y0, y, lst_poly, lst_ge):
        best_layer, best_index, ref_value = None, None, None
        if self.cnt_ref >= self.max_ref:
            return best_layer, best_index, ref_value

        self.cnt_ref += 1
        best_value = 0

        poly_i = lst_poly[0] # only work with input layer

        for ref_idx in range(len(poly_i.lw)):
            lw = poly_i.lw[ref_idx]
            up = poly_i.up[ref_idx]

            if best_value < (up - lw):
                best_layer = 0
                best_index = ref_idx
                best_value = up - lw
                ref_value = (lw + up) / 2

        return best_layer, best_index, ref_value


    # random refinement
    def __random_refinement(self, model, x0, x, y0, y, lst_poly, lst_ge):
        best_layer, best_index, ref_value = None, None, None
        if self.cnt_ref >= self.max_ref:
            return best_layer, best_index, ref_value

        self.cnt_ref += 1
        choice_lst = []

        for i in range(len(model.layers)):
            layer = model.layers[i]

            if i == 0 or not layer.is_poly_exact():
                poly_i = lst_poly[i]

                func = None if i == 0 else layer.func

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]

                    if ((i == 0 or func == sigmoid or func == tanh) and lw < up) \
                            or (func == relu and lw < 0 and up > 0):
                        ref_value = 0 if func == relu else (lw + up) / 2
                        choice_lst.append((i, ref_idx, ref_value))

        choice_idx = random.randrange(len(choice_lst))
        best_layer, best_index, ref_value = choice_lst[choice_idx]

        return best_layer, best_index, ref_value


    def __refine(self, lst_poly, ref_layer, ref_index, ref_value):
        lst_poly1, lst_poly2 = [], []

        for i in range(ref_layer + 1):
            lst_poly1.append(lst_poly[i].copy())
            lst_poly2.append(lst_poly[i].copy())

        lst_poly1[ref_layer].lw[ref_index] = ref_value
        lst_poly2[ref_layer].up[ref_index] = ref_value

        return lst_poly1, lst_poly2


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
