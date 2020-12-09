import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast
import os

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

        new_poly.le = self.le.copy()
        new_poly.ge = self.ge.copy()

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
    def __init__(self):
        self.has_ref = True
        self.has_tig = True
        self.max_ref = 20
        self.max_tig = 20
        self.cnt_ref = 0
        self.cnt_tig = 0
        self.ref_typ = 0
        self.tasks = []

        if self.has_ref:
            print('Run with refinement!')
        if self.has_tig:
            print('Run with input tighten!')

        if self.has_ref:
            print('Refinement type: ', end='')
            if self.ref_typ == 0:
                print('Greatest norm impact based on abs(coef * (lw,up))')


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

        adv = self.__find_adv(model, x0, y0, lw, up)

        if len(adv) == 0:
            x0_poly = Poly()

            x0_poly.lw, x0_poly.up = lw, up

            x0_poly.le = np.eye(len(x0) + 1)[:-1]
            x0_poly.ge = np.eye(len(x0) + 1)[:-1]

            lst_poly = [x0_poly]

            if self.has_tig:
                res = self.__verify_with_input_tighten(model, x0, y0, 0, lst_poly)
            else:
                res = self.__verify_without_input_tighten(model, x0, y0, 0, lst_poly)

            if res == 0:
                print('Unknown!')
            elif res == 1:
                print('The network is robust around x0!')

            print('Input tighten: {} times!'.format(self.cnt_tig))
            print('Refinement: {} times!'.format(self.cnt_ref))


    def __find_adv(self, model, x0, y0, lw, up):
        def obj_func(x, model, y0):
            output = model.apply(x)
            y0_score = output[0][y0]

            output = output - np.eye(output[0].size)[y0] * 1e9
            max_score = np.max(output)

            loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

            return loss + np.sum(x - x)

        x = x0.copy()

        args = (model, y0)
        jac = grad(obj_func)
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            valid = self.__validate_adv(model, res.x, y0)
            assert valid
            return res.x
        else:
            return np.empty(0)


    def __validate_adv(self, model, x, y0):
        y = np.argmax(model.apply(x), axis=1)[0]

        if y != y0:
            print('True adversarial sample found!')
            print('x = {}'.format(x))
            print('y = {}'.format(y))
            return True
        else:
            return False


    def __verify_without_input_tighten(self, model, x0, y0, idx, lst_poly):
        task = Task(idx, lst_poly)
        self.tasks.insert(0, task)

        while len(self.tasks) > 0:
            task = self.tasks.pop()
            res = self.__verify(model, x0, y0, task.idx, task.lst_poly)
            if res == 0 or res == 2: return res # False, unknown or with adv

        return 1 # True, robust


    def __verify_with_input_tighten(self, model, x0, y0, idx, lst_poly):
        res, x, y, lst_ge = self.__run(model, x0, y0, idx, lst_poly)

        if res:
            return 1 # True, robust
        elif self.__validate_adv(model, x, y0):
            return 2 # False, true adv found
        else:
            new_lst_poly = self.__input_tighten(model, y0, y, lst_poly)
            diff_lw = np.abs(lst_poly[0].lw - new_lst_poly[0].lw)
            diff_up = np.abs(lst_poly[0].up - new_lst_poly[0].up)

            # no progess with input tighten
            if np.all(diff_lw < 1e-9) and np.all(diff_up < 1e-9):
                if self.has_ref:
                    ref_layer, ref_index, ref_value = self.__choose_refinement(model, lst_poly, x, y0, y, lst_ge, self.ref_typ)
                else:
                    ref_layer = None

                if ref_layer != None:
                    lst_poly1, lst_poly2 = self.__refine(model, lst_poly, x, ref_layer, ref_index, ref_value)

                    task1 = Task(ref_layer, lst_poly1)
                    task2 = Task(ref_layer, lst_poly2)

                    self.tasks.insert(0, task1)
                    self.tasks.insert(1, task2)

                    while len(self.tasks) > 0:
                        task = self.tasks.pop()
                        res = self.__verify(model, x0, y0, task.idx, task.lst_poly)
                        if res == 0 or res == 2: return res # False, unknown or with adv

                    return 1 # True, robust
                else:
                    return 0 # False, unknown
            else:
                return self.__verify_with_input_tighten(model, x0, y0, 0, new_lst_poly)


    def __input_tighten(self, model, y0, y, lst_poly):
        len0 = len(lst_poly[0].lw)
        new_x0_poly = lst_poly[0].copy()

        if self.cnt_tig >= self.max_tig:
            return [new_x0_poly]

        self.cnt_tig += 1

        lw, up = self.__generate_bounds(model, lst_poly)
        constraints_eq, constraints_ge, lenx = self.__generate_constraints(model, y0, y, lst_poly)

        constraints_eq = np.array(constraints_eq)
        constraints_ge = np.array(constraints_ge)

        x = cp.Variable(lenx)
        constraints = [lw <= x, x <= up, constraints_eq @ x == 0, constraints_ge @ x >= 0]

        if len0 < 100 and len(constraints_eq) + len(constraints_eq) < 100:
            for i in range(len0):
                if (new_x0_poly.lw[i] < new_x0_poly.up[i]):
                    objective = cp.Minimize(x[i])
                    problem = cp.Problem(objective, constraints)
                    lw_i = round(problem.solve(solver=cp.CBC), 9)

                    objective = cp.Minimize(-x[i])
                    problem = cp.Problem(objective, constraints)
                    up_i = -round(problem.solve(solver=cp.CBC), 9)

                    new_x0_poly.lw[i] = lw_i
                    new_x0_poly.up[i] = up_i
        else:
            clonesX, clonesC = [], []

            for i in range(len0):
                clonesX.append(x)
                clonesC.append(constraints)

            num_cores = os.cpu_count()
            pool = multiprocessing.Pool(num_cores)
            zz = zip(range(len0), clonesX, clonesC, new_x0_poly.lw, new_x0_poly.up)
            for i, lw_i, up_i in pool.map(input_tighten, zz):
                new_x0_poly.lw[i] = lw_i
                new_x0_poly.up[i] = up_i
            pool.close()

        return [new_x0_poly]


    def __generate_bounds(self, model, lst_poly):
        lw = np.empty(0)
        up = np.empty(0)

        for poly in lst_poly:
            lw = np.concatenate((lw, poly.lw))
            up = np.concatenate((up, poly.up))

        return np.append(lw, 1), np.append(up, 1)


    def __generate_constraints(self, model, y0, y, lst_poly):
        lst_len = []
        for poly in lst_poly:
            lst_len.append(len(poly.lw))
        total_len = np.sum(lst_len)

        constraints_eq = list()
        constraints_ge = list()

        for i in range(len(lst_poly)):
            if i == 0: continue

            layer = model.layers[i-1]
            poly_i = lst_poly[i]
            poly_j = lst_poly[i-1]

            # weights for other variables
            w0 = np.zeros((lst_len[i], int(np.sum(lst_len[:i-1]))))
            w1_le = poly_i.le[:, :-1] # weights for previous layer variables
            w1_ge = poly_i.ge[:, :-1] # weights for previous layer variables
            w2 = np.eye(lst_len[i]) # weights for current layer variable
            # weights for other variables
            w3 = np.zeros((lst_len[i], total_len - np.sum(lst_len[:i+1])))
            b_le = poly_i.le[:, -1].reshape(lst_len[i], 1) # bias
            b_ge = poly_i.ge[:, -1].reshape(lst_len[i], 1) # bias

            if layer.is_poly_exact():
                coefs_eq = np.concatenate((w0, -w1_ge, w2, w3, -b_ge), axis=1)
                constraints_eq.extend(coefs_eq)
            else:
                coefs_le = np.concatenate((w0, w1_le, -w2, w3, b_le), axis=1)
                coefs_ge = np.concatenate((w0, -w1_ge, w2, w3, -b_ge), axis=1)

                for k in range(len(coefs_le)):
                    if poly_j.lw[k] < 0 and poly_j.up[k] > 0:
                        constraints_ge.append(coefs_le[k])
                        constraints_ge.append(coefs_ge[k])
                    else:
                        constraints_eq.append(coefs_ge[k])

        last_cons = np.zeros(total_len + 1)
        last_cons[-lst_len[-1] - 1 + y] = 1
        last_cons[-lst_len[-1] - 1 + y0] = -1

        constraints_ge.append(last_cons)

        return constraints_eq, constraints_ge, total_len + 1


    def __verify(self, model, x0, y0, idx, lst_poly):
        res, x, y, lst_ge = self.__run(model, x0, y0, idx, lst_poly)

        if res:
            return 1 # True, robust
        elif self.__validate_adv(model, x, y0):
            return 2 # False, true adv found
        elif not self.has_ref:
            return 0 # False, unknown
        else:
            ref_layer, ref_index, ref_value = self.__choose_refinement(model, lst_poly, x, y0, y, lst_ge, self.ref_typ)

            if ref_layer != None:
                lst_poly1, lst_poly2 = self.__refine(model, lst_poly, x, ref_layer, ref_index, ref_value)

                task1 = Task(ref_layer, lst_poly1)
                task2 = Task(ref_layer, lst_poly2)

                self.tasks.insert(0, task1)
                self.tasks.insert(1, task2)

                for task in self.tasks:
                    if task.idx > ref_layer: self.tasks.remove(task)

                return 1 # to continue
            else:
                return 0 # False, unknown


    def __run(self, model, x0, y0, idx, lst_poly):
        # print('\n############################\n')
        # print('idx = {}'.format(idx))
        # print('poly.lw = {}'.format(lst_poly[idx].lw))
        # print('poly.up = {}'.format(lst_poly[idx].up))

        if idx == len(model.layers):
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
    def __choose_refinement(self, model, lst_poly, x, y0, y, lst_ge, ref_typ):
        if ref_typ == 0:
            return self.__impact_refinement(model, lst_poly, x, y0, y, lst_ge)


    # impact refinement
    def __impact_refinement(self, model, lst_poly, x, y0, y, lst_ge):
        best_layer, best_index, ref_value = None, None, None
        if self.cnt_ref >= self.max_ref:
            return best_layer, best_index, ref_value

        self.cnt_ref += 1
        best_value = -1e9

        for i in range(len(model.layers)):
            layer = model.layers[i]
            sum_impact = 0

            if i == 0 or not layer.is_poly_exact():
                poly_i = lst_poly[i]
                ge_i = lst_ge[i]

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    cf = ge_i[ref_idx]

                    if (i == 0 and lw < up) or (lw < 0 and up > 0):
                        impact = max(abs(cf * lw), abs(cf * up))
                        sum_impact = sum_impact + impact

                if sum_impact > 0:
                    for ref_idx in range(len(poly_i.lw)):
                        lw = poly_i.lw[ref_idx]
                        up = poly_i.up[ref_idx]
                        cf = ge_i[ref_idx]

                        if (i == 0 and lw < up) or (lw < 0 and up > 0):
                            impact = max(abs(cf * lw), abs(cf * up))
                            norm_impact = impact / sum_impact
                            if best_value < norm_impact:
                                best_layer = i
                                best_index = ref_idx
                                best_value = norm_impact
                                ref_value = (lw + up) / 2 if i == 0 else 0

        return best_layer, best_index, ref_value


    def __refine(self, model, lst_poly, x, ref_layer, ref_index, ref_value):
        lst_poly1, lst_poly2 = [], []

        for i in range(ref_layer):
            lst_poly1.append(lst_poly[i].copy())
            lst_poly2.append(lst_poly[i].copy())

        x1_poly, x2_poly = Poly(), Poly()

        x1_poly.lw = lst_poly[ref_layer].lw.copy()
        x1_poly.up = lst_poly[ref_layer].up.copy()
        x1_poly.le = lst_poly[ref_layer].le.copy()
        x1_poly.ge = lst_poly[ref_layer].ge.copy()

        x2_poly.lw = lst_poly[ref_layer].lw.copy()
        x2_poly.up = lst_poly[ref_layer].up.copy()
        x2_poly.le = lst_poly[ref_layer].le.copy()
        x2_poly.ge = lst_poly[ref_layer].ge.copy()

        x1_poly.up[ref_index] = ref_value
        x2_poly.lw[ref_index] = ref_value

        lst_poly1.append(x1_poly)
        lst_poly2.append(x2_poly)

        return lst_poly1, lst_poly2


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
