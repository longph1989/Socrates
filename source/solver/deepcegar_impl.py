import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *
from poly_utils import *


class Poly():
    def __init__(self):
        self.lw = None
        self.up = None

        self.le = None
        self.ge = None

    def copy(self):
        new_poly = Poly()

        new_poly.lw = self.lw.copy()
        new_poly.up = self.up.copy()

        new_poly.le = self.le.copy()
        new_poly.ge = self.ge.copy()

        return new_poly

    def back_substitute(self, lst_poly, get_ineq=False):
        le_curr = self.le
        ge_curr = self.ge

        no_neurons = len(le_curr)

        if no_neurons <= 100 or len(lst_poly) <= 2:
            for i in range(no_neurons):
                args = (i, le_curr[i], ge_curr[i], lst_poly)
                _, lw_i, up_i, le, ge = back_substitute(args)
                self.lw[i] = lw_i
                self.up[i] = up_i

                # get_ineq only happens at the last step
                # no_neurons in this case always be 1
                if get_ineq:
                    lst_le = le
                    lst_ge = ge
        else:
            clones = []

            for i in range(no_neurons):
                clones.append(lst_poly)

            num_cores = 4
            pool = multiprocessing.Pool(num_cores)
            zz = zip(range(no_neurons), self.le, self.ge, clones)
            for i, lw_i, up_i, _, _ in pool.map(back_substitute, zz):
                self.lw[i] = lw_i
                self.up[i] = up_i
            pool.close()

        if get_ineq: return lst_le, lst_ge


class DeepCegarImpl():
    def __init__(self, max_ref):
        self.has_ref = True
        self.max_ref = 1
        self.cnt_ref = 0


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        eps = ast.literal_eval(read(spec['eps']))

        lw = np.maximum(model.lower, x0 - eps)
        up = np.minimum(model.upper, x0 + eps)

        if 'fairness' in spec:
            sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            for index in range(x0.size):
                if not (index in sensitive):
                    lw[index] = x0[index]
                    up[index] = x0[index]


        res, x = self.__find_adv(model, x0, y0, lw, up)
        if not res:
            self.__validate_adv(model, x, y0)
        else:
            x0_poly = Poly()

            x0_poly.lw = lw
            x0_poly.up = up

            x0_poly.le = np.eye(len(x0) + 1)[0:-1]
            x0_poly.ge = np.eye(len(x0) + 1)[0:-1]

            lst_poly = [x0_poly]

            # res = self.__verify_back_propagate(model, x0, y0, 0, lst_poly)
            res = self.__verify(model, x0, y0, 0, lst_poly)
            if res == 0:
                print('The network is robust around x0!')
            elif res == 1:
                print('Unknown!')
            elif res == 2:
                print('True adversarial sample found in verifiction!')


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
            return False, res.x
        else:
            return True, np.empty(0)


    def __validate_adv(self, model, x, y0):
        y = np.argmax(model.apply(x), axis=1)[0]

        if y != y0:
            print('True adversarial sample found!')
            print('x = {}'.format(x))
            print('y = {}'.format(y))
            return True
        else:
            return False


    def __verify_back_propagate(self, model, x0, y0, idx, lst_poly):
        res, x, y, lst_ge = self.__run(model, x0, y0, idx, lst_poly)

        if res:
            return 0 # True, robust
        elif not self.has_ref:
            return 1 # False, unknown
        else:
            if self.__validate_adv(model, x, y0):
                return 2 # False, with adv
            else:
                new_lst_poly = self.__back_propagate(model, y0, y, lst_poly)
                diff_lw = np.abs(lst_poly[0].lw - new_lst_poly[0].lw)
                diff_up = np.abs(lst_poly[0].up - new_lst_poly[0].up)

                # no progess with back propagation
                if np.all(diff_lw < 1e-3) and np.all(diff_up < 1e-3):
                    return self.__verify(model, x0, y0, 0, new_lst_poly)
                else:
                    return self.__verify_back_propagate(model, x0, y0, 0, new_lst_poly)


    def __back_propagate(self, model, y0, y, lst_poly):
        lw, up = self.__generate_bounds(model, lst_poly)
        constraints_eq, constraints_ge, lenx = self.__generate_constraints(model, y0, y, lst_poly)

        constraints_eq = np.array(constraints_eq)
        constraints_ge = np.array(constraints_ge)

        len0 = len(lst_poly[0].lw)
        new_x0_poly = lst_poly[0].copy()

        x = cp.Variable(lenx)
        constraints = [lw <= x, x <= up, constraints_eq @ x == 0, constraints_ge @ x >= 0]

        if len0 < 100 and len(constraints_eq) + len(constraints_eq) < 100:
            for i in range(len(lst_poly[0].lw)):
                objective = cp.Minimize(x[i])
                prob = cp.Problem(objective, constraints)
                lw_i = round(prob.solve(), 9)

                objective = cp.Minimize(-x[i])
                prob = cp.Problem(objective, constraints)
                up_i = -round(prob.solve(), 9)

                new_x0_poly.lw[i] = lw_i
                new_x0_poly.up[i] = up_i
        else:
            clonesX, clonesC = [], []

            for i in range(len0):
                clonesX.append(x)
                clonesC.append(constraints)

            num_cores = 4
            pool = multiprocessing.Pool(num_cores)
            zz = zip(range(len0), clonesX, clonesC)
            for i, lw_i, up_i in pool.map(back_propagate, zz):
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
            return 0 # True, robust
        elif not self.has_ref:
            return 1 # False, unknown
        else:
            if self.__validate_adv(model, x, y0):
                return 2 # False, with adv
            else:
                # ref_layer, ref_index, ref_value = self.__input_bisection(model, lst_poly, x, y0, y, lst_ge)
                # ref_layer, ref_index, ref_value = self.__inner_refinement(model, lst_poly, x, y0, y, lst_ge)
                # ref_layer, ref_index, ref_value = self.__cnt_refinement(model, lst_poly, x, y0, y, lst_ge)
                # ref_layer, ref_index, ref_value = self.__negative_impact_refinement(model, lst_poly, x, y0, y, lst_ge)
                ref_layer, ref_index, ref_value = self.__impact_refinement(model, lst_poly, x, y0, y, lst_ge)

                # this makes res1 and res2 not symmetric
                # there may be the chance res2 can find the true adv while
                # res1 is false with unknown and so res2 is not run
                # the problem is not too important for now because it is rare
                # and we focus on verification
                if ref_layer != -1:
                    lst_poly1, lst_poly2 = self.__refine(model, lst_poly, x, ref_layer, ref_index, ref_value)
                    res1 = self.__verify(model, x0, y0, ref_layer, lst_poly1)
                    if res1 == 0:
                        res2 = self.__verify(model, x0, y0, ref_layer, lst_poly2)
                        return res2
                    else:
                        return res1
                else:
                    return 1 # False, unknown


    def __run(self, model, x0, y0, idx, lst_poly):
        # print('\n############################\n')
        # print('idx = {}'.format(idx))
        # print('xi_poly.lw = {}'.format(lst_poly[idx].lw))
        # print('xi_poly.up = {}'.format(lst_poly[idx].up))

        if idx == len(model.layers):
            x = lst_poly[idx]
            no_neurons = len(x.lw)

            # print('x.lw = {}'.format(x.lw))
            # print('x.up = {}'.format(x.up))

            for y in range(no_neurons):
                if y != y0 and x.lw[y0] <= x.up[y]:
                    res_poly = Poly()

                    res_poly.lw = np.zeros(1)
                    res_poly.up = np.zeros(1)

                    res_poly.le = np.zeros([1, no_neurons + 1])
                    res_poly.ge = np.zeros([1, no_neurons + 1])

                    res_poly.ge[0,y0] = 1
                    res_poly.ge[0,y] = -1

                    lst_le, lst_ge = res_poly.back_substitute(lst_poly, True)
                    ge_x0 = lst_ge[0]

                    # print('y = {}'.format(y))
                    # print('res.lw = {}'.format(res_poly.lw[0]))

                    if res_poly.lw[0] <= 0:
                        res, x = self.__find_sus_adv(ge_x0, x0, lst_poly[0].lw, lst_poly[0].up)
                        return False, x, y, lst_ge

            return True, None, None, None
        else:
            xi_poly_curr = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(xi_poly_curr)
            return self.__run(model, x0, y0, idx + 1, lst_poly)


    def __find_sus_adv(self, ge_x0, x0, lw, up):
        def obj_func(x, ge_x0):
            res = 0
            for i in range(len(x)):
                res += x[i] * ge_x0[i]
            res += ge_x0[-1]
            return res if res >= 0 else np.sum(x - x)

        x = x0.copy()

        args = (ge_x0)
        jac = grad(obj_func)
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    # # input bisection with largest coefs
    # def __input_bisection(self, model, lst_poly, x, y0, y, lst_ge):
    #     best_layer = -1
    #     best_index = -1
    #     best_value = -1
    #     ref_value = 0
    #
    #     for i in range(1):
    #         ge = np.abs(lst_ge[i])
    #         poly_i = lst_poly[i]
    #
    #         for ref_idx in range(len(ge) - 1):
    #             lw = poly_i.lw[ref_idx]
    #             up = poly_i.up[ref_idx]
    #             if lw < up and (best_layer == -1 or best_value < ge[ref_idx]):
    #                 best_layer = i
    #                 best_index = ref_idx
    #                 best_value = ge[ref_idx]
    #                 ref_value = (lw + up) / 2
    #
    #     return best_layer, best_index, ref_value
    #
    #
    # # input or inner neuron refinement with largest coefs
    # def __inner_refinement(self, model, lst_poly, x, y0, y, lst_ge):
    #     best_layer = -1
    #     best_index = -1
    #     best_value = -1
    #     ref_value = 0
    #
    #     for i in range(len(model.layers)):
    #         layer = model.layers[i]
    #
    #         # if not layer.is_poly_exact():
    #         if i == 0 or not layer.is_poly_exact():
    #             ge = np.abs(lst_ge[i])
    #             poly_i = lst_poly[i]
    #
    #             for ref_idx in range(len(ge) - 1):
    #                 lw = poly_i.lw[ref_idx]
    #                 up = poly_i.up[ref_idx]
    #                 # if lw < 0 and up > 0 and (best_layer == -1 or best_value < ge[ref_idx]):
    #                 if ((i == 0 and lw < up) or (lw < 0 and up > 0)) and \
    #                         (best_layer == -1 or best_value < ge[ref_idx]):
    #                     best_layer = i
    #                     best_index = ref_idx
    #                     best_value = ge[ref_idx]
    #                     ref_value = (lw + up) / 2 if i == 0 else 0
    #
    #     return best_layer, best_index, ref_value
    #
    #
    # # count and choose the layer with less number of possibly refined neurons
    # def __cnt_refinement(self, model, lst_poly, x, y0, y, lst_ge):
    #     best_layer = -1
    #     best_cnt = 1e9
    #
    #     for i in range(len(model.layers)):
    #         layer = model.layers[i]
    #
    #         if i == 0 or not layer.is_poly_exact():
    #             cnt = 0
    #             poly_i = lst_poly[i]
    #
    #             for ref_idx in range(len(poly_i.lw)):
    #                 lw = poly_i.lw[ref_idx]
    #                 up = poly_i.up[ref_idx]
    #                 if (i == 0 and lw < up) or (lw < 0 and up > 0):
    #                     cnt = cnt + 1
    #
    #             if best_cnt > cnt and cnt > 0:
    #                 best_layer = i
    #                 best_cnt = cnt
    #
    #     if best_layer == -1:
    #         return -1, -1, -1
    #
    #     best_index = -1
    #     best_value = -1
    #     ref_val = 0
    #
    #     ge = np.abs(lst_ge[best_layer])
    #     poly_i = lst_poly[best_layer]
    #
    #     for ref_idx in range(len(ge) - 1):
    #         lw = poly_i.lw[ref_idx]
    #         up = poly_i.up[ref_idx]
    #         if ((best_layer == 0 and lw < up) or (lw < 0 and up > 0)) and \
    #                 (best_index == -1 or best_value < ge[ref_idx]):
    #             best_index = ref_idx
    #             best_value = ge[ref_idx]
    #             ref_value = (lw + up) / 2 if best_layer == 0 else 0
    #
    #     return best_layer, best_index, ref_value
    #
    #
    # # negative impact
    # def __negative_impact_refinement(self, model, lst_poly, x, y0, y, lst_ge):
    #     best_layer = -1
    #     best_index = -1
    #     best_value = -1
    #     ref_value = 0
    #
    #     for i in range(len(model.layers)):
    #         layer = model.layers[i]
    #         sum_impact = 0
    #
    #         if i == 0 or not layer.is_poly_exact():
    #             poly_i = lst_poly[i]
    #             ge_i = lst_ge[i]
    #
    #             for ref_idx in range(len(poly_i.lw)):
    #                 lw = poly_i.lw[ref_idx]
    #                 up = poly_i.up[ref_idx]
    #                 cf = ge_i[ref_idx]
    #
    #                 if (i == 0 and lw < up) or (lw < 0 and up > 0):
    #                     impact = min(cf * lw, cf * up)
    #                     if impact < 0:
    #                         sum_impact = sum_impact + impact
    #
    #             for ref_idx in range(len(poly_i.lw)):
    #                 lw = poly_i.lw[ref_idx]
    #                 up = poly_i.up[ref_idx]
    #                 cf = ge_i[ref_idx]
    #
    #                 if (i == 0 and lw < up) or (lw < 0 and up > 0):
    #                     impact = min(cf * lw, cf * up)
    #                     if impact < 0:
    #                         norm_impact = impact / sum_impact
    #                         if best_layer == -1 or best_value < norm_impact:
    #                             best_layer = i
    #                             best_index = ref_idx
    #                             best_value = norm_impact
    #                             ref_value = (lw + up) / 2 if i == 0 else 0
    #
    #     return best_layer, best_index, ref_value


    # impact refinement
    def __impact_refinement(self, model, lst_poly, x, y0, y, lst_ge):
        if self.cnt_ref == 0:
            # print('Refine! ')
            print('Refine! ', end='')

        self.cnt_ref += 1

        if self.cnt_ref > self.max_ref: return -1

        best_layer, best_index, best_value, ref_value = -1, -1, -1, 0

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
                            if best_layer == -1 or best_value < norm_impact:
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
