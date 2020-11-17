import autograd.numpy as np
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

        self.lt = None
        self.gt = None

    def copy(self):
        new_poly = Poly()

        new_poly.lw = self.lw.copy()
        new_poly.up = self.up.copy()

        new_poly.lt = self.lt.copy()
        new_poly.gt = self.gt.copy()

        return new_poly

    def back_substitute(self, lst_poly, get_ineq = False):
        lt_curr = self.lt
        gt_curr = self.gt

        no_neurons = len(lt_curr)

        if get_ineq:
            no_neurons_x0 = len(lst_poly[0].lw)

            lt_x0 = np.zeros([no_neurons, no_neurons_x0 + 1])
            gt_x0 = np.zeros([no_neurons, no_neurons_x0 + 1])

        if no_neurons <= 100 or len(lst_poly) <= 2:
            for i in range(no_neurons):
                args = (i, lt_curr[i], gt_curr[i], lst_poly)
                _, lw, up, lt, gt = back_substitute(args)
                self.lw[i] = lw
                self.up[i] = up

                if get_ineq:
                    lt_x0[i] = lt
                    gt_x0[i] = gt
        else:
            clones = []

            for i in range(no_neurons):
                clones.append(lst_poly)

            num_cores = 4
            pool = multiprocessing.Pool(num_cores)
            for i, lw, up, lt, gt in pool.map(back_substitute, zip(range(no_neurons), self.lt, self.gt, clones)):
                self.lw[i] = lw
                self.up[i] = up

                if get_ineq:
                    lt_x0[i] = lt
                    gt_x0[i] = gt
            pool.close()

        if get_ineq: return lt_x0, gt_x0


class DeepCegarImpl():
    def __init__(self, max_ref):
        self.has_ref = False
        self.max_ref = 20
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
            y = np.argmax(model.apply(x), axis=1)[0]

            print('True adversarial sample found!')
            print('x = {}'.format(x))
            print('y = {}'.format(y))
        else:
            x0_poly = Poly()

            x0_poly.lw = lw
            x0_poly.up = up

            x0_poly.lt = np.eye(len(x0) + 1)[0:-1]
            x0_poly.gt = np.eye(len(x0) + 1)[0:-1]

            lst_poly = [x0_poly]

            if self.__verify(model, x0, y0, 0, lst_poly):
                print('The network is robust around x0!')
            else:
                print('Unknown!')


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


    def __verify(self, model, x0, y0, idx, lst_poly):
        res, x = self.__run(model, x0, y0, idx, lst_poly)

        if res:
            return True
        elif not self.has_ref or len(x) == 0:
            return False
        else:
            y = np.argmax(model.apply(x), axis=1)[0]

            if y != y0:
                print('True adversarial sample found!')
                print('x = {}'.format(x))
                print('y = {}'.format(y))

                return False
            else:
                ref_layer, ref_index, ref_value = self.__choose_refinement(model, lst_poly, x)
                lst_poly1, lst_poly2 = self.__refine(model, lst_poly, x, ref_layer, ref_index, ref_value)

                if lst_poly1 != None and lst_poly2 != None:
                    if self.__verify(model, x0, y0, ref_layer, lst_poly1):
                        return self.__verify(model, x0, y0, ref_layer, lst_poly2)
                    else:
                        return False
                else:
                    return False


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

            for lbl in range(no_neurons):
                if lbl != y0 and x.lw[y0] <= x.up[lbl]:
                    res_poly = Poly()

                    res_poly.lw = np.zeros(1)
                    res_poly.up = np.zeros(1)

                    res_poly.lt = np.zeros([1, no_neurons + 1])
                    res_poly.gt = np.zeros([1, no_neurons + 1])

                    res_poly.gt[0,y0] = 1
                    res_poly.gt[0,lbl] = -1

                    lt_x0, gt_x0 = res_poly.back_substitute(lst_poly, True)

                    # print('res.lw = {}'.format(res_poly.lw[0]))
                    # print('lt_x0 = {}'.format(lt_x0))
                    # print('gt_x0 = {}'.format(gt_x0))

                    if res_poly.lw[0] < 0:
                        res, x = self.__find_sus_adv(gt_x0.reshape(-1), x0, lst_poly[0].lw, lst_poly[0].up)
                        return False, x

            return True, np.empty(0)
        else:
            xi_poly_curr = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(xi_poly_curr)
            return self.__run(model, x0, y0, idx + 1, lst_poly)


    def __find_sus_adv(self, gt_x0, x0, lw, up):
        def obj_func(x, gt_x0):
            res = 0
            for i in range(len(x)):
                res += x[i] * gt_x0[i]
            res += gt_x0[-1]
            return res if res >= 0 else np.sum(x - x)

        x = x0.copy()

        args = (gt_x0)
        jac = grad(obj_func)
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __choose_refinement1(model, lst_poly, x):
        best_layer = -1
        best_index = -1
        best_value = -1
        ref_value = 0

        for i in range(len(model.layers)):
            layer = model.layers[i]

            if not layer.is_poly_exact():
                tmp = model.apply_to(x, i)
                g = grad(model.apply_from)(tmp, i, y0=y0).reshape(-1)
                g = np.abs(g)

                poly_i = lst_poly[i]

                for ref_idx in range(len(g)):
                    if poly_i.lw[ref_idx] < 0 and poly_i.up[ref_idx] > 0:
                        if best_layer == -1 or best_value < g[ref_idx]:
                            best_layer = i
                            best_index = ref_idx
                            best_value = g[ref_idx]

        return best_layer, best_index, ref_value


    def __choose_refinement2(model, lst_poly, x):
        best_layer = -1
        best_index = -1
        best_value = -1
        ref_value = 0

        for i in range(1):
            layer = model.layers[i]

            g = grad(model.apply)(x, y0=y0).reshape(-1)
            g = np.abs(g)

            poly_i = lst_poly[i]

            for ref_idx in range(len(g)):
                if poly_i.lw[ref_idx] < poly_i.up[ref_idx]:
                    if best_layer == -1 or best_value < g[ref_idx]:
                        best_layer = i
                        best_index = ref_idx
                        best_value = g[ref_idx]
                        ref_value = (poly_i.lw[ref_idx] + poly_i.up[ref_idx]) / 2

        return best_layer, best_index, ref_value


    def __refine(self, model, lst_poly, x, ref_layer, ref_index, ref_value):
        if self.cnt_ref == 0:
            print('Refine! ', end='')

        self.cnt_ref += 1

        if self.cnt_ref > self.max_ref or best_layer == -1:
            return None, None

        lst_poly1 = []
        lst_poly2 = []

        for i in range(ref_layer):
            lst_poly1.append(lst_poly[i].copy())
            lst_poly2.append(lst_poly[i].copy())

        x1_poly = Poly()
        x2_poly = Poly()

        x1_poly.lw = lst_poly[ref_layer].lw.copy()
        x1_poly.up = lst_poly[ref_layer].up.copy()
        x1_poly.lt = lst_poly[ref_layer].lt.copy()
        x1_poly.gt = lst_poly[ref_layer].gt.copy()

        x2_poly.lw = lst_poly[ref_layer].lw.copy()
        x2_poly.up = lst_poly[ref_layer].up.copy()
        x2_poly.lt = lst_poly[ref_layer].lt.copy()
        x2_poly.gt = lst_poly[ref_layer].gt.copy()

        x1_poly.up[ref_index] = ref_value
        x2_poly.lw[ref_index] = ref_value

        lst_poly1.append(x1_poly)
        lst_poly2.append(x2_poly)

        return lst_poly1, lst_poly2


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
