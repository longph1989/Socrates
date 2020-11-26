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

    def back_substitute(self, lst_poly, get_ineq=False):
        lt_curr = self.lt
        gt_curr = self.gt

        no_neurons = len(lt_curr)

        if no_neurons <= 100 or len(lst_poly) <= 2:
            for i in range(no_neurons):
                args = (i, lt_curr[i], gt_curr[i], lst_poly)
                _, lw, up, lt, gt = back_substitute(args)
                self.lw[i] = lw
                self.up[i] = up

                # get_ineq only happens at the last step
                # no_neurons in this case always be 1
                if get_ineq:
                    lst_lt = lt
                    lst_gt = gt
        else:
            clones = []

            for i in range(no_neurons):
                clones.append(lst_poly)

            num_cores = 4
            pool = multiprocessing.Pool(num_cores)
            zz = zip(range(no_neurons), self.lt, self.gt, clones)
            for i, lw, up, _, _ in pool.map(back_substitute, zz):
                self.lw[i] = lw
                self.up[i] = up
            pool.close()

        if get_ineq: return lst_lt, lst_gt


class DeepCegarImpl():
    def __init__(self, max_ref):
        self.has_ref = True
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
            self.__validate_adv(model, x, y0)
        else:
            x0_poly = Poly()

            x0_poly.lw = lw
            x0_poly.up = up

            x0_poly.lt = np.eye(len(x0) + 1)[0:-1]
            x0_poly.gt = np.eye(len(x0) + 1)[0:-1]

            lst_poly = [x0_poly]

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


    def __verify(self, model, x0, y0, idx, lst_poly):
        res, x, y, lst_gt = self.__run(model, x0, y0, idx, lst_poly)

        if res:
            return 0 # True, robust
        elif not self.has_ref:
            return 1 # False, unknown
        else:
            if self.__validate_adv(model, x, y0):
                return 2 # False, with adv
            else:
                # ref_layer, ref_index, ref_value = self.__input_bisection(model, lst_poly, x, y0, y, lst_gt)
                # ref_layer, ref_index, ref_value = self.__inner_refinement(model, lst_poly, x, y0, y, lst_gt)
                # ref_layer, ref_index, ref_value = self.__cnt_refinement(model, lst_poly, x, y0, y, lst_gt)
                # ref_layer, ref_index, ref_value = self.__negative_impact_refinement(model, lst_poly, x, y0, y, lst_gt)
                ref_layer, ref_index, ref_value = self.__total_impact_refinement(model, lst_poly, x, y0, y, lst_gt)
                lst_poly1, lst_poly2 = self.__refine(model, lst_poly, x, ref_layer, ref_index, ref_value)

                # this makes res1 and res2 not symmetric
                # there may be the chance res2 can find the true adv while
                # res1 is false with unknown and so res2 is not run
                # the problem is not too important for now because it is rare
                # and we focus on verification
                if lst_poly1 != None and lst_poly2 != None:
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

                    res_poly.lt = np.zeros([1, no_neurons + 1])
                    res_poly.gt = np.zeros([1, no_neurons + 1])

                    res_poly.gt[0,y0] = 1
                    res_poly.gt[0,y] = -1

                    lst_lt, lst_gt = res_poly.back_substitute(lst_poly, True)
                    gt_x0 = lst_gt[0]

                    # print('y = {}'.format(y))
                    # print('res.lw = {}'.format(res_poly.lw[0]))

                    if res_poly.lw[0] <= 0:
                        res, x = self.__find_sus_adv(gt_x0, x0, lst_poly[0].lw, lst_poly[0].up)
                        return False, x, y, lst_gt

            return True, None, None, None
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


    # input bisection with largest coefs
    def __input_bisection(self, model, lst_poly, x, y0, y, lst_gt):
        best_layer = -1
        best_index = -1
        best_value = -1
        ref_value = 0

        for i in range(1):
            gt = np.abs(lst_gt[i])
            poly_i = lst_poly[i]

            for ref_idx in range(len(gt) - 1):
                lw = poly_i.lw[ref_idx]
                up = poly_i.up[ref_idx]
                if lw < up and (best_layer == -1 or best_value < gt[ref_idx]):
                    best_layer = i
                    best_index = ref_idx
                    best_value = gt[ref_idx]
                    ref_value = (lw + up) / 2

        return best_layer, best_index, ref_value


    # input or inner neuron refinement with largest coefs
    def __inner_refinement(self, model, lst_poly, x, y0, y, lst_gt):
        best_layer = -1
        best_index = -1
        best_value = -1
        ref_value = 0

        for i in range(len(model.layers)):
            layer = model.layers[i]

            # if not layer.is_poly_exact():
            if i == 0 or not layer.is_poly_exact():
                gt = np.abs(lst_gt[i])
                poly_i = lst_poly[i]

                for ref_idx in range(len(gt) - 1):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    # if lw < 0 and up > 0 and (best_layer == -1 or best_value < gt[ref_idx]):
                    if ((i == 0 and lw < up) or (lw < 0 and up > 0)) and \
                            (best_layer == -1 or best_value < gt[ref_idx]):
                        best_layer = i
                        best_index = ref_idx
                        best_value = gt[ref_idx]
                        ref_value = (lw + up) / 2 if i == 0 else 0

        return best_layer, best_index, ref_value


    # count and choose the layer with less number of possibly refined neurons
    def __cnt_refinement(self, model, lst_poly, x, y0, y, lst_gt):
        best_layer = -1
        best_cnt = 1e9

        for i in range(len(model.layers)):
            layer = model.layers[i]

            if i == 0 or not layer.is_poly_exact():
                cnt = 0
                poly_i = lst_poly[i]

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    if (i == 0 and lw < up) or (lw < 0 and up > 0):
                        cnt = cnt + 1

                if best_cnt > cnt and cnt > 0:
                    best_layer = i
                    best_cnt = cnt

        if best_layer == -1:
            return -1, -1, -1

        best_index = -1
        best_value = -1
        ref_val = 0

        gt = np.abs(lst_gt[best_layer])
        poly_i = lst_poly[best_layer]

        for ref_idx in range(len(gt) - 1):
            lw = poly_i.lw[ref_idx]
            up = poly_i.up[ref_idx]
            if ((best_layer == 0 and lw < up) or (lw < 0 and up > 0)) and \
                    (best_index == -1 or best_value < gt[ref_idx]):
                best_index = ref_idx
                best_value = gt[ref_idx]
                ref_value = (lw + up) / 2 if best_layer == 0 else 0

        return best_layer, best_index, ref_value


    # negative impact
    def __negative_impact_refinement(self, model, lst_poly, x, y0, y, lst_gt):
        best_layer = -1
        best_index = -1
        best_value = -1
        ref_value = 0

        for i in range(len(model.layers)):
            layer = model.layers[i]
            sum_impact = 0

            if i == 0 or not layer.is_poly_exact():
                poly_i = lst_poly[i]
                gt_i = lst_gt[i]

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    cf = gt_i[ref_idx]

                    if (i == 0 and lw < up) or (lw < 0 and up > 0):
                        impact = min(cf * lw, cf * up)
                        if impact < 0:
                            sum_impact = sum_impact + impact

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    cf = gt_i[ref_idx]

                    if (i == 0 and lw < up) or (lw < 0 and up > 0):
                        impact = min(cf * lw, cf * up)
                        if impact < 0:
                            norm_impact = impact / sum_impact
                            if best_layer == -1 or best_value < norm_impact:
                                best_layer = i
                                best_index = ref_idx
                                best_value = norm_impact
                                ref_value = (lw + up) / 2 if i == 0 else 0

        return best_layer, best_index, ref_value


    # total impact
    def __total_impact_refinement(self, model, lst_poly, x, y0, y, lst_gt):
        best_layer = -1
        best_index = -1
        best_value = -1
        ref_value = 0

        for i in range(len(model.layers)):
            layer = model.layers[i]
            sum_impact = 0

            if i == 0 or not layer.is_poly_exact():
                poly_i = lst_poly[i]
                gt_i = lst_gt[i]

                for ref_idx in range(len(poly_i.lw)):
                    lw = poly_i.lw[ref_idx]
                    up = poly_i.up[ref_idx]
                    cf = gt_i[ref_idx]

                    if (i == 0 and lw < up) or (lw < 0 and up > 0):
                        impact = max(abs(cf * lw), abs(cf * up))
                        sum_impact = sum_impact + impact

                if sum_impact > 0:
                    for ref_idx in range(len(poly_i.lw)):
                        lw = poly_i.lw[ref_idx]
                        up = poly_i.up[ref_idx]
                        cf = gt_i[ref_idx]

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
        if self.cnt_ref == 0:
            # print('Refine! ')
            print('Refine! ', end='')

        self.cnt_ref += 1

        # print('ref_layer = {}'.format(ref_layer))
        # print('ref_index = {}'.format(ref_index))
        # print('ref_value = {}'.format(ref_value))

        if self.cnt_ref > self.max_ref or ref_layer == -1:
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
