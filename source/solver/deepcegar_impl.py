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

    def back_substitute(self, lst_poly):
        lt_curr = self.lt
        gt_curr = self.gt
        
        no_neurons = len(lt_curr)
        
        if no_neurons <= 100 or len(lst_poly) <= 2:
            for i in range(no_neurons):
                args = (i, lt_curr[i], gt_curr[i], lst_poly)
                _, lw, up = back_substitute(args)
                self.lw[i] = lw
                self.up[i] = up
        else:
            clones = []
            
            for i in range(no_neurons):
                clones.append(lst_poly.copy())
                        
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            for i, lw, up in pool.map(back_substitute, zip(range(no_neurons), self.lt, self.gt, clones)):
                self.lw[i] = lw
                self.up[i] = up


class DeepCegarImpl():
    def __init__(self, max_ref):
        self.count_ref = 0
        self.max_ref = max_ref


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        eps = ast.literal_eval(read(spec['eps']))

        print('x0 = {}'.format(x0))
        print('y0 = {}'.format(y0))

        lw = np.maximum(model.lower, x0 - eps)
        up = np.minimum(model.upper, x0 + eps)

        res, x = self.__validate_x0(model, x0, y0, lw, up)
        if not res:
            y = np.argmax(model.apply(x), axis=1)[0]

            print('True adversarial sample found!')
            print('x = {}'.format(x))
            print('y = {}'.format(y))

            return
            
        x0_poly = Poly()

        x0_poly.lw = lw
        x0_poly.up = up

        x0_poly.lt = np.eye(len(x0) + 1)[0:-1]
        x0_poly.gt = np.eye(len(x0) + 1)[0:-1]

        lst_poly = [x0_poly]
        res = self.__verify(model, x0, y0, x0_poly, 0, lst_poly)

        if res:
            print('The network is robust around x0!')
        else:
            print('Unknown!')


    def __validate_x0(self, model, x0, y0, lw, up):
        x = x0

        args = (model, y0)
        jac = grad(self.__obj_func_x0)
        bounds = Bounds(lw, up)

        res = minimize(self.__obj_func_x0, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)
            
            
    def __obj_func_x0(self, x, model, y0):
        output = model.apply(x)
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        return loss + np.sum(x - x)


    def __verify(self, model, x0, y0, xi_poly_prev, idx, lst_poly):
        if idx == len(model.layers):
            x = xi_poly_prev
            no_neurons = len(x.lw)

            print('lw = {}'.format(x.lw))
            print('up = {}'.format(x.up))

            for lbl in range(no_neurons):
                if lbl != y0 and x.lw[y0] <= x.up[lbl]:
                    res_poly = Poly()

                    res_poly.lw = np.zeros(1)
                    res_poly.up = np.zeros(1)

                    res_poly.lt = np.zeros([1, no_neurons + 1])
                    res_poly.gt = np.zeros([1, no_neurons + 1])

                    res_poly.gt[0,y0] = 1
                    res_poly.gt[0,lbl] = -1

                    res_poly.back_substitute(lst_poly)

                    if res_poly.lw[0] < 0: return False

            return True
        else:
            xi_poly_curr = model.forward(xi_poly_prev, idx, lst_poly)

            if model.layers[idx].is_poly_exact():
                lst_poly.append(xi_poly_curr)
                return self.__verify(model, x0, y0, xi_poly_curr, idx + 1, lst_poly)

            res, x = self.__validate(model, x0, y0, xi_poly_curr, idx + 1)
            print('res = {}'.format(res))
            print('x = {}'.format(x))
            
            if not res:
                # a counter example is found, should be fake
                print('Fake adversarial sample found!')

                if self.count_ref >= self.max_ref:
                    return False
                else:
                    self.count_ref = self.count_ref + 1

                tmp = model.apply_to(x0, idx + 1)
                xi = x.reshape(tmp.shape)

                g = grad(model.apply_from)(xi, idx + 1, y0=y0)
                ref_ord = np.flip(np.argsort(g)).reshape(-1)
                
                func = model.layers[idx].func

                xi_poly_prev1, xi_poly_prev2 = self.__refine(xi_poly_prev, ref_ord, x, func)
                
                if xi_poly_prev1 == None or xi_poly_prev2 == None:
                    return False

                lst_poly1 = lst_poly[0:-1].copy()
                lst_poly2 = lst_poly[0:-1].copy()

                lst_poly1.append(xi_poly_prev1)
                lst_poly2.append(xi_poly_prev2)

                if self.__verify(model, x0, y0, xi_poly_prev1, idx, lst_poly1):
                    return self.__verify(model, x0, y0, xi_poly_prev2, idx, lst_poly2)
                else:
                    return False
            else:
                lst_poly.append(xi_poly_curr)
                return self.__verify(model, x0, y0, xi_poly_curr, idx + 1, lst_poly)
            
    
    
    def __refine(self, x_poly, ref_ord, x, func):
        lw = x_poly.lw
        up = x_poly.up
        
        ref_idx = -1
    
        if func == relu:
            for i in ref_ord:
                if lw[i] < 0 and up[i] > 0:
                    ref_idx = i
                    break
        else:
            for i in ref_ord:
                if lw[i] != up[i]:
                    ref_idx = idx
                    break
                    
        print('ref_idx = {}'.format(ref_idx))
        
        if ref_idx != -1:
            x1_poly = Poly()
            x2_poly = Poly()

            x1_poly.lw = x_poly.lw.copy()
            x1_poly.up = x_poly.up.copy()
            x1_poly.lt = x_poly.lt.copy()
            x1_poly.gt = x_poly.gt.copy()

            x2_poly.lw = x_poly.lw.copy()
            x2_poly.up = x_poly.up.copy()
            x2_poly.lt = x_poly.lt.copy()
            x2_poly.gt = x_poly.gt.copy()
            
            if func == relu:
                val = 0
            elif func == sigmoid:
                val = np.log(x_idx / (1 - x_idx))
            elif func == tanh:
                val = 0.5 * np.log((1 + x_idx) / (1 - x_idx))

            x1_poly.up[ref_idx] = val
            x2_poly.lw[ref_idx] = val
            
            return x1_poly, x2_poly
        else:
            return None, None
        
            
    def __validate(self, model, x0, y0, xi_poly, idx):
        leni = len(xi_poly.lw)
        x = np.zeros(leni)

        args = (model, x0, y0, idx)
        jac = grad(self.__obj_func)

        lw = np.concatenate([xi_poly.lw])
        up = np.concatenate([xi_poly.up])
        bounds = Bounds(lw, up)

        res = minimize(self.__obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __obj_func(self, x, model, x0, y0, idx):
        tmp = model.apply_to(x0, idx)

        x = x.reshape(tmp.shape)
        output = model.apply_from(x, idx)
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        return loss + np.sum(x - x)
       
       
    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
