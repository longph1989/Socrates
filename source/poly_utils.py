import autograd.numpy as np
import cvxpy as cp

def back_substitute(args):
    idx, le_curr, ge_curr, lst_poly = args

    lst_le, lst_ge = [le_curr], [ge_curr]
    best_lw, best_up = -1e9, 1e9

    for k, e in reversed(list(enumerate(lst_poly))):
        no_e_ns = len(e.lw)

        max_le_curr = np.maximum(le_curr[:-1], 0)
        min_le_curr = np.minimum(le_curr[:-1], 0)

        max_ge_curr = np.maximum(ge_curr[:-1], 0)
        min_ge_curr = np.minimum(ge_curr[:-1], 0)

        max_le_n0id = np.nonzero(max_le_curr)[0]
        min_le_n0id = np.nonzero(min_le_curr)[0]

        max_ge_n0id = np.nonzero(max_ge_curr)[0]
        min_ge_n0id = np.nonzero(min_ge_curr)[0]

        lw, up = ge_curr[-1], le_curr[-1]

        if len(max_ge_n0id) > 0:
            lw += np.sum(max_ge_curr[max_ge_n0id] * e.lw[max_ge_n0id])
        if len(min_ge_n0id) > 0:
            lw += np.sum(min_ge_curr[min_ge_n0id] * e.up[min_ge_n0id])

        if len(max_le_n0id) > 0:
            up += np.sum(max_le_curr[max_le_n0id] * e.up[max_le_n0id])
        if len(min_le_n0id) > 0:
            up += np.sum(min_le_curr[min_le_n0id] * e.lw[min_le_n0id])

        best_lw = max(best_lw, lw)
        best_up = min(best_up, up)

        if k > 0:
            no_coefs = len(e.le[0])

            le = np.zeros([no_coefs])
            ge = np.zeros([no_coefs])

            threshold = 1e3

            if len(max_le_n0id) > 0:
                if e.is_activation and no_coefs > threshold:
                    for i in max_le_n0id:
                        le[i] += max_le_curr[i] * e.le[i,i]
                        le[-1] += max_le_curr[i] * e.le[i,-1]        
                else:
                    le += np.sum(max_le_curr[max_le_n0id].reshape(len(max_le_n0id), 1) * e.le[max_le_n0id], axis=0)
            
            if len(min_le_n0id) > 0:
                if e.is_activation and no_coefs > threshold:
                    for i in min_le_n0id:
                        le[i] += min_le_curr[i] * e.ge[i,i]
                        le[-1] += min_le_curr[i] * e.ge[i,-1]
                else:
                    le += np.sum(min_le_curr[min_le_n0id].reshape(len(min_le_n0id), 1) * e.ge[min_le_n0id], axis=0)

            if len(max_ge_n0id) > 0:
                if e.is_activation and no_coefs > threshold:
                    for i in max_ge_n0id:
                        ge[i] += max_ge_curr[i] * e.ge[i,i]
                        ge[-1] += max_ge_curr[i] * e.ge[i,-1] 
                else:
                    ge += np.sum(max_ge_curr[max_ge_n0id].reshape(len(max_ge_n0id), 1) * e.ge[max_ge_n0id], axis=0)

            if len(min_ge_n0id) > 0:
                if e.is_activation and no_coefs > threshold:
                    for i in min_ge_n0id:
                        ge[i] += min_ge_curr[i] * e.le[i,i]
                        ge[-1] += min_ge_curr[i] * e.le[i,-1] 
                else:
                    ge += np.sum(min_ge_curr[min_ge_n0id].reshape(len(min_ge_n0id), 1) * e.le[min_ge_n0id], axis=0)
            
            le[-1] = le[-1] + le_curr[-1]
            ge[-1] = ge[-1] + ge_curr[-1]

            le_curr, ge_curr = le, ge

            lst_le.insert(0, le_curr)
            lst_ge.insert(0, ge_curr)

    return idx, best_lw, best_up, lst_le, lst_ge


def input_tighten(args):
    idx, x, constraints, lw_i, up_i = args

    if lw_i == up_i: return idx, lw_i, up_i

    objective = cp.Minimize(x[idx])
    problem = cp.Problem(objective, constraints)
    lw_i = round(problem.solve(solver=cp.CBC), 9)

    objective = cp.Minimize(-x[idx])
    problem = cp.Problem(objective, constraints)
    up_i = -round(problem.solve(solver=cp.CBC), 9)

    return idx, lw_i, up_i

