import autograd.numpy as np
import cvxpy as cp


def back_substitute(args):
    idx, le_curr, ge_curr, lst_poly = args
    lst_le = []
    lst_ge = []

    best_lw = -1e9
    best_up = 1e9

    for k, e in reversed(list(enumerate(lst_poly))):
        no_e_ns = len(e.le)
        no_coefs = len(e.le[0])

        lw = 0
        up = 0

        if k > 0:
            le = np.zeros([no_coefs])
            ge = np.zeros([no_coefs])

            for i in range(no_e_ns):
                if le_curr[i] > 0:
                    up = up + le_curr[i] * e.up[i]
                    le = le + le_curr[i] * e.le[i]
                elif le_curr[i] < 0:
                    up = up + le_curr[i] * e.lw[i]
                    le = le + le_curr[i] * e.ge[i]

                if ge_curr[i] > 0:
                    lw = lw + ge_curr[i] * e.lw[i]
                    ge = ge + ge_curr[i] * e.ge[i]
                elif ge_curr[i] < 0:
                    lw = lw + ge_curr[i] * e.up[i]
                    ge = ge + ge_curr[i] * e.le[i]

            lw = lw + ge_curr[-1]
            up = up + le_curr[-1]

            le[-1] = le[-1] + le_curr[-1]
            ge[-1] = ge[-1] + ge_curr[-1]

            best_lw = max(best_lw, lw)
            best_up = min(best_up, up)

            le_curr = le
            ge_curr = ge

            lst_le.insert(0, le_curr)
            lst_ge.insert(0, ge_curr)
        else:
            for i in range(no_e_ns):
                if le_curr[i] > 0:
                    up = up + le_curr[i] * e.up[i]
                elif le_curr[i] < 0:
                    up = up + le_curr[i] * e.lw[i]

                if ge_curr[i] > 0:
                    lw = lw + ge_curr[i] * e.lw[i]
                elif ge_curr[i] < 0:
                    lw = lw + ge_curr[i] * e.up[i]

            lw = lw + ge_curr[-1]
            up = up + le_curr[-1]

            best_lw = max(best_lw, lw)
            best_up = min(best_up, up)

    # return idx, best_lw, best_up, le_curr, ge_curr
    return idx, best_lw, best_up, lst_le, lst_ge


def back_propagate(args):
    idx, x, constraints = args

    objective = cp.Minimize(x[idx])
    prob = cp.Problem(objective, constraints)
    lw_i = round(prob.solve(), 9)

    objective = cp.Minimize(-x[idx])
    prob = cp.Problem(objective, constraints)
    up_i = -round(prob.solve(), 9)

    return idx, lw_i, up_i
