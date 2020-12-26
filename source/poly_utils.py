import autograd.numpy as np
import cvxpy as cp


def back_substitute(args):
    idx, le_curr, ge_curr, lst_poly = args

    lst_le, lst_ge = [le_curr], [ge_curr]
    best_lw, best_up = -1e9, 1e9

    for k, e in reversed(list(enumerate(lst_poly))):
        no_e_ns = len(e.lw)
        lw, up = 0, 0

        if k > 0:
            no_coefs = len(e.le[0])

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

            le_curr, ge_curr = le, ge

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
