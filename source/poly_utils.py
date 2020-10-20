import autograd.numpy as np


def back_substitute(args):
    idx, lt_curr, gt_curr, lst_poly = args
    
    lw = 0
    up = 0
    
    for k, e in reversed(list(enumerate(lst_poly))):
        lt_prev = e.lt
        gt_prev = e.gt

        no_e_ns = len(lt_prev)
        no_coefs = len(lt_prev[0])

        if k > 0:
            lt = np.zeros([no_coefs])
            gt = np.zeros([no_coefs])

            for i in range(no_e_ns):
                if lt_curr[i] > 0:
                    lt = lt + lt_curr[i] * lt_prev[i]
                elif lt_curr[i] < 0:
                    lt = lt + lt_curr[i] * gt_prev[i]

                if gt_curr[i] > 0:
                    gt = gt + gt_curr[i] * gt_prev[i]
                elif gt_curr[i] < 0:
                    gt = gt + gt_curr[i] * lt_prev[i]

            lt[-1] = lt[-1] + lt_curr[-1]
            gt[-1] = gt[-1] + gt_curr[-1]

            lt_curr = lt
            gt_curr = gt
        else:
            for i in range(no_e_ns):
                if lt_curr[i] > 0:
                    up = up + lt_curr[i] * e.up[i]
                elif lt_curr[i] < 0:
                    up = up + lt_curr[i] * e.lw[i]
            
                if gt_curr[i] > 0:
                    lw = lw + gt_curr[i] * e.lw[i]
                elif gt_curr[i] < 0:
                    lw = lw + gt_curr[i] * e.up[i]
            
            up = up + lt_curr[-1]
            lw = lw + gt_curr[-1]

    return idx, lw, up