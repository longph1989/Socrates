import autograd.numpy as np


def back_substitute(args):
    idx, lt_curr, gt_curr, lst_poly = args
    
    best_lw = -1e9
    best_up = 1e9
    
    for k, e in reversed(list(enumerate(lst_poly))):
        lw = 0
        up = 0

        no_e_ns = len(e.lt)
        no_coefs = len(e.lt[0])

        if k > 0:
            lt = np.zeros([no_coefs])
            gt = np.zeros([no_coefs])

            for i in range(no_e_ns):
                if lt_curr[i] > 0:
                    up = up + lt_curr[i] * e.up[i]
                    lt = lt + lt_curr[i] * e.lt[i]
                elif lt_curr[i] < 0:
                    up = up + lt_curr[i] * e.lw[i]
                    lt = lt + lt_curr[i] * e.gt[i]

                if gt_curr[i] > 0:
                    lw = lw + gt_curr[i] * e.lw[i]
                    gt = gt + gt_curr[i] * e.gt[i]
                elif gt_curr[i] < 0:
                    lw = lw + gt_curr[i] * e.up[i]
                    gt = gt + gt_curr[i] * e.lt[i]

            lw = lw + gt_curr[-1]
            up = up + lt_curr[-1]

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
            
            lw = lw + gt_curr[-1]
            up = up + lt_curr[-1]
            
            best_lw = max(best_lw, lw)
            best_up = min(best_up, up)

    return idx, best_lw, best_up, lt_curr, gt_curr