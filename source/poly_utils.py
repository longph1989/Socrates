import autograd.numpy as np
import time


def back_substitute(args):
    idx, lt_curr, gt_curr, lst_poly = args
    
    lw = 0
    up = 0
    
    t0 = time.time()
    for k, e in reversed(list(enumerate(lst_poly))):
        no_e_ns = len(e.lt)
        no_coefs = len(e.lt[0])

        if k > 0:
            lt = np.zeros([no_coefs])
            gt = np.zeros([no_coefs])

            for i in range(no_e_ns):
                if lt_curr[i] > 0:
                    lt = lt + lt_curr[i] * e.lt[i]
                elif lt_curr[i] < 0:
                    lt = lt + lt_curr[i] * e.gt[i]

                if gt_curr[i] > 0:
                    gt = gt + gt_curr[i] * e.gt[i]
                elif gt_curr[i] < 0:
                    gt = gt + gt_curr[i] * e.lt[i]

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
            
    t1 = time.time()
    # print('time = {}'.format(t1 - t0))

    return idx, lw, up, lt_curr, gt_curr