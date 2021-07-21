import numpy as np


# from ..util.log import *

# Returns the (epsilon, delta) values of the adaptive concentration inequality
# for the given parameters.
#
# n: int (number of samples)
# delta: float (parameter delta)
# return: epsilon (parameter epsilon)
def get_type(n, delta):
    n = np.float(n)
    b = -np.log(delta / 24.0) / 1.8
    epsilon = np.sqrt((0.6 * np.log(np.log(n) / np.log(1.1) + 1) + b) / n)
    return epsilon


# Run type inference to get the type judgement for the fairness criterion.
#
# c: float (minimum probability ratio to be fair)
# Delta: float (threshold on inequalities)
# n: int (number of samples)
# delta: float (parameter delta)
# E_A: float (estimate of p(offer | female))
# E_B: float (estimate of p(offer | male))
# is_causal: bool (whether to use the causal specification)
# is_log: bool (whether to print logging information)
# return: int * int | None (None - continue sampling, or (fairness, ambiguous) with 0 - not fair, 1 - fair, 0 - not ambiguous, 1 - ambiguous)
def get_fairness_type(c, Delta, n, delta, E_A, E_B, is_causal, is_log):
    # Step 1: Get (epsilon, delta) values from the adaptive concentration inequality for the current number of samples
    epsilon = get_type(n, delta)

    # Step 2: Check if |E_B| > epsilon_B
    if np.abs(E_B) <= epsilon:
        return None

    # logging
    if is_log:
        # log('INFO: n = {}, E_A = {}, E_B = {}'.format(n, E_A, E_B), INFO)
        print('INFO: n = {}, E_A = {}, E_B = {}'.format(n, E_A, E_B))

    # Step 3: Compute the type judgement for the fairness property (before the inequality)
    if is_causal:
        E_fair = E_A - E_B + c
        epsilon_fair = 2.0 * epsilon
        delta_fair = 2.0 * delta
    else:
        E_fair = E_A / E_B - (1 - c)
        epsilon_fair = epsilon / np.abs(E_B) + epsilon * (epsilon + np.abs(E_A)) / (
                    np.abs(E_B) * (np.abs(E_B) - epsilon))
        delta_fair = 2.0 * delta

    # logging
    if is_log:
        # log('INFO: n = {}, E_fair = {}, epsilon_fair = {}, delta_fair = {}'.format(n, E_fair, epsilon_fair, delta_fair), INFO)
        print(
            'INFO: n = {}, E_fair = {}, epsilon_fair = {}, delta_fair = {}'.format(n, E_fair, epsilon_fair, delta_fair))

    # Step 4: Check if fairness holds
    if E_fair - epsilon_fair > 0:
        return 1, 0

    # Step 5: Check if fairness does not hold
    if E_fair + epsilon_fair < 0:
        return 0, 0

    # Step 6: Check if fairness holds (ambiguously)
    if E_fair - epsilon_fair >= -Delta and epsilon_fair <= Delta:
        return 1, 1

    # Step 7: Check if fairness does not hold (ambiguously)
    if E_fair + epsilon_fair <= Delta and epsilon_fair <= Delta:
        return 0, 1

    # Step 6: Continue sampling
    return None


# model_A: int -> np.array([int]) (the model p(offer | female), takes number of samples as a parameter, output is {0, 1})
# model_B: int -> np.array([int]) (the model p(offer | male), takes number of samples as a parameter, output is {0, 1})
# c: float (minimum probability ratio to be fair)
# Delta: float (threshold on inequalities)
# delta: float (parameter delta)
# n_samples: int (number of samples per iteration)
# n_max: int (maximum number of iterations)
# is_causal: bool (whether to use the causal specification)
# int: log_iters
# return: (bool, int) (is fair, number of samples)
def verify(model_A, model_B, c, Delta, delta, n_samples, n_max, is_causal, log_iters):
    # Step 1: Initialization
    nE_A = 0.0
    nE_B = 0.0

    # Step 2: Iteratively sample and check whether fairness holds
    for i in range(n_max):
        # Step 2a: Sample points
        x = np.sum(model_A.sample(n_samples))
        y = np.sum(model_B.sample(n_samples))

        # Step 2b: Update statistics
        nE_A += x
        nE_B += y

        # Step 2c: Normalized statistics
        n = (i + 1) * n_samples
        E_A = nE_A / n
        E_B = nE_B / n

        # logging
        is_log = not log_iters is None and i % log_iters == 0

        # Step 2d: Get type judgement
        t = get_fairness_type(c, Delta, n, delta, E_A, E_B, is_causal, is_log)

        # Step 2e: Return if converged
        if not t is None:
            return t + (2 * n, E_A / E_B)

    # Step 3: Failed to verify after maximum number of samples
    return None
