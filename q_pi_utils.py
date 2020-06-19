import numpy as np
from piecewise.environment import (make_frozen_lake_8x8_train_env,
                                   make_frozen_lake_12x12_train_env)

NUM_ACTIONS = 4
ACTION_MAP = {0: "L", 1: "D", 2: "R", 3: "U"}

# blacklists for terminal states (y, x) coords
OLD_BLACKLISTED_STATES_8X8 = [(2, 3), (3, 5), (4, 3), (5, 1), (5, 2), (5, 6),
        (6, 1), (6, 4), (6, 6), (7, 3), (7, 7)]
env8 = make_frozen_lake_8x8_train_env()
BLACKLISTED_STATES_8X8 = [(y, x) for (x, y) in env8.terminal_states]
assert OLD_BLACKLISTED_STATES_8X8 == BLACKLISTED_STATES_8X8

env12 = make_frozen_lake_12x12_train_env()
BLACKLISTED_STATES_12X12 = [(y, x) for (x, y) in env12.terminal_states]


def calc_lcs_q_hat(lcs, num_x_pos, num_y_pos):
    """Calc q hat of lcs, with non-defined q-vals as NAN"""
    lcs_q_hat = np.full(shape=(num_y_pos, num_x_pos, NUM_ACTIONS),
                        fill_value=np.nan)
    blacklist = get_blacklist_for_shape(lcs_q_hat.shape)
    for x_pos in range(0, num_x_pos):
        for y_pos in range(0, num_y_pos):
            if (y_pos, x_pos) not in blacklist:
                situation = [x_pos, y_pos]
                match_set = lcs.gen_match_set(situation)
                prediction_arr = lcs.gen_prediction_array(match_set, situation)
                for action, pred in prediction_arr.items():
                    lcs_q_hat[y_pos, x_pos, action] = pred
    return lcs_q_hat


def get_hole_counts(lcs_q_hat):
    """Returns 5-tuple, indicating number of holes for each sub-q-func (each
    action), and last entry the total number."""
    (num_y_pos, num_x_pos, _) = lcs_q_hat.shape
    blacklist = get_blacklist_for_shape(lcs_q_hat.shape)
    hole_counts = []
    for action in range(NUM_ACTIONS):
        holes_for_action = 0
        for x_pos in range(0, num_x_pos):
            for y_pos in range(0, num_y_pos):
                if (y_pos, x_pos) not in blacklist:
                    if np.isnan(lcs_q_hat[y_pos, x_pos, action]):
                        holes_for_action += 1
        hole_counts.append(holes_for_action)
    hole_counts.append(sum(hole_counts)) 
    return tuple(hole_counts)


def calc_pi_from_q(q):
    """Calc pi of q in binary encoding."""
    (num_y_pos, num_x_pos, _) = q.shape
    blacklist = get_blacklist_for_shape(q.shape)
    pi = np.full(shape=(num_y_pos, num_x_pos, NUM_ACTIONS), fill_value=np.nan)
    for x_pos in range(0, num_x_pos):
        for y_pos in range(0, num_y_pos):
            if (y_pos, x_pos) not in blacklist:
                q_vals = q[y_pos, x_pos, :]
                assert len(q_vals) == NUM_ACTIONS
                assert np.nan not in q_vals
                max_q = np.max(q_vals)
                for action in range(NUM_ACTIONS):
                    pi[y_pos, x_pos, action] = int(q_vals[action] == max_q)
    return pi


def prettify_pi(pi):
    """Convert binary encoded pi into U,D,L,R coded pi"""
    (num_y_pos, num_x_pos, _) = pi.shape
    blacklist = get_blacklist_for_shape(pi.shape)
    pretty_pi = np.full(shape=(num_y_pos, num_x_pos), fill_value=None)
    for x_pos in range(0, num_x_pos):
        for y_pos in range(0, num_y_pos):
            if (y_pos, x_pos) not in blacklist:
                unpretty_actions = pi[y_pos, x_pos, :]
                pretty_actions = []
                for action in range(NUM_ACTIONS):
                    if unpretty_actions[action] == 1:
                        # advocated
                        pretty_actions.append(ACTION_MAP[action])
                    else:
                        pretty_actions.append("-")
                pretty_pi[y_pos, x_pos] = tuple(pretty_actions)
            else:
                pretty_pi[y_pos, x_pos] = tuple(['*'] * NUM_ACTIONS)
    return pretty_pi


def get_blacklist_for_shape(pi_shape):
    (num_y_pos, num_x_pos, _) = pi_shape
    assert num_y_pos == num_x_pos
    grid_size = num_y_pos
    if grid_size == 8:
        return BLACKLISTED_STATES_8X8
    elif grid_size == 12:
        return BLACKLISTED_STATES_12X12
    else:
        assert False, "invalid"


def calc_q_err(q_ref, q_hat):
    """Calc MAE for each sub q func and total mae - return 5-tuple."""
    (num_y_pos, num_x_pos, _) = q_ref.shape
    blacklist = get_blacklist_for_shape(q_ref.shape)
    saes = []
    maes = []
    for action in range(NUM_ACTIONS):
        q_ref_a = q_ref[:, :, action]
        q_hat_a = q_hat[:, :, action]
        sae = 0.0
        for x_pos in range(0, num_x_pos):
            for y_pos in range(0, num_y_pos):
                if (y_pos, x_pos) not in blacklist:
                    q_ref_val = q_ref_a[y_pos, x_pos]
                    q_hat_val = q_hat_a[y_pos, x_pos]
                    assert not np.isnan(q_hat_val)
                    sae += abs(q_ref_val - q_hat_val)
        saes.append(sae)
        num_frozen_cells = calc_num_frozen_cells(q_ref.shape)
        mae = sae / num_frozen_cells
        maes.append(mae)
    total_mae = sum(saes) / (calc_num_frozen_cells(q_ref.shape) * \
        NUM_ACTIONS)
    maes.append(total_mae)
    return maes


def calc_pi_err_matrix(pi_ref, pi_hat):
    """Calc error matrix between pi_ref, pi_hat."""
    (num_y_pos, num_x_pos, _) = pi_ref.shape
    blacklist = get_blacklist_for_shape(pi_ref.shape)
    error_matrix = np.full(shape=(num_y_pos, num_x_pos), fill_value=np.nan)
    for x_pos in range(0, num_x_pos):
        for y_pos in range(0, num_y_pos):
            if (y_pos, x_pos) not in blacklist:
                error_matrix[y_pos, x_pos] = \
                    calc_pi_cell_err(pi_ref[y_pos, x_pos, :],
                        pi_hat[y_pos, x_pos, :])
    return error_matrix


def calc_pi_correct_matrix(pi_error_matrix):
    pi_correct_matrix = np.full(pi_error_matrix.shape, fill_value=np.nan)
    (num_y_pos, num_x_pos) = pi_error_matrix.shape
    for y in range(num_y_pos):
        for x in range(num_x_pos):
            if pi_error_matrix[y,x] == 0:
                pi_correct_matrix[y,x] = 1
            elif pi_error_matrix[y,x] == 1:
                pi_correct_matrix[y,x] = 0
    return pi_correct_matrix

def calc_pi_cell_err(pi_ref_cell, pi_hat_cell):
    comparison = np.logical_and(pi_ref_cell, pi_hat_cell)
    num_same = list(comparison).count(True)
    if num_same >= 1:
        return 0
    else:
        return 1

def calc_pi_err(pi_ref, pi_hat):
    error_matrix = calc_pi_err_matrix(pi_ref, pi_hat)
    total_error = sum([elem for elem in error_matrix.flatten() if not np.isnan(elem)])
    assert total_error % 1 == 0
    return int(total_error)

def calc_num_frozen_cells(ref_shape):
    (num_y_pos, num_x_pos, _) = ref_shape
    blacklist = get_blacklist_for_shape(ref_shape)
    return (num_y_pos * num_x_pos) - len(blacklist)
