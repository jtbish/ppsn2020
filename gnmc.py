import copy
from q_pi_utils import *

NUM_ACTIONS = 4

def gnmc(lcs, lambda_, rho, num_x_pos, num_y_pos):
    clfrs_to_keep = []
    blacklist = get_blacklist_for_shape((num_y_pos, num_x_pos, NUM_ACTIONS))
    for x in range(num_x_pos):
        for y in range(num_y_pos):
            # only do compaction in non-terminals
            if (y, x) not in blacklist:
                situation = [x, y]
                match_set = lcs.gen_match_set(situation)
                for a in range(NUM_ACTIONS):
                    action_set = lcs.gen_action_set(match_set, a)
                    total_mass = sum([lambda_(clfr) for clfr in action_set])
                    target_mass = (1 - rho)*total_mass
                    sorted_action_set = sorted(action_set, key=lambda clfr: lambda_(clfr),
                            reverse=True)
                    mass_so_far = 0.0
                    num_kept = 0
                    for clfr in sorted_action_set:
                        if mass_so_far < target_mass:
                            clfrs_to_keep.append(clfr)
                            mass_so_far += lambda_(clfr)
                            num_kept += 1
                        else:
                            break
                    if rho == 0.0:
                        assert num_kept == len(sorted_action_set)

    # filter clfrs to keep to unique entries
    unique_clfrs_to_keep = []
    for clfr in clfrs_to_keep:
        if clfr not in unique_clfrs_to_keep:
            unique_clfrs_to_keep.append(clfr)

    # remove all clfrs that not marked for keeping
    lcs_compact = copy.deepcopy(lcs)
    for clfr in lcs.population:
        if clfr not in unique_clfrs_to_keep:
            lcs_compact.population.remove(clfr)

    return lcs_compact
