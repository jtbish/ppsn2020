import pickle
import argparse
from pathlib import Path
from q_pi_utils import *
from gnmc import gnmc
import numpy as np


def main():
    args = parse_args()

    print(args.experiment_dir)
    with open(Path(args.experiment_dir) / "trained_lcs.pkl",  "rb") as fp:
        lcs = pickle.load(fp)

    lambda_ = lambda clfr: 1/clfr.fitness
    rhos = np.arange(0.0, 0.99+0.01, 0.01)

    compact_lcss = {}

    num_x_pos = 8
    num_y_pos = 8
    for rho in rhos:
        lcs_compact = gnmc(lcs, lambda_, rho, num_x_pos, num_y_pos)
        lcs_q_hat = calc_lcs_q_hat(lcs_compact, num_x_pos, num_y_pos)
        hole_counts = get_hole_counts(lcs_q_hat)
        assert hole_counts[-1] == 0
        compact_lcss[rho] = lcs_compact

    with open(Path(args.experiment_dir) / "gnmc_fitness_antagonistic_lcss.pkl",
            "wb") as fp:
        pickle.dump(compact_lcss, fp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
