import pickle
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from q_pi_utils import *
import sys


def main():
    args = parse_args()
    with open(Path(args.experiment_dir) / "lcs_monitor.pkl",  "rb") as fp:
        lcss = pickle.load(fp)
    q_star = np.load(args.q_star_npy)
    pi_star = calc_pi_from_q(q_star)
    pi_errs = {}
    for time_step, lcs in lcss.items():
        if (time_step >= args.start_time_step) and \
                (time_step % args.time_step_freq == 0):
            q_hat = calc_lcs_q_hat(lcs, q_star.shape[1], q_star.shape[0])
            hole_counts = get_hole_counts(q_hat)
            if hole_counts[-1] != 0:
                print(f"{args.experiment_dir}: aborting, Qhat with holes "
                    f"{hole_counts} at time step {time_step}")
                sys.exit(1)
            pi_hat = calc_pi_from_q(q_hat)
            pi_errs[time_step] = calc_pi_err(pi_star, pi_hat)

    _do_plot(args.experiment_dir, pi_errs)
    _save_data(args.experiment_dir, pi_errs)


def _do_plot(experiment_dir, pi_errs):
    time_steps = list(pi_errs.keys())
    errs = list(pi_errs.values())
    plt.figure()
    plt.plot(time_steps, errs)
    plt.title("Pi err over training")
    plt.xlabel("Time step")
    plt.ylabel("Err between Pi_hat and Pi_star")
    plt.savefig(Path(experiment_dir) / "pi_plot.png")


def _save_data(experiment_dir, pi_errs):
    with open(Path(experiment_dir) / "pi_errs.pkl", "wb") as fp:
        pickle.dump(pi_errs, fp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-star-npy", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--start-time-step", type=int, required=True)
    parser.add_argument("--time-step-freq", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
