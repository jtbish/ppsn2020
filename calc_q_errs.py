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
    overall_maes = {}
    for time_step, lcs in lcss.items():
        if (time_step >= args.start_time_step) and \
                (time_step % args.time_step_freq == 0):
            q_hat = calc_lcs_q_hat(lcs, q_star.shape[1], q_star.shape[0])
            hole_counts = get_hole_counts(q_hat)
            if hole_counts[-1] != 0:
                print(f"{args.experiment_dir}: aborting, Qhat with holes "
                    f"{hole_counts} at time step {time_step}")
                sys.exit(1)
            maes = calc_q_err(q_star, q_hat)
            overall_mae = maes[-1]
            overall_maes[time_step] = overall_mae

    _do_plot(args.experiment_dir, overall_maes)
    _save_data(args.experiment_dir, overall_maes)


def _do_plot(experiment_dir, overall_maes):
    time_steps = list(overall_maes.keys())
    maes = list(overall_maes.values())
    plt.figure()
    plt.plot(time_steps, maes)
    plt.title("Q MAE over training")
    plt.xlabel("Time step")
    plt.ylabel("MAE between Q_hat and Q_star")
    plt.savefig(Path(experiment_dir) / "q_plot.png")


def _save_data(experiment_dir, overall_maes):
    with open(Path(experiment_dir) / "q_maes.pkl", "wb") as fp:
        pickle.dump(overall_maes, fp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-star-npy", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--start-time-step", type=int, required=True)
    parser.add_argument("--time-step-freq", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
