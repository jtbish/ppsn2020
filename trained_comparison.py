import argparse
import pickle
from pathlib import Path

import numpy as np

from q_pi_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-star-npy", required=True)
    parser.add_argument("--experiment-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    q_star = np.load(args.q_star_npy)
    experiment_path = Path(args.experiment_dir)
    lcs_pkl = experiment_path / "trained_lcs.pkl"
    with open(lcs_pkl, "rb") as fp:
        lcs = pickle.load(fp)

    (num_y_pos, num_x_pos, _) = q_star.shape
    lcs_q_hat = calc_lcs_q_hat(lcs, num_x_pos, num_y_pos)
    hole_counts = get_hole_counts(lcs_q_hat)

    pi_star = calc_pi_from_q(q_star)

    _report_q_err(q_star, lcs_q_hat, hole_counts, experiment_path)
    _report_pi_err(pi_star, lcs_q_hat, hole_counts, experiment_path)


def _report_q_err(q_star, lcs_q_hat, hole_counts, experiment_path):
    total_holes = hole_counts[-1]
    if total_holes != 0:
        lines = []
        lines.append(f"LCS Qhat hole count not zero: {hole_counts}, "
                f" can't measure err from Qstar.")
    else:
        lines = _format_q_err(q_star, lcs_q_hat)

    with open(experiment_path / "q_comparison.txt", "w") as fp:
        for line in lines:
            fp.write(f"{line}\n")


def _format_q_err(q_star, lcs_q_hat):
    lines = []
    maes = calc_q_err(q_star, lcs_q_hat)
    assert len(maes) == 5
    for action in range(NUM_ACTIONS):
        q_star_a = q_star[:, :, action]
        q_hat_a = lcs_q_hat[:, :, action]
        with np.printoptions(precision=4):
            lines.append(f"Q_{action} star")
            lines.append(str(q_star_a))
            lines.append(f"Q_{action} hat")
            lines.append(str(q_hat_a))
        lines.append(f"Q_{action} MAE: {maes[action]}")
    overall_mae = maes[4]
    lines.append(f"Overall MAE: {overall_mae}")
    return lines


def _report_pi_err(pi_star, lcs_q_hat, hole_counts, experiment_path):
    total_holes = hole_counts[-1]
    if total_holes != 0:
        lines = []
        lines.append(f"LCS Qhat hole count not zero: {hole_counts} "
                f", can't calculate LCS Pi hat")
    else:
        lcs_pi_hat = calc_pi_from_q(lcs_q_hat)
        lines = _format_pi_err(pi_star, lcs_pi_hat)
        
    with open(experiment_path / "pi_comparison.txt", "w") as fp:
        for line in lines:
            fp.write(f"{line}\n")


def _format_pi_err(pi_star, lcs_pi_hat):
    lines = []
    with np.printoptions(linewidth=350):
        lines.append("Pi star")
        lines.append(str(prettify_pi(pi_star)))
        lines.append("Pi hat")
        lines.append(str(prettify_pi(lcs_pi_hat)))
    pi_err = calc_pi_err(pi_star, lcs_pi_hat)
    lines.append(
        f"Error between pi_star & lcs_pi_hat: {pi_err}")
    return lines


if __name__ == "__main__":
    main()
