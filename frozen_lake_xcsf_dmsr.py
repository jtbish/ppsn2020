#!/usr/bin/python3
import argparse
import math
import logging

from piecewise.environment import make_frozen_lake_8x8_train_env
from piecewise.experiment import Experiment
from piecewise.lcs import make_custom_xcsf_from_canonical_base
from piecewise.lcs.component import XCSSubsumption, make_improved_xcs_ga
from piecewise.rule_repr import make_discrete_min_span_rule_repr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slip-prob", type=float, required=True)
    parser.add_argument("--lcs-seed", type=int, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--num-training-samples", type=int, required=True)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--beta-e", type=float, default=0.05)
    parser.add_argument("--epsilon-nought", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--r-nought", type=int, default=4)
    parser.add_argument("--m-nought", type=int, default=4)
    parser.add_argument("--lcs-monitor-freq", type=int, default=5000)
    return parser.parse_args()


def main(args):
    env_seed = 0
    env = make_frozen_lake_8x8_train_env(slip_prob=args.slip_prob, seed=env_seed)

    rule_repr = make_discrete_min_span_rule_repr(env)

    lcs_hyperparams = {
        "N": args.N,
        "beta": 0.1,
        "beta_e": args.beta_e,
        "alpha": 0.1,
        "epsilon_nought": args.epsilon_nought,
        "nu": 5,
        "gamma": args.gamma,
        "theta_ga": 50,
        "tau": 0.5,
        "chi": 1.0,
        "upsilon": 0.5,
        "mu": 0.05,
        "theta_del": 50,
        "delta": 0.1,
        "theta_sub": 50,
        "epsilon_I": args.epsilon_nought/10,
        "fitness_I": args.epsilon_nought/10,
        "theta_mna": len(env.action_set),
        "do_ga_subsumption": True,
        "do_as_subsumption": False,
        "p_explore": 0.5,
        "r_nought": args.r_nought,
        "m_nought": args.m_nought,
        "x_nought": 10,
        "eta": 0.1
    }

    subsumption = XCSSubsumption(rule_repr)
    rule_discovery = make_improved_xcs_ga(env.action_set, rule_repr,
            subsumption)
    lcs = make_custom_xcsf_from_canonical_base(
        env,
        rule_repr,
        lcs_hyperparams,
        seed=args.lcs_seed,
        subsumption=subsumption,
        rule_discovery=rule_discovery)

    experiment = Experiment(name=args.experiment_name,
                            env=env,
                            lcs=lcs,
                            num_training_samples=args.num_training_samples,
                            use_lcs_monitor=True,
                            lcs_monitor_freq=args.lcs_monitor_freq,
                            use_loop_monitor=True,
                            logging_level=logging.DEBUG,
                            var_args=args)
    experiment.run()
    experiment.save_results()


if __name__ == "__main__":
    args = parse_args()
    main(args)
