# ppsn2020
Supporting code for PPSN 2020 paper: "Optimality-based Analysis of XCSF Compaction in Discrete Reinforcement Learning"

## To setup environment (assuming Linux cmd line)
In cloned repo dir:
1. Make virtual env: ``python3 -m venv xcsf``
2. Activate it: ``source xcsf/bin/activate``
3. Install dependencies: ``pip3 install -r requirements.txt``

## To train XCSF as in the paper
Use ``frozen_lake_xcsf_dmsr.py`` run script. Hyperparameter args to script are pre-configured as specified in paper section 5.1, apart from variable args:
- slip probability (--slip-prob)
- seed for training lcs (--lcs-seed)
- name of experiment, used for output directory (--experiment-dir)
- number of training steps (--num-training-samples)

Example usage, replicating one of the runs from Fig. 3a in paper:
``python3 frozen_lake_xcsf_dmsr.py --slip-prob=0 --lcs-seed=0 --experiment-dir=test --num-training-samples=400000``

This would create an output directory ``test``, with contents looking like this:
<pre>total 132K
-rw-r--r-- 1 uqjbish3 uusers 769K Jun 19 14:26 experiment.log
-rw-r--r-- 1 uqjbish3 uusers  405 Jun 19 14:26 lcs_hyperparams.txt
-rw-r--r-- 1 uqjbish3 uusers    6 Jun 19 14:26 lcs_monitor.pkl
-rw-r--r-- 1 uqjbish3 uusers  94K Jun 19 14:26 loop_data_monitor.pkl
-rw-r--r-- 1 uqjbish3 uusers  357 Jun 19 14:26 python_env_info.txt
-rw-r--r-- 1 uqjbish3 uusers 3.0K Jun 19 14:26 run_script.py
-rw-r--r-- 1 uqjbish3 uusers  19K Jun 19 14:26 trained_lcs.pkl
-rw-r--r-- 1 uqjbish3 uusers  189 Jun 19 14:26 var_args.txt
</pre>

- ``experiment.log`` is a log file written to during training. By default log verbosity is debug so can get quite large. See run script to change this.
- ``lcs_hyperparams.txt`` gives full list of hyperparams used.
- ``lcs_monitor.pkl`` stores pickled LCS objects at increments during training.
- ``loop_data_monitor.pkl`` stores entire history of (s, a, r, s') experience tuples.
- ``python_env_info.pkl`` stores copy of virtualenv info.
- ``run_script.py`` is a copy of the run script invoked.
- ``trained_lcs.pkl`` is the final LCS model at end of training (also the most recent one in `lcs_monitor.pkl``).
- ``var_args.txt`` gives list of variable input args to run script.

## Notes
Other python scripts included were used to analyse the trained LCS models and produce other figures.
