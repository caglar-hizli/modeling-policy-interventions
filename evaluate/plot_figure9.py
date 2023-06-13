import argparse
import os.path

import numpy as np
import tensorflow as tf

from evaluate.plot_tm import compare_ga, compare_go, compare_mark_intensity
from experiment.real_world.treatment.run_gprpp_joint import prepare_init_args, prepare_tm_dirs


def compare_gprpp_fits(args):
    n_patient = len(args.patient_ids)
    if 'a' in args.action_components:
        labels = [r'$g_{{a,\pi_{}}}^{{*}}(t)$'.format(chr(i+65)) for i in range(n_patient)]
        ds = np.load(os.path.join(args.model_pred_dir, 'ga.npz'), allow_pickle=True)
        x, ga = ds['x'], ds['ga']
        compare_ga(x, ga, labels=labels, args=args)
    if 'o' in args.action_components:
        ds = np.load(os.path.join(args.model_pred_dir, 'go.npz'), allow_pickle=True)
        x, go = ds['x'], ds['go']
        labels = [r'$g_{{o,\pi_{}}}^{{*}}(t)$'.format(chr(i+65)) for i in range(n_patient)]
        compare_go(x, go, labels=labels, args=args)
    labels = [r'$p_{{\pi_{}}}(m \mid \tau)$'.format(chr(i+65)) for i in range(n_patient)]
    ds = np.load(os.path.join(args.model_pred_dir, 'f_mark.npz'), allow_pickle=True)
    x, f_mark = ds['x'], ds['f_mark']
    compare_mark_intensity(x, f_mark, labels=labels, args=args)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models.gprpp')
    parser.add_argument('--n_day_train', type=int, default=2)
    parser.add_argument("--n_day_test", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument('--action_components', type=str, default='bao')
    parser.add_argument("--patient_ids", type=str, default='1')
    parser.add_argument('--seed', type=int, default=1)
    init_args = parser.parse_args()
    print(init_args.patient_ids)
    init_args = prepare_init_args(init_args)
    init_args.patient_id = init_args.patient_ids[0]
    init_args = prepare_tm_dirs(init_args)
    print(init_args.patient_ids)
    compare_gprpp_fits(init_args)
