import os
import argparse
import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.utilities import to_default_float

from evaluate.plot_om import plot_fs_pred_multiple, compare_f_preds
from experiment.real_world.dataset.glucose_dataset import prepare_glucose_ds, load_glucose_dfs
from utils import constants_rw


def plot_outcome_pred(args):
    args.observational_outcome_ids = np.array(args.patient_ids.split(','), dtype=int)
    dfs = load_glucose_dfs(args, args.observational_outcome_ids)
    ds_alls = []
    ds_trains = []
    ds_tests = []
    for patient_idx, df in zip(args.observational_outcome_ids, dfs):
        ds_all, ds_train, ds_test = prepare_glucose_ds(df, args)
        ds_tests.append(ds_test)
        ds_trains.append(ds_train)
        ds_alls.append(ds_all)

    ds = np.load(os.path.join(args.outcome_model_pred_dir, 'f_outcome.npz'), allow_pickle=True)
    f_means, f_vars, ds_alls = ds['f_means'], ds['f_vars'], ds['ds_alls']
    plot_fs_pred_multiple(f_means, f_vars, ds_alls, args, run='all')

    ds = np.load(os.path.join(args.outcome_model_pred_dir, 'ft_outcome.npz'), allow_pickle=True)
    xnew, ft_means = ds['xnew'], ds['ft_means']
    compare_f_preds(xnew, ft_means, ylim=(0.0, 6.0),
                    ylabel=r'Treatment Response, $f_t(\tau)$',
                    title_str=r'Treatment Response Function Comparison (Carb= $3 \log$g)',
                    path=os.path.join(args.outcome_model_figures_dir, f'compare_ft_m3.pdf'))
    compare_f_preds(xnew, ft_means, ylim=(0.0, 6.0),
                    ylabel=r'Treatment Response, $f_t(\tau)$',
                    title_str=r'Treatment Response Function Comparison (Carb= $3 \log$g)',
                    path=os.path.join(args.outcome_model_figures_dir, f'compare_ft_m3.png'))

    ds = np.load(os.path.join(args.outcome_model_pred_dir, 'fb_outcome.npz'), allow_pickle=True)
    xnew, fb_means = ds['xnew'], ds['fb_means']
    compare_f_preds(xnew, fb_means, ylim=(0.0, 6.0),
                    ylabel=r'Baseline, $f_b(\tau)$',
                    title_str=r'Baseline Progression Comparison',
                    path=os.path.join(args.outcome_model_figures_dir, f'compare_fb.pdf'))
    compare_f_preds(xnew, fb_means, ylim=(0.0, 6.0),
                    ylabel=r'Baseline, $f_b(\tau)$',
                    title_str=r'Baseline Progression Comparison',
                    path=os.path.join(args.outcome_model_figures_dir, f'compare_fb.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='dataset/public_dataset.csv')
    parser.add_argument("--n_day_train", type=int, default=2)
    parser.add_argument("--n_day_test", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--patient_ids", type=str, default='0')
    parser.add_argument("--outcome_model_dir", type=str, default='models/sampler/outcome')
    parser.add_argument("--seed", type=int, default=1)
    init_args = parser.parse_args()
    init_dict = vars(init_args)
    init_dict.update(constants_rw.GENERAL_PARAMS)
    init_args = argparse.Namespace(**init_dict)
    np.random.seed(init_args.seed)
    init_args.outcome_model_dir = os.path.join(init_args.outcome_model_dir, f'outcome.p{init_args.patient_ids}')
    init_args.outcome_model_figures_dir = os.path.join(init_args.outcome_model_dir, 'figures')
    init_args.outcome_model_pred_dir = os.path.join(init_args.outcome_model_dir, 'outcome_pred')
    tf.random.set_seed(init_args.seed)
    plot_outcome_pred(init_args)
