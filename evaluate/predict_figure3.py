import argparse
import os
import numpy as np
import tensorflow as tf

from experiment.real_world.dataset.glucose_dataset import train_test_ds
from experiment.real_world.outcome.run_outcome_multiple_marked_hierarchical import train_ft
from experiment.real_world.treatment.run_gprpp_joint import get_glucose_dataset, prepare_tm_input
from utils.utils import get_relative_input_by_query


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='dataset/public_dataset.csv')
    parser.add_argument('--patient_id', type=int, default=8)
    parser.add_argument('--remove_night_time', action='store_true')
    parser.add_argument('--hours_day', type=float, default=24.0)
    parser.add_argument('--figure_dir', type=str, default='figures.figure3')
    parser.add_argument('--plot_functional_components', type=str, default='ao')
    parser.add_argument('--treatment_model_dir', type=str, default='models.gprpp')
    parser.add_argument('--outcome_model_dir', type=str, default='models.outcome')
    args = parser.parse_args()
    args.preprocess_actions = True
    args.domain = [0.0, 24.0]
    os.makedirs(args.figure_dir, exist_ok=True)
    #
    ds = get_glucose_dataset(args.patient_id, args)
    actions, outcome_tuples, baseline_times = prepare_tm_input([ds], args)
    args.plot_functional_components = args.plot_functional_components.split(',')
    Nc = len(args.plot_functional_components)
    labels = {
        'bao': r'$\lambda_{bao} = (\beta_0 + g_b + g_a + g_o)^2$',
        'ba': r'$\lambda_{ba} = (\beta_0 + g_b + g_a)^2$',
        'bo': r'$\lambda_{bo} = (\beta_0 + g_b + g_o)^2$',
        'ao': r'$\lambda_{ao} = (\beta_0 + g_a + g_o)^2$',
        'b': r'$\lambda_{b} = (\beta_0 + g_b)^2$',
    }
    Ds = {
        'bao': 4,
        'ba': 2,
        'bo': 3,
        'ao': 3,
        'b': -1,
    }
    dep_model_folders = [os.path.join(f'{args.treatment_model_dir}/f{ac}', f'action.p{args.patient_id}')
                         for ac in args.plot_functional_components if ac != 'b']
    dep_action_models = [tf.saved_model.load(os.path.join(f, 'time_intensity')) for f in dep_model_folders]
    ind_model_folder = os.path.join(f'{args.treatment_model_dir}/fb', f'action.p{args.patient_id}')
    ind_model = tf.saved_model.load(os.path.join(ind_model_folder, 'vbpp'))
    d = 1
    N_test = 200
    X_abs = np.linspace(*args.domain, N_test)
    #
    lambda_means = []
    baseline_time = baseline_times[d]
    action_time = actions[d]
    outcome_tuple = outcome_tuples[d]
    for action_model, ac in zip(dep_action_models + [ind_model], args.plot_functional_components):
        D = Ds[ac]
        if ac != 'b':
            args.D = D
            args.action_components = ac
            X = get_relative_input_by_query(X_abs, baseline_time, action_time, outcome_tuple, args)
            _, lambda_mean = action_model.predict_lambda_compiled(X)
            lambda_means.append(lambda_mean)
        else:
            _, lambda_mean = action_model.predict_lambda_compiled(X_abs.reshape(-1, 1))
            lambda_means.append(lambda_mean)

    np.savez(os.path.join(args.figure_dir, f'lambda_mean_f{Nc}.npz'), x=X_abs, lambda_means=lambda_means)

    outcome_args = argparse.Namespace(
        dataset=args.dataset,
        n_day_train=2,
        n_day_test=1,
        mins_hour=60.0,
        hours_day=24.0,
        T_treatment=3.0,
        mins_day=24*60.0,
        patient_ids=f'{args.patient_id}',
        observational_outcome_ids=[args.patient_id],
        treatment_covariates='STARCH,SUGAR',
        outcome_model_dir=args.outcome_model_dir,
        maxiter=2000,
        plot=False
    )

    args.remove_night_time = False
    ds = get_glucose_dataset(args.patient_id, args)
    outcome_args.outcome_model_figures_dir = os.path.join(outcome_args.outcome_model_dir, 'figures')
    os.makedirs(outcome_args.outcome_model_dir, exist_ok=True)
    os.makedirs(outcome_args.outcome_model_figures_dir, exist_ok=True)
    _, ds_train, ds_test = train_test_ds(ds, outcome_args)
    outcome_model_path = os.path.join(args.outcome_model_dir, f'outcome_model')
    if not os.path.exists(outcome_model_path):
        train_ft([ds_train], outcome_args)
    outcome_model = tf.saved_model.load(outcome_model_path)

    x_out, y_out, t_out, m_out = ds[0], ds[1], ds[2], ds[3]
    mask_x = np.logical_and(x_out > d*args.hours_day, x_out < (d+1)*args.hours_day)
    mask_t = np.logical_and(t_out > d*args.hours_day, t_out < (d+1)*args.hours_day)
    x_out, y_out = x_out[mask_x], y_out[mask_x]
    t_out, m_out = t_out[mask_t], m_out[mask_t]
    xnew = [x_out.astype(np.float64).reshape(-1, 1)]
    anew = [np.hstack([t_out.astype(np.float64).reshape(-1, 1), m_out.astype(np.float64).reshape(-1, 1)])]
    t_lengths = [ti.shape[0] for ti in anew]
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ft_mean, ft_var = outcome_model.predict_ft_w_tnew_compiled(xnew, anew, tnew_patient_idx)
    ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
    fb_mean, fb_var = outcome_model.predict_baseline_compiled(xnew)
    f_mean, f_var = outcome_model.predict_y_w_tnew_compiled(xnew, anew, tnew_patient_idx)

    np.savez(os.path.join(args.figure_dir, 'f_mean.npz'),
             x=x_out, f_mean=f_mean, f_var=f_var, fb_mean=fb_mean, fb_var=fb_var)
