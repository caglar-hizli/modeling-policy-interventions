import os
import numpy as np
import tensorflow as tf

from evaluate.plot_om import plot_ft_pred
from experiment.real_world.dataset import glucose_dataset
from models.treatment.gprpp.model import integrate_log_fn_sqr
from utils.utils import get_relative_input_joint


def predict_ft_marked_hier_compiled(model, args, oracle_model=None):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    for pidx in args.observational_outcome_ids:
        for m in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tnew = np.array([-0.001, m]).reshape(-1, 2)
            pidx_tf = tf.ones(1, dtype=tf.int32) * pidx
            ft_mean, ft_var = model.predict_ft_single_for_patient_compiled(xnew, tnew, pidx_tf[0])
            ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
            true_ft_mean = None
            if oracle_model is not None:
                pidx_r = pidx % len(args.outcome_sampler_patient_ids)
                oracle_pidx_tf = tf.ones(1, dtype=tf.int32) * args.outcome_sampler_patient_ids[pidx_r]
                true_ft_mean, _ = oracle_model.predict_ft_single_for_patient_compiled(xnew, tnew, oracle_pidx_tf[0])
            plot_ft_pred(xnew, ft_mean, ft_var, true_f_mean=true_ft_mean,
                         path=os.path.join(args.outcome_model_figures_dir, f'predict_ft_p{pidx}_m{m:.1f}.pdf'))


def compare_ft_marked_hier_compiled(model, args):
    xnew = np.linspace(0.0, args.T_treatment, 100).reshape(-1, 1)
    m = 3.0
    ft_means = []
    for pidx in args.observational_outcome_ids:
        tnew = np.array([-0.001, m]).reshape(-1, 2)
        pidx_tf = tf.ones(1, dtype=tf.int32) * pidx
        ft_mean, _ = model.predict_ft_single_for_patient_compiled(xnew, tnew, pidx_tf[0])
        ft_means.append(ft_mean)
    return xnew, ft_means


def compare_fb_compiled(model, args):
    N_test = 100
    xnew = [np.linspace(0.0, 24.0, N_test).astype(np.float64).reshape(-1, 1) for _ in args.observational_outcome_ids]
    fb_mean, fb_var = model.predict_baseline_compiled(xnew)
    fb_means = [fb_mean[i*N_test:(i+1)*N_test] for i in range(len(args.observational_outcome_ids))]
    return xnew[0], fb_means


def predict_f_shared_marked_hier_compiled(model, ds_plots):
    xnew = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_plots]
    anew = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                       ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_plots]
    t_lengths = [ti.shape[0] for ti in anew]
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ft_mean, ft_var = model.predict_ft_w_tnew_compiled(xnew, anew, tnew_patient_idx)
    ft_var = np.diag(ft_var.numpy()[0]).reshape(-1, 1)
    fb_mean, fb_var = model.predict_baseline_compiled(xnew)
    f_mean, f_var = ft_mean + fb_mean, ft_var + fb_var
    f_means, f_vars = [fb_mean, ft_mean, f_mean], [fb_var, ft_var, f_var]
    return f_means, f_vars


def predict_gprpp_ll_lbo(model, actions, outcome_tuples, baseline_times, args):
    rel_at_actions, rel_at_all_points, abs_all_points = get_relative_input_joint(actions, outcome_tuples,
                                                                                 baseline_times, args, D=args.D)
    evs_start = [np.array([baseline_time[0].item(),
                           action[0] if len(action) > 0 else 24.0,
                           outcome_tuple[0, 0]])
                 for action, outcome_tuple, baseline_time in zip(actions, outcome_tuples, baseline_times)]
    data_term = predict_gprpp_data_term(model, rel_at_actions)
    integral_term = model.predict_integral_term_compiled(abs_all_points, rel_at_all_points, evs_start)
    ll_lbo = data_term - integral_term
    return ll_lbo, data_term, integral_term


def predict_gprpp_data_term(model, rel_events):
    data_term = 0.0
    for rel_event in rel_events:
        if rel_event.shape[0] > 0:
            mean, var = model.predict_f_compiled(Xnew=rel_event)
            data_term += tf.reduce_sum(integrate_log_fn_sqr(mean, var))
    return data_term


def predict_vbpp_data_term(model, new_events):
    data_term = 0.0
    for evs in new_events:
        if evs.shape[0] > 0:
            mean, var = model.predict_f_compiled(Xnew=evs)
            data_term += tf.reduce_sum(integrate_log_fn_sqr(mean, var))
    return data_term


def predict_vbpp_ll_lbo(model, new_events):
    data_term = predict_vbpp_data_term(model, new_events)
    integral_term = model.predict_integral_term_compiled()
    integral_term = len(new_events) * integral_term
    ll_lbo = data_term + integral_term
    return ll_lbo, data_term, -integral_term


def predict_vbpp(model, domain):
    X = glucose_dataset.domain_grid(domain, 100)
    lambda_mean, lower, upper = model.predict_lambda_and_percentiles(X)
    lower = lower.numpy().flatten()
    upper = upper.numpy().flatten()
    return X, lambda_mean, lower, upper


def predict_log_density(model, ds_plots):
    Xnew = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_plots]
    tnew = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                       ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_plots]
    t_lengths = [ti.shape[0] for ti in tnew]
    Ynew = [ds[1].reshape(-1, 1) for ds in ds_plots]
    N_patients = len(t_lengths)
    patient_order_arr = np.arange(N_patients, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ll = model.predict_log_density_for_patients_compiled(Xnew, Ynew, tnew, tnew_pidx=tnew_patient_idx)
    ll_sum = ll.numpy().flatten().sum()
    return ll_sum
