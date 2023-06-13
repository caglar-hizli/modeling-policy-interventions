import argparse
import os.path
import pandas as pd

import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR
from gpflow.utilities import to_default_float

from experiment.real_world.dataset.glucose_dataset import get_glucose_dataset
from experiment.predict import predict_gprpp_ll_lbo
from evaluate.plot_tm import plot_fm_pred, plot_gprpp_results
from models.treatment.gprpp.kernel import MaskedSE, MaskedMarkedSE
from models.treatment.gprpp.model import GPRegressiveMarkedPP
from utils import constants_rw
from utils.utils import get_relative_input_joint


def demo(args):
    # Prepare data
    ds = get_glucose_dataset(args.patient_id, args)
    time_intensity = run_treatment_time_intensity([ds], args)
    mark_intensity = run_treatment_mark_intensity_pooled([ds], args)
    return time_intensity, mark_intensity


def prepare_tm_input(ds, args):
    outcome_tuples = []
    actions = []
    baseline_times = []
    for ds_patient in ds:
        x, y, t, _ = ds_patient
        D = np.max(np.concatenate([x-1e-6, t])) // 24
        for d in range(int(D + 1)):
            mask_t = np.logical_and(d * 24.0 < t, t < (d+1) * 24.0)
            t_day = t[mask_t] - d * 24.0
            sampling_offset = t_day[0] - 0.5 if args.remove_night_time else 0.0
            mask_x = np.logical_and(d * 24.0 + sampling_offset < x, x < (d+1) * 24.0)
            x_day = x[mask_x] - d * 24.0
            y_day = y[mask_x]

            actions.append(t_day)
            outcome_tuples.append(np.stack([x_day, y_day]).T)
            baseline_times.append(np.zeros((1, 1)))

    return actions, outcome_tuples, baseline_times


def build_gprpp_Z(args):
    Z_array = []
    if 'b' in args.action_components:
        Z_array.append(np.linspace(0.0, 24.0, args.M_times))
    if 'a' in args.action_components:
        Z_array.append(np.linspace(*args.action_time_domain, args.M_times))
    if 'o' in args.action_components:
        Z_array.append(np.ones(args.M_times))
        Z_array.append(np.linspace(*args.mark_domain, args.M_times))
    Z = np.stack(Z_array).T
    return Z


def build_gprpp_kernel(args):
    variance_init, variance_prior = args.variance_init, args.variance_prior
    lengthscales_init = args.lengthscales_init
    lengthscales_prior = args.lengthscales_prior
    optimize_variance, optimize_lengthscales = args.optimize_variance, args.optimize_lengthscales
    #
    kernel = []
    d = 0
    # Baseline-dependent Kernel
    if 'b' in args.action_components:
        k0 = MaskedSE(variance=variance_init[0], lengthscales=lengthscales_init[0], active_dims=[0])
        prepare_variable_for_optim(k0.variance, variance_prior, optimize_variance)
        prepare_variable_for_optim(k0.lengthscales, lengthscales_prior, optimize_lengthscales)
        kernel = k0
        d += 1
    # Action-dependent Kernel
    if 'a' in args.action_components:
        kd = MaskedSE(variance=variance_init[d],
                      lengthscales=lengthscales_init[d], active_dims=[d])
        prepare_variable_for_optim(kd.variance, variance_prior, optimize_variance)
        prepare_variable_for_optim(kd.lengthscales, lengthscales_prior, optimize_lengthscales)
        d += 1
        kernel = kernel + kd if kernel else kd
    # Outcome-dependent Kernel
    if 'o' in args.action_components:
        kd = MaskedMarkedSE(variance=variance_init[d],
                            lengthscales=[lengthscales_init[d], lengthscales_init[d+1]], active_dims=[d, d+1])
        prepare_variable_for_optim(kd.variance, variance_prior, optimize_variance)
        prepare_variable_for_optim(kd.lengthscales, lengthscales_prior, optimize_lengthscales)
        d += 1
        kernel = kernel + kd if kernel else kd

    return kernel


def prepare_variable_for_optim(variable, prior_value, optimize_variable):
    if optimize_variable:
        variable.prior = tfp.distributions.HalfNormal(to_default_float(prior_value))
    else:
        gpflow.utilities.set_trainable(variable, False)


def save_gprpp_model(model: GPRegressiveMarkedPP, model_path, args):
    model.predict_lambda_compiled = tf.function(
        model.predict_lambda,
        input_signature=[tf.TensorSpec(shape=[None, args.D], dtype=tf.float64)]
    )
    model.predict_f_compiled = tf.function(
        model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, args.D], dtype=tf.float64)]
    )
    model.predict_integral_term_compiled = tf.function(
        model.predict_integral_term,
        input_signature=[
            [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
            [tf.TensorSpec(shape=[None, args.D], dtype=tf.float64)],
            [tf.TensorSpec(shape=[3], dtype=tf.float64)],
        ]
    )
    tf.saved_model.save(model, model_path)


def save_gpr_model(model: GPR, model_path):
    model.predict_f_compiled = tf.function(
        model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_f_samples_compiled = tf.function(
        model.predict_f_samples,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    tf.saved_model.save(model, model_path)


def build_gprpp_model(args, domain, kernel, Z):
    beta0, marked = args.beta0, args.marked
    M = args.M_times
    time_dims, mark_dims = args.time_dims, args.mark_dims
    #
    inducing_points = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(M)
    q_S = np.eye(M)
    model = GPRegressiveMarkedPP(inducing_points, kernel, domain, q_mu, q_S, beta0=beta0,
                                 time_dims=time_dims, mark_dims=mark_dims, marked=marked)
    gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
    gpflow.utilities.set_trainable(model.beta0, False)
    gpflow.utilities.print_summary(model)
    return model


def run_treatment_time_intensity(ds, args, oracle_model=None):
    actions, outcome_tuples, baseline_times = prepare_tm_input(ds, args)
    actions_train, actions_test = actions[:args.n_day_train], actions[args.n_day_train:]
    outcome_tuples_train, outcome_tuples_test = outcome_tuples[:args.n_day_train], outcome_tuples[args.n_day_train:]
    baseline_times_train, baseline_times_test = baseline_times[:args.n_day_train], baseline_times[args.n_day_train:]
    rel_at_actions, rel_at_all_points, abs_all_points = get_relative_input_joint(actions_train, outcome_tuples_train,
                                                                                 baseline_times_train, args, D=args.D)
    model_path = os.path.join(args.patient_model_dir, 'time_intensity')
    domain = np.array(args.domain, float).reshape(1, 2)
    args = set_time_domain(args, rel_at_all_points)
    if args.marked:
        marks = np.concatenate([marked[:, -1] for marked in rel_at_all_points])
        args = set_mark_domain(args, marks)
    if not os.path.exists(model_path):
        evs_start = []
        for action, outcome_tuple in zip(actions_train, outcome_tuples_train):
            ev_start = []
            if 'b' in args.action_components:
                ev_start.append(baseline_times[0].item())
            if 'a' in args.action_components:
                ev_start.append(action[0] if len(action) > 0 else domain[0, -1])
            if 'o' in args.action_components:
                ev_start.append(outcome_tuple[0, 0] if len(outcome_tuple) > 0 else domain[0, -1])
            evs_start.append(np.array(ev_start))
        for ev_start in evs_start:
            assert ev_start.shape[0] == args.Dt
        # TODO: load model from disk, but then loose uncertainty feature w/o making predict_percentiles a tf function
        kernel = build_gprpp_kernel(args)
        Z = build_gprpp_Z(args)
        gpflow.utilities.print_summary(kernel)
        model = build_gprpp_model(args, domain=domain, kernel=kernel, Z=Z)

        def objective_closure():
            return -model.elbo(abs_all_points, rel_at_actions, rel_at_all_points, evs_start)

        min_logs = gpflow.optimizers.Scipy().minimize(objective_closure, model.trainable_variables,
                                                      compile=True,
                                                      options={"disp": True,
                                                               "maxiter": args.maxiter})

        save_gprpp_model(model, model_path, args)
        plot_gprpp_results(baseline_times, actions, outcome_tuples, action_model=model,
                           model_figures_dir=args.model_figures_dir, args=args, oracle_model=oracle_model)

    saved_model = tf.saved_model.load(model_path)
    if args.compute_nll:
        df_ll = pd.DataFrame.from_dict({'train_ll_lbo': [0.0],
                                        'train_data_term': [0.0],
                                        'train_integral_term': [0.0]})
        for ai, oi, bi in zip(actions_train, outcome_tuples_train, baseline_times_train):
            ll_lbo, data_term, integral_term = predict_gprpp_ll_lbo(saved_model, [ai], [oi], [bi], args)
            df_ll['train_ll_lbo'] += ll_lbo.numpy().item()
            df_ll['train_data_term'] += data_term.numpy().item()
            df_ll['train_integral_term'] += integral_term.numpy().item()

        if args.n_day_train < 3:
            ll_lbo, data_term, integral_term = predict_gprpp_ll_lbo(saved_model, actions_test, outcome_tuples_test,
                                                                    baseline_times_test, args)
            df_ll_test = pd.DataFrame.from_dict({'test_ll_lbo': [ll_lbo.numpy().item()],
                                                 'test_data_term': [data_term.numpy().item()],
                                                 'test_integral_term': [integral_term.numpy().item()]})
            df_ll = pd.concat([df_ll, df_ll_test], axis=1)

        df_ll.to_csv(os.path.join(args.model_metric_dir, 'll_metrics.csv'), index=False)
    return saved_model


def run_treatment_mark_intensity_pooled(ds, args):
    model_path = os.path.join(args.patient_model_dir, 'mark_intensity')
    if not os.path.exists(model_path):
        t_day = np.concatenate([ds_patient[2] % 24 for ds_patient in ds])
        m_day = np.concatenate([ds_patient[3] for ds_patient in ds])
        X = t_day.astype(np.float64).reshape(-1, 1)
        Y = m_day.astype(np.float64).reshape(-1, 1)
        kb = gpflow.kernels.SquaredExponential(variance=2.0, lengthscales=5.0, active_dims=[0])
        kb.variance.prior = tfp.distributions.HalfNormal(to_default_float(2.0))
        kb.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(5.0))
        model = gpflow.models.GPR(data=(X, Y), kernel=kb)
        gpflow.utilities.print_summary(model)
        min_logs = gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables,
                                                      compile=True, options={"disp": True, "maxiter": 2000})
        gpflow.utilities.print_summary(model)
        Xnew = np.linspace(0.0, 24.0, 50).astype(np.float64).reshape(-1, 1)
        f_mean, f_var = model.predict_f(Xnew)
        plot_fm_pred(t_day, m_day, Xnew, f_mean, f_var,
                     path=os.path.join(args.model_figures_dir, f'f_mark.pdf'))
        save_gpr_model(model, model_path)
    saved_model = tf.saved_model.load(model_path)
    return saved_model


def get_run_identifier(args):
    output_id = f'action.p{args.patient_id}'
    return output_id


def prepare_tm_dirs(args):
    run_id = get_run_identifier(args)
    args.patient_model_dir = os.path.join(args.model_dir, run_id)
    args.model_figures_dir = os.path.join(args.patient_model_dir, 'figures')
    args.model_metric_dir = os.path.join(args.patient_model_dir, 'metric')
    args.model_pred_dir = os.path.join(args.model_dir, 'action_pred')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.model_pred_dir, exist_ok=True)
    os.makedirs(args.patient_model_dir, exist_ok=True)
    os.makedirs(args.model_metric_dir, exist_ok=True)
    os.makedirs(args.model_figures_dir, exist_ok=True)
    return args


def set_mark_domain(args, marks):
    marks = marks[np.isfinite(marks)]
    if len(marks) > 0:
        mark_domain = (np.quantile(marks, 0.0), np.quantile(marks, 1.0))
    else:
        mark_domain = (4.0, 8.0)
    args.mark_domain = mark_domain
    return args


def set_time_domain(args, rel_events):
    if 'a' in args.action_components:
        relative_times_for_action_dim = np.concatenate([rel_at_all[:, args.action_dim] for rel_at_all in rel_events])
        relative_times_for_action_dim = relative_times_for_action_dim[np.isfinite(relative_times_for_action_dim)]
        args.action_time_domain = (0.0, np.quantile(relative_times_for_action_dim, 0.95))
    else:
        args.action_time_domain = args.domain

    return args


def prepare_init_args(args):
    args.patient_ids = [int(s) for s in args.patient_ids.split(',')]
    init_dict = vars(args)
    init_dict.update(constants_rw.GENERAL_PARAMS)
    init_dict.update(constants_rw.TREATMENT_PARAMS)
    init_dict.update(constants_rw.GPRPP_PARAMS)
    if init_dict['action_components'] == 'bao':
        init_dict.update(constants_rw.GPRPP_PARAMS_FBAO)
    elif init_dict['action_components'] == 'ba':
        init_dict.update(constants_rw.GPRPP_PARAMS_FBA)
    elif init_dict['action_components'] == 'bo':
        init_dict.update(constants_rw.GPRPP_PARAMS_FBO)
    elif init_dict['action_components'] == 'ao':
        init_dict.update(constants_rw.GPRPP_PARAMS_FAO)
    elif init_dict['action_components'] == 'b':
        init_dict.update(constants_rw.GPRPP_PARAMS_FB)
    else:
        raise ValueError('Wrong action component combination!')
    args = argparse.Namespace(**init_dict)
    args.D = args.Dt + args.Dm
    args.marked = args.Dm > 0
    args.time_dims = list(range(args.Dt))
    args.mark_dims = list(range(args.Dt, args.D))
    return args


def predict_ga(models, args):
    X = np.full((100, args.D), np.inf, dtype=float)
    X_flat = np.linspace(*args.action_time_domain, 100)
    X[:, args.action_dim] = X_flat
    ga = [model[0].predict_lambda_compiled(X)[0] for model in models]
    ga_path = os.path.join(args.model_pred_dir, f'ga.npz')
    np.savez(ga_path, x=X_flat, ga=ga, allow_pickle=True)


def predict_go(models, args):
    N_test = 100
    t_grid = np.linspace(*args.domain, N_test + 1)
    m_grid = np.linspace(*args.mark_domain, N_test + 1)
    xx, yy = np.meshgrid(t_grid, m_grid)
    X_plot = np.vstack((xx.flatten(), yy.flatten())).T
    X = np.full((X_plot.shape[0], args.D), np.inf, dtype=float)
    X[:, args.outcome_dim] = X_plot[:, 0]
    X[:, args.outcome_dim + 1] = X_plot[:, 1]

    idx = int(N_test // 2)
    go = []
    for model in models:
        f_pred, lambda_pred = model[0].predict_lambda_compiled(X)
        f_pred_2d = f_pred.numpy().reshape(*xx.shape)
        f_mark_effect = f_pred_2d[:, idx]
        go.append(f_mark_effect)
    go_path = os.path.join(args.model_pred_dir, f'go.npz')
    np.savez(go_path, x=m_grid, go=go, allow_pickle=True)


def predict_mark_intensity(models, args):
    N_test = 100
    X_flat = np.linspace(0.0, 24.0, N_test)
    f_mark = []
    for model in models:
        f_mean, _ = model[1].predict_f_compiled(X_flat.reshape(-1, 1))
        f_mean = f_mean.numpy().flatten()
        f_mark.append(f_mean)

    mark_path = os.path.join(args.model_pred_dir, f'f_mark.npz')
    np.savez(mark_path, x=X_flat, f_mark=f_mark, allow_pickle=True)


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
    init_args = prepare_init_args(init_args)
    np.random.seed(init_args.seed)
    tf.random.set_seed(init_args.seed)
    models = []
    for patient_id in init_args.patient_ids:
        init_args.patient_id = patient_id
        prepare_tm_dirs(init_args)
        model = demo(init_args)
        models.append(model)
    if 'a' in init_args.action_components:
        predict_ga(models, init_args)
    if 'o' in init_args.action_components:
        predict_go(models, init_args)
    predict_mark_intensity(models, init_args)
