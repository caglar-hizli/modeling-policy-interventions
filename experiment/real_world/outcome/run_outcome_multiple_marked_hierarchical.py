import os
import argparse
import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.utilities import to_default_float

from evaluate.plot_om import plot_fs_pred_multiple, compare_f_preds
from experiment.predict import predict_ft_marked_hier_compiled, \
    predict_f_shared_marked_hier_compiled, compare_ft_marked_hier_compiled, compare_fb_compiled
from experiment.real_world.dataset.glucose_dataset import prepare_glucose_ds, load_glucose_dfs
from models.outcome.piecewise_se.model_multiple_marked_hierarchical import TRSharedMarkedHier
from utils import constants_rw


def run_outcome_demo(args):
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

    outcome_model_path = os.path.join(args.outcome_model_dir, f'outcome_model')
    if not os.path.exists(outcome_model_path):
        train_ft(ds_trains, args)
    model = tf.saved_model.load(outcome_model_path)

    predict_ft_marked_hier_compiled(model, args)
    f_means, f_vars = predict_f_shared_marked_hier_compiled(model, ds_tests)
    f_mean = f_means[-1]
    offset = 0
    for i, ds_plot in enumerate(ds_tests):
        predict_shape = ds_plot[0].shape[0]
        f_means_i = f_mean[offset:offset + predict_shape]
        mse = np.mean(np.square(f_means_i.numpy().flatten() - ds_plot[1]))
        print(f'Patient[{i}], mse={mse:.2f}')
    plot_fs_pred_multiple(f_means, f_vars, ds_tests, args, run='test')
    f_means, f_vars = predict_f_shared_marked_hier_compiled(model, ds_alls)
    np.savez(os.path.join(args.outcome_model_pred_dir, f'f_outcome.npz'), f_means=f_means, f_vars=f_vars,
             ds_alls=ds_alls, allow_pickle=True)

    xnew, ft_means = compare_ft_marked_hier_compiled(model, args)
    np.savez(os.path.join(args.outcome_model_pred_dir, f'ft_outcome.npz'), ft_means=ft_means, xnew=xnew,
             ds_alls=ds_alls, allow_pickle=True)

    xnew, fb_means = compare_fb_compiled(model, args)
    np.savez(os.path.join(args.outcome_model_pred_dir, f'fb_outcome.npz'), fb_means=fb_means, xnew=xnew,
             ds_alls=ds_alls, allow_pickle=True)
    return model


def train_ft(ds_trains, args):
    model = build_outcome_model_glucose(ds_trains, args)
    gpf.utilities.print_summary(model)
    opt = gpf.optimizers.Scipy()
    min_logs = opt.minimize(model.training_loss,
                            model.trainable_variables,
                            compile=True,
                            options={"disp": True,
                                     "maxiter": args.maxiter})
    gpf.utilities.print_summary(model)
    save_om_multiple_marked_hier(model, args.outcome_model_dir, args)


def build_outcome_model_glucose(ds_trains, args):
    N = len(ds_trains)
    X = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_trains]
    t = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                    ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_trains]
    Y = [ds[1].reshape(-1, 1) for ds in ds_trains]
    model = TRSharedMarkedHier(data=(X, Y), t=t, T_treatment=args.T_treatment,
                               patient_ids=args.observational_outcome_ids,
                               baseline_kernels=[get_baseline_kernel() for _ in range(N)],
                               treatment_base_kernel=get_treatment_base_kernel(),
                               mean_functions=[gpf.mean_functions.Zero() for _ in range(N)],
                               noise_variance=1.0,
                               train_noise=True, )
    return model


def get_baseline_kernel():
    # kb_se = get_se_kernel_long_ell()
    kb_per = gpf.kernels.Periodic(base_kernel=get_se_kernel_periodic(), period=24.0)
    gpf.utilities.set_trainable(kb_per.period, False)
    return gpf.kernels.Constant() + kb_per


def get_matern_kernel_st_noise():
    k = gpf.kernels.Matern12(variance=0.05, lengthscales=0.3)
    gpf.utilities.set_trainable(k, False)
    return k


def get_se_kernel_long_ell():
    kb_se = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=20.0)
    kb_se.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kb_se.lengthscales.prior = tfp.distributions.Gamma(to_default_float(20.0), to_default_float(1.0))
    return kb_se


def get_se_kernel_periodic():
    kb_se = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=10.0)
    kb_se.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    # kb_se.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(0.1))
    kb_se.lengthscales.prior = tfp.distributions.Gamma(to_default_float(10.0), to_default_float(1.0))
    return kb_se


def get_treatment_base_kernel():
    kse = gpf.kernels.SquaredExponential(variance=1.0, lengthscales=0.5, active_dims=[0])
    kse.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kse.lengthscales.prior = tfp.distributions.Gamma(to_default_float(1.0), to_default_float(2.0))
    return kse


def save_om_multiple_marked_hier(model: TRSharedMarkedHier, output_dir, args=None):
    model.predict_ft_w_tnew_compiled = tf.function(
        model.predict_ft_w_tnew,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32)]
    )
    model.predict_f_w_tnew_compiled = tf.function(
        model.predict_f_w_tnew,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32)]
    )
    model.predict_y_w_tnew_compiled = tf.function(
        model.predict_y_w_tnew,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32)]
    )
    model.predict_ft_w_tnew_conditional_compiled = tf.function(
        model.predict_ft_w_tnew_conditional,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32),
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32)]
    )
    model.predict_baseline_compiled = tf.function(
        model.predict_baseline,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N]
    )
    model.predict_baseline_conditional_compiled = tf.function(
        model.predict_baseline_conditional,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         [tf.TensorSpec(shape=[None, 2], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=[None, ], dtype=tf.int32),
                         [tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N]
    )
    model.predict_baseline_samples_compiled = tf.function(
        model.predict_baseline_samples,
        input_signature=[[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)] * model.N,
                         tf.TensorSpec(shape=None, dtype=tf.int32)]
    )
    model.predict_ft_single_for_patient_compiled = tf.function(
        model.predict_ft_single_for_patient,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64),
                         tf.TensorSpec(shape=[None, 2], dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.int32)]
    )
    tf.saved_model.save(model, os.path.join(output_dir, f'outcome_model'))


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
    os.makedirs(init_args.outcome_model_dir, exist_ok=True)
    os.makedirs(init_args.outcome_model_figures_dir, exist_ok=True)
    os.makedirs(init_args.outcome_model_pred_dir, exist_ok=True)
    tf.random.set_seed(init_args.seed)
    run_outcome_demo(init_args)
