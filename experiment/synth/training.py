import os
import numpy as np
import gpflow
import tensorflow as tf

from evaluate.plot_om import plot_fs_pred_multiple
from experiment.synth.sample_train_data import get_joint_dep_samplers
from experiment.real_world.outcome.run_outcome_multiple_marked_hierarchical import save_om_multiple_marked_hier, \
    get_baseline_kernel, get_treatment_base_kernel
from experiment.real_world.treatment.run_vbpp import train_vbpp
from evaluate.plot import plot_joint_intensity_train
from experiment.predict import predict_ft_marked_hier_compiled, predict_f_shared_marked_hier_compiled
from experiment.real_world.treatment.run_gprpp_joint import save_gprpp_model, \
    run_treatment_time_intensity, run_treatment_mark_intensity_pooled, save_gpr_model, prepare_tm_dirs
from models.benchmarks.schulam.run_schulam_saria import train_schulam
from utils.tm_utils import get_tm_run_args
from models.outcome.piecewise_se.model_multiple_marked_hierarchical import TRSharedMarkedHier


def train_models(dataset_train, args):
    train_treatment_model(dataset_train['dataset'], args)
    train_outcome_model(dataset_train['dataset'], args, true_f_means=dataset_train['true_f_mean'])
    if args.plot_joint_train:
        plot_joint_train_fit(dataset_train['dataset'], dataset_train['true_f_mean'], args)
    # Benchmarks
    # (Hua et al., 2021) is trained with Rscript, so not running here
    train_vbpp_treatment_model(dataset_train['dataset'], args)
    train_schulam_outcome_model(dataset_train['dataset'], args)


def train_treatment_model(ds_train, args):
    oracle_sampler = get_joint_dep_samplers(args)
    action_sampler, outcome_sampler = oracle_sampler
    for porder, patient_id in enumerate(args.observational_action_ids):
        action_time_sampler, action_mark_sampler = action_sampler[porder]
        tm_run_args = get_tm_run_args(patient_id, args.policy_model_dir, args)
        prepare_tm_dirs(tm_run_args)
        ds_pid = [ds for i, ds in enumerate(ds_train) if i % 2 == porder]
        if not os.path.exists(os.path.join(tm_run_args.patient_model_dir, 'time_intensity')):
            run_treatment_time_intensity(ds_pid, tm_run_args, oracle_model=action_time_sampler)
        if not os.path.exists(os.path.join(tm_run_args.patient_model_dir, 'mark_intensity')):
            run_treatment_mark_intensity_pooled(ds_pid, tm_run_args)


def train_outcome_model(ds_train, args, true_f_means):
    outcome_model_path = os.path.join(args.outcome_model_dir, 'outcome_model')
    oracle_sampler = get_joint_dep_samplers(args)
    action_sampler, outcome_sampler = oracle_sampler
    if not os.path.exists(outcome_model_path):
        args.patient_ids = list(range(args.n_patient))
        model = build_outcome_model(ds_train, args)
        gpflow.utilities.print_summary(model)
        opt = gpflow.optimizers.Scipy()
        min_logs = opt.minimize(model.training_loss,
                                model.trainable_variables,
                                compile=True,
                                options={"disp": True,
                                         "maxiter": args.outcome_maxiter})
        gpflow.utilities.print_summary(model)
        save_om_multiple_marked_hier(model, args.outcome_model_dir, args)
        trained_model = tf.saved_model.load(outcome_model_path)
        predict_ft_marked_hier_compiled(trained_model, args, oracle_model=outcome_sampler)
        f_means, f_vars = predict_f_shared_marked_hier_compiled(trained_model, ds_train)
        plot_fs_pred_multiple(f_means, f_vars, ds_train, args)
        plot_fs_pred_multiple(f_means, f_vars, ds_train, args, plot_var=False, true_f_means=true_f_means)


def train_vbpp_treatment_model(ds_train, args):
    for porder, patient_id in enumerate(args.observational_action_ids):
        policy_dir = args.policy_model_dirs[patient_id]
        if not os.path.exists(os.path.join(policy_dir, 'vbpp')):
            ds_pid = [ds for i, ds in enumerate(ds_train) if i % 2 == porder]
            events = []
            for ds in ds_pid:
                t = ds[2]
                ts = [t[np.logical_and(t > i * 24.0, t < (i+1) * 24.0)] % 24.0 for i in range(args.n_day_train)]
                events += ts
            events = [ev.reshape(-1, 1) for ev in events]
            train_vbpp(events, patient_id, policy_dir, np.array(args.domain).reshape(1, 2), args)


def build_outcome_model(ds_trains, args):
    N = len(ds_trains)
    X = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_trains]
    t = [np.hstack([ds[2].astype(np.float64).reshape(-1, 1),
                    ds[3].astype(np.float64).reshape(-1, 1)]) for ds in ds_trains]
    Y = [ds[1].reshape(-1, 1) for ds in ds_trains]
    model = TRSharedMarkedHier(data=(X, Y), t=t, T_treatment=args.T_treatment,
                               patient_ids=args.patient_ids,
                               baseline_kernels=[get_baseline_kernel() for _ in range(N)],
                               treatment_base_kernel=get_treatment_base_kernel(),
                               mean_functions=[gpflow.mean_functions.Zero() for _ in range(N)],
                               noise_variance=1.0,
                               train_noise=True, )
    return model


def train_schulam_outcome_model(patient_ds, args):
    train_ds_schulam, outcome_model_schulam = train_schulam(patient_ds, args)
    args.train_ds_schulam = train_ds_schulam
    args.outcome_model_schulam = outcome_model_schulam


def plot_joint_train_fit(datasets, true_f_means, args):
    baseline_times = [np.zeros((1, 1)) for _ in datasets]
    est_action_gprpp = {
        i: [tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], intensity_name))
            for intensity_name in ['time_intensity', 'mark_intensity']]
        for i, pi in enumerate(args.observational_action_ids)
    }
    est_outcome_model = tf.saved_model.load(os.path.join(args.outcome_model_dir, 'outcome_model'))
    model = (est_action_gprpp, est_outcome_model)
    plot_joint_intensity_train(model, datasets, baseline_times, args,
                               oracle_model=get_joint_dep_samplers(args), true_f_means=true_f_means)