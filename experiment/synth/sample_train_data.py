import os
import argparse
import numpy as np
import tensorflow as tf

from experiment.real_world.dataset.glucose_dataset import remove_closeby
from experiment.real_world.outcome.run_outcome_multiple_marked_hierarchical import run_outcome_demo
from evaluate.plot_om import plot_fs_pred
from sample.sample_joint import sample_joint_tuples
from models.benchmarks.hua.model_hua_from_mcmc import get_dosage_hua
from utils import constants_synth
from utils.utils import prepare_dirs, add_parser_arguments


def load_train_data(args):
    sample_path = os.path.join(args.sample_output_dir, f'patients.npz')
    if not os.path.exists(sample_path):
        patient_sampler = get_joint_dep_samplers(args)
        patient_datasets = []
        true_f_means = []
        for pidx in range(args.n_patient):
            action_pidx = pidx % len(args.observational_action_ids)
            outcome_pidx = pidx % len(args.observational_outcome_ids)
            actions, outcomes, true_f_mean_pidx, true_f_var_pidx = sample_joint_tuples(patient_sampler, action_pidx,
                                                                                       outcome_pidx, args.n_day_train,
                                                                                       args=args)
            ds_pidx = (outcomes[:, 0], outcomes[:, 1], actions[:, 0], actions[:, 1])
            patient_datasets.append(ds_pidx)
            true_f_means.append(true_f_mean_pidx)
            print(f'Patient[{pidx}], #treatments: {len(actions[:, 0])}')
            plot_fs_pred(ds_pidx[0], true_f_mean_pidx, true_f_var_pidx,
                         [r'$\mathbf{f_b}$', r'$\mathbf{f_a}$', r'$\mathbf{f}$'],
                         ['tab:orange', 'tab:green', 'tab:blue'], ds_pidx, pidx, args,
                         path=os.path.join(args.sample_figures_dir, f'pidx{pidx}.pdf'), plot_var=False)
        true_f_means = [np.concatenate([fms[0] for fms in true_f_means]),
                        np.concatenate([fms[1] for fms in true_f_means]),
                        np.concatenate([fms[2] for fms in true_f_means])]
        np.savez(sample_path, ds=patient_datasets, true_f_means=true_f_means, allow_pickle=True)
    else:
        dataset_dict = np.load(sample_path, allow_pickle=True)
        true_f_means = dataset_dict['true_f_means']
        patient_datasets = dataset_dict['ds']
    save_r_data(patient_datasets, args)
    return {'dataset': patient_datasets, 'true_f_mean': true_f_means}


def save_r_data(patient_datasets, args):
    Tr_t, Dr_t, Yr_t, idr_t = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}
    Tr_o, Dr_o, Yr_o, idr_o = [], [], [], []
    for pidx in range(args.n_patient):
        action_pidx = pidx % len(args.observational_action_ids)
        ds = patient_datasets[pidx]
        actions, outcomes = np.stack([ds[2], ds[3]]).T, np.stack([ds[0], ds[1]]).T
        Tr_ti, Dr_ti, Yr_ti, idr_ti = transform_to_r_treatment_data(actions, outcomes, pidx)
        Tr_oi, Yr_oi, Dr_oi, idr_oi = transform_to_r_outcome_data(actions, outcomes, pidx, args)
        Tr_t[action_pidx].append(Tr_ti)
        Dr_t[action_pidx].append(Dr_ti)
        Yr_t[action_pidx].append(Yr_ti)
        idr_t[action_pidx].append(idr_ti)
        Tr_o.append(Tr_oi)
        Dr_o.append(Dr_oi)
        Yr_o.append(Yr_oi)
        idr_o.append(idr_oi)
    save_r_treatment_data(Dr_t, Tr_t, Yr_t, idr_t, args)
    save_r_outcome_data(Dr_o, Tr_o, Yr_o, idr_o, args)


def save_r_treatment_data(Dr, Tr, Yr, idr, args):
    for k in Dr.keys():
        Dr_k = np.concatenate(Dr[k])
        Tr_k = np.concatenate(Tr[k])
        Yr_k = np.concatenate(Yr[k])
        idr_k = np.concatenate(idr[k])
        cr_k = np.ones_like(Yr_k, dtype=int)
        sr_k = np.ones_like(Yr_k) * 25.0
        # X0_inds_k = np.stack([idr_k * 1e-1, np.random.randn(len(idr_k)) * 1e-1]).T
        np.savez(os.path.join(args.sample_output_dir, f'treatment_data_p{k}_s{args.seed}.npz'),
                 D=Dr_k, Y=Yr_k, Ts=Tr_k, id=idr_k, censor=cr_k, surv_time=sr_k, Npat=args.n_patient//2,
                 allow_pickle=True)


def save_r_outcome_data(Dr, Tr, Yr, idr, args):
    Dr_ = np.concatenate(Dr)
    Tr_ = np.concatenate(Tr)
    Yr_ = np.concatenate(Yr)
    idr_ = np.concatenate(idr)
    cr_ = np.ones_like(Yr_, dtype=int)
    sr_ = np.ones_like(Yr_) * 25.0
    np.savez(os.path.join(args.sample_output_dir, f'outcome_data_s{args.seed}.npz'),
             D=Dr_, Y=Yr_, Ts=Tr_, id=idr_, censor=cr_, surv_time=sr_, Npat=args.n_patient,
             allow_pickle=True)


def transform_to_r_outcome_data(actions, outcomes, pidx, args):
    x, y = outcomes[:, 0], outcomes[:, 1]
    Dr_i = get_dosage_hua(x, actions, args)
    idr_i = np.ones_like(x, dtype=int) * (pidx+1)
    return x, y, Dr_i, idr_i


def transform_to_r_treatment_data(actions, outcomes, pidx):
    Tr_i, Dr_i = np.copy(actions[:, 0]), np.copy(actions[:, 1])
    # Tr_i, Dr_i = remove_closeby(Tr_i, Dr_i, threshold=0.5)
    Yr_i = []
    for ai in Tr_i:
        mask_smaller = outcomes[:, 0] < ai
        yi = outcomes[:, 1][mask_smaller][-1]
        Yr_i.append(yi)
    Yr_i = np.array(Yr_i)
    idr_i = np.ones_like(Yr_i, dtype=int) * (pidx // 2 + 1)
    return Tr_i, Dr_i, Yr_i, idr_i


def get_joint_dep_samplers(args):
    action_sampler_dict = {}
    for i, action_id in enumerate(args.observational_action_ids):
        action_model_path = os.path.join(args.sampler_dir, f'action.p{action_id}')
        time_intensity = tf.saved_model.load(os.path.join(action_model_path, 'time_intensity'))
        mark_intensity = tf.saved_model.load(os.path.join(action_model_path, 'mark_intensity'))
        action_sampler_dict[i] = (time_intensity, mark_intensity)
    outcome_str = ','.join(str(e) for e in args.outcome_sampler_patient_ids)
    model_path = os.path.join(args.sampler_dir, f'outcome.p{outcome_str}', 'outcome_model')
    outcome_oracle_model = tf.saved_model.load(model_path)
    return action_sampler_dict, outcome_oracle_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patient", type=int, default=20)
    parser.add_argument("--n_day_train", type=int, default=2)  # same for obs
    parser.add_argument("--n_day_test", type=int, default=1)  # same for test and int
    parser.add_argument("--sampler_dir", type=str, default='sampler')
    parser.add_argument("--model_output_dir", type=str, default='models')
    parser.add_argument("--sample_output_dir", type=str, default='samples')
    parser.add_argument("--seed", type=int, default=1)
    parser = add_parser_arguments(parser, constants_synth.GENERAL_PARAMS)
    parser = add_parser_arguments(parser, constants_synth.SAMPLER_PARAMS)
    parser = add_parser_arguments(parser, constants_synth.GPRPP_PARAMS)
    parser = add_parser_arguments(parser, constants_synth.GPRPP_FAO)
    #
    init_args = parser.parse_args()

    # First sampler - equal size
    init_args.action_sampler_train_fnc = None
    init_args.train_ds_schulam = None
    init_args.outcome_sampler_train_fnc = run_outcome_demo
    #
    np.random.seed(init_args.seed)
    tf.random.set_seed(init_args.seed)
    #
    init_args = prepare_dirs(init_args)
    load_train_data(init_args)
