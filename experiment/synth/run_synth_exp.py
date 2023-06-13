import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from experiment.synth.sample_train_data import load_train_data, get_joint_dep_samplers
from experiment.synth.training import train_models
from experiment.real_world.outcome.run_outcome_multiple_marked_hierarchical import run_outcome_demo
from evaluate.plot import plot_multiple_sampling
from evaluate.plot_om import compare_trajectory_pair
from sample.sample_multiple import sample_multiple_trajectories_int, sample_multiple_trajectories_cf, get_lambda_x
from models.benchmarks.hua.model_hua_from_mcmc import TimeIntensityHua, OutcomeModelHua, load_treatment_params_hua, \
    load_outcome_params_hua
from utils import constants_synth
from utils.utils import prepare_dirs, add_parser_arguments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_synth_exp(args):
    # Sample semi-synthetic training data
    dataset_train = load_train_data(args)
    # Train models
    train_models(dataset_train, args)

    # Sample test data from observational, interventional and counterfactual distributions
    # Observational
    target_distribution = 'observational'
    predict_period_start = args.n_day_train * args.hours_day
    period = [predict_period_start, predict_period_start + max(args.horizons)]
    obs_exp_ids = args.exp_ids[target_distribution]
    run_multiple_sampling_interventional(obs_exp_ids, period, target_distribution, args)
    report_trajectory_mse(obs_exp_ids, period, args.horizons, target_distribution, init_args)
    # Interventional
    target_distribution = 'interventional'
    int_exp_ids = args.exp_ids[target_distribution]
    run_multiple_sampling_interventional(int_exp_ids, period, target_distribution, args)
    report_trajectory_mse(int_exp_ids, period, args.horizons, target_distribution, init_args)
    # Counterfactual
    target_distribution = 'counterfactual'
    period = [0.0, max(args.horizons)]
    cf_exp_ids = args.exp_ids[target_distribution]
    run_multiple_sampling_counterfactual(cf_exp_ids, dataset_train['dataset'], period, target_distribution, args)
    report_trajectory_mse(cf_exp_ids, period, args.horizons, target_distribution, init_args)


def run_multiple_sampling_interventional(exp_ids, period, target_distribution, args):
    suffix = args.target_distribution_suffix[target_distribution]
    sample_path = os.path.join(args.model_evaluate_dirs[target_distribution], f'patient_test_multiple_{suffix}.npz')
    oracle_model_types = ('gprpp', 'oracle_gp')
    oracle_model = get_oracle_model(target_distribution, args)
    models = [oracle_model]
    model_types = [oracle_model_types]
    for exp_id in exp_ids:
        compared_model = get_estimated_model(exp_id, target_distribution, args)
        estimated_model_types = get_estimated_model_types(exp_id)
        models.append(compared_model)
        model_types.append(estimated_model_types)
    if not os.path.exists(sample_path):
        print(f'Sampling multiple models for policy [{target_distribution}], for seed {args.seed}...')
        patient_multiple_datasets, algorithm_logs, pp_intensity_logs = [], [], []
        for pidx in range(args.n_patient):
            print(f'Sampling Patient [{pidx}]...')
            action_pidx = pidx % len(args.observational_action_ids)
            outcome_pidx = pidx % len(args.observational_outcome_ids)
            patient_models = [(m[0][action_pidx], m[1]) for m in models]
            actions, outcomes, algorithm_log = sample_multiple_trajectories_int(patient_models, model_types, pidx,
                                                                                period, args=args)
            multiple_ds = [(outcome[:, 0], outcome[:, 1], action[:, 0], action[:, 1])
                           for action, outcome in zip(actions, outcomes)]
            pp_intensity_log = get_pp_intensity_log(patient_models, model_types, multiple_ds, algorithm_log, period, args)

            patient_multiple_datasets.append(multiple_ds)
            algorithm_logs.append(algorithm_log)
            pp_intensity_logs.append(pp_intensity_log)
        np.savez(sample_path, ds=patient_multiple_datasets, algorithm_log=algorithm_logs,
                 pp_intensity_log=pp_intensity_logs, allow_pickle=True)


def get_pp_intensity_log(models, model_types, multiple_ds, algorithm_log, period, args):
    pp_intensity_log = []
    baseline_time = np.array([period[0]])
    candidates = [l[2][0] for l in algorithm_log if l[2][1]]
    for i, (model, ds, model_type) in \
            enumerate(zip(models, multiple_ds, model_types)):
        time_intensity = model[0][0]
        xo, yo = ds[0], ds[1]
        action_time = ds[2]
        outcome_tuple = np.stack([xo, yo]).T
        x = np.sort(np.concatenate([action_time, xo, candidates]))
        lambdaXa_o = get_lambda_x(time_intensity, model_type[0], x,
                                  baseline_time, action_time, outcome_tuple, args)
        pp_intensity_log.append((x, lambdaXa_o))
    return pp_intensity_log


def run_multiple_sampling_counterfactual(exp_ids, train_ds, period, target_distribution, args):
    suffix = args.target_distribution_suffix[target_distribution]
    sample_path = os.path.join(args.model_evaluate_dirs[target_distribution], f'patient_test_multiple_{suffix}.npz')
    oracle_model_obs = get_oracle_model('observational', args)
    oracle_model_int = ({0: oracle_model_obs[0][1], 1: oracle_model_obs[0][0]}, oracle_model_obs[-1])
    oracle_model_types = ('gprpp', 'oracle_gp')
    models_obs = [oracle_model_obs]
    models_int = [oracle_model_int]
    model_types = [oracle_model_types]
    use_noise_posteriors = [True]
    for exp_id in exp_ids:
        compared_model_int = get_estimated_model(exp_id, 'interventional', args)
        compared_model_obs = ({0: compared_model_int[0][1], 1: compared_model_int[0][0]}, compared_model_int[-1])
        estimated_model_types = get_estimated_model_types(exp_id)
        models_obs.append(compared_model_obs)
        models_int.append(compared_model_int)
        model_types.append(estimated_model_types)
        use_noise_posteriors.append(exp_id not in ['observational', 'interventional'])

    patient_multiple_datasets = []
    if not os.path.exists(sample_path):
        print(f'Sampling multiple models for policy counterfactual, for seed {args.seed}...')
        algorithm_logs = []
        for pidx in range(args.n_patient):
            dsi_cf = train_ds[pidx]
            action_obs, outcome_obs = np.stack([dsi_cf[2], dsi_cf[3]]).T, np.stack([dsi_cf[0], dsi_cf[1]]).T
            print(f'Sampling Patient [{pidx}]...')
            action_pidx = pidx % len(args.observational_action_ids)
            outcome_pidx = pidx % len(args.observational_outcome_ids)
            patient_models_obs = [(m[0][action_pidx], m[1]) for m in models_obs]
            patient_models_int = [(m[0][action_pidx], m[1]) for m in models_int]
            actions, outcomes = sample_multiple_trajectories_cf(patient_models_obs, patient_models_int,
                                                                model_types, use_noise_posteriors,
                                                                action_obs, outcome_obs, pidx, period, args=args)
            multiple_ds = [(outcome[:, 0], outcome[:, 1], action[:, 0], action[:, 1])
                           for action, outcome in zip(actions, outcomes)]
            patient_multiple_datasets.append(multiple_ds)
        np.savez(sample_path, ds=patient_multiple_datasets, allow_pickle=True)


def report_trajectory_mse(exp_ids, period, horizons, target_distribution, args):
    suffix = args.target_distribution_suffix[target_distribution]
    sample_path = os.path.join(args.model_evaluate_dirs[target_distribution], f'patient_test_multiple_{suffix}.npz')
    ds_dict = np.load(sample_path, allow_pickle=True)
    patient_multiple_datasets = ds_dict['ds']
    out_figures_folder = os.path.join(args.model_evaluate_dirs[target_distribution], 'figures')
    os.makedirs(out_figures_folder, exist_ok=True)
    error = []
    for pidx, multiple_ds in enumerate(patient_multiple_datasets):
        mse = get_horizon_error(multiple_ds, ['oracle'] + exp_ids, pidx, horizons, out_figures_folder)
        error.append(mse)
        if target_distribution in ['observational', 'interventional']:
            plot_sampling_exp_subset(ds_dict, exp_ids, pidx, period, out_figures_folder)
    error = np.stack(error)
    pd.DataFrame(error).to_csv(os.path.join(args.model_evaluate_dirs[target_distribution],
                                            f'{target_distribution}_error.csv'))
    return error


def plot_sampling_exp_subset(ds_dict, exp_ids, pidx, period, out_figures_folder):
    plot_exp_ids = ['oracle', 'vbpp', 'hua_nptr', 'observational', 'interventional']
    model_strs = ['oracle', 'ab3', 'ab1', 'our', 'our']
    algorithm_log = ds_dict['algorithm_log'][pidx]
    pp_logs = ds_dict['pp_intensity_log'][pidx]
    multiple_ds = ds_dict['ds'][pidx]
    all_exp_ids = ['oracle'] + exp_ids
    plot_ds = [ds for exp_id, ds in zip(all_exp_ids, multiple_ds) if exp_id in plot_exp_ids]
    plot_pp_logs = [pp_log for pp_log, exp_id in zip(pp_logs, all_exp_ids) if exp_id in plot_exp_ids]
    plot_multiple_sampling(pidx, plot_ds, algorithm_log, plot_pp_logs, exp_ids, model_strs,
                           period, out_figures_folder)


def get_horizon_error(multiple_ds, exp_ids, pidx, horizons, out_folder):
    plot_exp_ids = ['oracle', 'vbpp', 'hua_nptr', 'observational', 'interventional']
    actions = [np.stack([ds[2], ds[3]]).T for ds in multiple_ds]
    outcomes = [np.stack([ds[0], ds[1]]).T for ds in multiple_ds]
    mse_arr = []
    horizon = horizons[0]
    for i in range(1, len(outcomes)):
        diff = outcomes[i][:, 1] - outcomes[0][:, 1]
        x_start = outcomes[0][0, 0]
        mask_x = outcomes[0][:, 0] < (x_start+horizon)
        mse = np.mean(np.square(diff[mask_x]))
        mse_arr.append(mse)
        file_name = os.path.join(out_folder, f'outcome_pair_pidx{pidx}_{int(horizon)}h.pdf')
        actions_horizon = [action[action[:, 0] < (x_start+horizon), :] for action in actions]
        outcomes_horizon = [ou[mask_x, :] for ou in outcomes]
        compare_trajectory_pair(actions_horizon, outcomes_horizon, exp_ids, plot_exp_ids, file_name)
    return mse_arr


def get_estimated_model_types(exp_name):
    if exp_name == 'vbpp_schulam':
        estimated_model_types = ('vbpp', 'schulam')
    elif exp_name == 'gprpp_schulam':
        estimated_model_types = ('gprpp', 'schulam')
    elif exp_name == 'gprpp_hua':
        estimated_model_types = ('gprpp', 'hua')
    elif exp_name == 'vbpp':
        estimated_model_types = ('vbpp', 'estimated_gp')
    elif exp_name == 'hua':
        estimated_model_types = ('hua', 'hua')
    elif exp_name == 'hua_nptr':
        estimated_model_types = ('hua', 'estimated_gp')
    elif exp_name in ['observational', 'interventional', 'counterfactual']:
        estimated_model_types = ('gprpp', 'estimated_gp')
    else:
        raise ValueError('Experiment does not exist!')
    return estimated_model_types


def get_oracle_model(target_distribution, args):
    oracle_model = get_joint_dep_samplers(args)
    if target_distribution == 'interventional':
        oracle_model = [
            {0: oracle_model[0][1],
             1: oracle_model[0][0]},
            oracle_model[-1]
        ]
    return oracle_model


def get_estimated_model(exp_name, target_distribution, args):
    if exp_name == 'vbpp_schulam':
        compared_action_vbpp = {
            i: [tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'vbpp')),
                tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'mark_intensity'))]
            for i, pi in enumerate(args.observational_action_ids)
        }
        estimated_model = (compared_action_vbpp, args.outcome_model_schulam)
    elif exp_name == 'gprpp_schulam':
        compared_action_vbpp = {
            i: [tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'time_intensity')),
                tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'mark_intensity'))]
            for i, pi in enumerate(args.observational_action_ids)
        }
        estimated_model = (compared_action_vbpp, args.outcome_model_schulam)
    elif exp_name == 'gprpp_hua':
        compared_action_vbpp = {
            i: [tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'time_intensity')),
                tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'mark_intensity'))]
            for i, pi in enumerate(args.observational_action_ids)
        }
        outcome_model_hua = OutcomeModelHua(*load_outcome_params_hua(
            os.path.join(args.hua_sample_output_dir, f'mcmc_outcome_i20000_s{args.seed}.npz')))
        estimated_model = (compared_action_vbpp, outcome_model_hua)
    elif exp_name == 'vbpp':
        compared_action_vbpp = {
            i: [tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'vbpp')),
                tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'mark_intensity'))]
            for i, pi in enumerate(args.observational_action_ids)
        }
        compared_outcome_model = tf.saved_model.load(os.path.join(args.outcome_model_dir, 'outcome_model'))
        estimated_model = (compared_action_vbpp, compared_outcome_model)
    elif exp_name == 'hua':
        compared_action_hua = {
            i: [TimeIntensityHua(
                *load_treatment_params_hua(
                    os.path.join(args.hua_sample_output_dir, f'mcmc_treatment{i}_i20000_s{args.seed}.npz')
                )
            ),
                tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'mark_intensity'))]
            for i, pi in enumerate(args.observational_action_ids)
        }
        outcome_model_hua = OutcomeModelHua(*load_outcome_params_hua(
            os.path.join(args.hua_sample_output_dir, f'mcmc_outcome_i20000_s{args.seed}.npz')))
        estimated_model = (compared_action_hua, outcome_model_hua)
    elif exp_name == 'hua_nptr':
        compared_action_hua = {
            i: [TimeIntensityHua(
                *load_treatment_params_hua(
                    os.path.join(args.hua_sample_output_dir, f'mcmc_treatment{i}_i20000_s{args.seed}.npz')
                )
            ),
                tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], 'mark_intensity'))]
            for i, pi in enumerate(args.observational_action_ids)
        }
        compared_outcome_model = tf.saved_model.load(os.path.join(args.outcome_model_dir, 'outcome_model'))
        estimated_model = (compared_action_hua, compared_outcome_model)
    elif exp_name in ['observational', 'interventional', 'counterfactual']:
        compared_action_gprpp = {
            i: [tf.saved_model.load(os.path.join(args.policy_model_dirs[pi], intensity_name))
                for intensity_name in ['time_intensity', 'mark_intensity']]
            for i, pi in enumerate(args.observational_action_ids)
        }
        compared_outcome_model = tf.saved_model.load(os.path.join(args.outcome_model_dir, 'outcome_model'))
        estimated_model = (compared_action_gprpp, compared_outcome_model)
    else:
        raise ValueError('Experiment does not exist!')
    if target_distribution in ['interventional', 'counterfactual'] and exp_name != 'observational':
        estimated_model = (
            {0: estimated_model[0][1],
             1: estimated_model[0][0]},
            estimated_model[1]
        )
    return estimated_model


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
    run_synth_exp(init_args)
