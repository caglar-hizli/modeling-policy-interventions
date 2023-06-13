import os
import logging
import numpy as np


def get_relative_input_joint(action_times, outcome_tuples, baseline_times, args, D):
    T = args.domain[1]
    rel_at_actions = []
    rel_at_all_points = []
    abs_all_points = []
    for action_time, outcome_tuple, baseline_time in zip(action_times, outcome_tuples, baseline_times):
        # Relative times to events: needed for the data term
        outcome_time = outcome_tuple[:, 0]
        rel_tuple_at_action = get_relative_input_by_query(action_time, baseline_time, action_time, outcome_tuple, args)\
            if len(action_time) > 0 else np.array([])
        rel_at_actions.append(rel_tuple_at_action)
        # Relative times to all: needed for the integral term
        all_events = []
        if 'b' in args.action_components:
            all_events.append(baseline_time.flatten())
        if 'a' in args.action_components:
            all_events.append(action_time.flatten())
        if 'o' in args.action_components:
            all_events.append(outcome_time.flatten())
        abs_time_point = np.sort(np.concatenate(all_events))
        rel_tuple_at_all_points = get_relative_input_by_query(abs_time_point, baseline_time, action_time,
                                                              outcome_tuple, args)
        # Add last region to relative times to all points for the integral computation
        last_action_time = action_time[-1] if len(action_time) > 0 else 0.0
        last_region = []
        if 'b' in args.action_components:
            last_region.append(T)
        if 'a' in args.action_components:
            last_region.append(T - last_action_time)
        if 'o' in args.action_components:
            last_region.append(T - outcome_tuple[-1, 0])
            last_region.append(outcome_tuple[-1, 1])
        last_region = np.array(last_region)
        rel_tuple_at_all_points_shifted = np.concatenate([rel_tuple_at_all_points[1:, :], last_region.reshape(1, -1)])
        rel_at_all_points.append(rel_tuple_at_all_points_shifted)
        abs_all_points.append(abs_time_point.reshape(-1, 1))
    return rel_at_actions, rel_at_all_points, abs_all_points


def get_relative_input_by_query(query_points, baseline_time, action_time, outcome_tuple, args):
    Nq = query_points.shape[0]
    rel_tuple_at_all_points = np.full((Nq, args.D), np.inf)
    d = 0
    # Relative time to baseline == absolute time
    if 'b' in args.action_components:
        rel_tuple_at_all_points[:, d] = query_points - baseline_time
        d += 1
    # Relative time to actions
    if 'a' in args.action_components:
        for i, xi in enumerate(query_points):
            smaller_action_idx = np.where(action_time < xi)[0]
            if len(smaller_action_idx) > 0:
                largest_smaller_idx = np.max(smaller_action_idx)
                rel_tuple_at_all_points[i, d] = xi - action_time[largest_smaller_idx]
        d += 1

    if 'o' in args.action_components:
        # Relative time to outcomes
        for i, xi in enumerate(query_points):
            smaller_outcome_idx = np.where(outcome_tuple[:, 0] < xi)[0]
            if len(smaller_outcome_idx) > 0:
                largest_smaller_idx = np.max(smaller_outcome_idx)
                rel_tuple_at_all_points[i, d:] = outcome_tuple[largest_smaller_idx, :]
                rel_tuple_at_all_points[i, d] = xi - rel_tuple_at_all_points[i, d]  # Relative time to outcome
        d += 1

    return rel_tuple_at_all_points


def set_logger(log_name):
    logging.basicConfig(filename=log_name,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


def get_metric_identifier(args):
    output_id = f'metric.nci{args.classifier_n_train_day}.ncf{args.classifier_cf_n_train_day}.' \
                f'ncfp{args.classifier_cf_n_train_per_day}'
    return output_id


def get_oracle_classifier_identifier(cl_name, cl_target, args):
    cl_type = 'cf' if 'c' in cl_name else 'int'
    actions_str = ','.join(map(str, cl_target[0]))
    outcomes_str = ','.join(map(str, cl_target[1]))
    output_id = f'oracle.{cl_type}.{cl_name}'
    output_id += f'.nc{args.classifier_n_train_day}' if cl_type == 'int' else \
        f'.nc{args.classifier_cf_n_train_day}.ncp{args.classifier_cf_n_train_per_day}'
    output_id += f'.pa{actions_str}.po{outcomes_str}'
    return output_id


def get_pred_classifier_identifier(patient_idx, cl_name, args):
    patient_classifier_targets = get_patient_classifier_targets(patient_idx, args, is_sorted=False)
    target = patient_classifier_targets[cl_name]
    actions_str = ','.join(map(str, target[0]))
    outcomes_str = ','.join(map(str, target[1]))
    output_id = f'pred.int.{cl_name}.np{args.n_patient}.ntr{args.n_day_train}.pi{patient_idx}' \
                f'.nc{args.classifier_n_train_day}.pa{actions_str}.po{outcomes_str}'
    return output_id


def get_run_identifier(args):
    action_str = ','.join(str(e) for e in args.observational_action_ids)
    outcome_str = ','.join(str(e) for e in args.observational_outcome_ids)
    output_id = f'np{args.n_patient}.ntr{args.n_day_train}.pa{action_str}.po{outcome_str}' \
                f'.no{args.n_outcome}.t{args.regular_measurement_times}'
    return output_id


def get_oracle_classifier_targets(args):
    classifier_targets = set()
    for patient_idx in args.cl_patient_ids:
        patient_classifier_targets = get_patient_classifier_targets(patient_idx, args, is_sorted=True)
        classifier_targets = classifier_targets.union(patient_classifier_targets.values())
    return classifier_targets


def get_oracle_classifier_targets_dict(args):
    classifier_targets_dict = {cl_target: set() for cl_target in args.classifier_targets}
    for patient_idx in args.cl_patient_ids:
        patient_classifier_targets = get_patient_classifier_targets(patient_idx, args, is_sorted=True)
        for k, v in classifier_targets_dict.items():
            classifier_targets_dict[k] = v.union((patient_classifier_targets[k],))

    return classifier_targets_dict


def get_patient_classifier_targets(patient_idx, args, is_sorted=True):
    patient_classifier_targets = {}
    if 'oi' in args.classifier_targets:
        actions_to_classify = (args.patient_observational_action_ids[patient_idx],
                               args.patient_interventional_action_ids[patient_idx])
        outcomes_to_classify = (args.patient_observational_outcome_ids[patient_idx],
                                args.patient_interventional_outcome_ids[patient_idx])
        # Sort by actions ids for oracle classifiers, so that we don't learn equivalent with ids
        # in different order for multiple patients
        if is_sorted:
            actions_sort_idx = np.argsort(actions_to_classify)
            actions_to_classify = tuple(np.take(actions_to_classify, actions_sort_idx))
            outcomes_to_classify = tuple(np.take(outcomes_to_classify, actions_sort_idx))
        obs_vs_int_classifier_target = (actions_to_classify, outcomes_to_classify)
        patient_classifier_targets['oi'] = obs_vs_int_classifier_target

    if 'ii' in args.classifier_targets:
        int_vs_int_classifier_target = ((args.patient_interventional_action_ids[patient_idx],
                                         args.patient_interventional_action_ids[patient_idx]),
                                        (args.patient_interventional_outcome_ids[patient_idx],
                                         args.patient_interventional_outcome_ids[patient_idx]))
        patient_classifier_targets['ii'] = int_vs_int_classifier_target

    if 'ic' in args.classifier_targets:
        int_vs_cf_classifier_target = ((args.patient_interventional_action_ids[patient_idx],
                                        (args.patient_observational_action_ids[patient_idx],
                                         args.patient_interventional_action_ids[patient_idx])),
                                       (args.patient_interventional_outcome_ids[patient_idx],
                                        (args.patient_observational_outcome_ids[patient_idx],
                                         args.patient_interventional_outcome_ids[patient_idx])))
        patient_classifier_targets['ic'] = int_vs_cf_classifier_target

    if 'oc' in args.classifier_targets:
        int_vs_cf_classifier_target = ((args.patient_observational_action_ids[patient_idx],
                                        (args.patient_observational_action_ids[patient_idx],
                                         args.patient_interventional_action_ids[patient_idx])),
                                       (args.patient_observational_outcome_ids[patient_idx],
                                        (args.patient_observational_outcome_ids[patient_idx],
                                         args.patient_interventional_outcome_ids[patient_idx])))
        patient_classifier_targets['oc'] = int_vs_cf_classifier_target

    if 'cc' in args.classifier_targets:
        cf_vs_cf_classifier_target = (((args.patient_observational_action_ids[patient_idx],
                                        args.patient_interventional_action_ids[patient_idx]),
                                       (args.patient_observational_action_ids[patient_idx],
                                        args.patient_interventional_action_ids[patient_idx])),
                                      ((args.patient_observational_outcome_ids[patient_idx],
                                        args.patient_interventional_outcome_ids[patient_idx]),
                                       (args.patient_observational_outcome_ids[patient_idx],
                                        args.patient_interventional_outcome_ids[patient_idx])))
        patient_classifier_targets['cc'] = cf_vs_cf_classifier_target

    return patient_classifier_targets


def target_not_exists(target_to_check, classifier_targets):
    for classifier_target in classifier_targets:
        if np.array_equal(target_to_check, classifier_target):
            return True
    return False


def get_oracle_classifier_output_dir(cl_name, cl_target, args):
    classifier_id = get_oracle_classifier_identifier(cl_name, cl_target, args)
    sample_output_dir = os.path.join(args.classifier_sample_output_dir, classifier_id)
    model_output_dir = os.path.join(args.classifier_model_output_dir, classifier_id)
    return sample_output_dir, model_output_dir


def get_pred_classifier_output_dir(patient_idx, cl_name, args):
    classifier_id = get_pred_classifier_identifier(patient_idx, cl_name, args)
    sample_output_dir = os.path.join(args.classifier_sample_output_dir, classifier_id)
    model_output_dir = os.path.join(args.classifier_model_output_dir, classifier_id)
    return sample_output_dir, model_output_dir


def add_parser_arguments(parser, param_dict):
    for k, v in param_dict.items():
        parser.add_argument(f'--{k}', default=v)
    return parser


def prepare_dirs(args):
    run_id = get_run_identifier(args)
    #
    args.hua_sample_output_dir = os.path.join(args.sample_output_dir, 'mcmc_output')
    args.model_output_dir = os.path.join(args.model_output_dir, f'joint.seed{args.seed}')
    args.sample_output_dir = os.path.join(args.sample_output_dir, f'joint.seed{args.seed}')
    args.model_output_dir = os.path.join(args.model_output_dir, run_id)
    args.sample_output_dir = os.path.join(args.sample_output_dir, run_id)
    args.model_figures_dir = os.path.join(args.model_output_dir, 'figures')
    args.model_evaluate_dir = os.path.join(args.model_output_dir, 'evaluate')
    args.sample_figures_dir = os.path.join(args.sample_output_dir, 'figures')

    args.model_evaluate_dirs = {target_distribution: os.path.join(args.model_evaluate_dir,
                                                                  f'policy_{target_distribution}')
                              for target_distribution in args.target_distributions}
    args.policy_model_dir = os.path.join(args.model_output_dir, f'action')
    args.policy_model_dirs = {k: os.path.join(args.policy_model_dir, f'action.p{k}')
                              for k in args.observational_action_ids}
    args.outcome_model_dir = os.path.join(args.model_output_dir, 'outcome')
    args.outcome_model_figures_dir = os.path.join(args.outcome_model_dir, 'figures')
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.model_figures_dir, exist_ok=True)
    os.makedirs(args.model_evaluate_dir, exist_ok=True)
    os.makedirs(args.outcome_model_dir, exist_ok=True)
    os.makedirs(args.policy_model_dir, exist_ok=True)
    os.makedirs(args.sample_output_dir, exist_ok=True)
    os.makedirs(args.hua_sample_output_dir, exist_ok=True)
    os.makedirs(args.sample_figures_dir, exist_ok=True)
    os.makedirs(args.outcome_model_figures_dir, exist_ok=True)
    for model_eval_dir in args.model_evaluate_dirs.values():
        os.makedirs(model_eval_dir, exist_ok=True)
    for policy_model_dir in args.policy_model_dirs.values():
        os.makedirs(policy_model_dir, exist_ok=True)
    return args


def get_tm_label(args):
    if args.action_components == 'bao':
        label = r'$\lambda_{bao}(t) = (\beta_0 + f_b(t) + f_a(t) + f_o(t))^2$'
    elif args.action_components == 'ao':
        label = r'$\lambda_{ao}(t) = (\beta_0 + f_a(t) + f_o(t))^2$'
    elif args.action_components == 'ba':
        label = r'$\lambda_{ba}(t) = (\beta_0 + f_b(t) + f_a(t))^2$'
    elif args.action_components == 'a':
        label = r'$\lambda_{a}(t) = (\beta_0 + f_a(t))^2$'
    elif args.action_components == 'bo':
        label = r'$\lambda_{bo}(t) = (\beta_0 + f_b(t) + f_o(t))^2$'
    elif args.action_components == 'b':
        label = r'$\lambda_{b}(t) = (\beta_0 + f_b(t))^2$'
    else:
        raise ValueError('Wrong action component combination!')
    return label
