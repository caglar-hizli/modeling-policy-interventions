from collections import deque

import numpy as np

from sample.sample_joint import get_closest_future_time
from models.benchmarks.schulam.simulations.common import predict_outcome_trajectory
from models.benchmarks.hua.model_hua_from_mcmc import get_dosage_hua, get_relative_input_hua
from utils.utils import get_relative_input_by_query


def sample_multiple_trajectories_int(models, model_types, pidx, time_domain, args):
    noise_std = 0.1
    tm_types, om_types = [model_type[0] for model_type in model_types], [model_type[1] for model_type in model_types]
    outcome_models = [m[1] for m in models]
    time_intensities = [m[0][0] for m in models]
    mark_intensities = [m[0][1] for m in models]
    x = np.linspace(time_domain[0], time_domain[1], args.n_outcome).astype(np.float64)
    n_patient_oracle = len(args.observational_outcome_ids)
    fb_means = [
        get_fb_mean_w_args(om, om_type=om_type, x=x, patient_idx=pidx, args=args)
        for om_type, om in zip(om_types, outcome_models)
    ]
    outcomes = [np.stack([x, fb_mean]).T for fb_mean in fb_means]
    #
    T = time_domain[-1]
    current_t = x[0]
    until_t = get_closest_future_time(current_t, x, T)
    baseline_time = np.zeros((1, 1))
    actions = [np.zeros((0, 2)).astype(np.float64) for _ in models]
    interval = (current_t, until_t)
    algorithm_log = []
    ub_noise, acc_noise = np.random.uniform(size=100), np.random.uniform(size=100)
    ub_noise_q, acc_noise_q = deque([n for n in ub_noise]), deque([n for n in acc_noise])
    while interval[0] < time_domain[-1]:
        u1 = ub_noise_q.pop()
        u2 = acc_noise_q.pop()
        lambda_sup = [get_lambda_sup(time_intensity, tm_type, interval,
                                     baseline_time, action[:, 0], outcome, args)
                      for time_intensity, action, outcome, tm_type in
                      zip(time_intensities, actions, outcomes, tm_types)]
        lambda_sup = np.max(lambda_sup)
        ti = current_t + -1 / lambda_sup * np.log(u1)
        if ti > until_t:
            candidate_point = (ti, False)
            current_t = until_t
            until_t = get_closest_future_time(current_t, x, T)
        else:
            candidate_point = (ti, True)
            ti = np.array([ti])
            for i, (time_intensity, mark_intensity, outcome_model, om_type, tm_type) in \
                    enumerate(zip(time_intensities, mark_intensities, outcome_models, om_types, tm_types)):
                accepted = thin_candidate_point(ti, u2, time_intensity, tm_type, lambda_sup,
                                                baseline_time, actions[i][:, 0], outcomes[i], args)
                if accepted:
                    mark, _ = mark_intensities[i].predict_f_compiled(np.array([[ti.item() % args.hours_day]]))
                    actions[i] = np.concatenate([actions[i], [[ti.item(), mark.numpy().item()]]])
                    f_mean = get_f_mean_w_args(outcome_model, om_type=om_type, x=x, action=actions[i], patient_idx=pidx,
                                               fb_mean=fb_means[i], args=args)
                    outcomes[i] = np.stack([x, f_mean]).T

            current_t = ti.item()
            until_t = get_closest_future_time(current_t, x, T)

        algorithm_log.append([interval, lambda_sup, candidate_point, [u1, u2]])
        interval = (current_t, until_t)
    for outcome in outcomes:
        outcome_noise = np.random.randn(args.n_outcome) * noise_std
        outcome[:, 1] += outcome_noise
    return actions, outcomes, algorithm_log


def sample_multiple_trajectories_cf(
        models_obs, models_cf, model_types, use_noise_posteriors,
        action_obs, outcome_obs, pidx, time_domain, args):
    noise_std = 0.1
    n_patient_oracle = len(args.observational_outcome_ids)
    T = time_domain[-1]
    tm_types, om_types = [model_type[0] for model_type in model_types], [model_type[1] for model_type in model_types]
    outcome_models = [m[1] for m in models_obs]
    time_intensities_obs = [m[0][0] for m in models_obs]
    time_intensities_cf = [m[0][0] for m in models_cf]
    mark_intensities_cf = [m[0][1] for m in models_cf]
    #
    x = np.linspace(time_domain[0], time_domain[1], args.n_outcome).astype(np.float64)
    action_time_obs = action_obs[:, 0]
    all_event_times_obs = np.sort(np.concatenate([action_time_obs, x]))
    #
    fb_means_cf = [
        get_fb_mean_w_args(om, om_type=om_type, x=x, patient_idx=pidx, args=args)
        for om_type, om in zip(om_types, outcome_models)
    ]
    outcomes_cf = [np.stack([x, fb_mean]).T for fb_mean in fb_means_cf]
    f_means = [
        get_f_mean_w_args(om, om_type=om_type, x=x, action=action_obs, patient_idx=pidx, args=args)
        for om_type, om in zip(om_types, outcome_models)
    ]
    noises = [outcome_obs[:, 1] - f_mean if use_noise else np.random.randn(outcome_obs.shape[0]) * noise_std
              for f_mean, use_noise in zip(f_means, use_noise_posteriors)]
    baseline_time = np.zeros((1, 1))
    actions_cf = [np.zeros((0, 2)).astype(np.float64) for _ in range(len(models_cf))]
    current_t = x[0]
    until_t = get_closest_future_time(current_t, all_event_times_obs, T)
    interval = (current_t, until_t)
    while interval[0] < time_domain[-1]:
        # Resample rejected
        lambda_sup_obs = [get_lambda_sup(time_intensity, tm_type, interval,
                                         baseline_time, action_time_obs, outcome_obs, args)
                          for time_intensity, tm_type, use_noise in
                          zip(time_intensities_obs, tm_types, use_noise_posteriors) if use_noise]
        lambda_sup_cf = [get_lambda_sup(time_intensity, tm_type, interval,
                                        baseline_time, action_cf[:, 0], outcome_cf, args)
                         for time_intensity, action_cf, outcome_cf, tm_type in
                         zip(time_intensities_cf, actions_cf, outcomes_cf, tm_types)]
        lambda_sup = np.max(lambda_sup_obs+lambda_sup_cf)
        noise_vars_rej = np.random.uniform(size=2)
        u1, u2 = noise_vars_rej
        ti = current_t + -1 / lambda_sup * np.log(u1)
        noise_rej_resample, noise_acc_resample = np.random.uniform(), np.random.uniform()
        if ti > until_t:
            for i, (int_obs, int_cf, mark_int_cf, fb_mean_cf, use_noise, tm_type, om_type) in \
                    enumerate(zip(time_intensities_obs, time_intensities_cf, mark_intensities_cf, fb_means_cf,
                                  use_noise_posteriors, tm_types, om_types)):
                if use_noise and (until_t in action_time_obs):
                    lambda_acc_obs = get_lambda_x(int_obs, tm_type, np.array([until_t]), baseline_time,
                                                  action_time_obs, outcome_obs, args)
                    lambda_acc_cf = get_lambda_x(int_cf, tm_type, np.array([until_t]), baseline_time,
                                                 actions_cf[i][:, 0], outcomes_cf[i], args)
                    u_transformed = noise_acc_resample * lambda_acc_obs.item() / lambda_sup
                    acc_acc = u_transformed <= (lambda_acc_cf.item() / lambda_sup)
                    if acc_acc:
                        mark, _ = mark_int_cf.predict_f_compiled(np.array([[until_t % args.hours_day]]))
                        actions_cf[i] = np.concatenate([actions_cf[i], [[until_t, mark.numpy().item()]]])
                        f_mean = get_f_mean_w_args(outcome_models[i], om_type=om_type, x=x, action=actions_cf[i],
                                                   patient_idx=pidx, fb_mean=fb_mean_cf, args=args)
                        outcomes_cf[i] = np.stack([x, f_mean]).T
            current_t = until_t
            until_t = get_closest_future_time(current_t, x, T)
        else:
            ti = np.array([ti])
            for i, (int_obs, int_cf, mark_int_cf, fb_mean_cf, use_noise, tm_type, om_type) in \
                    enumerate(zip(time_intensities_obs, time_intensities_cf, mark_intensities_cf, fb_means_cf,
                                  use_noise_posteriors, tm_types, om_types)):
                target_intensity_for_candidates = int_obs if use_noise else int_cf
                sample_rejected = True if use_noise else False

                ti_acc = thin_candidate_point(ti, u2, target_intensity_for_candidates,
                                              tm_type, lambda_sup, baseline_time,
                                              action_time_obs, outcome_obs, args,
                                              sample_rejected=sample_rejected)

                rej_acc = False
                if ti_acc and use_noise:
                    lambda_rej_obs = get_lambda_x(int_obs, tm_type, ti, baseline_time,
                                                  action_time_obs, outcome_obs, args)
                    lambda_rej_cf = get_lambda_x(int_cf, tm_type, ti, baseline_time,
                                                 actions_cf[i][:, 0], outcomes_cf[i], args)
                    u_transformed = (1.0 - lambda_rej_obs/lambda_sup) * noise_rej_resample + lambda_rej_obs/lambda_sup
                    rej_acc = u_transformed <= (lambda_rej_cf / lambda_sup)

                if rej_acc or (ti_acc and not use_noise):
                    current_t = ti.item()
                    mark, _ = mark_int_cf.predict_f_compiled(np.array([[current_t % args.hours_day]]))
                    actions_cf[i] = np.concatenate([actions_cf[i], [[current_t, mark.numpy().item()]]])
                    f_mean = get_f_mean_w_args(outcome_models[i], om_type=om_type, x=x, action=actions_cf[i],
                                               patient_idx=pidx, fb_mean=fb_mean_cf, args=args)
                    outcomes_cf[i] = np.stack([x, f_mean]).T

            current_t = ti.item()
            until_t = get_closest_future_time(current_t, all_event_times_obs, T)
        interval = (current_t, until_t)

    for outcome_cf, noise in zip(outcomes_cf, noises):
        outcome_cf[:, 1] += noise
    return actions_cf, outcomes_cf


def get_fb_mean_w_args(outcome_model, om_type, x, patient_idx, args):
    n_patient_model = len(args.observational_outcome_ids) if om_type == 'oracle_gp' else args.n_patient
    pidx = patient_idx % n_patient_model
    return get_fb_mean(outcome_model, om_type, x, pidx, n_patient_model, train_ds_schulam=args.train_ds_schulam)


def get_fb_mean(outcome_model, om_type, x, pidx, n_patient_model, train_ds_schulam=None):
    if om_type == 'schulam':
        fb_mean = predict_outcome_trajectory(outcome_model, train_ds_schulam, pidx, xnew=x % 24.0, tnew=[])
    elif om_type == 'hua':
        fb_mean = outcome_model.predict_baseline(x, pidx)
    elif om_type == 'oracle_gp' or om_type == 'estimated_gp':
        fb_mean = get_gp_baseline(outcome_model, x, pidx, n_patient_model)
    else:
        raise ValueError('Wrong model type!')
    return fb_mean


def get_f_mean_w_args(outcome_model, om_type, x, action, patient_idx, args, fb_mean=None):
    n_patient_model = len(args.observational_outcome_ids) if om_type == 'oracle_gp' else args.n_patient
    pidx = patient_idx % n_patient_model
    if om_type == 'schulam':
        f_mean = predict_outcome_trajectory(outcome_model, args.train_ds_schulam, pidx, xnew=x % 24, tnew=action[:, 0])
    elif om_type == 'hua':
        ds = get_dosage_hua(x, action, args)
        f_mean = outcome_model.predict_outcome(x % 24.0, ds, pidx)
    elif om_type == 'oracle_gp' or om_type == 'estimated_gp':
        if fb_mean is None:
            fb_mean = get_gp_baseline(outcome_model, x, pidx, n_patient_model)
        f_mean = get_gp_f(outcome_model, x, action, fb_mean, pidx, n_patient_model)
    else:
        raise ValueError('Wrong model type!')
    return f_mean


def get_f_mean(outcome_model, om_type, x, action, pidx, n_patient_model, args, fb_mean=None, train_ds_schulam=None):
    if om_type == 'schulam':
        f_mean = predict_outcome_trajectory(outcome_model, train_ds_schulam, pidx, xnew=x % 24, tnew=action[:, 0])
    elif om_type == 'hua':
        ds = get_dosage_hua(x, action, args)
        f_mean = outcome_model.predict_outcome(x % 24.0, ds, pidx)
    elif om_type == 'oracle_gp' or om_type == 'estimated_gp':
        if fb_mean is None:
            fb_mean = get_gp_baseline(outcome_model, x, pidx, n_patient_model)
        f_mean = get_gp_f(outcome_model, x, action, fb_mean, pidx, n_patient_model)
    else:
        raise ValueError('Wrong model type!')
    return f_mean


def get_gp_f(outcome_model, x, action, fb_mean, patient_idx, n_patient_model):
    n_outcome = x.shape[0]
    t_lengths = [action.shape[0]] * n_patient_model
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ft_mean, _ = outcome_model.predict_ft_w_tnew_compiled([x.reshape(-1, 1) for _ in range(n_patient_model)],
                                                          [action for _ in range(n_patient_model)],
                                                          tnew_patient_idx)
    ft_mean = ft_mean.numpy().flatten()[n_outcome*patient_idx:n_outcome*(patient_idx+1)]
    f = fb_mean + ft_mean
    return f


def get_gp_baseline(outcome_model, x, pidx, n_patient_model):
    n_outcome = x.shape[0]
    fb_mean, _ = outcome_model.predict_baseline_compiled([x.reshape(-1, 1) for _ in range(n_patient_model)])
    fb_mean = fb_mean.numpy().flatten()[n_outcome*pidx:n_outcome*(pidx + 1)]
    return fb_mean


def thin_candidate_point(ti, u2, time_intensity, time_intensity_type,
                         lambda_sup, baseline_time, action_time, outcome_tuple,
                         args, sample_rejected=False):
    lambda_ti = get_lambda_x(time_intensity, time_intensity_type, ti, baseline_time, action_time, outcome_tuple, args)
    lambda_ti = lambda_ti.item()
    intensity_val = (lambda_sup - lambda_ti) if sample_rejected else lambda_ti
    accept = u2 <= (intensity_val / lambda_sup)
    return accept


def get_lambda_sup(time_intensity, time_intensity_type, interval, baseline_time, action_time, outcome_tuple, args):
    N = 40
    t1, t2 = interval
    xx = np.linspace(t1, t2, N+1)[1:]
    lambda_xx = get_lambda_x(time_intensity, time_intensity_type, xx, baseline_time, action_time, outcome_tuple, args)
    lambda_sup = np.max(lambda_xx)
    return lambda_sup


def get_lambda_x(time_intensity, time_intensity_type, xx, baseline_time, action_time, outcome_tuple, args):
    if time_intensity_type == 'gprpp':
        X = get_relative_input_by_query(xx, baseline_time, action_time, outcome_tuple, args)
        lambdaX = time_intensity.predict_lambda_compiled(X)[1].numpy().flatten()
    elif time_intensity_type == 'vbpp':
        # VBPP is defined over [0, 24], assuming exchangeable days
        lambdaX = time_intensity.predict_lambda_compiled(xx.reshape(-1, 1) % args.hours_day)[1].numpy().flatten()
    elif time_intensity_type == 'hua':
        X, Y = get_relative_input_hua(xx, action_time, outcome_tuple)
        lambdaX = time_intensity.predict_lambda_compiled(X, Y)
    else:
        raise ValueError('Wrong treatment model type!')
    return lambdaX
