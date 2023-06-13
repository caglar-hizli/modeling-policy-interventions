import numpy as np

from utils.utils import get_relative_input_by_query


def sample_joint_tuples(patient_sampler, action_pidx, outcome_pidx, n_day, args):
    n_patient_oracle = len(args.outcome_sampler_patient_ids)
    T = args.hours_day * n_day
    args.No = args.n_outcome * n_day
    noise_std = 0.1
    action_sampler, outcome_sampler = patient_sampler
    action_time_sampler, action_mark_sampler = action_sampler[action_pidx]
    lambda_fnc = action_time_sampler.predict_lambda_compiled
    noise = np.random.randn(args.No) * noise_std
    x = np.linspace(0.0, T, args.No) if args.regular_measurement_times else np.sort(np.random.uniform(0.0, T, args.No))
    x = x.astype(np.float64)
    fb_mean, fb_var = outcome_sampler.predict_baseline_compiled([x.reshape(-1, 1) for _ in range(n_patient_oracle)])
    fb_mean = fb_mean.numpy().flatten()[args.No*outcome_pidx:args.No*(outcome_pidx + 1)]
    fb_var = fb_var.numpy().flatten()[args.No*outcome_pidx:args.No*(outcome_pidx + 1)]
    ft_mean, ft_var = np.zeros_like(x), np.zeros_like(x)
    f = fb_mean + ft_mean
    y = f + noise
    outcome_tuple = np.stack([x, y]).T
    #
    current_t = x[0]
    until_t = get_closest_future_time(current_t, x, T)
    action_day, mark_day, action_marked = [], [], np.zeros((0, 2))
    while current_t < T:
        ti, accepted = thinning_interval(lambda_fnc, current_t, until_t, np.array(action_day), outcome_tuple, args)
        if len(ti) == 0:
            current_t = until_t
            until_t = get_closest_future_time(current_t, x, T)
        else:
            if accepted:
                action_day.append(ti[0])
                # current_t = accepted[0] + jitter  # simple PP means no instantaneous effects
                mark, _ = action_mark_sampler.predict_f_compiled(np.array([[ti[0] % 24.0]]))
                mark_day.append(mark.numpy().item())
                action_marked = np.stack([action_day, mark_day]).T.astype(np.float64)
                y, f, ft_mean, ft_var = update_outcome_tuple(outcome_sampler,
                                                             [x.reshape(-1, 1) for _ in range(n_patient_oracle)],
                                                             [action_marked for _ in range(n_patient_oracle)],
                                                             fb_mean, noise, outcome_pidx, args)
                outcome_tuple = np.stack([x, y]).T
            current_t = ti.item()
            until_t = get_closest_future_time(current_t, x, T)

    f_means = [fb_mean, ft_mean, fb_mean+ft_mean]
    f_vars = [fb_var, ft_var, fb_var+ft_var]
    return action_marked, outcome_tuple, f_means, f_vars


def get_closest_future_time(current_t, xd, max_time):
    larger_event_idx = np.where(xd > current_t)[0]
    if len(larger_event_idx) > 0:
        smallest_larger_idx = np.min(larger_event_idx)
        return xd[smallest_larger_idx]
    else:
        return max_time


def thinning_interval(lambda_fnc, t1, t2, action_time, outcome_tuple, args):
    """
    :param lambda_fnc:
    :param t1:
    :param t2:
    :param action_time: (N,)
    :param outcome_tuple: (N, 2)
    :param args:
    :return:
    """
    N = 40
    baseline_time = np.zeros((1, 1))
    xx = np.linspace(t1, t2, N+1)[1:]
    X = get_relative_input_by_query(xx, baseline_time, action_time, outcome_tuple, args)
    _, lambdaX = lambda_fnc(X)
    lambda_sup = np.max(lambdaX)  # + 0.05
    ti = t1 + np.random.exponential(1 / lambda_sup, size=1)
    if ti.item() > t2:
        return [], False

    X = get_relative_input_by_query(ti, baseline_time, action_time, outcome_tuple, args)
    _, lambdaX = lambda_fnc(X)
    lambda_vals = lambdaX.numpy().flatten().item()
    accept = np.random.uniform(0.0, 1.0) <= (lambda_vals / lambda_sup)
    return ti, accept


def update_outcome_tuple(outcome_sampler, xs, actions, fb, noise, patient_idx, args):
    t_lengths = [ti.shape[0] for ti in actions]
    Np = len(t_lengths)
    patient_order_arr = np.arange(Np, dtype=np.int32)
    tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
    ft_mean, ft_var = outcome_sampler.predict_ft_w_tnew_compiled(xs, actions, tnew_patient_idx)
    ft_mean = ft_mean.numpy().flatten()[args.No*patient_idx:args.No*(patient_idx+1)]
    ft_var = ft_var.numpy().flatten()[args.No*patient_idx:args.No*(patient_idx+1)]
    f = fb + ft_mean
    y = f + noise
    return y, f, ft_mean, ft_var
