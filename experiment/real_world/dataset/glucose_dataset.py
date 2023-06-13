import numpy as np
import pandas as pd


action_domain = [0.0, 24.0]


def load_glucose_dfs(args, patient_ids):
    glucose = pd.read_csv(args.dataset)
    dfs = [glucose[glucose['id'] == idx] for idx in patient_ids]
    return dfs


def get_glucose_dataset(patient_id, args):
    df_patient = load_glucose_dfs(args, [patient_id])[0]
    df_meals = df_patient[~np.isnan(df_patient.STARCH)]
    df_meals['MTOT'] = df_meals.SUGAR + df_meals.STARCH
    df_glucose = df_patient[~np.isnan(df_patient.glucose)]
    is_meal_sufficient = df_meals.MTOT > 1.0
    t = df_meals[is_meal_sufficient].time.to_numpy() / 60.0
    m = np.log(df_meals[is_meal_sufficient].MTOT.to_numpy())
    y = df_glucose.glucose.to_numpy()
    x = df_glucose.time.to_numpy() / 60.0
    #
    if args.preprocess_actions:
        t, m = preprocess_action_times(t, m, x, y)
    return x, y, t, m


def preprocess_action_times(t, m, x, y):
    tnew, mnew = np.copy(t), np.copy(m)
    tnew, mnew = remove_closeby(tnew, mnew)
    tnew = update_time_w_grad(tnew, x, y)
    tnew, mnew = remove_closeby(tnew, mnew)
    return tnew, mnew


def remove_closeby(t, m, threshold=2.0):
    idx = 0
    while idx < len(t):
        ti = t[idx]
        future = t[t > ti]
        if len(future) > 0 and len(future[future - ti < threshold]) > 0:
            removal_idx = [np.argwhere(t == tf).item() for tf in future[future - ti < threshold]]
            t = np.delete(t, removal_idx, None)
            m = np.delete(m, removal_idx, None)
        idx += 1
    return t, m


def update_time_w_grad(t, x, y):
    dy = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    dy2 = (dy[1:] + dy[:-1]) / 2
    x2 = x[2:]
    tnew = []
    threshold = 0.5
    for j, ti in enumerate(t):
        dy_past = dy2[x2 < ti]
        t_update = 0.0
        if np.abs(dy_past[-1]) > threshold:
            dy_negative = dy_past[-1] < 0.0
            if dy_negative:
                dy_search = dy2[x2 >= ti]
                x2_search = x2[x2 >= ti]
                dy_pos_count = 0
                for s, dyi in enumerate(dy_search):
                    if np.abs(dyi) < threshold:
                        t_update = x2_search[s] - 0.01
                        break
                    elif dyi > threshold:
                        dy_pos_count += 1
                        if dy_pos_count >= 2:
                            t_update = x2_search[s-1] - 0.01
                            break
                    else:
                        dy_pos_count = 0
            else:
                dy_search = dy_past[::-1]
                x2_search = x2[x2 < ti]
                x2_search = x2_search[::-1]
                dy_neg_count = 0
                for s, dyi in enumerate(dy_search):
                    if np.abs(dyi) < threshold:
                        t_update = x2_search[s] + 0.01
                        break
                    elif dyi < threshold:
                        dy_neg_count += 1
                        if dy_neg_count >= 2:
                            t_update = x2_search[s-1] + 0.01
                            break
                    else:
                        dy_neg_count = 0

        tnew.append(t_update if t_update > 0.0 else ti)
    tnew = np.array(tnew)
    return tnew


def domain_grid(domain, num_points):
    return np.linspace(domain.min(axis=1), domain.max(axis=1), num_points)


def get_daily_treatments(t, args):
    t_days = []
    for d in range(args.n_day_train+args.n_day_test):
        mask = np.logical_and(t >= d*args.hours_day, t < (d+1)*args.hours_day)
        t_day = t[mask]
        t_days.append(t_day - d*args.hours_day)
    return t_days


def ds_from_df(df, t_covariate_names):
    y_idx = ~df['glucose'].isnull().values
    y = df['glucose'][y_idx].values
    x = df['time'][y_idx].values

    # not null && covariates > threshold
    rx_idx = ((~df[t_covariate_names[0]].isnull()) & (df[t_covariate_names].sum(axis=1) > 10)).values
    t_covar = df[t_covariate_names][rx_idx].values
    t_covar = np.log(np.sum(t_covar, axis=1))
    t = df['time'][rx_idx].values
    yt = ((df['glucose'].fillna(method='bfill') + df['glucose'].fillna(method='ffill')) / 2)[rx_idx].values

    t_idx = np.nonzero(rx_idx)[0]
    x_idx = np.nonzero(y_idx)[0]

    return (x, y, x_idx), (t, t_covar, yt, t_idx)


def prepare_glucose_ds(df, args):
    (x, y, _), (t, m, _, _) = ds_from_df(df, args.treatment_covariates)

    x = x / args.mins_hour
    t = t / args.mins_hour

    t_end_train = args.hours_day * args.n_day_train
    t_start_test = args.hours_day * args.n_day_train
    t_end_test = args.hours_day * (args.n_day_train + args.n_day_test)

    idx_x_train = x <= t_end_train
    idx_t_train = t <= t_end_train
    patient_train = (x[idx_x_train], y[idx_x_train], t[idx_t_train], m[idx_t_train])

    idx_x_test = np.logical_and(x > t_start_test, t_end_test >= x)
    idx_t_test = np.logical_and(t > t_start_test, t_end_test >= t)
    patient_test = (x[idx_x_test], y[idx_x_test], t[idx_t_test], m[idx_t_test])
    patient_all = (x, y, t, m)
    return patient_all, patient_train, patient_test


def train_test_ds(ds, args):
    x, y, t, m = ds

    t_end_train = args.hours_day * args.n_day_train
    t_start_test = args.hours_day * args.n_day_train
    t_end_test = args.hours_day * (args.n_day_train + args.n_day_test)

    idx_x_train = x <= t_end_train
    idx_t_train = t <= t_end_train
    patient_train = (x[idx_x_train], y[idx_x_train], t[idx_t_train], m[idx_t_train])

    idx_x_test = np.logical_and(x > t_start_test, t_end_test >= x)
    idx_t_test = np.logical_and(t > t_start_test, t_end_test >= t)
    patient_test = (x[idx_x_test], y[idx_x_test], t[idx_t_test], m[idx_t_test])
    patient_all = (x, y, t, m)
    return patient_all, patient_train, patient_test
