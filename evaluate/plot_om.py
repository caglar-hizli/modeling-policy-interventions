import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_fs_pred(xnew, f_means, f_vars, f_labels, f_colors, ds, patient_id, args,
                 show_data=True, path=None, plot_var=True, show_test=False):
    fig, ax1 = plt.subplots(figsize=(15, 4))
    lines, labels = [], f_labels
    for fm, fv, col in zip(f_means, f_vars, f_colors):
        line_gp = plot_gp_pred(xnew, fm, fv, color=col, plot_var=plot_var)
        lines.append(line_gp)
    ax2 = ax1.twinx()
    if show_data:
        line1, bar2 = plot_joint_data(ds, axes=(ax1, ax2))
        lines = [line1, bar2] + lines
        labels = [r'$Y$', r'$\mathbf{a}$'] + labels
    if show_test:
        ax2.vlines(args.n_day_train * args.hours_day, *ax2.get_ylim(), linestyle="--", color="grey")
        ax2.axvspan(args.n_day_train * args.hours_day,
                    (args.n_day_train+args.n_day_test) * args.hours_day, color="grey", alpha=0.1)
    ax1.legend(lines, labels, loc='lower left', fontsize=14, framealpha=1.0)
    ax1.set_ylabel(r'Blood glucose (mmol/l), $Y$', fontsize=14)
    ax1.set_xlabel(r'Time (hours), $\tau$', fontsize=14)
    ax2.set_ylabel(r'Carbohydrate in. (log g), $m$', fontsize=14)
    plt.title('Train' + ('/Test' if show_test else '')
              + f'Fit of Outcome Model for Patient {patient_id}', fontsize=18)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ft_pred(Xnew, f_mean, f_var, ds=None, true_f_mean=None, show_data=True, show_var=True, path=None):
    plt.figure(figsize=(8, 4))
    if true_f_mean is not None:
        plt.plot(Xnew, true_f_mean, 'red', lw=2, label='true f')
    plt.plot(Xnew, f_mean, 'tab:blue', lw=2, label=r'$f_a(\tau, (t_0,m_0)=(0.0, 1.0))$')
    plt.fill_between(
        Xnew[:, 0],
        f_mean[:, 0] - 1.96 * np.sqrt(f_var[:, 0]),
        f_mean[:, 0] + 1.96 * np.sqrt(f_var[:, 0]),
        color='tab:blue',
        alpha=0.2,
    )
    plt.legend(fontsize=16)
    plt.title('Treatment Response Function for Fixed mark=1.0', fontsize=20)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.ylabel(r'$f_a(\tau, \mathbf{a})$', fontsize=16)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compare_fs_pred(xnew, f_means, f_vars, f_labels, f_colors, true_f_means, path=None, plot_var=True):
    fig, ax1 = plt.subplots(figsize=(15, 4))
    lines, labels = [], f_labels
    for fm, fv, col, tfm in zip(f_means, f_vars, f_colors, true_f_means):
        line_gp = plot_gp_pred(xnew, fm, fv, color=col, plot_var=plot_var)
        lines.append(line_gp)
        _ = plot_gp_pred(xnew, tfm, fv, color='black', ls='dashed', plot_var=False)
    ax1.legend(lines, labels, loc=2, fontsize=14, framealpha=1.0)
    plt.savefig(path)
    plt.close()


def plot_fs_pred_multiple(f_means, f_vars, ds_plots, args, plot_var=True, true_f_means=None, run='all'):
    offset = 0
    xnew = [ds[0].astype(np.float64).reshape(-1, 1) for ds in ds_plots]
    for i, ds_plot in enumerate(ds_plots):
        predict_shape = xnew[i].shape[0]
        f_means_i = [ff[offset:offset + predict_shape] for ff in f_means]
        f_vars_i = [ff[offset:offset + predict_shape] for ff in f_vars]
        file_name = f'f_{run}_fit_id{i}_v{plot_var}.pdf'
        plot_fs_pred(xnew[i], f_means_i, f_vars_i,
                     [r'$\mathbf{f_b}$', r'$\mathbf{f_a}$', r'$\mathbf{f}$'],
                     ['tab:orange', 'tab:green', 'tab:blue'], ds_plot, args.patient_ids[i],
                     args, show_data=True, plot_var=plot_var,
                     path=os.path.join(args.outcome_model_figures_dir, file_name), show_test=run == 'all')
        if true_f_means is not None:
            file_name = f'compare_f_{run}_fit_id{i}_v{plot_var}.pdf'
            true_f_means_i = [ff[offset:offset + predict_shape] for ff in true_f_means]
            compare_fs_pred(xnew[i], f_means_i, f_vars_i,
                            [r'$\mathbf{f_b}$', r'$\mathbf{f_a}$', r'$\mathbf{f}$'],
                            ['tab:orange', 'tab:green', 'tab:blue'], true_f_means_i,
                            plot_var=False, path=os.path.join(args.outcome_model_figures_dir, file_name))
        offset += predict_shape


def plot_trajectory_pair(outcome_pair, path):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    for outcome, label in zip(outcome_pair, [r'$f_{oracle}$', r'$f_{estimated}$']):
        ax1.plot(outcome[:, 0], outcome[:, 1], label=label)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_ft_comparison(xnew, f_means, f_vars, path=None):
    plt.figure(figsize=(8, 4))
    for f_mean, f_var, label, color in zip(f_means, f_vars, [r'$f_a$ Bas.', r'$f_a$ Ope.'], ['tab:blue', 'tab:red']):
        plot_gp_pred(xnew, f_mean, f_var, color=color, label=label)
    label_response_curve()
    plt.savefig(path)
    plt.close()


def label_response_curve():
    plt.legend(fontsize=13)
    plt.title('Treatment Response Function for Fixed mark=1.0', fontsize=20)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.ylabel(r'$f_a(\tau, \mathbf{a})$', fontsize=16)
    plt.tight_layout()


def plot_gp_pred(xnew, f_mean, f_var, color='tab:blue', label='f Pred', plot_var=True, ls='solid'):
    line_gp, = plt.plot(xnew, f_mean, color, lw=2, label=label, zorder=2, ls=ls)
    if plot_var:
        plt.fill_between(
            xnew[:, 0],
            f_mean[:, 0] - 1.96 * np.sqrt(f_var[:, 0]),
            f_mean[:, 0] + 1.96 * np.sqrt(f_var[:, 0]),
            color=color,
            alpha=0.2,
        )
    return line_gp


def plot_patient_df(df_patient, axes):
    is_meal = df_patient.y.isna()
    t = df_patient.t[is_meal].values
    x = df_patient.t[~is_meal].values
    x = x / 60
    t = t / 60
    meal_in = df_patient.SUGAR + df_patient.STARCH
    m = meal_in[is_meal].values
    y = df_patient.y[~is_meal].values
    return plot_joint_data((x, y, t, m), axes)


def plot_joint_data(ds, axes=None):
    if axes is None:
        fig, ax1 = plt.subplots(figsize=(15, 4))
        ax2 = ax1.twinx()

    x, y, t, m = ds

    day_min, day_max = x.min() // 24, x.max() // 24
    meal_bar_width = (day_max - day_min + 1) * (1 / 6)
    ax1, ax2 = axes
    #
    line1, = ax1.plot(x, y, 'kx', ms=5, alpha=0.5, label='Glucose, $y(t)$')
    ylim1 = ax1.get_ylim()
    ax1.set_ylim(ylim1[0] - 1.0, ylim1[1])
    bar2 = ax2.bar(t, m, color=(0.1, 0.1, 0.1, 0.1), edgecolor='red', width=meal_bar_width,
                   label=r'Meal, $\mathbf{a}$')
    # Widen ylim2, so that max meal is under min glucose
    ylim2 = ax2.get_ylim()
    ylim_max2_wide = ylim2[1] * (1.0+ylim1[1]-ylim1[0])
    if np.isfinite(ylim2[0]):
        ax2.set_ylim(ylim2[0], ylim_max2_wide)
    #
    ax1.vlines([24 * i for i in range(int(day_min), int(day_max) + 1)], 0.0, ylim1[1], colors='grey', alpha=0.5)
    return line1, bar2


def compare_trajectory_pair(action_pair, outcome_pair, exp_ids, plot_exp_ids, path):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    lines = []
    colors = mcolors.TABLEAU_COLORS
    color_names = list(colors)
    o_labels = [r'$\mathbf{Y}_{' + exp_id[:3] + '}$' for exp_id in exp_ids if exp_id in plot_exp_ids]
    a_labels = [r'$\mathbf{a}_{' + exp_id[:3] + '}$' for exp_id in exp_ids if exp_id in plot_exp_ids]
    outcome_pair_plot = [oi for exp_id, oi in zip(exp_ids, outcome_pair) if exp_id in plot_exp_ids]
    action_pair_plot = [ai for exp_id, ai in zip(exp_ids, action_pair) if exp_id in plot_exp_ids]
    for outcome, cn in zip(outcome_pair_plot, color_names):
        line, = ax1.plot(outcome[:, 0], outcome[:, 1], '-o', color=colors[cn])
        lines.append(line)

    ax2 = ax1.twinx()
    ylim1 = ax1.get_ylim()
    ax1.set_ylim(ylim1[0] - 1.0, ylim1[1])
    bars = []
    for action, cn in zip(action_pair_plot, color_names):
        bar = ax2.bar(action[:, 0], action[:, 1], color=colors[cn], edgecolor=colors[cn], width=1/6, alpha=0.5)
        bars.append(bar)
    ylim2 = ax2.get_ylim()
    ylim_max2_wide = ylim2[1] * (1.0+ylim1[1]-ylim1[0])
    if np.isfinite(ylim2[0]):
        ax2.set_ylim(ylim2[0], ylim_max2_wide)

    ax1.legend(lines+bars, o_labels+a_labels, loc=2, fontsize=16)
    plt.savefig(path)
    plt.close()


def compare_f_preds(Xnew, f_means, ylabel, title_str, ylim=None, path=None):
    plt.figure(figsize=(8, 4))
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green']
    for i, (f_mean, color) in enumerate(zip(f_means, colors)):
        label = f'PG {i}'
        plt.plot(Xnew, f_mean, '--', lw=2, label=label)
    plt.legend(fontsize=16)
    plt.title(title_str, fontsize=18)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
