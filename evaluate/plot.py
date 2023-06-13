import os
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils.utils import get_relative_input_by_query, get_tm_label


def plot_joint_intensity_train(model, datasets, baseline_times, args, oracle_model=None, true_f_means=None):
    N_test = 400
    period = (0.0, args.n_day_train*args.hours_day)
    for pidx, (baseline_time, ds) in enumerate(zip(baseline_times, datasets)):
        action_time_intensity, outcome_model = model[0][pidx % 2][0], model[1]
        X_abs = np.linspace(*period, N_test)
        action_time, action_mark, outcome_tuple = ds[2], ds[3], np.stack([ds[0], ds[1]]).T
        X = get_relative_input_by_query(X_abs, baseline_time, action_time, outcome_tuple, args)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1],
                                                                                        'wspace': 0.0,
                                                                                        'hspace': 0.0})
        Nx = outcome_tuple.shape[0]
        xnew = [outcome_tuple[:, 0].astype(np.float64).reshape(-1, 1) for _ in range(args.n_patient)]
        anew = [np.hstack([action_time.reshape(-1, 1), action_mark.reshape(-1, 1)]).astype(np.float64)
                for _ in range(args.n_patient)]
        t_lengths = [ti.shape[0] for ti in anew]
        Np = len(t_lengths)
        patient_order_arr = np.arange(Np, dtype=np.int32)
        tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
        ft_mean, ft_var = outcome_model.predict_ft_w_tnew_compiled(xnew, anew, tnew_patient_idx)
        fb_mean, fb_var = outcome_model.predict_baseline_compiled(xnew)
        f_mean, f_var = ft_mean + fb_mean, ft_var + fb_var

        plt.subplot(2, 1, 1)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlim(np.min(X_abs), np.max(X_abs))
        plt.plot(outcome_tuple[:, 0], fb_mean[Nx*pidx:Nx*(pidx+1)], '--', lw=2, color='tab:orange',
                 label=r"$\mathbf{f_b}$")
        plt.plot(outcome_tuple[:, 0], f_mean[Nx*pidx:Nx*(pidx+1)], '--', lw=2, color='tab:red',
                 label=r"$\mathbf{f}=\mathbf{f_b}+\mathbf{f_a}$")
        if true_f_means is None:
            oracle_outcome_model = oracle_model[1]
            xnew = [outcome_tuple[:, 0].astype(np.float64).reshape(-1, 1)
                    for _ in range(len(args.outcome_sampler_patient_ids))]
            anew = [np.hstack([action_time.reshape(-1, 1), action_mark.reshape(-1, 1)]).astype(np.float64)
                    for _ in range(len(args.outcome_sampler_patient_ids))]
            t_lengths = [ti.shape[0] for ti in anew]
            Np = len(t_lengths)
            patient_order_arr = np.arange(Np, dtype=np.int32)
            tnew_patient_idx = np.repeat(patient_order_arr, t_lengths)
            ft_mean, ft_var = oracle_outcome_model.predict_ft_w_tnew_compiled(xnew, anew, tnew_patient_idx)
            fb_mean, fb_var = oracle_outcome_model.predict_baseline_compiled(xnew)
            f_mean, f_var = ft_mean + fb_mean, ft_var + fb_var
            pidx_eff = pidx % len(args.outcome_sampler_patient_ids)
            plt.plot(outcome_tuple[:, 0], fb_mean[Nx * pidx_eff:Nx * (pidx_eff + 1)], lw=2, color='tab:cyan',
                     alpha=0.25, label=r"$\mathbf{f_{b,oracle}}$")
            plt.plot(outcome_tuple[:, 0], f_mean[Nx * pidx_eff:Nx * (pidx_eff + 1)], lw=2, color='tab:blue',
                     alpha=0.25, label=r"$\mathbf{f_{oracle}}$")
        else:
            fb_mean, f_mean = true_f_means[0], true_f_means[2]
            plt.plot(outcome_tuple[:, 0], fb_mean[Nx * pidx:Nx * (pidx + 1)], lw=2, color='tab:cyan',
                     alpha=0.25, label=r"$\mathbf{f_{b,oracle}}$")
            plt.plot(outcome_tuple[:, 0], f_mean[Nx * pidx:Nx * (pidx + 1)], lw=2, color='tab:blue',
                     alpha=0.25, label=r"$\mathbf{f_{oracle}}$")

        plt.plot(outcome_tuple[:, 0], outcome_tuple[:, 1], 'kx', label=r"Outcomes $\mathbf{o}$")
        plt.plot()
        ylim = plt.gca().get_ylim()
        # plt.vlines(outcome_tuple[:, 0], *ylim, colors='black', alpha=0.1)
        plt.ylabel(r'$y(\tau)$', fontsize=16)
        plt.xticks([])
        plt.legend(loc='upper right', fontsize=12, framealpha=1.0)

        plt.subplot(2, 1, 2)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlim(np.min(X_abs), np.max(X_abs))
        lw = 1.5
        label = get_tm_label(args)

        _, lambda_mean = action_time_intensity.predict_lambda_compiled(X)
        plt.plot(X_abs, lambda_mean, '--', color='tab:red', lw=lw, label=label)
        _ = plt.xticks(np.arange(0, period[-1]+1))
        ylim = plt.gca().get_ylim()
        if oracle_model is not None:
            oracle_action_time_intensity = oracle_model[0][pidx % 2][0]
            _, oracle_lambda_mean = oracle_action_time_intensity.predict_lambda_compiled(X)
            plt.plot(X_abs, oracle_lambda_mean, color='tab:blue', lw=2, alpha=0.25, label=r'$\lambda_{ao,oracle}$')
        plt.vlines(action_time, *ylim, colors='green', linewidth=2, label=r'Treatments $\mathbf{a}$', zorder=-1)
        plt.vlines(outcome_tuple[:, 0], *ylim, colors='black', alpha=0.05)
        plt.ylabel(r'$\lambda(\tau)$', fontsize=16)
        plt.xlabel(r'Time $\tau$', fontsize=16)
        plt.legend(loc='upper right', fontsize=12, framealpha=1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(args.model_figures_dir, f'joint_train_p{pidx}.pdf'))
        plt.close()


def plot_multiple_sampling(pidx, patient_datasets, algorithm_log, pp_logs, exp_ids, model_strs, period, out_folder):
    N_plot = len(exp_ids)
    accept_noise = np.array([l[3][1] for l in algorithm_log if l[2][1]])
    candidates_lambda_ub = np.array([l[1] for l in algorithm_log if l[2][1]])
    lambda_ub = [l[1]+0.1 for l in algorithm_log]
    baseline_time = np.array([period[0]])
    accept_noise_scaled = accept_noise * candidates_lambda_ub
    x_ub = [l[0][1] for l in algorithm_log]
    _ = plt.subplots(2, 1, sharex=True, figsize=(15, 6), gridspec_kw={'height_ratios': [1.5, 1],
                                                                        # 'wspace': 0.0, 'hspace': 0.00
                                                                        })
    #
    plt.subplot(2, 1, 1)
    colors = mcolors.TABLEAU_COLORS
    color_names = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:red']
    ms = np.concatenate([ds[3] for ds in patient_datasets])
    mmax = np.max(ms)
    fs_legend = 13
    fs_label = 16
    for i, (ds, exp_id, model_str, cn) \
            in enumerate(zip(patient_datasets, exp_ids, model_strs, color_names)):
        xo, yo = ds[0], ds[1]
        to, mo = ds[2], ds[3]
        distribution_str = 'observational' if exp_id == 'observational' else 'interventional'
        if i == 0:
            lines, = plt.plot(xo, yo, '-o', color=colors[cn], alpha=0.7,
                              lw=4, label=r'$\mathbf{Y}^{'+model_str+'}_{' + distribution_str[:3] + '}$')
        elif i < N_plot - 1:
            lines, = plt.plot(xo, yo, '--o', color=colors[cn], lw=2, alpha=0.25,
                              label=r'$\mathbf{Y}^{'+model_str+'}_{' + distribution_str[:3] + '}$')
        else:
            lines, = plt.plot(xo, yo, '--o', color=colors[cn], lw=4, alpha=1.0,
                              label=r'$\mathbf{Y}^{'+model_str+'}_{' + distribution_str[:3] + '}$')

    for i, (ds, exp_id, model_str, cn) \
            in enumerate(zip(patient_datasets, exp_ids, model_strs, color_names)):
        xo, yo = ds[0], ds[1]
        to, mo = ds[2], ds[3]
        distribution_str = 'observational' if exp_id == 'observational' else 'interventional'
        if i == 0:
            plt.bar(to, (mo/mmax)*0.5, alpha=0.7,
                    bottom=3.0, color=colors[cn], edgecolor='black', width=0.2, lw=2,
                    label=r'$\mathbf{a}^{'+model_str+'}_{' + distribution_str[:3] + '}$')
            # plt.vlines(to, 3.0, 3.0+,
            #            lw=6, colors=colors[cn], alpha=1.0)
        elif i < N_plot-1:
            plt.bar(to-(i-2)*0.2+0.1, (mo / mmax) * 0.5, bottom=2.5,
                    color=colors[cn], edgecolor='black', width=0.2, lw=2, alpha=0.25,
                    label=r'$\mathbf{a}^{'+model_str+'}_{' + distribution_str[:3] + '}$')
        else:
            plt.bar(to - (i - 2) * 0.2 + 0.1, (mo / mmax) * 0.5, bottom=2.5,
                    color=colors[cn], edgecolor='black', width=0.2, lw=2, alpha=1.0,
                    label=r'$\mathbf{a}_{' + distribution_str[:3] + '}$')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False)

    plt.legend(fontsize=fs_legend, loc='upper left', ncol=2, framealpha=1.0)
    plt.ylabel(r'Glucose (mmol/l), $Y$', fontsize=fs_label)

    plt.subplot(2, 1, 2)
    # plt.title(f'OA={action_time_oracle_str}\nEA={action_time_est_str}')
    plt.step([period[0]]+x_ub, [lambda_ub[0]]+lambda_ub, '-',
             color='grey', lw=2, label=r'$\lambda_{ub}$', alpha=0.5)
    candidates = [l[2][0] for l in algorithm_log if l[2][1]]
    plt.plot(candidates, accept_noise_scaled, "gx", markersize=13, label='Accept')
    plt.vlines(candidates, np.zeros_like(candidates_lambda_ub), accept_noise_scaled,
               lw=1, colors='green', alpha=0.5)
    for i, (pp_log, ds, exp_id, model_str, cn) in \
            enumerate(zip(pp_logs, patient_datasets, exp_ids, model_strs, color_names)):
        x, lambdaXa_o = pp_log
        distribution_str = 'observational' if exp_id == 'observational' else 'interventional'
        if i == 0:
            plt.plot(x, lambdaXa_o, '-', color=colors[cn], lw=4, alpha=0.7,
                     label=r'$\lambda^{'+model_str+'}_{' + distribution_str[:3] + '}$')
        elif N_plot - 1:
            plt.plot(x, lambdaXa_o, '--', color=colors[cn], lw=2, alpha=0.25,
                     label=r'$\lambda^{'+model_str+'}_{' + distribution_str[:3] + '}$')
        else:
            plt.plot(x, lambdaXa_o, '--', color=colors[cn], lw=4, alpha=1.0,
                     label=r'$\lambda^{'+model_str+'}_{' + distribution_str[:3] + '}$')

    plt.legend(fontsize=fs_legend-2, loc='upper left', framealpha=1.0)
    plt.xticks(np.arange(period[0], period[1]+1))
    plt.ylabel(r'Intensity, $\lambda(\tau)$', fontsize=fs_label)
    plt.xlabel(r'Time (hours), $\tau$', fontsize=fs_label)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'compare_pair_sampling_{pidx}.pdf'))
    plt.close()
