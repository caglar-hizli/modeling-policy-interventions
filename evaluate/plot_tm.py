import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from evaluate.plot_om import plot_gp_pred
from utils.utils import get_tm_label, get_relative_input_by_query


def plot_gprpp_results(baseline_times, actions, outcome_tuples, action_model, args, model_figures_dir,
                       oracle_model=None):
    plot_treatment_intensity_train(action_model, baseline_times, actions, outcome_tuples, model_figures_dir, args,
                                   plot_confidence=False, oracle_model=oracle_model)
    if 'b' in args.action_components:
        plot_trig_kernel_baseline_comp(action_model, actions, model_figures_dir, args, plot_confidence=False,
                                       oracle_model=oracle_model)
    if 'a' in args.action_components:
        plot_trig_kernel_action_comp(action_model, model_figures_dir, args, plot_confidence=False,
                                     oracle_model=oracle_model)
    if 'o' in args.action_components:
        plot_trig_kernel_outcome_comp(action_model, model_figures_dir, args, oracle_model=oracle_model)


def plot_vbpp(events, X, lambda_mean, upper, lower, title, true_lambda_mean=None,
              plot_path=None, plot_data=False):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=30)
    plt.xlabel('Hours', fontsize=12)
    plt.xlabel(r'$\lambda(\cdot)$', fontsize=12)
    plt.xlim(X.min(), X.max())
    if true_lambda_mean is not None:
        plt.plot(X, true_lambda_mean, 'blue', lw=2, label='Lambda True')
    plt.plot(X, lambda_mean, 'red', lw=2, label='Lambda Pred')
    plt.fill_between(X.flatten(), lower, upper, color='red', alpha=0.2)
    if plot_data:
        cmap = plt.cm.get_cmap('Dark2')
        for d, ev in enumerate(events):
            if d < 7:
                plt.vlines(ev, np.zeros_like(ev), [np.max(upper) + 0.1] * len(ev),
                           linestyles='--',
                           label=f'Day {int(d)}',
                           colors=cmap(d))
            else:
                plt.vlines(ev, np.zeros_like(ev), [np.max(upper) + 0.1] * len(ev),
                           linestyles='--',
                           colors=cmap(7))
    plt.legend(fontsize=18, loc='upper left')
    plt.savefig(plot_path)
    plt.close()


def plot_fm_pred(xs, ms, xnew, f_mean, f_var, path=None):
    plt.figure(figsize=(10, 4))
    plot_gp_pred(xnew, f_mean, f_var, label=r'$f_m(\cdot)$')
    plt.plot(xs, ms, 'x', ms=10, label=f'Mark')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compare_fm_pred(action_models, args):
    plt.figure(figsize=(10, 4))
    Xnew = np.stack([np.linspace(0.0, 24.0, 50), np.zeros(50)]).astype(np.float64).T
    for model, label, color in zip(action_models,
                                   [r'$f_m(\cdot)$ Baseline', r'$f_m(\cdot)$ Operation'],
                                   ['tab:blue', 'tab:orange']):
        f_mean, f_var = model.predict_f(Xnew)
        plot_gp_pred(Xnew[:, 0].reshape(-1, 1), f_mean, f_var, color=color, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_figures_dir, f'compare_fm.pdf'))
    plt.close()


def plot_treatment_intensity_train(model, baseline_times, actions, outcome_tuples, model_figures_dir, args,
                                   plot_confidence=True, oracle_model=None):
    N_test = 200
    for d, (baseline_time, action_time, outcome_tuple) in enumerate(zip(baseline_times, actions, outcome_tuples)):
        X_abs = np.linspace(*args.domain, N_test)
        X = get_relative_input_by_query(X_abs, baseline_time, action_time, outcome_tuple, args)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4), gridspec_kw={'height_ratios': [2, 1],
                                                                                        'wspace': 0.0,
                                                                                        'hspace': 0.0})
        plt.subplot(2, 1, 1)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlim(np.min(X_abs), np.max(X_abs))
        lw = 1.5
        label = get_tm_label(args)

        _, lambda_mean = model.predict_lambda(X)
        plt.plot(X_abs, lambda_mean, 'b--', lw=lw, label=label)
        if oracle_model is not None:
            _, oracle_lambda_mean = oracle_model.predict_lambda_compiled(X)
            plt.plot(X_abs, oracle_lambda_mean, 'g--', lw=lw, label=r'$\lambda_{oracle}$')

        _ = plt.xticks(np.linspace(0.0, 20.0, 5), fontsize=12)
        ylim = plt.gca().get_ylim()
        plt.vlines(action_time, *ylim, colors='red', linewidth=2.0, label=r'Treatments $\mathbf{a}$', zorder=-1)
        plt.vlines(outcome_tuple[:, 0], *ylim, colors='black', alpha=0.05)
        plt.ylabel(r'$\lambda(t)$', fontsize=16)
        plt.legend(loc='upper left', fontsize=12)

        plt.subplot(2, 1, 2)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlim(np.min(X_abs), np.max(X_abs))
        plt.plot(outcome_tuple[:, 0], outcome_tuple[:, 1], 'kx', label=r"Outcomes $\mathbf{o}$")
        ylim = plt.gca().get_ylim()
        plt.vlines(outcome_tuple[:, 0], *ylim, colors='black', alpha=0.1)
        plt.ylabel(r'$y(t)$', fontsize=16)
        plt.xlabel(r'Time $t$', fontsize=16)
        plt.xticks([])
        plt.legend(loc='upper left', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(model_figures_dir, f'train_fit_d{d}_c{str(plot_confidence)[0]}.pdf'))
        plt.close()


def plot_trig_kernel_baseline_comp(model, actions, model_figures_dir, args, plot_confidence=True, oracle_model=None):
    d = 0
    action = actions[d]
    X = np.full((100, args.D), np.inf, dtype=float)
    X_flat = np.linspace(*args.domain, 100)
    X[:, 0] = X_flat
    lambda_mean, lower, upper = model.predict_lambda_and_percentiles(X)
    lower = lower.numpy().flatten()
    upper = upper.numpy().flatten()
    f_mean, lambda_mean = model.predict_lambda(X)

    plt.figure(figsize=(12, 6))
    plt.xlim(np.min(X_flat), np.max(X_flat))
    # plt.plot(X_flat, lambda_mean, 'r--', label=r'$\lambda(t)$')
    plt.plot(X_flat, f_mean, 'g--', label=r'$f(t)$', alpha=0.2)
    if oracle_model is not None:
        f_mean_oracle, _ = oracle_model.predict_lambda_compiled(X)
        plt.plot(X_flat, f_mean, 'r--', label=r'$f_{oracle}$', alpha=0.2)

    if plot_confidence:
        plt.fill_between(X_flat, lower, upper, color='red', alpha=0.2, label='Confidence')

    _ = plt.xticks(np.linspace(*args.action_time_domain, 10), fontsize=12)
    ylim = plt.gca().get_ylim()
    plt.vlines(action, *ylim, colors='blue')
    # _ = plt.yticks(np.linspace(0.0, 0.2, 5), fontsize=12)
    plt.xlabel(r'Time, $t$', fontsize=16)
    plt.ylabel(r'$\lambda(t)$', fontsize=16)
    plt.title(r'Estimated $\lambda(t)$', fontsize=20)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_d{d}_Dt0_c{str(plot_confidence)[0]}.pdf'))
    plt.close()


def plot_trig_kernel_action_comp(model, model_figures_dir, args, plot_confidence=True, oracle_model=None):
    X = np.full((100, args.D), np.inf, dtype=float)
    X_flat = np.linspace(*args.action_time_domain, 100)
    X[:, args.action_dim] = X_flat
    f_mean, lambda_mean = model.predict_lambda(X)

    plt.figure(figsize=(12, 6))
    plt.xlim(np.min(X_flat), np.max(X_flat))
    # plt.plot(X_flat, lambda_mean, 'r--', label=r'$\lambda(t)$')
    plt.plot(X_flat, f_mean, 'g--', label=r'$f(t)$', alpha=0.2)
    if oracle_model is not None:
        f_mean_oracle, _ = oracle_model.predict_lambda_compiled(X)
        plt.plot(X_flat, f_mean_oracle, 'r--', label=r'$f_{oracle}$', alpha=0.2)

    if plot_confidence:
        _, lower, upper = model.predict_lambda_and_percentiles(X)
        lower = lower.numpy().flatten()
        upper = upper.numpy().flatten()
        plt.fill_between(X_flat, lower, upper, color='red', alpha=0.2, label='Confidence')

    _ = plt.xticks(np.linspace(*args.action_time_domain, 10), fontsize=12)
    plt.xlabel(r'Time, $t$', fontsize=16)
    plt.ylabel(r'$\lambda(t)$', fontsize=16)
    plt.title(r'Estimated $\lambda(t)$', fontsize=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_Dt1_c{str(plot_confidence)[0]}.pdf'))
    plt.close()


def plot_trig_kernel_outcome_comp(model, model_figures_dir, args, oracle_model=None):
    N_test = 100
    t_grid = np.linspace(*args.domain, N_test + 1)
    m_grid = np.linspace(4.0, 8.5, N_test + 1)
    xx, yy = np.meshgrid(t_grid, m_grid)
    X_plot = np.vstack((xx.flatten(), yy.flatten())).T
    X = np.full((X_plot.shape[0], args.D), np.inf, dtype=float)
    X[:, args.outcome_dim] = X_plot[:, 0]
    X[:, args.outcome_dim+1] = X_plot[:, 1]
    f_pred, lambda_pred = model.predict_lambda(X)
    lambda_pred_2d = lambda_pred.numpy().reshape(*xx.shape)
    f_pred_2d = f_pred.numpy().reshape(*xx.shape)

    plt.figure(figsize=(12, 6))
    mark_ids = np.linspace(0, N_test, 5).astype(int)
    for m_idx in mark_ids:
        m = yy[int(m_idx), 0]
        lambda_pred_m = lambda_pred_2d[int(m_idx)]
        plt.plot(t_grid, lambda_pred_m, lw=2, label=r'$\lambda(\cdot,$ ' + f'{m:.1f})')
    plt.title('Estimated Marked Intensity 1D', fontsize=24)
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_marked_1d.pdf'))
    plt.close()

    plt.figure(figsize=(12, 6))
    mark_ids = np.linspace(0, N_test, 5).astype(int)
    for m_idx in mark_ids:
        m = yy[int(m_idx), 0]
        f_pred_m = f_pred_2d[int(m_idx)]
        plt.plot(t_grid, f_pred_m, lw=2, label=r'$f(\cdot,$ ' + f'{m:.1f})')

    plt.title('Estimated Marked Intensity 1D', fontsize=24)
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'f_est_marked_1d.pdf'))
    plt.close()

    plt.figure(figsize=(12, 6))
    _ = plt.contourf(
        xx,
        yy,
        lambda_pred_2d,
        20,
        # [0.5],  # plot the p=0.5 contour line only
        cmap="RdGy_r",
        linewidths=2.0,
        # zorder=100,
    )
    plt.colorbar()
    plt.title('Estimated Marked Intensity 2D', fontsize=24)
    plt.savefig(os.path.join(model_figures_dir, f'lambda_est_marked_2d.pdf'))
    plt.close()

    X = np.full((N_test+1, args.D), np.inf, dtype=float)
    X[:, args.outcome_dim] = 12.0
    X[:, args.outcome_dim + 1] = m_grid
    f_pred, lambda_pred = model.predict_lambda(X)
    plt.figure(figsize=(12, 6))
    plt.plot(m_grid, f_pred, 'b', label=r'Estm. $f(12,\mathbf{m})$')
    if oracle_model is not None:
        f_mean_oracle, _ = oracle_model.predict_lambda_compiled(X)
        plt.plot(m_grid, f_mean_oracle, 'g--', label=r'$f_{oracle}$', alpha=0.2)

    plt.legend(fontsize=18, loc='upper right')
    plt.xlabel(r'Marks, $\mathbf{m}$', fontsize=18)
    plt.ylabel(r'$\lambda(\cdot, cdot)$', fontsize=18)
    plt.title('Estimated vs. True Mark Effect', fontsize=24)
    #
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.linspace(args.mark_domain[0], args.mark_domain[1], 10))
    plt.tight_layout()
    plt.savefig(os.path.join(model_figures_dir, f'mark_effect_est.pdf'))
    plt.close()


def compare_ga(x, ga, labels, args):
    plt.figure(figsize=(6, 4))
    plt.xlim(x.min(), x.max())
    for label, gi in zip(labels, ga):
        plt.plot(x, gi, '--', linewidth=3, label=label, alpha=1.0)

    _ = plt.xticks(np.linspace(x[0], x[-1], 10), fontsize=12)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.ylabel(r'$g^*_a(\tau)$', fontsize=16)
    plt.title(r'Treatment-Dep. Functions $g^*_a(\tau)$', fontsize=18)
    plt.legend(fontsize=14 if len(labels) <= 2 else 8, loc='lower right', framealpha=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_pred_dir, f'compare_ga.pdf'))
    plt.savefig(os.path.join(args.model_pred_dir, f'compare_ga.png'))
    plt.close()


def compare_go(x, go, labels, args):
    plt.figure(figsize=(6, 4))
    for label, f_mark_effect in zip(labels, go):
        plt.plot(x, f_mark_effect, '--', linewidth=3, label=label)
    plt.xlabel(r'Mark, $m$', fontsize=16)
    plt.ylabel(r'$g^*_o(\cdot, m)$', fontsize=16)
    plt.title(r'Outcome-Dep. Functions $g^*_o(\tau)$', fontsize=18)
    plt.legend(fontsize=14 if len(labels) <= 2 else 8, loc='upper right', framealpha=1.0)
    #
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.linspace(x[0], x[-1], 10))
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_pred_dir, f'compare_go.pdf'))
    plt.savefig(os.path.join(args.model_pred_dir, f'compare_go.png'))
    plt.close()


def compare_mark_intensity(x, f_mark, labels, args):
    plt.figure(figsize=(6, 4))
    plt.xlim(np.min(x), np.max(x))
    for label, fi in zip(labels, f_mark):
        plt.plot(x, fi, '--', linewidth=3, label=label, alpha=1.0)

    _ = plt.xticks(np.linspace(0.0, 24.0, 10), fontsize=12)
    plt.ylabel(r'Carb. Intake (Dosage, $\log g$)', fontsize=16)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.title(r'Mark Intensity Functions $p(m \mid \tau)$', fontsize=18)
    plt.legend(loc='lower right', fontsize=14, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_pred_dir, f'compare_mark_intensity.pdf'))
    plt.savefig(os.path.join(args.model_pred_dir, f'compare_mark_intensity.png'))
    plt.close()
