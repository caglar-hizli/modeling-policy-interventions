import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from experiment.real_world.treatment.run_gprpp_joint import get_glucose_dataset, prepare_tm_input

# Uncomment for latex fonts, latex should be available in path
# plt.rcParams['text.usetex'] = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='dataset/public_dataset.csv')
    parser.add_argument('--patient_id', type=int, default=8)
    parser.add_argument('--remove_night_time', action='store_true')
    parser.add_argument('--hours_day', type=float, default=24.0)
    parser.add_argument('--figure_dir', type=str, default='figures.figure3')
    parser.add_argument('--plot_functional_components', type=str, default='ao')
    parser.add_argument('--treatment_model_dir', type=str, default='models.gprpp')
    parser.add_argument('--outcome_model_dir', type=str, default='models.outcome')
    args = parser.parse_args()
    args.preprocess_actions = True
    args.domain = [0.0, 24.0]
    os.makedirs(args.figure_dir, exist_ok=True)
    #
    ds = get_glucose_dataset(args.patient_id, args)
    actions, outcome_tuples, baseline_times = prepare_tm_input([ds], args)
    args.plot_functional_components = args.plot_functional_components.split(',')
    Nc = len(args.plot_functional_components)
    labels = {
        'bao': r'$\lambda_{bao} = (\beta_0 + g_b + g_a + g_o)^2$',
        'ba': r'$\lambda_{ba} = (\beta_0 + g_b + g_a)^2$',
        'bo': r'$\lambda_{bo} = (\beta_0 + g_b + g_o)^2$',
        'ao': r'$\lambda_{ao} = (\beta_0 + g_a + g_o)^2$',
        'b': r'$\lambda_{b} = (\beta_0 + g_b)^2$',
    }
    d = 1
    baseline_time = baseline_times[d]
    action_time = actions[d]
    outcome_tuple = outcome_tuples[d]

    dst = np.load(os.path.join(args.figure_dir, f'lambda_mean_f{Nc}.npz'), allow_pickle=True)
    X_abs, lambda_means = dst['x'], dst['lambda_means']

    args.remove_night_time = False
    ds = get_glucose_dataset(args.patient_id, args)
    x_out, y_out, t_out, m_out = ds[0], ds[1], ds[2], ds[3]
    mask_x = np.logical_and(x_out > d*args.hours_day, x_out < (d+1)*args.hours_day)
    mask_t = np.logical_and(t_out > d*args.hours_day, t_out < (d+1)*args.hours_day)
    x_out, y_out = x_out[mask_x], y_out[mask_x]
    t_out, m_out = t_out[mask_t], m_out[mask_t]
    xnew = [x_out.astype(np.float64).reshape(-1, 1)]
    anew = [np.hstack([t_out.astype(np.float64).reshape(-1, 1), m_out.astype(np.float64).reshape(-1, 1)])]
    t_lengths = [ti.shape[0] for ti in anew]
    Np = len(t_lengths)

    dso = np.load(os.path.join(args.figure_dir, 'f_mean.npz'), allow_pickle=True)
    x_out, f_mean, f_var, fb_mean, fb_var = dso['x'], dso['f_mean'], dso['f_var'], dso['fb_mean'], dso['fb_var']

    x_out, t_out = x_out % 24.0, t_out % 24.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True,
                                   gridspec_kw={'height_ratios': [1.25, 1], 'wspace': 0.0, 'hspace': 0.05})
    ax1_ylim = (2.5, 7.5)
    plt.subplot(2, 1, 1)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xlim(np.min(x_out), np.max(x_out))
    label1 = r"Outcomes $\mathbf{o}$"
    lines1, = ax1.plot(x_out, y_out, 'kx', label=label1)
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False)

    ax12 = ax1.twinx()
    label2 = r'Treatments, $(\mathbf{t},\mathbf{m})$'
    bar2 = ax12.bar(t_out, m_out, color='cyan', edgecolor='black', width=0.2, lw=2,
                    label=label2)
    ax12.set_ylim(0.0, np.max(m_out) * (ax1_ylim[1] - ax1_ylim[0]))
    ax12.set_ylabel(r'Carb., $m$ (log g)', fontsize=16, loc='bottom')
    ax12.set_yticks([0.0, 2.0, 4.0])
    ax12.set_yticklabels(['0.0', '2.0', '4.0'], fontsize=12)
    plt.yticks()
    # ax12.vlines(t_out - 0.1, 2.0, 2.8, color='black', linewidth=3.0, )
    # ax12.scatter(t_out - 0.1, [3.0] * len(t_out), marker='o', edgecolor='black',
    #             linewidth=3.0, facecolor='cyan', s=100, label=r'Treatments $\mathbf{a}$')
    label3 = r"$f_b$"
    lines3, = ax1.plot(x_out, fb_mean, 'tab:olive', label=label3)
    ax1.fill_between(
        x_out,
        fb_mean[:, 0] - np.sqrt(fb_var[:, 0]),
        fb_mean[:, 0] + np.sqrt(fb_var[:, 0]),
        color='tab:olive',
        alpha=0.2,
    )
    label4 = r"$y = f_b + f_a + \epsilon$"
    lines4, = ax1.plot(x_out, f_mean, 'r', label=label4)
    ax1.fill_between(
        x_out,
        f_mean[:, 0] - np.sqrt(f_var[:, 0]),
        f_mean[:, 0] + np.sqrt(f_var[:, 0]),
        color='r',
        alpha=0.2,
    )
    ax1.set_ylim(*ax1_ylim)
    ax1.set_yticks([4.0, 5.0, 6.0, 7.0])
    ax1.set_yticklabels(['4.0', '5.0', '6.0', '7.0'], fontsize=12)
    # plt.vlines(action_time - 0.1, *ylim, colors='cyan', linewidth=1.5,
    #            zorder=-1, alpha=1.0)
    # plt.vlines(outcome_tuple[:, 0], ylim[0], outcome_tuple[:, 1], colors='black', alpha=0.1)
    ax1.set_ylabel(r'Glucose, $y$ (mmol/l)', fontsize=16)
    # plt.xlabel(r'Time $t$', fontsize=16)
    ax1.vlines(outcome_tuple[0, 0] - 0.1, *ax1_ylim, linestyle="--", color="grey")
    ax1.axvspan(0.0, outcome_tuple[0, 0] - 0.1, color="grey", alpha=0.1)
    ax1.annotate("Night\nTime", xytext=(0.5, ax1_ylim[1] - 1.75), textcoords='data',
                 xycoords='data', xy=(outcome_tuple[0, 0] - 0.1, ax1_ylim[1] - 1.4),
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"), )
    # plt.xticks([])
    ax1.legend([lines1, bar2, lines3, lines4],
               [label1, label2, label3, label4],
               loc='upper right', fontsize=12, framealpha=1.0)

    plt.subplot(2, 1, 2)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xlim(np.min(X_abs), np.max(X_abs))
    lw = 1.5

    colors = ['tab:blue', 'tab:pink', 'tab:olive', 'tab:green', 'tab:orange']
    plt.vlines(action_time - 0.1, -0.2, -0.1, color='black', linewidth=3.0, )
    plt.scatter(action_time - 0.1, [-0.05] * len(action_time), marker='o', edgecolor='black',
                linewidth=3.0, facecolor='cyan', s=100, label=r'Treatment times $\mathbf{t}$')
    for ac, c, lambda_mean in zip(args.plot_functional_components, colors, lambda_means):
        plt.plot(X_abs, lambda_mean, '--', color=c, lw=lw, label=labels[ac])
    ylim = plt.gca().get_ylim()
    plt.vlines(outcome_tuple[0, 0] - 0.1, *ylim, linestyle="--", color="grey")
    plt.axvspan(0.0, outcome_tuple[0, 0] - 0.1, color="grey", alpha=0.1)
    plt.annotate("Night\nTime", xytext=(0.5, ylim[1] - 0.27), textcoords='data',
                 xycoords='data', xy=(outcome_tuple[0, 0] - 0.1, ylim[1] - 0.2),
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"), )
    _ = plt.xticks(np.linspace(0.0, 20.0, 5), fontsize=12)
    plt.ylabel(r'Intensity $\lambda(t)$', fontsize=16)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=12)
    plt.xlabel(r'Time, $\tau$', fontsize=16)
    plt.legend(loc='upper right', fontsize=12 if Nc <= 2 else 10, framealpha=1.0)

    plt.tight_layout()
    filename = f'train_fit_d{d}_p{args.patient_id}'
    filename += '_fig3.pdf' if Nc <= 2 else '_fig7.pdf'
    plt.savefig(os.path.join(args.figure_dir, filename))
    plt.close()
