import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Uncomment for latex fonts, latex should be available in path
# plt.rcParams['text.usetex'] = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_ids', type=str, default='0,6,18')
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--hours_day', type=float, default=24.0)
    parser.add_argument('--output_dir', type=str, default='figures.figure12')
    parser.add_argument('--sample_dir', type=str, default='figures.figure12')
    args = parser.parse_args()
    args.patient_ids_int = [int(s) for s in args.patient_ids.split(',')]
    args.preprocess_actions = True
    args.domain = [0.0, 24.0]
    os.makedirs(args.output_dir, exist_ok=True)
    #
    args.remove_night_time = False
    sample_path = os.path.join(args.sample_dir, f'patient_test_multiple_int.npz')
    dataset_dict = np.load(sample_path, allow_pickle=True)
    patient_pair_datasets_oracle = dataset_dict['ds']
    ds_oracle = [pair_ds[0] for pair_ds in patient_pair_datasets_oracle if len(pair_ds[0][2]) > 0]
    N = 3
    fig, axes = plt.subplots(N, 1, figsize=(13, 7.5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1],
                                          # 'wspace': 0.0, 'hspace': 0.05
                                          })
    plt.xlim(24.0, 48.0)
    ax1_ylim = (2.5, 7.5)
    circle_centers_all = [[(27.3, 5.2), (33.0, 4.8), (38.5, 5.0), (42.5, 5.5), (47.0, 6.0)],
                          [(28.5, 5.5), (32.5, 5.5), (37.0, 4.5), (41.5, 5.0)],
                          [(30, 5.5), (40.0, 4.5), (45.5, 5.2)]]
    for n, (pidx, circle_centers_pidx) in enumerate(zip(args.patient_ids_int, circle_centers_all)):
        ds = ds_oracle[pidx]
        x_out, y_out, t_out, m_out = ds[0], ds[1], ds[2], ds[3]
        action_time = t_out
        outcome_tuple = np.stack([x_out, y_out]).T

        ax = axes[n]
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        label1 = r"Outcomes $\mathbf{o}$"
        label2 = r'Treatments, $(\mathbf{t},\mathbf{m})$'
        label3 = r'Response'
        lines1, = ax.plot(x_out, y_out, 'k-o', label=label1)
        ax_twin = ax.twinx()
        bar2 = ax_twin.bar(t_out, m_out, color='cyan', edgecolor='black', width=0.2, lw=2,
                           label=label2)
        ax_twin.set_ylim(0.0, np.max(m_out) * (ax1_ylim[1] - ax1_ylim[0]))
        ax_twin.set_ylabel('Carb. (log g)', fontsize=16)
        ax_twin.set_yticks([0.0, 2.0, 4.0])
        ax_twin.set_yticklabels(['0.0', '2.0', '4.0'], fontsize=12)
        ax.set_ylim(*ax1_ylim)
        ax.set_yticks([4.0, 5.0, 6.0, 7.0], fontsize=12)
        ax.set_ylabel(r'Glucose (mmol/l)', fontsize=16)
        p = []
        for cc in circle_centers_pidx:
            circle = plt.Circle(cc, 1.2, color='red', fill=False)
            p = ax.add_patch(circle)

        ax.legend([lines1, bar2, p], [label1, label2, label3], loc='upper left', fontsize=12, framealpha=1.0)

        if n < len(args.patient_ids_int)-1:
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                labelbottom=False)
        else:
            ax.set_xticks(np.linspace(24.0, 48.0, 7), fontsize=12)
            ax.set_xlabel(r'Time, $\tau$', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'fig12_p{args.patient_ids}.pdf'))
    plt.close()
