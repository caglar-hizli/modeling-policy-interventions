import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from experiment.real_world.dataset.glucose_dataset import load_glucose_dfs, prepare_glucose_ds
from utils import constants_rw

# Uncomment for latex fonts, latex should be available in path
# plt.rcParams['text.usetex'] = True


def plot_samples(pidx, sample_patient_ids, args):
    args.remove_night_time = False
    sample_path = os.path.join(args.sample_dir, 'patients.npz')
    dataset_dict = np.load(sample_path, allow_pickle=True)
    patient_pair_datasets_oracle = dataset_dict['ds']
    ds_oracle = [pair_ds for pair_ds in patient_pair_datasets_oracle if len(pair_ds[2]) > 0]
    fig, axes = plt.subplots(len(sample_patient_ids), 1, figsize=(10, 7.5),
                             sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    for n, idx in enumerate(sample_patient_ids):
        ds = ds_oracle[idx]
        ax = axes[n]
        plot_joint_data(ax, ds, n == len(args.patient_ids_int) - 1)

    plt.tight_layout()
    patient_str = ','.join([str(s) for s in sample_patient_ids])
    plt.savefig(os.path.join(args.output_dir, f'fig11_synth_data_p{pidx}_s{patient_str}.pdf'))
    plt.close()


def plot_joint_data(ax, ds, plot_xticks):
    plt.xlim(0.0, 24.0)
    ax_ylim = (2.5, 9.0)
    x_out, y_out, t_out, m_out = ds[0], ds[1], ds[2], ds[3]
    action_time = t_out
    outcome_tuple = np.stack([x_out, y_out]).T
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    label1 = r"Outcomes $\mathbf{o}$"
    label2 = r'Treatments, $(\mathbf{t},\mathbf{m})$'
    label3 = r'Response'
    lines1, = ax.plot(x_out, y_out, 'kx', label=label1)
    ax_twin = ax.twinx()
    bar2 = ax_twin.bar(t_out, m_out, color='cyan', edgecolor='black', width=0.2, lw=2,
                       label=label2)
    ax_twin.set_ylim(0.0, np.max(m_out) * (ax_ylim[1] - ax_ylim[0]))
    ax_twin.set_ylabel('Carb. (log g)', fontsize=16)
    ax_twin.set_yticks([0.0, 2.0, 4.0])
    ax_twin.set_yticklabels(['0.0', '2.0', '4.0'], fontsize=12)
    ax.set_ylim(*ax_ylim)
    ax.set_yticks([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], fontsize=12)
    ax.set_ylabel(r'Glucose (mmol/l)', fontsize=16)
    ax.legend([lines1, bar2], [label1, label2], loc='upper left', fontsize=12, framealpha=1.0)
    if plot_xticks:
        ax.set_xticks(np.linspace(0.0, 24.0, 7), fontsize=12)
        ax.set_xlabel(r'Time, $\tau$', fontsize=16)
    else:
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            labelbottom=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_day_train", type=int, default=3)
    parser.add_argument("--n_day_test", type=int, default=0)
    parser.add_argument("--patient_ids", type=str, default='3,8,12')
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument("--treatment_covariates", type=str, default='SUGAR,STARCH')
    parser.add_argument('--output_dir', type=str, default='figures.figure11')
    parser.add_argument('--sample_dir', type=str, default='figures.figure11')
    init_args = parser.parse_args()
    init_dict = vars(init_args)
    init_dict.update(constants_rw.GENERAL_PARAMS)
    init_args = argparse.Namespace(**init_dict)
    init_args.patient_ids_int = [int(s) for s in init_args.patient_ids.split(',')]
    init_args.preprocess_actions = True
    init_args.domain = [0.0, 24.0]
    os.makedirs(init_args.output_dir, exist_ok=True)
    for pidx, sample_patient_ids in zip(init_args.patient_ids_int, [[12, 15, 27], [1, 4, 25], [29, 41, 47]]):
        plot_samples(pidx, sample_patient_ids, init_args)

    for pidx in init_args.patient_ids_int:
        dfs = load_glucose_dfs(init_args, [pidx])
        ds, _, _ = prepare_glucose_ds(dfs[0], init_args)
        for d in range(3):
            x, y, t, m = ds
            mask_x = np.logical_and(x > d*init_args.hours_day, x < (d+1)*init_args.hours_day)
            mask_t = np.logical_and(t > d*init_args.hours_day, t < (d+1)*init_args.hours_day)
            ds_day = (x[mask_x] % 24.0, y[mask_x], t[mask_t] % 24.0, m[mask_t])
            plt.figure(figsize=(10, 3))
            ax = plt.gca()
            plot_joint_data(ax, ds_day, True)
            plt.tight_layout()
            plt.savefig(os.path.join(init_args.output_dir, f'fig11_real_data_p{pidx}_d{d}.pdf'))
            plt.close()
