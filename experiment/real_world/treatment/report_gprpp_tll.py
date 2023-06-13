import argparse
import os

import numpy as np
import pandas as pd


def report_gprpp_test_ll(patient_ids, model_dir):
    action_components_array = ['b', 'ba', 'bo', 'ao', 'bao']
    test_ll_pidx = {ac: [] for ac in action_components_array}
    for patient_id in patient_ids:
        patient_test_lls = []
        for action_comp in action_components_array:
            print(patient_id, action_comp)
            run_id = f'action.p{patient_id}'
            metric_path = os.path.join(model_dir, f'f{action_comp}', run_id, 'metric', 'll_metrics.csv')
            df = pd.read_csv(metric_path)
            test_ll_pidx[action_comp].append(df['test_ll_lbo'].item())
            patient_test_lls.append(df['test_ll_lbo'].item())
        max_idx_test = np.argmax(patient_test_lls)
        print(f'For patient {patient_id}, test_lls: {patient_test_lls}')
        print(f"For Patient {patient_id}, "
              f"Max comp: {action_components_array[max_idx_test]},  Max test ll: {patient_test_lls[max_idx_test]}")

    mean_tll = {k: np.mean(v) for k, v in test_ll_pidx.items()}
    std_tll = {k: np.sqrt(np.var(v) / len(patient_ids)) for k, v in test_ll_pidx.items()}
    print(f'Mean tll test: {mean_tll}')
    print(f'Std tll test: {std_tll}')
    df_test_ll = pd.DataFrame.from_dict(test_ll_pidx)
    os.makedirs('report.gprpp', exist_ok=True)
    df_test_ll.to_csv(os.path.join('out/report.gprpp', 'test_ll_gprpp.csv'), index=False)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models.gprpp')
    parser.add_argument("--patient_ids", type=str, default='0,1')
    init_args = parser.parse_args()
    patient_ids = [int(s) for s in init_args.patient_ids.split(',')]
    report_gprpp_test_ll(patient_ids, init_args.model_dir)
