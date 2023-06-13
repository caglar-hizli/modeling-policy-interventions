import argparse
import os

import numpy as np
import pandas as pd

from utils import constants_synth
from utils.utils import add_parser_arguments, prepare_dirs, get_run_identifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patient", type=int, default=20)
    parser.add_argument("--n_day_train", type=int, default=2)  # same for obs
    parser.add_argument("--n_day_test", type=int, default=1)  # same for test and int
    parser.add_argument("--sampler_dir", type=str, default='sampler')
    parser.add_argument("--model_output_dir", type=str, default='models')
    parser.add_argument("--sample_output_dir", type=str, default='samples')
    parser.add_argument("--seeds", type=str, default='1,2,3,4,5,6,7,8,9,10')
    parser = add_parser_arguments(parser, constants_synth.GENERAL_PARAMS)
    parser = add_parser_arguments(parser, constants_synth.SAMPLER_PARAMS)
    parser = add_parser_arguments(parser, constants_synth.GPRPP_PARAMS)
    parser = add_parser_arguments(parser, constants_synth.GPRPP_FAO)

    horizons = [24]
    init_args = parser.parse_args()
    seeds_int = [int(s) for s in init_args.seeds.split(',')]
    obs_error, int_error, cf_error = [], [], []
    run_id = get_run_identifier(init_args)
    for s in seeds_int:
        path = os.path.join(init_args.model_output_dir, f'joint.seed{s}', run_id,
                            'evaluate/policy_observational/observational_error.csv')
        obs_error_seed = pd.read_csv(path)
        if s == 1:
            obs_error = obs_error_seed
        else:
            obs_error = pd.concat([obs_error, obs_error_seed])

    print(obs_error.mean())
    print(np.sqrt(obs_error.var() / len(obs_error)))

    int_exp_ids = ['vbpp_schulam', 'gprpp_schulam', 'hua', 'gprpp_hua',
                   'vbpp', 'hua_nptr', 'observational', 'interventional']
    for s in seeds_int:
        path = os.path.join(init_args.model_output_dir, f'joint.seed{s}', run_id,
                            'evaluate/policy_interventional/interventional_error.csv')
        int_error_seed = pd.read_csv(path)
        if s == 1:
            int_error = int_error_seed
        else:
            int_error = pd.concat([int_error, int_error_seed])
    print(int_error.mean())
    print(np.sqrt(int_error.var() / len(int_error)))

    cf_exp_ids = ['vbpp_schulam', 'gprpp_schulam', 'hua', 'gprpp_hua',
                  'vbpp', 'hua_nptr', 'observational', 'interventional', 'counterfactual']
    for s in seeds_int:
        path = os.path.join(init_args.model_output_dir, f'joint.seed{s}', run_id,
                            f'evaluate/policy_counterfactual/counterfactual_error.csv')
        cf_error_seed = pd.read_csv(path)
        if s == 1:
            cf_error = cf_error_seed
        else:
            cf_error = pd.concat([cf_error, cf_error_seed])
    print(cf_error.mean())
    print(np.sqrt(cf_error.var() / len(cf_error)))
