GENERAL_PARAMS = {
    'T_treatment': 3.0,
    'hours_day': 24.0,
    'dataset': 'dataset/public_dataset.csv',
    'treatment_covariates': ['SUGAR', 'STARCH'],
    'maxiter': 5000,
    'outcome_maxiter': 1000,
    'plot_joint_train': False,
    'horizons': [24.0],
    'target_distributions': ['observational', 'interventional', 'counterfactual'],
    'target_distribution_suffix': {'observational': 'int',
                                   'interventional': 'int',
                                   'counterfactual': 'cf'},
    'exp_ids': {
        'observational': ['vbpp_schulam', 'gprpp_schulam', 'hua', 'gprpp_hua', 'vbpp', 'hua_nptr', 'observational'],
        'interventional': ['vbpp_schulam', 'gprpp_schulam', 'hua', 'gprpp_hua',
                           'vbpp', 'hua_nptr', 'observational', 'interventional'],
        'counterfactual': ['vbpp_schulam', 'gprpp_schulam', 'hua', 'gprpp_hua', 'vbpp', 'hua_nptr',
                           'observational', 'interventional', 'counterfactual']
    }
}
SAMPLER_PARAMS = {
    'n_outcome': 40,
    'regular_measurement_times': True,
    'noise_std': 0.1,
    'outcome_sampler_patient_ids': [3, 8, 12],
    'observational_action_ids': [4, 13],
    'observational_outcome_ids': [3, 8, 12],
}
# GPRPP
GPRPP_PARAMS = {
    'domain': [0.0, 24.0],
    'preprocess_actions': False,
    'share_variance': False,
    'share_lengthscales': False,
    'optimize_variance': False,
    'optimize_lengthscales': False,
    'remove_night_time': False,
    'compute_nll': False,
    'M_times': 20,
    'beta0': 0.1,
}
# GPRPP model specific
GPRPP_FAO = {
    'action_components': 'ao',
    'Dt': 2,
    'marked_dt': [1],
    'D': 3,
    'marked': True,
    'variance_init': [0.05, 0.15],
    'lengthscales_init': [1.0, 100.0, 2.5],
    'action_dim': 0,
    'outcome_dim': 1,
    'time_dims': [0, 1],
    'mark_dims': [2],
}