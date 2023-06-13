GENERAL_PARAMS = {
    'T_treatment': 3.0,
    'hours_day': 24.0,
    'mins_hour': 60.0,
    'dataset': 'dataset/public_dataset.csv',
    'treatment_covariates': ['SUGAR', 'STARCH'],
}
# TREATMENT PARAMS
TREATMENT_PARAMS = {
    'preprocess_actions': True,
}
# GPRPP PARAMS
GPRPP_PARAMS = {
    'domain': [0.0, 24.0],
    'share_variance': False,
    'share_lengthscales': False,
    'optimize_variance': False,
    'optimize_lengthscales': False,
    'remove_night_time': True,
    'compute_nll': True,
    'M_times': 20,
    'beta0': 0.1,
}
# GPRPP F_BAO
GPRPP_PARAMS_FBAO = {
    'variance_init': [0.1, 0.05, 0.15],
    'lengthscales_init': [7.0, 1.0, 100.0, 2.5],
    'variance_prior': [1.0, 0.5, 0.5],
    'lengthscales_prior': [7.0, 1.0, 100.0, 2.5],
    'Dt': 3,
    'Dm': 1,
    'action_dim': 1,
    'outcome_dim': 2
}
# GPRPP F_BA
GPRPP_PARAMS_FBA = {
    'variance_init': [0.1, 0.05],
    'lengthscales_init': [7.0, 1.0],
    'variance_prior': [1.0, 0.5],
    'lengthscales_prior': [7.0, 1.0],
    'Dt': 2,
    'Dm': 0,
    'action_dim': 1,
}
# GPRPP F_BO
GPRPP_PARAMS_FBO = {
    'variance_init': [0.1, 0.15],
    'lengthscales_init': [7.0, 100.0, 2.5],
    'variance_prior': [1.0, 0.5],
    'lengthscales_prior': [7.0, 100.0, 2.5],
    'Dt': 2,
    'Dm': 1,
    'outcome_dim': 1
}
# GPRPP F_AO
GPRPP_PARAMS_FAO = {
    'variance_init': [0.05, 0.15],
    'lengthscales_init': [1.0, 100.0, 2.5],
    'variance_prior': [0.5, 0.5],
    'lengthscales_prior': [1.0, 100.0, 2.5],
    'Dt': 2,
    'Dm': 1,
    'action_dim': 0,
    'outcome_dim': 1
}
# GPRPP F_B
GPRPP_PARAMS_FB = {
    'variance_init': [0.1],
    'lengthscales_init': [7.0],
    'variance_prior': [1.0],
    'lengthscales_prior': [7.0],
    'Dt': 1,
    'Dm': 0
}