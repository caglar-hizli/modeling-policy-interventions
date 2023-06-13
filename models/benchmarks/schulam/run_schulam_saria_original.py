import logging
import pickle
import os

import numpy as np

from models.benchmarks.schulam.bsplines import BSplines
from models.benchmarks.schulam.simulations import common
from models.benchmarks.schulam.simulations import mixture
from models.benchmarks.schulam.model import bspline_mixture_rx


def simulation():
    rng = np.random.RandomState(0)

    # output_dir = 'prediction_output'
    output_dir = 'schulam_prediction_output'
    os.makedirs(output_dir, exist_ok=True)

    class_coef = np.array([
        [ 1.0,  0.9,  0.0, -0.5, -1.0],  # rapidly decline
        [ 1.0,  1.0,  0.5,  0.2,  0.2],  # mild decline
        [-0.3, -0.2, -0.2, -0.3, -0.2]   # stable
    ])

    low, high, n_bases, degree = 0.0, 24.0, class_coef.shape[1], 2
    rx_win, rx_effect = 2.0, 1.0
    n_samples, n_train = 500, 300
    avg_obs = 15
    ln_a, ln_l, noise_scale = 2.0 * np.log(0.2), np.log(8.0), 0.1
    policy1_params =  2.0, -0.5,    0.0  # History window, policy weight, policy bias
    policy2_params =  2.0,  0.5,    0.0
    policy3_params =  2.0,  1.0,    0.0
    policy4_params =  2.0, -0.5,    0.0
    prediction_times = [12.0]
    time = prediction_times[0]

    basis = BSplines(low, high, n_bases, degree, boundaries='space')

    population = mixture.PopulationModel(basis, len(class_coef))
    # population.sample_class_prob(rng)
    population.set_class_prob([0.35, 0.35, 0.3])
    population.set_class_coef(class_coef)

    obs_proc = common.ObservationTimes(low, high, avg_obs)
    policy1 = common.TreatmentPolicy(*policy1_params, rx_win, rx_effect)
    policy2 = common.TreatmentPolicy(*policy2_params, rx_win, rx_effect)
    policy3 = common.TreatmentPolicy(*policy3_params, rx_win, rx_effect)
    policy4 = common.TreatmentPolicy(*policy4_params, rx_win, 0.0)  # Policy 1 with no effect.

    classes, params = population.sample(rng, size=n_samples)
    trajectories = [common.Trajectory(w, basis) for w in params]
    sampled = [common.sample_trajectory(t, obs_proc, ln_a, ln_l, noise_scale, rng) for t in trajectories]

    data_set = {
        'n_train'      : n_train,
        'classes'      : classes,
        'params'       : params,
        'trajectories' : trajectories,
        'sampled'      : sampled,
    }

    with open(os.path.join(output_dir, 'data_set.pkl'), 'wb') as f:
        pickle.dump(data_set, f)

    test1 = common.truncate_treat_data_set(sampled[n_train:], time, policy1, rng)

    # Scenario 1: Train and test on policy 1

    train1 = common.treat_data_set(sampled[:n_train], policy1, rng)

    base1 = train_model(train1, time, basis, 3, 2.0)
    prop1 = train_model(train1, high, basis, 3, 2.0)

    with open(os.path.join(output_dir, 'base1.pkl'), 'wb') as f:
        pickle.dump(base1, f)

    with open(os.path.join(output_dir, 'prop1.pkl'), 'wb') as f:
        pickle.dump(prop1, f)

    base_pred1 = common.evaluate_model(base1, test1, prediction_times)
    prop_pred1 = common.evaluate_model(prop1, test1, prediction_times)

    base_pred1.to_csv(os.path.join(output_dir, 'base_predictions1.csv'), index=False)
    prop_pred1.to_csv(os.path.join(output_dir, 'prop_predictions1.csv'), index=False)

    # Scenario 2: Training on policy 2 and test on policy 1

    train2 = common.treat_data_set(sampled[:n_train], policy2, rng)
    # test2 = common.truncate_treat_data_set(sampled[n_train:], time, policy1, rng)

    base2 = train_model(train2, time, basis, 3, 2.0)
    prop2 = train_model(train2, high, basis, 3, 2.0)

    with open(os.path.join(output_dir, 'base2.pkl'), 'wb') as f:
        pickle.dump(base2, f)

    with open(os.path.join(output_dir, 'prop2.pkl'), 'wb') as f:
        pickle.dump(prop2, f)

    base_pred2 = common.evaluate_model(base2, test1, prediction_times)
    prop_pred2 = common.evaluate_model(prop2, test1, prediction_times)

    base_pred2.to_csv(os.path.join(output_dir, 'base_predictions2.csv'), index=False)
    prop_pred2.to_csv(os.path.join(output_dir, 'prop_predictions2.csv'), index=False)

    # Scenario 3: Train on confounding policy, test on

    treat1 = common.make_treatment_func(policy1, rng)
    treat4 = common.make_treatment_func(policy4, rng)

    treatments = [treat1, treat4]

    treatment_probs = [
        (0.2, 0.8),
        (0.9, 0.1),
        (0.5, 0.5)
    ]

    train3 = []
    for z, (y, t) in zip(classes[:n_train], sampled[:n_train]):
        treatment = rng.choice(treatments, p=treatment_probs[z])
        train3.append(treatment(y, t))

    # test3 = []
    # for z, (y, t) in zip(classes[n_train:], sampled[n_train:]):
    #     treatment = rng.choice(treatments, p=treatment_probs[z])
    #     test3.append(treatment(y, t))

    base3 = train_model(train3, time, basis, 3, 2.0)
    prop3 = train_model(train3, high, basis, 3, 2.0)

    with open(os.path.join(output_dir, 'base3.pkl'), 'wb') as f:
        pickle.dump(base3, f)

    with open(os.path.join(output_dir, 'prop3.pkl'), 'wb') as f:
        pickle.dump(prop3, f)

    base_pred3 = common.evaluate_model(base3, test1, prediction_times)
    prop_pred3 = common.evaluate_model(prop3, test1, prediction_times)

    base_pred3.to_csv(os.path.join(output_dir, 'base_predictions3.csv'), index=False)
    prop_pred3.to_csv(os.path.join(output_dir, 'prop_predictions3.csv'), index=False)


def train_model(data, rx_history, basis, n_classes, rx_w):
    model = bspline_mixture_rx.TreatmentBSplineMixture(basis, 3, 2.0)
    data = common.hide_treatments(data, rx_history)
    model.fit(data)
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    simulation()
