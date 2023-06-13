import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import to_default_float

from evaluate.plot_tm import plot_vbpp
from experiment.real_world.dataset.glucose_dataset import get_daily_treatments
from experiment.predict import predict_vbpp, predict_vbpp_ll_lbo
from models.treatment.vbpp import VBPP
from experiment.real_world.dataset import glucose_dataset
from utils import constants_rw


def build_glucose_action_model(events, domain, M=20, is_max=False):
    kernel = gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=5.0)
    kernel.variance.prior = tfp.distributions.HalfNormal(to_default_float(1.0))
    kernel.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(5.0))
    Z = glucose_dataset.domain_grid(domain, M)
    feature = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(M)
    q_S = np.eye(M)
    model = VBPP(feature, kernel, domain, q_mu, q_S, beta0=0.1, num_observations=len(events))
    gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
    gpflow.utilities.set_trainable(model.beta0, False)
    return model


def run_vbpp_glucose(args):
    datasets = [glucose_dataset.get_glucose_dataset(patient_id, args) for patient_id in args.patient_ids]
    lambda_means = []
    domain = glucose_dataset.action_domain
    test_lls = {}
    for patient_idx, ds in zip(args.patient_ids, datasets):
        patient_dir = os.path.join(args.model_dir, f'action.p{patient_idx}')
        os.makedirs(patient_dir, exist_ok=True)
        os.makedirs(os.path.join(patient_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(patient_dir, 'metric'), exist_ok=True)
        model_path = os.path.join(patient_dir, 'vbpp')
        t = ds[2]
        t_days = get_daily_treatments(t, args)
        t_days_train, t_days_test = t_days[:args.n_day_train], t_days[args.n_day_train:]
        events = [np.array(ev, float).reshape(-1, 1) for ev in t_days_train]
        events_test = [np.array(ev, float).reshape(-1, 1) for ev in t_days_test]
        domain = np.array(glucose_dataset.action_domain, float).reshape(1, 2)

        if not os.path.exists(model_path):
            train_vbpp(events, patient_idx, patient_dir, domain, args)

        saved_model = tf.saved_model.load(model_path)
        X = glucose_dataset.domain_grid(domain, 100)
        _, lambda_mean = saved_model.predict_lambda_compiled(X)
        lambda_means.append(lambda_mean.numpy().flatten())
        #
        ll_lbo, data_term, integral_term = predict_vbpp_ll_lbo(saved_model, events_test)
        df_ll = pd.DataFrame.from_dict({'test_ll_lbo': [ll_lbo.numpy().item()],
                                        'test_data_term': [data_term.numpy().item()],
                                        'test_integral_term': [integral_term.numpy().item()]})
        df_ll.to_csv(os.path.join(patient_dir, 'metric',  'll_metrics.csv'), index=False)
        print(f'For Patient {patient_idx}, test_ll: {ll_lbo}')

    X = glucose_dataset.domain_grid(domain, 100).flatten()
    plt.figure(figsize=(12, 6))
    for i, (lambda_mean, lower, upper) in enumerate(zip(lambda_means)):
        plt.plot(X, lambda_mean, lw=2, label=f'Patient {args.patient_ids[i]} - ' + r'$\lambda(\cdot)$')
    plt.xlabel('Hours', fontsize=16)
    plt.ylabel(r'$\lambda(\cdot)$', fontsize=16)
    plt.title('Ground Truth Meal Intensities', fontsize=24)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'compare_meal_profiles.pdf'))
    plt.close()


def train_vbpp(events, patient_idx, patient_dir, domain, args):
    def objective_closure():
        return -model.elbo(events)

    model = build_glucose_action_model(events, domain)
    # gpflow.utilities.print_summary(model)
    gpflow.optimizers.Scipy().minimize(objective_closure, model.trainable_variables,
                                       compile=True,
                                       options={"disp": False,
                                                "maxiter": args.maxiter}
                                       )
    gpflow.utilities.print_summary(model)
    X, lambda_mean, lower, upper = predict_vbpp(model, domain)
    plot_vbpp(events, X, lambda_mean, upper, lower,
              title=f'Daily Meal Profile of Patient {patient_idx}',
              plot_path=os.path.join(patient_dir, 'figures', f'vbpp_pred.pdf'))
    save_vbpp_model(model, patient_dir)
    return lambda_mean, lower, model, upper


def save_vbpp_model(model: VBPP, output_dir):
    model.predict_lambda_compiled = tf.function(
        model.predict_lambda,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_f_compiled = tf.function(
        model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
    )
    model.predict_integral_term_compiled = tf.function(
        model.predict_integral_term,
        input_signature=[]
    )
    tf.saved_model.save(model, os.path.join(output_dir, f'vbpp'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models.vbpp')
    parser.add_argument("--n_day_train", type=int, default=2)
    parser.add_argument("--n_day_test", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--patient_ids", type=str, default='1')
    init_args = parser.parse_args()
    init_args.patient_ids = [int(s) for s in init_args.patient_ids.split(',')]
    init_dict = vars(init_args)
    init_dict.update(constants_rw.GENERAL_PARAMS)
    init_dict.update(constants_rw.TREATMENT_PARAMS)
    init_args = argparse.Namespace(**init_dict)
    os.makedirs(init_args.model_dir, exist_ok=True)
    run_vbpp_glucose(init_args)
