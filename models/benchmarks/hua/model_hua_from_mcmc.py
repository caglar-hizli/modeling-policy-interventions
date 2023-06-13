import numpy as np
from scipy.stats import gamma


class TimeIntensityHua:
    def __init__(self, mu, theta_a, k, a, rate, thinned_iters):
        self.mu = mu[thinned_iters-1]  # (M,)
        self.theta_a = theta_a[:, thinned_iters-1]  # (2, M)
        self.k = k[thinned_iters-1]  # (M,)
        self.a = a[thinned_iters-1]  # (M,)
        self.rate = rate[thinned_iters-1]  # (M,)

    def predict_lambda_compiled(self, X, Y):
        alpha = self.k.reshape(-1, 1) / (1 + np.exp(self.theta_a.T @ np.stack([np.ones_like(Y), Y])))  # (M, len(X))
        mask_Y = np.isinf(Y)
        alpha[:, mask_Y] = 0.0
        gamma_pdf_val = np.stack([gamma.pdf(X, a=a, scale=1/r) for a, r in zip(self.a, self.rate)])  # (M, len(X))
        mask_X = np.isinf(X)
        gamma_pdf_val[:, mask_X] = 0.0
        intensity = np.exp(self.mu).reshape(-1, 1) + alpha * gamma_pdf_val  # (M, len(X))
        return intensity.mean(0)


class TimeIntensityHuaMean:
    def __init__(self, mu, theta_a, k, a, rate, thinned_iters):
        self.mu = mu[thinned_iters-1].mean()  # (M,)
        self.theta_a = theta_a[:, thinned_iters-1].mean(1)  # (2, M)
        self.k = k[thinned_iters-1].mean()  # (M,)
        self.a = a[thinned_iters-1].mean()  # (M,)
        self.rate = rate[thinned_iters-1].mean()  # (M,)

    def predict_lambda_compiled(self, X, Y):
        alpha = self.k / (1 + np.exp(self.theta_a.reshape(1, 2) @ np.stack([np.ones_like(Y), Y]))).flatten()
        mask_Y = np.isinf(Y)
        alpha[mask_Y] = 0.0
        gamma_pdf_val = gamma.pdf(X, a=self.a, scale=1/self.rate)  # (M, len(X))
        mask_X = np.isinf(X)
        gamma_pdf_val[mask_X] = 0.0
        intensity = np.exp(self.mu) + alpha * gamma_pdf_val  # (M, len(X))
        return intensity


class OutcomeModelHua:
    def __init__(self, beta_l, b_il, thinned_iters):
        self.beta_l = beta_l[:, thinned_iters-1]  # (4, M)
        self.b_il = b_il[:, :, thinned_iters-1]  # (3, n_patient, M)
        self.thinned_iters = thinned_iters

    def predict_outcome(self, X, dosage, pidx):
        Z = np.stack([np.ones_like(X), dosage, X, X**2])
        mean_fixed = self.beta_l.T @ Z  # (M, len(X))
        R = np.stack([np.ones_like(X), dosage, X])
        mean_random = self.b_il[:, pidx].T @ R  # (M, len(X))
        return mean_fixed.mean(0) + mean_random.mean(0)

    def predict_baseline(self, X, pidx):
        return self.predict_outcome(X=X, dosage=np.zeros_like(X), pidx=pidx)


def get_relative_input_hua(t_query, action_time, outcome_tuple):
    Xs, Ys = [], []
    x, y = outcome_tuple[:, 0], outcome_tuple[:, 1]
    for ti in t_query:
        earlier_action_idx = np.where(action_time < ti)[0]
        if len(earlier_action_idx) > 0:
            closest_of_earlier_idx = np.max(earlier_action_idx)
            closest_of_earlier_actions = action_time[closest_of_earlier_idx]
            Xs.append(ti - closest_of_earlier_actions)
            #
            earlier_outcome_idx = np.where(x < closest_of_earlier_actions)[0]
            if len(earlier_outcome_idx) > 0:
                closest_of_earlier_outcomes = np.max(earlier_outcome_idx)
                Ys.append(y[closest_of_earlier_outcomes])
            else:
                later_outcome_idx = np.where(x > closest_of_earlier_actions)[0]
                closest_of_later_outcomes = np.min(later_outcome_idx)
                Ys.append(y[closest_of_later_outcomes])
        else:
            Ys.append(np.inf)
            Xs.append(np.inf)

    return np.array(Xs), np.array(Ys)


def get_dosage_hua(x, actions, args):
    Dr_i = np.zeros_like(x)
    action_times, action_dosages = actions[:, 0], actions[:, 1]
    for ti, di in zip(action_times, action_dosages):
        mask_treatment_effect = np.logical_and(x >= ti, x < ti + args.T_treatment)
        Dr_i[mask_treatment_effect] = di
    Dr_i = np.array(Dr_i)
    return Dr_i


def load_treatment_params_hua(path):
    mcmc = np.load(path, allow_pickle=True)
    thinned_iters = mcmc['thin_ids'].astype(int)
    mu = mcmc['mu']
    v1 = mcmc['beta_v'][0, 0]
    v2 = mcmc['beta_v'][0, 1]
    k = mcmc['k']
    theta_a = mcmc['theta_a']
    alpha = k.reshape(-1, 1) / (1 + np.exp(theta_a.T @ np.stack([np.ones_like(mcmc['Y']), mcmc['Y']])))
    nu = np.exp(v1)
    a = np.exp(v2) + 1
    rate = (a - 1) / nu
    return mu, theta_a, k, a, rate, thinned_iters


def load_outcome_params_hua(path):
    mcmc = np.load(path, allow_pickle=True)
    thinned_iters = mcmc['thin_ids'].astype(int)
    beta_l = mcmc['beta_l']
    b_il = mcmc['b_il']
    return beta_l, b_il, thinned_iters
