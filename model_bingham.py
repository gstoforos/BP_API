import numpy as np
from scipy.optimize import curve_fit

def r2_score(y_true, y_pred):
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def fit_bingham(shear_rates, shear_stresses, flow_rate, diameter, density, re_critical=4000):
    # Bingham model: τ = τ₀ + μ * γ̇
    def model(gamma_dot, tau0, mu):
        return tau0 + mu * gamma_dot

    popt, _ = curve_fit(model, shear_rates, shear_stresses, bounds=(0, np.inf))
    tau0, mu = popt
    k = mu  # Rename for consistency

    predicted = model(np.array(shear_rates), *popt)
    r2 = r2_score(shear_stresses, predicted)

    n = 1
    avg_shear_rate = np.mean(shear_rates)
    mu_app = tau0 / avg_shear_rate + mu if avg_shear_rate != 0 else float('inf')

    velocity = flow_rate / (np.pi * (diameter / 2) ** 2)
    re = (density * velocity * diameter) / mu_app

    q_critical = (np.pi * diameter ** 2 / 4) * ((re_critical * mu_app) / (density * diameter))

    equation = f"τ = {tau0:.3f} + {mu:.3f}·γ̇"

    return {
        "equation": equation,
        "tau0": tau0,
        "k": k,
        "n": n,
        "r2": r2,
        "mu_app": mu_app,
        "re": re,
        "re_critical": re_critical,
        "q_critical": q_critical
    }
