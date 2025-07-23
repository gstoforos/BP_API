import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def fit_bingham(data):
    try:
        gamma_dot = np.array(data.get("shear_rates", []), dtype=float)
        sigma = np.array(data.get("shear_stresses", []), dtype=float)

        if len(gamma_dot) != len(sigma) or len(gamma_dot) == 0:
            return {"error": "Invalid or empty data provided."}

        def model(gamma_dot, sigma0, mu): return sigma0 + mu * gamma_dot

        popt, _ = curve_fit(model, gamma_dot, sigma, bounds=(0, np.inf))
        sigma0, mu = popt
        predictions = model(gamma_dot, *popt)
        r2 = r2_score(sigma, predictions)

        mu_app = mu  # For Bingham, apparent viscosity equals plastic viscosity

        return {
            "model": "Bingham Plastic",
            "tau0": round(sigma0, 6),
            "k": round(mu, 6),
            "r2": round(r2, 6),
            "mu_app": round(mu_app, 6),
            "re": None,
            "equation": f"τ = {sigma0:.3g} + {mu:.3g}·γ̇"
        }

    except Exception as e:
        return {"error": f"Fitting failed: {str(e)}"}
