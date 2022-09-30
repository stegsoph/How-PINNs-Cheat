from scipy.integrate import solve_ivp 
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS FOR OBTAINING THE REFERENCE SOLUTION USING TRADITIONAL ODE SOLVERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pendulum(X,t,l1=1,l2=1,m1=1,m2=1,g=9.81):
    """ Return the first derivatives of X = [theta_1, theta_2, omega_1, omega_2] using the nonlinear state equations of the double pendulum
    Parameters
    ----------
    X : list
        System vector X = [theta_1, theta_2, omega_1, omega_2]
    l1, l2, m1, m2, g : System parameters
        Pendulum rod mass (m1,m2), length (l1, l2), gravitational constant

    Returns
    -------
    dX: list
        First derivative of the vector X' = [theta_1', theta_2', omega_1', omega_2']
    """

    th1, th2, w1, w2 = X

    k1 = -g * ((2 * m1) + m2) * np.sin(th1)
    k2 = m2 * g * np.sin(th1 - (2 * th2))
    k3 = 2 * np.sin(th1 - th2) * m2
    k4 = ((w2**2) * l2) + ((w1**2) * l1 * np.cos(th1 - th2))
    k5 = m2 * np.cos((2 * th1) - (2 * th2))
    k6 = 2 * np.sin(th1 - th2)
    k7 = ((w1**2) * l1 * (m1 + m2))
    k8 = g * (m1 + m2) * np.cos(th1)
    k9 = (w2**2) * l2 * m2 * np.cos(th1 - th2)
    dX = [
        w1,
        w2,
        (k1 - k2 - (k3 * k4)) / (l1 * ((2 * m1) + m2 - k5)),
        (k6 * (k7 + k8 + k9)) / (l2 * ((2 * m1) + m2 - k5))
    ]

    return dX

def exact_RK(t_eval,y0,l1=1,l2=1,m1=1,m2=1,g=9.81, method = 'RK45'):
    """ Numerically integrate the system of ordinary differential equations describing the double pendulum given an initial value:
    Parameters
    ----------
    t_eval : array
        Computational domain
    l1, l2, m1, m2, g : System parameters
        Pendulum rod mass (m1,m2), length (l1, l2), gravitational constant
    method: string or OdeSolver, optional
        Integration method to use: 'RK45' (default)

    Returns
    -------
    X_true: list
        Reference solution of the system vector X = [theta_1, theta_2, omega_1, omega_2]
    """

    sol = solve_ivp(lambda t_eval, X: pendulum(X,t_eval,l1,l2,m1,m2,g), 
                    [t_eval.min(), t_eval.max()], y0, method=method, t_eval=t_eval)
    X_true = sol.y

    return X_true

