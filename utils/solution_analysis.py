from utils.exact import pendulum
import numpy as np
import scipy.signal
import pandas as pd

def residuals_from_y(y, x, xmax):
    """
    Return the mean physics residuals 
        res_i = mean( |theta_i'' - f_i(theta_1,theta_2,theta_1',theta_2')| )
    and the mean magnitude of the second order derivatives 
        mean( | theta_i'' | )
    given the solution trajectories y and evaluation time t, 
    derivatives are computed using the central finite difference method. 

    Parameters
    ----------
    y : np.array with shape (4, len(x))
      System vector containing the solution trajectories: [theta_1,theta_2,theta_1',theta_2']
    x : np.array with shape (len(x),)
    Returns
    -------
    (res_1_mean, res_2_mean, deriv_1_mean, deriv_2_mean): 4-tuple
        
    """
    n_col = len(x)

    dX = pendulum(y,x) 
    x_to_xmax = x[(x<=xmax)]
    y_to_xmax = y[:,(x<=xmax)]
    dX_to_xmax = [ dX_line[(x<=xmax)] for dX_line in dX ]

    x, y, dX = x_to_xmax, y_to_xmax, dX_to_xmax

    x1 = y[0,:]
    x2 = y[1,:]
    x3 = np.gradient(x1,xmax/n_col)
    x4 = np.gradient(x2,xmax/n_col)
    dx3 = np.gradient(x3,xmax/n_col)
    dx4 = np.gradient(x4,xmax/n_col)

    res_1 = dx3 - dX[2]
    res_2 = dx4 - dX[3]
    res_1_abs = np.abs(res_1[20:-20])
    res_2_abs = np.abs(res_2[20:-20])

    res_1_mean = np.mean(res_1_abs)
    res_2_mean = np.mean(res_2_abs)

    deriv_1_mean = np.mean( np.abs(dX[2]) )
    deriv_2_mean = np.mean( np.abs(dX[3]) )
    
    return (res_1_mean, res_2_mean, deriv_1_mean, deriv_2_mean)


def bandpower(ylist, fs, f0, nfft=0, window='hamming'):
    """
    Return the relative fraction of total signal power above a certain frequency f0

    Parameters
    ----------
    ylist : np.array with shape (2, len(x)) 
        Reduced system vector containing the angles: [theta_1,theta_2]
    fs : float
        Sampling frequency
    f0 : float
        Boundary frequency
    nfft : int
        Number of zero-padding points
    window: str
        Window function
    Returns
    -------
    power : float
        Relative fraction of total signal power above f0
    yf_RK45_list : list
        List containing the periodogram of theta_1 and theta_2
    f : array
        Frequency vector 
    """
    power = 0
    yf_RK45_list = []

    for y in ylist: 
        f, Pxx = scipy.signal.periodogram(y, fs = fs, scaling = 'spectrum', nfft = nfft+len(y), window = window)
        ind_min = np.argmax(f > f0) - 1
        
        power_temp = np.trapz(Pxx[ind_min::], f[ind_min::])
        total_power = np.trapz(Pxx[:], f[:])
        power_norm = power_temp / total_power

        yf_RK45_list.append(Pxx)
        power+=power_norm
        
    return power, yf_RK45_list, f


def get_theta_start_stop(data):
    """
    Simple helper function to obtain true initial condition and predicted initial condition given a list of PINN log files

    Parameters
    ----------
    data : list of dict
        Containing all information from PINN training and prediction 
    Returns
    -------
    theta_start: float
        Given initial angle
    th1_final, th2_final: floats
        Predicted initial angle
    """
    df = pd.DataFrame(data)
    df['theta'] = df['y0'].apply(lambda x: x[0]*180/np.pi)
    df['th1_final'] = df['y_pred'].apply(lambda x: x[0][0]*180/np.pi)
    df['th2_final'] = df['y_pred'].apply(lambda x: x[0][1]*180/np.pi)

    theta_start = df['theta'].to_numpy()
    th1_final = df['th1_final'].to_numpy()
    th2_final = df['th2_final'].to_numpy()

    return theta_start, th1_final, th2_final