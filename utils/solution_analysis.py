from utils.exact import pendulum, exact_RK
import numpy as np
import scipy.signal
import pandas as pd



def res_median_from_y(y,x,xmax):

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

    res_x3 = dx3 - dX[2]
    res_x4 = dx4 - dX[3]
    res_x3_abs = np.abs(res_x3[20:-20])
    res_x4_abs = np.abs(res_x4[20:-20])

    res_x3_mean = np.mean(res_x3_abs)
    res_x4_mean = np.mean(res_x4_abs)

    res_x3_rhs_mean = np.mean( np.abs(dX[2]) )
    res_x4_rhs_mean = np.mean( np.abs(dX[3]) )
    
    return (res_x3_mean, res_x4_mean, res_x3_rhs_mean, res_x4_rhs_mean)


def bandpower(ylist, fs, fmin, nfft=0, window='blackman'):
    
    power = 0
    Pxx_list = []

    for y in ylist: 
        f, Pxx = scipy.signal.periodogram(y, fs = fs, scaling = 'spectrum', nfft = nfft+len(y), window = window)
        ind_min = np.argmax(f > fmin) - 1
        
        power_temp = np.trapz(Pxx[ind_min::], f[ind_min::])
        total_power = np.trapz(Pxx[:], f[:])
        power_norm = power_temp / total_power

        Pxx_list.append(Pxx)
        power+=power_norm
        
    return power, Pxx_list, f


def get_theta_start_stop(data):

    df_pinn = pd.DataFrame(data)
    df_pinn['theta'] = df_pinn['y0'].apply(lambda x: x[0]*180/np.pi)
    df_pinn['th1_final'] = df_pinn['y_pred'].apply(lambda x: x[0][0]*180/np.pi)
    df_pinn['th2_final'] = df_pinn['y_pred'].apply(lambda x: x[0][1]*180/np.pi)

    pinn_theta_start = df_pinn['theta'].to_numpy()
    pinn_th1_final = df_pinn['th1_final'].to_numpy()
    pinn_th2_final = df_pinn['th2_final'].to_numpy()
    lambda_IC = df_pinn['lambda_IC'].to_numpy()[0]

    return pinn_theta_start, pinn_th1_final, pinn_th2_final, lambda_IC