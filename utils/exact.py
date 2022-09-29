from scipy.integrate import solve_ivp 
import numpy as np

def pendulum(X,t,l1=1,l2=1,m1=1,m2=1,g=9.81):
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
    
    ##############################################################################
    # RUNGE KUTTA - solve ivp
    ##############################################################################
    sol = solve_ivp(lambda t_eval, X: pendulum(X,t_eval,l1,l2,m1,m2,g), 
                    [t_eval.min(), t_eval.max()], y0, method=method, t_eval=t_eval)
    X_true = sol.y
    return X_true


def data_generation_rk(config, data_type='full',  domain_type = 'inside', method = 'RK45',n_col=1000):
  
    if config['n_data_'+data_type] == 0: 
      x_data, y_data = np.array(np.NaN),np.empty((4,1))
      y_data[:] = np.NaN
      
    else:
      y0 = config['y0']
      l1, l2, m1, m2, g = config['l1'], config['l2'], config['m1'], config['m2'], config['g']

      # whole data domain, starting from x0
      x = np.linspace(config['x_domain'][0],config['x_domain'][1], n_col)
      y_rk45 = exact_RK(x, y0, l1, l2, m1, m2, g, method=method)

      # select only data points from INSIDE the specified data domain:
      if domain_type == 'inside':
        x_temp, y_temp = x[x<=config['data_domain_'+data_type][1]], y_rk45[:,x<=config['data_domain_'+data_type][1]]
        x_temp, y_temp = x_temp[x_temp>=config['data_domain_'+data_type][0]], y_rk45[:,x>=config['data_domain_'+data_type][0]]

      # select only data points OUTSIDE of the specified data domain:
      else:
        index_lower = x<=config['data_domain_'+data_type][0]
        inder_upper = x>=config['data_domain_'+data_type][1]
        x_lower, y_lower = x[index_lower], y_rk45[:, index_lower]
        x_upper, y_upper = x[inder_upper], y_rk45[:, inder_upper]
        x_temp = np.concatenate((x_lower,x_upper))
        y_temp = np.concatenate((y_lower,y_upper),axis=1)
    
      # select the specified amount of data points, given by N=config['n_data']
      # uniformly due to linspace: index from 0 to the end of the array with N steps
      idx_data = np.linspace(0,len(x_temp),config['n_data_'+data_type], dtype = int, endpoint=False)
      # indexing
      x_data, y_data = x_temp[idx_data], y_temp[::,idx_data]

    return x_data, y_data
    
def data_config_update(config, domain_type='inside'):
    
    config_temp = {}
    x_data_full,y_data_full = data_generation_rk(config, data_type='full', domain_type=domain_type)
    
    config_temp.update( {'y_data_full':  y_data_full.tolist(), 'x_data_full': x_data_full.tolist()} )

    x_data_partial,y_data_partial = data_generation_rk(config, data_type='partial', domain_type=domain_type)
    config_temp.update( {'y_data_partial':  y_data_partial.tolist(), 'x_data_partial': x_data_partial.tolist()} )

    config.update(config_temp)

    return config