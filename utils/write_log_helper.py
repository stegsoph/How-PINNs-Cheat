from pathlib import Path
import numpy as np
from decimal import Decimal

def check_pi(y, div_str = '_'):
    y_copy = []
    for yi in y:
        if yi == 0:
            y_copy.append('0')
        elif abs(Decimal(yi/np.pi).as_tuple().exponent) == 0: # no decimals
            y_copy.append(str(Decimal(yi/np.pi))+'pi')
        elif abs(Decimal(np.pi/yi).as_tuple().exponent) == 0: # no decimals
            y_copy.append('pi'+div_str+str(Decimal(np.pi/yi)))
        else:
            y_copy.append(str(np.round(yi,3)))
    return y_copy
            

def return_log_path(config):

    if 'log_order' in config:
      order = config['log_order']
    else:
      order = ['y0', 't', 'layers', 'epochs', 'loss']
            
    y0 = config['y0']
    y0_str = check_pi(y0)
    y0_str = ','.join(y0_str)
    sys_params = [config['l1'],config['l2'],config['m1'],config['m2']]
    
    p = Path()
    #---------------------------------------------------------------------
    # create the path according to given order
    #---------------------------------------------------------------------

    for string in order: 

        # initial condition
        if string == 'y0':
          p = Path(p, 'y0=[' + y0_str+']') 

        # sys parameters: L1,L2,m1,m2 
        elif string == 'sys_par':
          p = Path(p, 'sys_par_' + str(sys_params)) 

        # computational domain t & preprocessing domain
        elif string == 't':
          if config['norm_flag'] == True:
            norm_str = '_norm_'+str(config['preprocessing_domain']) 
          else: 
            norm_str = ''
          p = Path(p, 't_'+ str(config['x_domain'])+norm_str) 
        
        # NN parameters: hidden layers & neurons
        elif string == 'layers':
          p = Path(p, str(config['n_hidden'])+'x'+str(config['n_neurons'])) 

        # epochs
        elif string == 'epochs':
          p = Path(p, 'n_epochs_' + str(config['n_epochs']))

        # loss weights
        elif string == 'loss': 
          p = Path(p, 'lambda_IC_' + str(config['lambda_IC']))

    #---------------------------------------------------------------------
    file_name = 'SD_'+str(config['seed_data'])+'_SP_'+str(config['seed_pinn'])+'.json'
        
    return p, file_name