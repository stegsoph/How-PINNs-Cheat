from distutils.spawn import find_executable
from utils.angle_helpers import angles2xy, wrap_angles
from utils.write_log_helper import check_pi

import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################

class PostProcessing():

  def __init__(self, log, idx=0, alpha=1, IN_COLAB=0, method = 'RK45'):
    self.log = log
    self.idx = idx
    self.alpha = alpha
    self.method = method


  def estimation_error(self, domain=None, log=None):
    
    if domain == None:
      domain = self.log['x_domain']
      
    if log==None:
      log = self.log
    
    x_line = np.array( log['x_line'] ).squeeze()
    idx_domain = [ np.argmin( np.abs( x_line - domain[0] ) ),
                   np.argmin( np.abs( x_line - domain[1] ) ) ]
      
    y_PINN = np.array( log['y_pred'] )
    y_exact = np.array( self.exact_sol() ).T

    if y_PINN.shape[0] != y_exact.shape[0]:
      y_exact = np.array( self.exact_sol() )
      
    error_sq = ( y_exact - y_PINN ) ** 2  
    
    return error_sq, idx_domain


  def exact_sol(self):
    
    y_rk45 = exact_RK(np.array(self.log['x_line']).squeeze(), self.log['y0'], 
                      self.log['l1'], self.log['l2'], 
                      self.log['m1'], self.log['m2'], 
                      self.log['g'], method=self.method)
    return y_rk45
  
  
  def exact_continuation(self, t_cont=0, direction='positive'):
    
    x = np.array(self.log['x_line']).squeeze()
    y = np.array(self.log['y_pred'])

    idx_new = x >= t_cont
    x_new = x[idx_new]
    y_new = y[idx_new,:]
    y0_new = y_new[0,:]
    
    y_cont = exact_RK(x_new, y0_new, 
                      self.log['l1'], self.log['l2'], 
                      self.log['m1'], self.log['m2'], 
                      self.log['g'], method=self.method)

    return y_cont, x_new
  
