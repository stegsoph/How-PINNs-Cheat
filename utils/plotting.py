import matplotlib.pyplot as plt
import numpy as np

from utils.angle_helpers import wrap_angles
from utils.exact import exact_RK

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS FOR PLOTTING 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def latex_float(f):
    """
    Creates a legend without any duplicate labels 

    Parameters
    ----------
    f : float
      Float to be transformed into a LaTeX string that is formatted with the scientific notation: has a single digit to the left of the decimal point.
    Returns
    -------
    float_str: str
        LaTeX string that is formatted floating number
    """
    float_str = "{:.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def legend_without_duplicate_labels(ax, bbox, idx=None, alpha=0, ncol=1, loc='best', fontsize=9):
    """
    Creates a legend without any duplicate labels 

    Parameters
    ----------
    ax : Axes object
    bbox: 2-tuple, or 4-tuple of floats
        A 2-tuple (x, y) places the corner of the legend specified by loc at x, y.
        If a 4-tuple or BboxBase is given, then it specifies the bbox (x, y, width, height) that the legend is placed in. 
    idx: list
        Specifies the legend entries that are displayed
    alpha: float 
        The alpha transparency of the legend's background
    ncol: int
        The number of columns that the legend has.
    loc: str or pair of floats
        The location of the legend.
    fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        The font size of the legend
    """

    handles, labels = ax.get_legend_handles_labels()
    unique = np.array([(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]])
    if idx!=None:  
        unique = unique[np.array(idx)]
    
    ax.legend(*zip(*unique),framealpha=alpha, bbox_to_anchor=bbox, ncol=ncol,loc=loc, fontsize=fontsize)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_optimal_figsize(fig_width_pt = 397.48499, scale=0.495, height_factor=None):
  """
  Returns the optimal figsize for LaTeX figures 

  Parameters
  ----------
  fig_width_pt : float
      Obtained from LaTeX: with \the\textwidth 
  scale: float
      Specify how much space the figure should fill (0-1 range) 
  height_factor: float, None
      Specify the ratio between height and width
      If None: using golden ratio
  Returns
  -------
  fig_size: list
      [figure width, figure height]
  """
  fig_width_pt = fig_width_pt*scale       # Scaled figure width
  inches_per_pt = 1.0/72.27               # Convert pt to inches
  if height_factor==None:
    height_factor = (np.sqrt(5)-1.0)/2.0  # Golden ratio
  fig_width = fig_width_pt*inches_per_pt  # width in inches
  fig_height = fig_width*height_factor    # height in inches
  fig_size = [fig_width,fig_height]
  return fig_size

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def init_plot_style(use_tex=False):
  """
  Initialize the plot style for pyplot.

  Parameters
  ----------
  use_tex : Boolean
      True - use LaTeX to render text (requires a working tex installation)
  """
  
  plt.rcParams.update({'figure.figsize': (4, 3)})
  plt.rcParams.update({'figure.dpi' : 150 })
  plt.rcParams.update({'lines.linewidth': 1.2})
  plt.rcParams.update({'lines.markersize': 4})
  plt.rcParams.update({'lines.markeredgewidth': 1})

  if use_tex:
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'font.family': 'serif',
                        'font.serif': 'computer modern'})

  plt.rcParams.update({ 'font.size' : 10,
                        'axes.labelsize' : 10,
                        'legend.fontsize': 10,
                        'xtick.labelsize' : 10,
                        'ytick.labelsize' : 10,
  })


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER CLASS FOR PLOTTING THE SOLUTION TRAJECTORIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Plotting():

  """
  A class used to simplify plotting the trajectories of the double pendulum
  ...

  Methods
  -------
  plotting_yt_sol(axes, exact=True, color='r')
    Plots the predicted and reference trajectories

  plotting_losses(axes):
    Plots the losses over the epochs and the residuals over the computational domain

  plotting_continuation(axes,t_cont=0):
    Plots the RK solution with the predicted initial condition of the PINN at time step t_cont
    Parameters

  """
    
  def __init__(self, log, use_tex=False, method = 'RK45', alpha=1):
    """
        Parameters
        ----------
        log : dict
            The dictionary containing the PINN parameters and predictions from training
        use_tex : Boolean
            True - use LaTeX to render text (requires a working tex installation)
        method : str
            Integration method to use for scipy.integrate.solve_ivp
            -> 'RK45' (default)
        alpha : float or None
            Set the alpha value used for transparency of the lines (0-1 range) 
    """
    self.log = log
    self.alpha = alpha
    self.method = method
    init_plot_style(use_tex)


  def plotting_yt_sol(self,axes,exact=True, color='r'):
    """Plots the PINN trajectories (theta_1, theta_2) in axes[0] and (omega_1, omega_2) in axes[1], along with the reference solution if specified

    Parameters
      ----------
      axes : Flattened axes array
      exact : Boolean
          Specifies if the RK reference solution is visualized
      color : str
          Set the color of the line.
    """
    x = np.array( self.log['x_line'] )
    y = np.array( self.log['y_pred'] )
    y0 = np.array( self.log['y0'] )
    y_rk45 = exact_RK(np.array(self.log['x_line']).squeeze(), self.log['y0'], 
                      self.log['l1'], self.log['l2'], 
                      self.log['m1'], self.log['m2'], 
                      self.log['g'], method=self.method)

    if exact:
      axes[0].plot(x, wrap_angles(y_rk45[0]), 'b', alpha=self.alpha, label = self.method+': $\\theta_1$')
      axes[0].plot(x, wrap_angles(y_rk45[1]), 'b--', alpha=self.alpha, label = self.method+': $\\theta_2$')
      axes[1].plot(x, y_rk45[2], 'b', alpha=self.alpha, label = self.method+': $\omega_1$')
      axes[1].plot(x, y_rk45[3], 'b--', alpha=self.alpha, label = self.method+': $\omega_2$')
    
    axes[0].plot(x, wrap_angles(y[:,0]), c=color, alpha=self.alpha, label = 'PINN: $\\theta_1$')
    axes[0].plot(x, wrap_angles(y[:,1]), c=color, ls='--', alpha=self.alpha, label = 'PINN: $\\theta_2$')
    axes[1].plot(x, y[:,2], c=color, alpha=self.alpha, label = 'PINN: $\omega_1$')
    axes[1].plot(x, y[:,3], c=color, ls='--', alpha=self.alpha, label = 'PINN: $\omega_2$')

    axes[0].scatter(x[0], wrap_angles(y0[0]),c='g',marker='x', label='IC')
    axes[0].scatter(x[0], wrap_angles(y0[1]),c='g',marker='x')
    axes[1].scatter(x[0], (y0[2]),c='g',marker='x', label='IC')
    axes[1].scatter(x[0], (y0[3]),c='g',marker='x')
    
    for ax in axes:
      ax.set_xlabel('$t$')
      ax.grid()

  def plotting_losses(self,axes):
    """Plots the losses over the epochs in axes[0] and the residuals over the computational domain in axes[1]

    Parameters
      ----------
      axes : Flattened axes array
    """

    losses, res = [], []
    for key in ['loss_IC', 'loss_Fx1', 'loss_Fx2', 'loss_Fx3', 'loss_Fx4']:
      losses.append(self.log[key])
    for key in ['res_Fx1', 'res_Fx2', 'res_Fx3', 'res_Fx4']:
      res.append(self.log[key])

    loss_train = np.sum(np.array(losses),axis=0)
    x = np.array( self.log['x_line'] )
    x_epoch = np.arange(len(losses[0]))*self.log['freq_log']

    axes[0].plot(x_epoch, losses[1], 'C0', alpha=self.alpha, label = r'L$_{F \theta 1}$')
    axes[0].plot(x_epoch, losses[2], 'C1', alpha=self.alpha, label = r'L$_{F \theta 2}$')
    axes[0].plot(x_epoch, losses[3], 'C2', alpha=self.alpha, label = r'L$_{F \omega 1}$')
    axes[0].plot(x_epoch, losses[4], 'C3', alpha=self.alpha, label = r'L$_{F \omega 2}$')
    axes[0].plot(x_epoch, losses[0], 'C4', alpha=self.alpha, label = 'L$_{IC}$')
    axes[0].plot(x_epoch, loss_train, 'C6', ':', alpha=self.alpha, label = 'L$_{train}$')
 
    axes[1].plot(x,res[0], 'C0', alpha=self.alpha, label = r'res$_{F \theta 1}$')
    axes[1].plot(x,res[1], 'C1', alpha=self.alpha, label = r'res$_{F \theta 2}$')
    axes[1].plot(x,res[2], 'C2', alpha=self.alpha, label = r'res$_{F \omega 1}$')
    axes[1].plot(x,res[3], 'C3', alpha=self.alpha, label = r'res$_{F \omega 2}$')

    axes[0].set_xlabel('epochs')
    axes[1].set_xlabel('$t$')

    for ax in axes:
      ax.set_yscale('log')
      ax.grid()
      
  def plotting_continuation(self,axes,t_cont=0):
    """Plots the RK solution with the predicted initial condition of the PINN at time step t_cont
    Parameters
      ----------
      axes : Flattened axes array
      t_cont : float
        Specifies the time step from where the RK solution is evaluated with the corresponding PINN prediction
    """
    x = np.array(self.log['x_line']).squeeze()
    y = np.array(self.log['y_pred'])

    idx_new = x >= t_cont
    x_new = x[idx_new]
    y_new = y[idx_new,:]
    y0_new = y_new[0,:]
    
    y_cont = exact_RK(x_new, y0_new, method=self.method)

    color = 'tab:cyan'

    axes[0].plot(x_new, wrap_angles(y_cont[0]), '-', c=color, label = self.method+'$_{\mathrm{shifted}}$: $\\theta_1$')
    axes[0].plot(x_new, wrap_angles(y_cont[1]), '--', c=color, label = self.method+'$_{\mathrm{shifted}}$: $\\theta_2$')

    axes[1].plot(x_new, y_cont[2], '-', c=color, label = self.method+'$_{\mathrm{shifted}}$: $\omega_1$')
    axes[1].plot(x_new, y_cont[3], '--', c=color, label = self.method+'$_{\mathrm{shifted}}$: $\omega_2$')
