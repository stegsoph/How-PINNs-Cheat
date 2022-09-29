import matplotlib.pyplot as plt
import numpy as np

from utils.angle_helpers import wrap_angles
from utils.postprocessing import PostProcessing
from utils.exact import exact_RK

def latex_float(f):
    float_str = "{:.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def legend_without_duplicate_labels(ax, bbox, idx=None,alpha=0,ncol=1,loc='best',fontsize=9,delete_strings=False):
    handles, labels = ax.get_legend_handles_labels()
    if delete_strings:
      labels = [elem[:-12] for elem in labels]
    unique = np.array([(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]])
    if idx!=None:  
        unique = unique[np.array(idx)]
    
    ax.legend(*zip(*unique),framealpha=alpha, bbox_to_anchor=bbox, ncol=ncol,loc=loc, fontsize=fontsize)

def get_optimal_figsize(fig_width_pt = 455.24408, scale=0.495, height_factor=None):
  fig_width_pt = fig_width_pt*scale
  inches_per_pt = 1.0/72.27               # Convert pt to inches
  if height_factor==None:
    height_factor = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
  fig_width = fig_width_pt*inches_per_pt  # width in inches
  fig_height =fig_width*height_factor       # height in inches
  fig_size = [fig_width,fig_height]
  return fig_size

def init_plot_style(use_tex=False):
    """Initialize the plot style for pyplot.
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


class Plotting(PostProcessing):
  
  def __init__(self, log, idx=0, alpha=1, use_tex=False, method = 'RK45'):
    self.log = log
    self.idx = idx
    self.alpha = alpha
    self.method = method

    init_plot_style(use_tex)

    
  def plotting_continuation(self,axes,t_cont=0):
    
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


  def plotting_yt_sol(self,axes,title=None,exact=True, color='r'):

    x = np.array( self.log['x_line'] )
    y = np.array( self.log['y_pred'] )
    y0 = np.array( self.log['y0'] )
    y_rk45 = self.exact_sol()

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
    
    if (self.idx == 0): 
      if title==None:
        title=' '*len(axes)
      for ax, tit in zip(axes, title):
        ax.tick_params(axis='x', which='minor')
        ax.set_xlabel('t')
        ax.legend(ncol = 4, loc='best')
        ax.set_title(tit)
    
  def plotting_losses(self,axes,title=None):
    losses, res = self.get_losses()
    loss_train = np.sum(np.array(losses),axis=0)
    x = np.array( self.log['x_line'] )
    x_epoch = np.arange(len(losses[0]))*self.log['freq_log']

    axes[0].plot(x_epoch, losses[1], 'C0', alpha=self.alpha, label = r'L$_{F \theta 1}$')
    axes[0].plot(x_epoch, losses[2], 'C1', alpha=self.alpha, label = r'L$_{F \theta 2}$')
    axes[0].plot(x_epoch, losses[3], 'C2', alpha=self.alpha, label = r'L$_{F \omega 1}$')
    axes[0].plot(x_epoch, losses[4], 'C3', alpha=self.alpha, label = r'L$_{F \omega 2}$')
    axes[0].plot(x_epoch, losses[0], 'C4', alpha=self.alpha, label = 'L$_{IC}$')
    axes[0].plot(x_epoch, loss_train, 'tab:orange', ':', alpha=self.alpha, label = 'L$_{train}$')
 
    axes[1].plot(x,res[0], 'C0', alpha=self.alpha, label = r'res$_{F \theta 1}$')
    axes[1].plot(x,res[1], 'C1', alpha=self.alpha, label = r'res$_{F \theta 2}$')
    axes[1].plot(x,res[2], 'C2', alpha=self.alpha, label = r'res$_{F \omega 1}$')
    axes[1].plot(x,res[3], 'C3', alpha=self.alpha, label = r'res$_{F \omega 2}$')

    if (self.idx == 0): 
      if title==None:
        title=' '*len(axes)
      axes[0].set_xlabel('\# epochs')
      axes[1].set_xlabel('t')

      for ax, tit in zip(axes, title):
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.set_title(tit)
