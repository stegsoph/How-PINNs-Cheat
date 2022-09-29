import numpy as np
import tensorflow as tf

from numpy.random import random

class DataLoader():
    
    def __init__(self, config):

        np.random.seed(config['seed_data'])
        
        # read system settings
        x_domain = config['x_domain']
        self.x_min, self.x_max = min(x_domain), max(x_domain)
        self.x_range = self.x_max - self.x_min
        
        # read data settings
        self.n_col = config['n_col']
        
        
    def x_line(self, N=1000):
        
        x_line = np.linspace(self.x_min, self.x_max, N)
        x_line = np.expand_dims(x_line, axis=1)
       
        return tf.convert_to_tensor(x_line, dtype=tf.float32)
    
    def sample_collocation(self):
        
        x_col = self.x_min + self.x_range * random(self.n_col)
        x_col = np.expand_dims(x_col, axis=1)

        return tf.convert_to_tensor(x_col, dtype=tf.float32)

        

        
