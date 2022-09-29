import tensorflow as tf

# activate for easier debugging - can jump inside of tf functions
tf.config.run_functions_eagerly(False)


class Loss():
    
    def __init__(self, pinn, config):
        
        # save neural network
        self.pinn = pinn

        # save IC
        self.x0 = config['x0']
        self.y0 = config['y0']
        self.l1, self.l2, self.m1, self.m2, self.g = config['l1'], config['l2'], config['m1'], config['m2'], config['g']
        self.x_max = config['x_domain'][1]
        self.n_epochs = config['n_epochs']

       
    def IC(self):
      
        x0 = tf.constant([[self.x0]], dtype=tf.float32)
        y0 = tf.constant(self.y0, dtype=tf.float32)
        y0_pred = tf.squeeze(self.pinn.y_pred_func(x0))
        
        diff = y0 - y0_pred
        loss = tf.reduce_mean(tf.square(diff)) 
        
        return loss
      
    def ODE_equations(self, x_col):
      
        with tf.GradientTape() as t:
            t.watch(x_col)
            y = self.pinn.y_pred_func(x_col)
            dx = tf.expand_dims(y[:,2:], axis=2)
        ddx = t.batch_jacobian(dx, x_col) 
                 
        dx1 = dx[:,0]
        dx2 = dx[:,1]
        dx3 = ddx[:,0,0]
        dx4 = ddx[:,1,0]
        x1 = tf.expand_dims(y[:,0], axis=1)
        x2 = tf.expand_dims(y[:,1], axis=1)
        x3 = tf.expand_dims(y[:,2], axis=1)
        x4 = tf.expand_dims(y[:,3], axis=1)
                
        k1 = -self.g * ((2 * self.m1) + self.m2) * tf.math.sin(x1)
        k2 = self.m2 * self.g * tf.math.sin(x1 - (2 * x2))
        k3 = 2 * tf.math.sin(x1 - x2) * self.m2
        k4 = ((x4**2) * self.l2) + ((x3**2) * self.l1 * tf.math.cos(x1 - x2))
        k5 = self.m2 * tf.math.cos((2 * x1) - (2 * x2))
        k6 = 2 * tf.math.sin(x1 - x2)
        k7 = ((x3**2) * self.l1 * (self.m1 + self.m2))
        k8 = self.g * (self.m1 + self.m2) * tf.math.cos(x1)
        k9 = (x4**2) * self.l2 * self.m2 * tf.math.cos(x1 - x2)

        ddx1 = (k1 - k2 - (k3 * k4)) / (self.l1 * ((2 * self.m1) + self.m2 - k5))
        ddx2 = (k6 * (k7 + k8 + k9)) / (self.l2 * ((2 * self.m1) + self.m2 - k5))

        # Residuals 
        res_x1 = dx1 - x3
        res_x2 = dx2 - x4
        res_x3 = ( dx3 - ddx1 )
        res_x4 = ( dx4 - ddx2 )
        
        res_vec = [res_x1,res_x2,res_x3,res_x4]
        
        return res_vec
      
      
    def F(self, x_col):
        
        res_vec = self.ODE_equations(x_col)
        res_x1, res_x2, res_x3, res_x4 = res_vec
        
        loss_Fx1 = tf.reduce_mean(tf.square(res_x1))    
        loss_Fx2 = tf.reduce_mean(tf.square(res_x2))    
        loss_Fx3 = tf.reduce_mean(tf.square(res_x3)) 
        loss_Fx4 = tf.reduce_mean(tf.square(res_x4)) 

        return loss_Fx1, loss_Fx2, loss_Fx3, loss_Fx4
    

    def F_residuals(self, x_col):
                
        res_vec = self.ODE_equations(x_col)
        res_x1, res_x2, res_x3, res_x4 = res_vec
        
        res_Fx1 = tf.math.abs(res_x1) 
        res_Fx2 = tf.math.abs(res_x2) 
        res_Fx3 = tf.math.abs(res_x3) 
        res_Fx4 = tf.math.abs(res_x4) 

        return res_Fx1, res_Fx2, res_Fx3, res_Fx4

        
      