import tensorflow as tf

from pinn.data_loader import DataLoader
from pinn.loss_functions import Loss
from pinn.callback import CustomCallback
import numpy as np

# tf.config.run_functions_eagerly(True)

##############################################################################    
    
class PhysicsInformedNN(tf.keras.Sequential):
    
    def __init__(self, config, verbose=False): 
                     
        # call parent constructor & build NN
        super().__init__(name='PINN')   

        # build the neural network
        self.build_network(config, verbose)  
          
        # create data_loader instance
        self.data_loader = DataLoader(config)  
        
        # create loss instance
        self.loss = Loss(self, config)
        
        # create callback instance
        self.callback = CustomCallback(config, self.neural_net)   
        self.save_data = config.get('save_data',False)

        # preprocessing setting
        self.norm = config.get('norm_flag',True)
        self.x_domain = config['x_domain']
        self.preprocessing_domain = config['preprocessing_domain']
        
        # parameters for training
        self.n_epochs = config['n_epochs']
        self.learning_rate = config['learning_rate']
        self.decay_rate = config['decay_rate']
        self.alpha_IC = config['alpha_IC']

        print('*** PINN build & initialized ***')


  ##############################################################################
  # Build the network
  ##############################################################################

    def build_network(self, config, verbose):
        
        # set random seeds
        tf.random.set_seed(config['seed_pinn']) 
        
        # layer settings
        n_hidden = config['n_hidden']     
        n_neurons = config['n_neurons']      
        activation = config['activation']

        # create subnetwork for scaling input
        self.neural_net = tf.keras.Sequential()

        # build input layers
        self.neural_net.add( tf.keras.layers.InputLayer(input_shape=(1,),
                                                        name="input"))
      
        # build hidden layers
        for i in range(n_hidden):
            self.neural_net.add(tf.keras.layers.Dense(units=n_neurons, 
                                                      activation=activation, 
                                                      name=f"hidden_{i}"))
           
        self.neural_net.add(tf.keras.layers.Dense(units=2,
                                                  activation=None,
                                                  name="output"))
          
        if verbose:
            self.neural_net.summary() 

  ##############################################################################
  # PINN functions
  ##############################################################################

  
    # --------------------------------------------------------------------------
    # Normalization of input vector
    def normalization(self, X):
      print('Feature scaling performed')
      X_norm = (X - self.x_domain[0]) / (self.x_domain[1] - self.x_domain[0])
      X_norm = X_norm * (self.preprocessing_domain[1] - self.preprocessing_domain[0]) + self.preprocessing_domain[0]
      return X_norm

    # --------------------------------------------------------------------------
    # Call the neural network with normalized input: 
    # Input: t
    # Output: [th1, th2]
    def call(self, X):
        if self.norm == True:
          X_norm = self.normalization(X)
        else:
          X_norm = X
        return self.neural_net(X_norm)

    # --------------------------------------------------------------------------
    # Predict the full system vector: 
    # Input: t
    # Output: [th1, th2, w1, w2]
    def y_pred_func(self, x_line):

        with tf.GradientTape() as t:
            t.watch(x_line)
            theta = self(x_line)
        omega = t.batch_jacobian(theta, x_line)[:,:,0] 

        y_pred_full = tf.concat((theta, omega), axis=1)

        return y_pred_full     


  ##############################################################################
  ##############################################################################  
   
    def train(self):
         
        # learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate)  
                                
        # Adam optimizer with default settings for momentum
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) 

        print("Training started...")
        for epoch in range(self.n_epochs):
            
            x_col = self.data_loader.sample_collocation() 

            # perform one train step
            train_logs = self.train_step(x_col)

            # provide logs to callback class
            self.callback.write_logs(train_logs, epoch)
                                
        # final prediction
        log = self.validate()
        self.callback.write_logs(log, self.n_epochs)
        
        # save logs
        self.callback.save_logs()
        print("Training finished!")


    @tf.function
    def train_step(self, x_col):
       
        # open a GradientTape to record forward/loss pass     
        with tf.GradientTape(persistent=True) as tape:
            
            # Data loss
            loss_IC = self.loss.IC() 

            # Physics loss
            loss_Fx1, loss_Fx2, loss_Fx3, loss_Fx4 = self.loss.F(x_col)
            
            # Weighted physics loss
            loss_F_weighted = ( loss_Fx1 + loss_Fx2 + loss_Fx3 + loss_Fx4 ) 

            # Training loss
            loss_train = ( self.alpha_IC * loss_IC + ( 1 - self.alpha_IC ) * loss_F_weighted  )

        # retrieve gradients 
        grads_F = tape.gradient(loss_F_weighted, self.neural_net.weights)    
        grads_IC = tape.gradient(loss_IC, self.neural_net.weights)

        del tape      

        # final gradient for optimization step       
        grads = [ ( self.alpha_IC * g_IC + ( 1 - self.alpha_IC ) * g_F  ) for g_F, g_IC in zip(grads_F, grads_IC)]   
          
        # perform single GD step 
        self.optimizer.apply_gradients(zip( grads, self.neural_net.weights))     
        
        # save logs for recording
        train_logs = {'loss_train': loss_train,
                      'loss_IC': loss_IC,
                      'loss_Fx1': loss_Fx1, 
                      'loss_Fx2': loss_Fx2,
                      'loss_Fx3': loss_Fx3,
                      'loss_Fx4': loss_Fx4, 
                      }
        
        return train_logs

    ##############################################################################
    ##############################################################################

    def validate(self):

        # equally spaced points
        x_line = self.data_loader.x_line()

        # get final prediction
        y_pred_full = self.y_pred_func(x_line)
        
        # get final loss values
        res_Fx1, res_Fx2, res_Fx3, res_Fx4 = self.loss.F_residuals(x_line)
        
        log = {'x_line': x_line.numpy().tolist(),
               'y_pred': y_pred_full.numpy().tolist(),
               'res_Fx1': res_Fx1.numpy().tolist(),
               'res_Fx2': res_Fx2.numpy().tolist(),
               'res_Fx3': res_Fx3.numpy().tolist(),
               'res_Fx4': res_Fx4.numpy().tolist(),
               }
        return log
    
    
    
    
    
        