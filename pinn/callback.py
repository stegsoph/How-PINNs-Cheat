import json
import numpy as np

from pathlib import Path
from utils.write_log_helper import return_log_path


class CustomCallback():

    def __init__(self, config, neural_net):      
      
        self.config = config
        self.neural_net = neural_net
        
        # save the data in a json file or not
        self.flag_save_data = config['save_data']

        # determines digits for 'fancy' log printing
        self.n_epochs = config['n_epochs']
        self.digits = int(np.log10(self.n_epochs)+1) 
            
        # create log from config file
        self.log = config.copy()
        
        # log and print freq
        self.freq_log = config['freq_log']
        self.freq_print = config['freq_print']
        self.weights_log = config.get('weights_log', 
                                      np.linspace(0,config['n_epochs']-1,10).tolist())
        
        # keys to be printed
        self.keys_print = config['keys_print']
        
        # name of log folder 
        if 'log_name' in config:
          print('log_name given')
          log_name = config['log_name']
        else:
          print('log_name not given')
          log_name = config['version']
          
        if not isinstance(log_name, list):
          log_name = [log_name]
        
        directory = config['directory']
        
        if self.flag_save_data:  
            # create model folder
            log_path, self.file_name = return_log_path(config)
            self.model_path = Path(directory, 'logs', *log_name, log_path)
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            print(self.model_path)
            
        
    def write_logs(self, logs, epoch):  
        
        # record logs only with specified frequency
        if (epoch % self.freq_log) == 0 or (epoch == self.n_epochs) == 1:

            # record logs in log file
            for key, item in logs.items():
                # append if list already exists
                try:
                    self.log[key].append(item.numpy().astype(np.float64))
                # create list otherwise
                except KeyError:
                    try:
                        self.log[key] = [item.numpy().astype(np.float64)]
                    # if list is given 
                    except AttributeError:
                        self.log[key] = item
                  
        
        # print logs only with specified frequency
        if (epoch % self.freq_print) == 0:
            
            s = f"{epoch:{self.digits}}/{self.n_epochs}"
            for key in self.keys_print:
                try:
                    s += f" | {key}: {logs[key]:2.2e}"
                except:
                    pass
            print(s) 
            
    def save_logs(self):
        
        # save log file 
        if self.flag_save_data:            
            log_file = self.model_path.joinpath(self.file_name)
            print("file is saved @: ", self.model_path.joinpath(self.file_name))        
            with open(log_file, "w") as f:
                json.dump(self.log, f, indent=2)
    
            print("*** logs saved ***")
        else: 
            print("*** logs not saved (as specified) ***")
            
