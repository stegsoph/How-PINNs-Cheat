import yaml

def load_config(file, config_update=None, verbose=True): 

    with open(file) as f:
        config = yaml.full_load(f)
    
    if config_update:
        config.update(config_update)
        
    if verbose:
        for key, item in config.items():
            print(key,":", item)
        
    return config


def save_config(config, file_path):
    
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    