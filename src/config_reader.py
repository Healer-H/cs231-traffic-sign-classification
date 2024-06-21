import yaml
import os

def read_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        if 'None' in config['model']['random_forest']['max_depth']:
            config['model']['random_forest']['max_depth'].remove('None')
            config['model']['random_forest']['max_depth'].append(None)
            
            
    return config

config = read_config()
