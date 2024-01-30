import json
import random
import numpy as np

def get_data(args):
    """
    Args:
        args

    Returns:
        raw json train/test data Tuple
    """
    
    with open(f"{args.raw_train_path}", "r") as json_file:
        raw_train_data = json.load(json_file)
    
    with open(f"{args.raw_test_path}", "r") as json_file:
        raw_test_data = json.load(json_file)
    
    return raw_train_data, raw_test_data

def split_train_validation_data(args, data):
    data_size = len(data)
    validation_size = int(data_size * args.validation_ratio)
    validation_indices = random.sample(range(data_size), validation_size)
    
    train_data = [data[i] for i in range(data_size) if i not in validation_indices]
    validation_data = [data[i] for i in validation_indices]
    
    return train_data, validation_data

def set_global_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)