import json
import random
import numpy as np
import torch

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


def check_gpu():

    num_cuda_devices = torch.cuda.device_count()

    if num_cuda_devices > 0:
        for i in range(num_cuda_devices):
            device = torch.cuda.get_device_name(i)
            memory_info = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB로 변환
            print(f"GPU {i + 1}: {device} - Available: {memory_info:.2f} GB")
    else:
        print("No Gpu Available")