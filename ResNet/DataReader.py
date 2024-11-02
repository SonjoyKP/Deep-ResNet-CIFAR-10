import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    x_train_list, y_train_list = [], []
    
    # Loop through data batches (1 to 5 are training batches)
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        with open(file_path, 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
            x_train_list.append(batch[b'data'])
            y_train_list.extend(batch[b'labels'])
    
    # Convert lists to numpy arrays
    x_train = np.concatenate(x_train_list, axis=0).astype(np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    
    # Load test data
    test_file_path = os.path.join(data_dir, 'test_batch')
    with open(test_file_path, 'rb') as file:
        test_batch = pickle.load(file, encoding='bytes')
        x_test = test_batch[b'data'].astype(np.float32)
        y_test = np.array(test_batch[b'labels'], dtype=np.int32)
    
    # Normalize the data to [0, 1] range by dividing by 255.0
    x_train /= 255.0
    x_test /= 255.0
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid