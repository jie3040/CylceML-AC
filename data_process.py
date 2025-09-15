import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os

class FaultDiagnosisDataset:
    """Dataset class for fault diagnosis with meta-learning support"""
    
    def __init__(self, data_path: str, num_samples_per_class: int = 5, data_length: int = 1024):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the .npz file
            num_samples_per_class: Number of samples to use per fault class
            data_length: Length of each sample (default 512 for FFT, 128 for other)
        """
        self.data_path = data_path
        self.num_samples_per_class = num_samples_per_class
        self.data_length = data_length
        self.num_fault_classes = 9  # 9 fault types
        
        # Load and process data
        self.domain_A_data, self.domain_B_data, self.test_data = self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load data from npz file and process into required format"""
        
        # Load the npz file
        data = np.load(self.data_path)
        
        # Process Domain A (healthy samples) - class 0
        domain_A_train_X = data['domain_A_train_X']
        domain_A_train_Y = data['domain_A_train_Y']
        
        # Ensure labels are 0 for healthy samples
        domain_A_train_Y = np.zeros(len(domain_A_train_X), dtype=np.int64)
        
        # Ensure data length matches requirement
        if domain_A_train_X.shape[1] != self.data_length:
            # Truncate or pad as needed
            if domain_A_train_X.shape[1] > self.data_length:
                domain_A_train_X = domain_A_train_X[:, :self.data_length]
            else:
                padding = np.zeros((domain_A_train_X.shape[0], 
                                   self.data_length - domain_A_train_X.shape[1]))
                domain_A_train_X = np.concatenate([domain_A_train_X, padding], axis=1)
        
        domain_A_data = {
            'X': domain_A_train_X.astype(np.float32),
            'Y': domain_A_train_Y
        }
        
        # Process Domain B (fault samples) - classes 1-9
        domain_B_data = {}
        for i in range(self.num_fault_classes):
            # Load data for each fault type
            X_key = f'domain_B_train_X_{i}'
            Y_key = f'domain_B_train_Y_{i}'
            
            if X_key in data:
                domain_B_X = data[X_key][:self.num_samples_per_class]
                domain_B_Y = data[Y_key][:self.num_samples_per_class]
                
                # Ensure labels are correct (1-9 for fault types)
                domain_B_Y = np.full(len(domain_B_X), i+1, dtype=np.int64)
                
                # Ensure data length matches requirement
                if domain_B_X.shape[1] != self.data_length:
                    #print(domain_B_X.shape[1])
                    if domain_B_X.shape[1] > self.data_length:
                        domain_B_X = domain_B_X[:, :self.data_length]
                    else:
                        padding = np.zeros((domain_B_X.shape[0], 
                                          self.data_length - domain_B_X.shape[1]))
                        domain_B_X = np.concatenate([domain_B_X, padding], axis=1)
                
                domain_B_data[i] = {
                    'X': domain_B_X.astype(np.float32),
                    'Y': domain_B_Y
                }
            else:
                print(f"Warning: {X_key} not found in data file")
                # Create dummy data if not found
                domain_B_data[i] = {
                    'X': np.random.randn(self.num_samples_per_class, self.data_length).astype(np.float32),
                    'Y': np.full(self.num_samples_per_class, i+1, dtype=np.int64)
                }
        
        # Process test data
        test_X = data['test_X'] if 'test_X' in data else None
        test_Y = data['test_Y'] if 'test_Y' in data else None
        
        if test_X is not None and test_Y is not None:
            # Ensure data length matches requirement
            if test_X.shape[1] != self.data_length:
                #print('1')
                if test_X.shape[1] > self.data_length:
                    test_X = test_X[:, :self.data_length]
                else:
                    padding = np.zeros((test_X.shape[0], 
                                      self.data_length - test_X.shape[1]))
                    test_X = np.concatenate([test_X, padding], axis=1)
            
            test_data = {
                'X': test_X.astype(np.float32),
                'Y': test_Y.astype(np.int64)
            }
        else:
            test_data = None
        
        return domain_A_data, domain_B_data, test_data
    
    def get_combined_domain_B_data(self):
        """Get all domain B data combined (for SVM training)"""
        X_list = []
        Y_list = []
        
        for i in range(self.num_fault_classes):
            if i in self.domain_B_data:
                X_list.append(self.domain_B_data[i]['X'])
                Y_list.append(self.domain_B_data[i]['Y'])
        
        if X_list:
            domain_B_combined_X = np.concatenate(X_list, axis=0)
            domain_B_combined_Y = np.concatenate(Y_list, axis=0)
            return domain_B_combined_X, domain_B_combined_Y
        else:
            return None, None
    
    def prepare_meta_learning_tasks(self, support_size: int = 2, query_size: int = 3):
        """
        Prepare tasks for meta-learning
        
        Args:
            support_size: Number of samples in support set
            query_size: Number of samples in query set
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        # Get indices for domain A samples
        num_domain_A_samples = len(self.domain_A_data['X'])
        
        for task_id in range(self.num_fault_classes):
            if task_id not in self.domain_B_data:
                continue
                
            # Domain B samples for this task
            domain_B_X = self.domain_B_data[task_id]['X']
            domain_B_Y = self.domain_B_data[task_id]['Y']
            
            # Ensure we have enough samples
            if len(domain_B_X) < support_size + query_size:
                print(f"Warning: Task {task_id} has insufficient samples. Skipping.")
                continue
            
            # Split domain B data into support and query
            support_idx = np.arange(support_size)
            query_idx = np.arange(support_size, support_size + query_size)
            
            # Random sample from domain A
            domain_A_idx = np.random.choice(num_domain_A_samples, 
                                          support_size + query_size, 
                                          replace=False)
            
            task_data = {
                'task_id': task_id,
                # For auxiliary classifier training
                'support': (
                    torch.FloatTensor(domain_B_X[support_idx]),
                    torch.LongTensor(domain_B_Y[support_idx])
                ),
                'query': (
                    torch.FloatTensor(domain_B_X[query_idx]),
                    torch.LongTensor(domain_B_Y[query_idx])
                ),
                # For generator/discriminator training
                'support_A': (
                    torch.FloatTensor(self.domain_A_data['X'][domain_A_idx[:support_size]]),
                    torch.LongTensor(self.domain_A_data['Y'][domain_A_idx[:support_size]])
                ),
                'query_A': (
                    torch.FloatTensor(self.domain_A_data['X'][domain_A_idx[support_size:]]),
                    torch.LongTensor(self.domain_A_data['Y'][domain_A_idx[support_size:]])
                ),
                'support_B': (
                    torch.FloatTensor(domain_B_X[support_idx]),
                    torch.LongTensor(domain_B_Y[support_idx])
                ),
                'query_B': (
                    torch.FloatTensor(domain_B_X[query_idx]),
                    torch.LongTensor(domain_B_Y[query_idx])
                )
            }
            tasks.append(task_data)
        
        return tasks
    
    def get_test_data(self):
        """Get test data if available"""
        if self.test_data is not None:
            return (torch.FloatTensor(self.test_data['X']),
                   torch.LongTensor(self.test_data['Y']))
        else:
            return None, None


def load_fault_diagnosis_data(data_path: str, 
                             num_samples_per_class: int = 5,
                             data_length: int = 512,
                             support_size: int = 2,
                             query_size: int = 3):
    """
    Convenience function to load data and prepare for training
    
    Args:
        data_path: Path to the .npz file
        num_samples_per_class: Number of samples per fault class to use
        data_length: Length of each sample
        support_size: Size of support set for meta-learning
        query_size: Size of query set for meta-learning
        
    Returns:
        dataset: FaultDiagnosisDataset object
        tasks: List of prepared meta-learning tasks
    """
    dataset = FaultDiagnosisDataset(data_path, num_samples_per_class, data_length)
    tasks = dataset.prepare_meta_learning_tasks(support_size, query_size)
    
    return dataset, tasks


def create_data_loader_generator(dataset: FaultDiagnosisDataset,
                                support_size: int = 2,
                                query_size: int = 3,
                                shuffle_tasks: bool = True):
    """
    Create a generator that yields batches of tasks
    
    Args:
        dataset: FaultDiagnosisDataset object
        support_size: Size of support set
        query_size: Size of query set
        shuffle_tasks: Whether to shuffle tasks each epoch
        
    Yields:
        List of task dictionaries
    """
    while True:
        tasks = dataset.prepare_meta_learning_tasks(support_size, query_size)
        if shuffle_tasks:
            np.random.shuffle(tasks)
        yield tasks


# Example usage
if __name__ == '__main__':
    # Path to your data
    data_path = '/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_2/github/dataset_fft_for_cyclegan_case1_512.npz'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating dummy data for testing...")
        
        # Create dummy data for testing
        dummy_data = {}
        
        # Domain A (healthy)
        dummy_data['domain_A_train_X'] = np.random.randn(1000, 512).astype(np.float32)
        dummy_data['domain_A_train_Y'] = np.zeros(1000, dtype=np.int64)
        
        # Domain B (faults)
        for i in range(9):
            dummy_data[f'domain_B_train_X_{i}'] = np.random.randn(10, 512).astype(np.float32)
            dummy_data[f'domain_B_train_Y_{i}'] = np.full(10, i+1, dtype=np.int64)
        
        # Test data
        dummy_data['test_X'] = np.random.randn(500, 512).astype(np.float32)
        dummy_data['test_Y'] = np.random.randint(0, 10, 500).astype(np.int64)
        
        # Save dummy data
        np.savez('dummy_data.npz', **dummy_data)
        data_path = 'dummy_data.npz'
    
    # Load dataset
    dataset, tasks = load_fault_diagnosis_data(
        data_path=data_path,
        num_samples_per_class=5,
        data_length=512,
        support_size=2,
        query_size=3
    )
    
    print(f"Loaded dataset with:")
    print(f"  Domain A samples: {len(dataset.domain_A_data['X'])}")
    print(f"  Domain B classes: {len(dataset.domain_B_data)}")
    print(f"  Number of tasks: {len(tasks)}")
    
    # Get combined domain B data for SVM
    domain_B_X, domain_B_Y = dataset.get_combined_domain_B_data()
    if domain_B_X is not None:
        print(f"  Combined Domain B samples: {len(domain_B_X)}")
    
    # Get test data
    test_X, test_Y = dataset.get_test_data()
    print(test_Y)
    if test_X is not None:
        print(f"  Test samples: {len(test_X)}")
    
    # Example of task structure
    if tasks:
        print(f"\nExample task structure:")
        print(f"  Support A shape: {tasks[0]['support_A'][0].shape}")
        print(f"  Support B shape: {tasks[0]['support_B'][0].shape}")
        print(f"  Query A shape: {tasks[0]['query_A'][0].shape}")
        print(f"  Query B shape: {tasks[0]['query_B'][0].shape}")