import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SVMEvaluator:
    """SVM-based evaluator for generated samples"""
    
    def __init__(self, data_length: int = 512):
        self.data_length = data_length
        self.scaler = None
        self.classifier = None
        
    def scalar_stand(self, train_X: np.ndarray, test_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize training and test data"""
        self.scaler = preprocessing.StandardScaler().fit(train_X)
        train_X_scaled = self.scaler.transform(train_X)
        test_X_scaled = self.scaler.transform(test_X)
        return train_X_scaled, test_X_scaled
    
    def generate_augmented_samples(self, 
                                  generator: nn.Module,
                                  domain_A_X: np.ndarray,
                                  add_quantity: int = 995,
                                  num_classes: int = 9,
                                  device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples using the generator
        
        Args:
            generator: Trained generator model (g_AB)
            domain_A_X: Healthy samples
            add_quantity: Number of samples to generate per class
            num_classes: Number of fault classes to generate
            device: Device to run generation on
            
        Returns:
            Generated samples and their labels
        """
        generator.eval()
        generated_samples = []
        labels = []
        
        with torch.no_grad():
            for class_id in range(1, num_classes + 1):  # Classes 1-9 for faults
                # Create labels for this class
                class_labels = torch.full((add_quantity,), class_id, dtype=torch.long).to(device)
                
                # Randomly select domain A samples
                indices = np.random.choice(len(domain_A_X), add_quantity, replace=True)
                selected_samples = torch.FloatTensor(domain_A_X[indices]).to(device)
                
                # Generate samples
                generated = generator(selected_samples, class_labels)
                
                # Move to CPU and store
                generated_samples.append(generated.cpu().numpy())
                labels.append(np.full((add_quantity, 1), class_id))
        
        # Combine all generated samples
        generated_samples = np.vstack(generated_samples)
        labels = np.vstack(labels)
        
        return generated_samples, labels
    
    def evaluate_with_svm(self,
                         generator: nn.Module,
                         domain_A_X: np.ndarray,
                         domain_A_Y: np.ndarray,
                         domain_B_X: np.ndarray,
                         domain_B_Y: np.ndarray,
                         test_X: np.ndarray,
                         test_Y: np.ndarray,
                         add_quantity: int = 995,
                         device: str = 'cuda',
                         verbose: bool = False) -> dict:
        """
        Evaluate the generator by training SVM on augmented data
        
        Args:
            generator: Trained generator model
            domain_A_X, domain_A_Y: Healthy samples and labels
            domain_B_X, domain_B_Y: Fault samples and labels
            test_X, test_Y: Test samples and labels
            add_quantity: Number of synthetic samples per class
            device: Device for generation
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate synthetic samples
        generated_X, generated_Y = self.generate_augmented_samples(
            generator, domain_A_X, add_quantity, num_classes=9, device=device
        )
        
        # Combine all training data
        # Original fault samples + Generated samples
        augmented_fault_X = np.vstack([domain_B_X, generated_X])
        augmented_fault_Y = np.vstack([domain_B_Y.reshape(-1, 1), generated_Y])
        
        # Add healthy samples
        total_train_X = np.vstack([augmented_fault_X, domain_A_X])
        total_train_Y = np.vstack([augmented_fault_Y, domain_A_Y.reshape(-1, 1)])
        
        # Flatten labels
        total_train_Y = total_train_Y.ravel()
        test_Y = test_Y.ravel()
        
        # Standardize data
        train_X_scaled, test_X_scaled = self.scalar_stand(total_train_X, test_X)
        
        # Train SVM classifier
        self.classifier = SVC(C=100, gamma=0.01, kernel='rbf', random_state=0)
        self.classifier.fit(train_X_scaled, total_train_Y)
        
        # Predict
        Y_pred = self.classifier.predict(test_X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(test_Y, Y_pred)
        cm = confusion_matrix(test_Y, Y_pred)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': Y_pred,
            'true_labels': test_Y,
            'num_generated': len(generated_X),
            'num_original': len(domain_B_X),
            'num_total_train': len(total_train_X)
        }
        
        if verbose:
            print(f"\n=== SVM Evaluation Results ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Training samples: {len(total_train_X)} "
                  f"(Original: {len(domain_B_X) + len(domain_A_X)}, "
                  f"Generated: {len(generated_X)})")
            print(f"Test samples: {len(test_X)}")
            print("\nClassification Report:")
            print(classification_report(test_Y, Y_pred))
        
        return results


def integrate_evaluation_into_training(model, dataset, epoch, device='cuda', verbose=True):
    """
    Function to integrate evaluation into the main training loop
    
    Args:
        model: CycleML_AC model instance
        dataset: FaultDiagnosisDataset instance
        epoch: Current training epoch
        device: Device to use
        verbose: Whether to print results
        
    Returns:
        Accuracy score
    """
    # Initialize evaluator
    evaluator = SVMEvaluator(data_length=dataset.data_length)
    
    # Get data from dataset
    domain_A_X = dataset.domain_A_data['X']
    domain_A_Y = dataset.domain_A_data['Y']
    
    # Combine domain B data
    domain_B_X, domain_B_Y = dataset.get_combined_domain_B_data()
    
    # Get test data
    test_X, test_Y = dataset.get_test_data()
    
    if test_X is None:
        print("Warning: No test data available for evaluation")
        return 0.0
    
    # Convert test tensors to numpy if needed
    if isinstance(test_X, torch.Tensor):
        test_X = test_X.cpu().numpy()
        test_Y = test_Y.cpu().numpy()
    
    # Evaluate
    results = evaluator.evaluate_with_svm(
        generator=model.g_AB,
        domain_A_X=domain_A_X,
        domain_A_Y=domain_A_Y,
        domain_B_X=domain_B_X,
        domain_B_Y=domain_B_Y,
        test_X=test_X,
        test_Y=test_Y,
        add_quantity=995,  # Number of samples to generate per class
        device=device,
        verbose=verbose
    )
    
    if verbose:
        print(f"[Epoch {epoch}] SVM Evaluation Accuracy: {results['accuracy']:.4f}")
    
    return results['accuracy']


# Modified train method for CycleML_AC class
def train_with_evaluation(self, epochs, tasks_data_loader, dataset, num_tasks=9, 
                         eval_frequency=10, save_frequency=100):
    """
    Enhanced training loop with periodic evaluation
    
    Args:
        epochs: Number of training epochs
        tasks_data_loader: Generator for task data
        dataset: FaultDiagnosisDataset instance for evaluation
        num_tasks: Number of meta-learning tasks
        eval_frequency: How often to evaluate (in epochs)
        save_frequency: How often to save models
    """
    import datetime
    start_time = datetime.datetime.now()
    
    # Track evaluation results
    eval_history = {
        'epochs': [],
        'accuracies': [],
        'best_accuracy': 0.0,
        'best_epoch': 0
    }
    
    for epoch in range(epochs):
        # Get batch of tasks
        tasks_data = next(tasks_data_loader)
        
        # Step 1: Meta-train auxiliary classifier
        print(f"Epoch {epoch}: Training auxiliary classifier...")
        self.meta_train_auxiliary_classifier(tasks_data)
        
        # Step 2: Meta-train discriminators
        print(f"Epoch {epoch}: Training discriminators...")
        self.meta_train_discriminator(
            self.d_1, self.g_BA, self.d1_optimizer_inner, self.d1_optimizer_outer,
            tasks_data, source_domain='B'
        )
        self.meta_train_discriminator(
            self.d_2, self.g_AB, self.d2_optimizer_inner, self.d2_optimizer_outer,
            tasks_data, source_domain='A'
        )
        
        # Step 3: Meta-train generators
        print(f"Epoch {epoch}: Training generators...")
        self.meta_train_generators(tasks_data)
        
        elapsed_time = datetime.datetime.now() - start_time
        
        # Periodic evaluation with SVM
        if epoch % eval_frequency == 0 and epoch > 0:
            print(f"\n{'='*50}")
            print(f"Running SVM evaluation at epoch {epoch}...")
            
            accuracy = integrate_evaluation_into_training(
                model=self,
                dataset=dataset,
                epoch=epoch,
                device=self.device,
                verbose=True
            )
            
            # Track results
            eval_history['epochs'].append(epoch)
            eval_history['accuracies'].append(accuracy)
            
            # Update best results
            if accuracy > eval_history['best_accuracy']:
                eval_history['best_accuracy'] = accuracy
                eval_history['best_epoch'] = epoch
                # Save best models
                self.save_models(f'best_epoch_{epoch}')
                print(f"New best accuracy: {accuracy:.4f} at epoch {epoch}")
            
            print(f"{'='*50}\n")
        
        # Regular progress reporting
        if epoch % 10 == 0:
            with torch.no_grad():
                # Get a sample task for monitoring
                eval_task = tasks_data[0]
                samples_A, labels_A = eval_task['query_A']
                samples_B, labels_B = eval_task['query_B']
                
                samples_A = samples_A.to(self.device)
                labels_A = labels_A.to(self.device).long()
                samples_B = samples_B.to(self.device)
                labels_B = labels_B.to(self.device).long()
                
                # Generate fake samples
                fake_B = self.g_AB(samples_A, labels_B)
                fake_A = self.g_BA(samples_B, labels_A)
                
                # Discriminator predictions
                d1_real = self.d_1(samples_A)
                d1_fake = self.d_1(fake_A)
                d2_real = self.d_2(samples_B)
                d2_fake = self.d_2(fake_B)
                
                # AC predictions
                ac_pred_A = self.ac(samples_A)
                ac_pred_B = self.ac(samples_B)
                ac_acc_A = (ac_pred_A.argmax(1) == labels_A).float().mean()
                ac_acc_B = (ac_pred_B.argmax(1) == labels_B).float().mean()
                
                print(f"[Epoch {epoch}] D1_real: {d1_real.mean():.4f}, "
                      f"D1_fake: {d1_fake.mean():.4f}, "
                      f"D2_real: {d2_real.mean():.4f}, "
                      f"D2_fake: {d2_fake.mean():.4f}, "
                      f"AC_acc_A: {ac_acc_A:.4f}, AC_acc_B: {ac_acc_B:.4f}, "
                      f"Time: {elapsed_time}")
        
        # Save models periodically
        if epoch % save_frequency == 0 and epoch > 0:
            self.save_models(epoch)
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("Final evaluation...")
    final_accuracy = integrate_evaluation_into_training(
        model=self,
        dataset=dataset,
        epoch=epochs,
        device=self.device,
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Best accuracy: {eval_history['best_accuracy']:.4f} at epoch {eval_history['best_epoch']}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"{'='*50}")
    
    return eval_history