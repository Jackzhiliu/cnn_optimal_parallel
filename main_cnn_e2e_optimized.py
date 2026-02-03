import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import numpy as np

# Import custom modules
from helper_optimized import (
    inc_train_2_layer_e2e_optimized,
    train_multi_layer_parallel_optimized,
    process_batch_sample_by_sample,
    ConvergenceMonitor,
    DEVICE_
)
import my_module1 as mm
from my_module1 import m0, m1, m2, m3, m4, m5, m6, m7

# ============ Global Configuration ============
# Train loader batch size (larger = fewer iterations = faster)
BATCH_SIZE = 64
# DataLoader workers for faster I/O (tune for your environment)
NUM_WORKERS = 4
LEARNING_RATE = 0.005  # Adjustable
NUM_EPOCHS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Running on device:", DEVICE)

mm.DEVICE = DEVICE
DEVICE_[0] = DEVICE

# Enable cuDNN benchmarking (speeds up fixed input sizes)
torch.backends.cudnn.benchmark = True
# Enable TF32 on A100/AMPERE for matmul/conv
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Set save path
FOLDER_e2e = mm.FOLDER_e2e
if not os.path.exists(FOLDER_e2e):
    os.makedirs(FOLDER_e2e)

# ============ Data Loading ============

def get_datasets():
    """
    Get training and validation datasets
    
    Optional datasets:
    - SVHN (current)
    - CIFAR10
    - Fashion-MNIST
    - KMNIST
    """
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # SVHN dataset
    train_set = datasets.SVHN(
        root='./data', 
        split='train', 
        transform=train_transform,
        download=True
    )
    
    val_dataset = datasets.SVHN(
        root='./data', 
        split='train', 
        transform=val_transform,
        download=True
    )
    
    # Split train and validation sets (70% train, 30% val)
    torch.manual_seed(42)
    train_size = int(0.7 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, _ = random_split(train_set, [train_size, val_size])
    
    print(f'Train set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, num_models=8):
    """
    Create dataloaders for training
    
    Note: All models share the SAME train_loader to ensure they process
    the same samples in each iteration (required by the algorithm).
    """
    pin_mem = DEVICE.type == 'cuda'
    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        persistent_workers=NUM_WORKERS > 0
    )
    
    # Single shared train loader for all models
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        **loader_kwargs
    )
    
    # Separate val loaders (can be shared, but keeping separate for flexibility)
    val_loaders = []
    for i in range(num_models):
        val_loader = DataLoader(
            val_dataset, 
            batch_size=64,  # Can use a larger batch for evaluation
            shuffle=False,
            **loader_kwargs
        )
        val_loaders.append(val_loader)
    
    return train_loader, val_loaders


# ============ Model Management ============

def get_set_weight(model_src, model_dst, layer_indices):
    """
    Copy weights from the source model to the target model
    Used for progressive training (adding depth layer by layer)
    """
    for idx in layer_indices:
        w = model_src.layers[idx].weights.weight.data
        model_dst.layers[idx].weights.weight.data = w.clone()


def save_best_model(model_num, epoch, acc_lst, best_accu, model):
    """Save the best-performing model"""
    if acc_lst[1] > best_accu:
        name = os.path.join(FOLDER_e2e, f'model{model_num}_best.pkl')
        torch.save(model, name)
        best_accu = acc_lst[1]
        model.save_best_accu(model_num, epoch, acc_lst)
        print(f'✓ Model {model_num} best accuracy: {acc_lst[1]:.2f}% (epoch {epoch})')
    
    # Save the latest model
    name_last = os.path.join(FOLDER_e2e, f'model{model_num}_last.pkl')
    torch.save(model, name_last)
    
    return best_accu


# ============ Optimized Training Functions ============

def train_model_optimized(model, model_num, train_loader, val_loader, 
                         conv_layer_config, learning_rate, epochs):
    """
    Optimized model training function
    
    Args:
        model: CNN model
        model_num: Model index
        train_loader: Training dataloader
        val_loader: Validation dataloader
        conv_layer_config: Convolution layer config (index, pool_type, ker, stri)
        learning_rate: Learning rate
        epochs: Number of training epochs
    
    Returns:
        model: Trained model
        best_accu: Best accuracy
    """
    best_accu = 0
    conv_idx, pool_type, ker, stri = conv_layer_config
    
    # Create convergence monitor
    monitor = ConvergenceMonitor(window_size=100)
    
    print(f'\n{"="*60}')
    print(f'Start training model {model_num}')
    print(f'Conv layer index: {conv_idx}, Pooling: {pool_type}, Learning rate: {learning_rate}')
    print(f'{"="*60}\n')
    
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Train sample by sample (function splits batch internally)
        for batch_idx, (x, y) in enumerate(train_loader):
            inc_train_2_layer_e2e_optimized(
                model=model,
                batch_idx=batch_idx,
                epoch_idx=epoch,
                x=x,
                y=y,
                ker=ker,
                stri=stri,
                pool_layer=pool_type,
                epochs=epochs,
                gain=learning_rate,
                auto=True  # Automatically adjust learning rate to ensure convergence
            )
        
        # Evaluate once per epoch
        if epoch % 1 == 0:
            acc_lst = model.evaluate_both(model_num, train_loader, val_loader)
            elapsed_time = time.time() - t0
            
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train acc: {acc_lst[0]:.2f}% | '
                  f'Val acc: {acc_lst[1]:.2f}% | '
                  f'Time: {elapsed_time:.1f}s')
            
            # Save the best model
            best_accu = save_best_model(model_num, epoch, acc_lst, best_accu, model)
    
    print(f'\n✓ Model {model_num} training complete, best val acc: {best_accu:.2f}%')
    print(f'Total time: {time.time() - t0:.1f}s\n')
    
    return model, best_accu


def train_sequential_models_optimized(models, train_loader, val_loaders, 
                                      configs, learning_rate, epochs):
    """
    Train multiple models with correct sequential dependency (progressively increasing depth)
    
    IMPORTANT: The 8 models are NOT independent! They must:
    1. Process the SAME sample at each iteration
    2. Copy weights BETWEEN each model's training step (not just at epoch start)
    
    This preserves the algorithm's convergence guarantees while keeping I/O and compute optimizations.
    
    Args:
        models: List of models
        train_loader: Single training dataloader (all models share the same data)
        val_loaders: List of validation dataloaders
        configs: Config list [(conv_idx, pool_type, ker, stri, copy_from_indices)]
        learning_rate: Learning rate
        epochs: Number of training epochs
    
    Returns:
        models: Trained models
        best_accus: List of best accuracies
    """
    best_accus = [0] * len(models)
    num_models = len(models)
    t0 = time.time()
    
    for epoch in range(epochs):
        print(f'\n{"#"*60}')
        print(f'Global Epoch {epoch+1}/{epochs}')
        print(f'{"#"*60}\n')
        
        t_epoch = time.time()
        t_train = time.time()
        
        # Create data iterator for this epoch (all models share the SAME data)
        data_iter = iter(train_loader)
        num_batches = len(train_loader)
        
        # Process each batch
        for batch_idx in range(num_batches):
            try:
                x, y = next(data_iter)
            except StopIteration:
                print('StopIteration')
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            
            # Train all models SEQUENTIALLY on the SAME sample
            # with weight copying between each model
            for i, (model, config) in enumerate(zip(models, configs)):
                conv_idx, pool_type, ker, stri, copy_indices = config
                
                # Copy weights from previous model BEFORE training this model
                if i > 0 and copy_indices:
                    get_set_weight(models[i-1], model, copy_indices)
                
                # Train this model on the current sample
                inc_train_2_layer_e2e_optimized(
                    model=model,
                    batch_idx=batch_idx,
                    epoch_idx=epoch,
                    x=x,
                    y=y,
                    ker=ker,
                    stri=stri,
                    pool_layer=pool_type,
                    epochs=epochs,
                    gain=learning_rate,
                    auto=True
                )
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                elapsed = time.time() - t_train
                pct = (batch_idx + 1) / num_batches * 100
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                print(f'  Batch {batch_idx+1}/{num_batches} ({pct:.1f}%) - {elapsed:.1f}s - {batches_per_sec:.1f} batches/s')
        
        train_time = time.time() - t_train
        print(f'\nTraining time: {train_time:.1f}s')
        print(f'Total elapsed: {time.time() - t0:.1f}s')
        
        # Evaluate all models
        t_eval = time.time()
        print(f'\nEpoch {epoch+1} evaluation results:')
        for i, (model, val_loader) in enumerate(zip(models, val_loaders)):
            acc_lst = model.evaluate_both(i, train_loader, val_loader)
            print(f'  Model {i}: Train={acc_lst[0]:.2f}%, Val={acc_lst[1]:.2f}%')
            
            best_accus[i] = save_best_model(i, epoch, acc_lst, best_accus[i], model)
        eval_time = time.time() - t_eval
        
        print(f'Train: {train_time:.1f}s | Eval: {eval_time:.1f}s | Epoch total: {time.time() - t_epoch:.1f}s')
    
    return models, best_accus


# ============ Main Training Flow ============

def main():
    """Main training function"""
    print("="*60)
    print("Optimized E2E-CNN Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Optimizations: TF32, AMP, cudnn.benchmark, object reuse")
    print(f"Per-sample update: ✓ maintained")
    print("="*60 + "\n")
    
    # Load data
    train_dataset, val_dataset = get_datasets()
    train_loader, val_loaders = create_dataloaders(train_dataset, val_dataset, num_models=8)
    
    # Create models (progressively increasing depth)
    # Note: All models use the same train_loader for initialization
    N_CLASSES = 10
    
    print("Creating models...")
    model0 = m0(N_CLASSES, train_loader).float()
    model1 = m1(N_CLASSES, train_loader).float()
    model2 = m2(N_CLASSES, train_loader).float()
    model3 = m3(N_CLASSES, train_loader).float()
    model4 = m4(N_CLASSES, train_loader).float()
    model5 = m5(N_CLASSES, train_loader).float()
    model6 = m6(N_CLASSES, train_loader).float()
    model7 = m7(N_CLASSES, train_loader).float()
    
    models = [model0, model1, model2, model3, model4, model5, model6, model7]
    
    # Configure each model
    # Format: (conv_layer_idx, pool_type, kernel, stride, copy_from_indices)
    configs = [
        (-4, 'max', 2, 2, []),           # model0
        (-4, 'max', 2, 2, [0]),          # model1: copy layer 0 from model0
        (-3, False, 2, 2, [0, 2]),       # model2
        (-4, 'max', 2, 2, [0, 2, 4]),    # model3
        (-3, False, 2, 2, [0, 2, 4, 5]), # model4
        (-4, 'max', 2, 2, [0, 2, 4, 5, 7]),      # model5
        (-3, False, 2, 2, [0, 2, 4, 5, 7, 8]),   # model6
        (-4, 'max', 2, 2, [0, 2, 4, 5, 7, 8, 10]) # model7
    ]
    
    # Start training with optimized sequential execution
    print("\nStart training all models (optimized sequential)...\n")
    print("Optimizations applied:")
    print("  - TF32 / AMP for faster matmul/conv")
    print("  - cudnn.benchmark for fixed input sizes")
    print("  - ParallelFilterLearningSystem object reuse")
    print(f"  - Batch size {BATCH_SIZE} to reduce iterations\n")
    
    # Optimized sequential training: correct algorithm with reduced overhead
    models, best_accus = train_sequential_models_optimized(
        models, train_loader, val_loaders, configs, LEARNING_RATE, NUM_EPOCHS
    )
    
    # Print final results
    print("\n" + "="*60)
    print("Training complete! Final results:")
    print("="*60)
    for i, best_accu in enumerate(best_accus):
        print(f"Model {i}: Best validation accuracy = {best_accu:.2f}%")
    print("="*60 + "\n")
    
    return models, best_accus


if __name__ == "__main__":
    try:
        models, best_accus = main()
        print("\n✓ All training tasks completed!")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
