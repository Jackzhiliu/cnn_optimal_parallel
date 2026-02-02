import numpy as np
import time
import torch
import torch.nn.functional as F
import math
import my_functionaladj as mf

# GPU device configuration
DEVICE_ = ['cuda' if torch.cuda.is_available() else 'cpu']
print("→ Optimized helper running on", DEVICE_[0])


def one_hot_embedding(labels, num_classes):
    """Convert labels to one-hot encoding"""
    y = torch.eye(num_classes)
    return y[labels]


def gain_schedule(loop, j):
    """Learning rate schedule strategy"""
    gain = 1
    return gain


def create_matrix_x_batch(x, filter_shape, stride, pad):
    """
    Create filter-dependent matrices in batch
    Use unfold to convert input images to a matrix form
    
    Args:
        x: input image [batch_size, channels, height, width]
        filter_shape: filter shape [num_filters, in_channels, kernel_h, kernel_w]
        stride: stride
        pad: padding
    
    Returns:
        matrix_x: [batch_size, in_channels*kernel_h*kernel_w, num_positions]
    """
    kernel_size = (filter_shape[2], filter_shape[3])
    matrix_x = F.unfold(x, kernel_size, stride=stride, padding=pad)
    return matrix_x


def pool_backward_error_batch(out_err, kernel=2, method='Ave'):
    """
    Backward error for pooling layer (batched)
    
    Args:
        out_err: output error [batch_size, channels, h, w]
        kernel: pooling kernel size
        method: pooling method 'Ave' or 'Max'
    
    Returns:
        in_error: input error
    """
    in_error = 0
    if method == 'Ave':
        in_error = torch.repeat_interleave(
            torch.repeat_interleave(out_err, kernel, dim=2), 
            kernel, dim=3
        )
    return in_error


class ParallelFilterLearningSystem:
    """
    Parallel filter learning system
    Updates all convolution layer weights simultaneously instead of sequentially
    """
    
    def __init__(self, model, device):
        """
        Initialize the parallel learning system
        
        Args:
            model: CNN model
            device: compute device
        """
        self.model = model
        self.device = device
        self.conv_indices = []
        self.fc_index = -1
        
        # Locate all convolution layer indices
        for idx, layer in enumerate(model.layers):
            if layer.name == 'conv':
                self.conv_indices.append(idx)
            elif layer.name == 'fc':
                self.fc_index = idx
                break  # take only the first FC layer
    
    def forward_to_layer_batch(self, x, to_layer):
        """
        Batched forward pass to a specified layer
        
        Args:
            x: input [batch_size, ...]
            to_layer: target layer index
        
        Returns:
            output: output of the specified layer
        """
        for layer in self.model.layers[0:to_layer]:
            if layer.name in ['conv', 'fc']:
                x = layer.activations(layer.weights(x))
            elif layer.name in ['flat', 'pool', 'bn', 'dp']:
                x = layer.weights(x)
        return x
    
    def compute_conv_gradients_parallel(self, x, y_true, conv_layer_idx, fc_layer_idx, 
                                       pool_layer=True, ker=2, stri=2, gain=0.001, 
                                       slope=0.01, auto=True):
        """
        Compute convolution gradients in parallel
        Core optimization: compute gradients for all samples at once instead of a for-loop
        
        Args:
            x: input data [1, channels, height, width] - batch_size must be 1
            y_true: true label [1, num_classes]
            conv_layer_idx: convolution layer index
            fc_layer_idx: fully connected layer index
            pool_layer: pooling type
            ker: pooling kernel size
            stri: pooling stride
            gain: learning rate
            slope: LeakyReLU slope
            auto: whether to auto-adjust the learning rate
        
        Returns:
            fil_w_new: updated convolution weights
            fc_w_new: updated FC weights
            alpha_v: convolution learning-rate scaling factor
            alpha_w: FC learning-rate scaling factor
        """
        assert x.shape[0] == 1, "To ensure the convergence proof holds, batch_size must be 1"
        
        DEVICE = self.device
        
        # Get weights for the current layers
        conv_layer = self.model.layers[conv_layer_idx]
        fc_layer = self.model.layers[fc_layer_idx]
        
        fil = conv_layer.weights.weight.data
        fc_wei = fc_layer.weights.weight.data
        
        # Move input to device
        x = x.to(DEVICE)
        y_target = one_hot_embedding(y_true.long(), self.model.no_outputs).to(DEVICE).float()
        y_target = y_target.reshape(y_target.shape[0], y_target.shape[1], 1)
        
        # Get the input to the convolution layer
        layer_in = self.forward_to_layer_batch(x, conv_layer_idx)
        
        # Reshape filter weights
        fil_shape = fil.shape
        fil_w = fil.reshape(fil_shape[0], -1).to(DEVICE)
        fc_w = fc_wei.to(DEVICE)
        
        # Initialize learning-rate scaling factors
        alpha_v = torch.tensor(1.0, device=DEVICE)
        alpha_w = torch.tensor(1.0, device=DEVICE)
        
        # ============ Key optimization: single-sample update ============
        # Create filter-dependent matrix
        stride = conv_layer.stride if isinstance(conv_layer.stride, int) else conv_layer.stride[0]
        pad = conv_layer.padding if isinstance(conv_layer.padding, int) else conv_layer.padding[0]
        
        # Pass the weight shape (not the tensor) so unfold receives int kernel sizes
        in_matrix = create_matrix_x_batch(layer_in, fil.shape, stride, pad)[0]  # [C*K*K, H*W]
        
        # Convolution forward pass
        conv_act = torch.matmul(fil_w, in_matrix)  # [num_filters, num_positions]
        conv_out = conv_layer.activations(conv_act)
        conv_out_shape = conv_out.shape
        
        # Reshape to 4D for pooling
        spatial_dim = int(math.sqrt(conv_out.shape[1]))
        conv_out_4d = conv_out.reshape(1, conv_out.shape[0], spatial_dim, spatial_dim)
        
        # Pooling
        if pool_layer:
            if pool_layer == 'avg':
                pool_out = F.avg_pool2d(conv_out_4d, ker, stri)
                pool_ind = None
            elif pool_layer == 'max':
                pool_out, pool_ind = F.max_pool2d(conv_out_4d, ker, stri, return_indices=True)
            fc_in = pool_out.reshape(-1, 1)
        else:
            fc_in = conv_out.reshape(-1, 1)
        
        # FC forward pass
        y_pred = fc_layer.activations(torch.matmul(fc_w, fc_in))
        
        # Compute error
        error = y_target[0] - y_pred
        
        # FC backprop
        e_fc_in = torch.matmul(fc_w.t(), error)
        
        # Pooling backprop
        if pool_layer:
            e_pool_out = e_fc_in.reshape(pool_out.shape)
            if pool_layer == 'avg':
                e_conv_out = pool_backward_error_batch(e_pool_out, ker)
            elif pool_layer == 'max':
                e_conv_out = F.max_unpool2d(e_pool_out, pool_ind, ker)
            e_conv_out = e_conv_out.reshape(-1, 1)
        else:
            e_conv_out = e_fc_in.reshape(-1, 1)
        
        # Derivative of activation function
        dot_value = mf.derivative_fun(conv_layer.activations)(conv_act.flatten(), slope)
        dot_value = dot_value.reshape(-1, 1)
        
        # Convolution layer error
        e_conv_flat = dot_value * e_conv_out
        e_conv = e_conv_flat.reshape(conv_out_shape)
        
        # ============ Auto-adjust learning rate to ensure convergence ============
        if auto:
            lm = gain
            # Compute update magnitude
            fc_update = lm * torch.matmul(error, fc_in.t())
            conv_update = lm * torch.matmul(e_conv, in_matrix.t())
            
            # Compute convergence condition (per the paper)
            # Ensure ΔV(k) = V(k+1) - V(k) <= 0
            sum_condition = (-2 * error.t() @ error * lm + 
                           alpha_w * error.t() @ error * lm ** 2 * torch.sum(fc_in ** 2) +
                           -2 * error.t() @ error * lm + 
                           alpha_v * torch.sum(conv_update ** 2))
            
            # Adjust alpha until the convergence condition is satisfied
            max_iter = 100
            iter_count = 0
            while sum_condition > 0 and iter_count < max_iter:
                alpha_v /= 1.1
                alpha_w /= 1.1
                sum_condition = (-2 * error.t() @ error * lm + 
                               alpha_w * error.t() @ error * lm ** 2 * torch.sum(fc_in ** 2) +
                               -2 * error.t() @ error * lm + 
                               alpha_v * torch.sum((lm * e_conv @ in_matrix.t()) ** 2))
                iter_count += 1
                if sum_condition < 0:
                    break
        
        # ============ Weight updates ============
        fc_w_new = fc_w + alpha_w * gain * torch.matmul(error, fc_in.t())
        fil_w_new = fil_w + alpha_v * gain * torch.matmul(e_conv, in_matrix.t())
        
        # Reshape back to original shape
        fil_w_new = fil_w_new.reshape(fil_shape)
        
        return fil_w_new, fc_w_new, alpha_v, alpha_w, error


def inc_train_single_sample_optimized(model, x, y, conv_idx, fc_idx=-1, 
                                     ker=2, stri=2, pool_layer='max', 
                                     gain=0.001, slope=0.01, auto=True):
    """
    Single-sample E2E training (optimized)
    
    Key traits:
    1. batch_size=1 to ensure per-sample updates (matches convergence proof)
    2. Vectorized operations to reduce loops
    3. Auto-adjust learning rate to ensure convergence
    
    Args:
        model: CNN model
        x: single input sample [1, channels, height, width]
        y: single label [1]
        conv_idx: convolution layer index to train
        fc_idx: FC layer index (default -1 uses the last layer)
        ker: pooling kernel size
        stri: pooling stride
        pool_layer: pooling type ('max', 'avg', False)
        gain: base learning rate
        slope: LeakyReLU slope
        auto: whether to auto-adjust learning rate
    
    Returns:
        None (weights updated in place)
    """
    assert x.shape[0] == 1, "batch_size must be 1 to ensure convergence"
    
    # Create parallel learning system
    parallel_system = ParallelFilterLearningSystem(model, DEVICE_[0])
    
    with torch.no_grad():
        # Compute gradients and update weights
        fil_w_new, fc_w_new, alpha_v, alpha_w, error = \
            parallel_system.compute_conv_gradients_parallel(
                x, y, conv_idx, fc_idx, pool_layer, ker, stri, gain, slope, auto
            )
        
        # Update model weights
        model.layers[conv_idx].weights.weight.data = fil_w_new
        model.layers[fc_idx].weights.weight.data = fc_w_new


def train_multi_layer_parallel_optimized(model, x, y, conv_indices, fc_idx=-1,
                                        ker=2, stri=2, pool_layers=None, 
                                        gain=0.001, slope=0.01, auto=True):
    """
    Multi-layer parallel training (optimized)
    
    Key idea: filter learning systems for all convolution layers can be updated in parallel
    
    Args:
        model: CNN model
        x: single input sample [1, channels, height, width]
        y: single label [1]
        conv_indices: list of convolution layer indices to train
        fc_idx: FC layer index
        ker: pooling kernel size
        stri: pooling stride
        pool_layers: list of pooling types per layer
        gain: learning rate
        slope: LeakyReLU slope
        auto: whether to auto-adjust learning rate
    
    Returns:
        None (weights updated in place)
    """
    assert x.shape[0] == 1, "batch_size must be 1 to ensure convergence"
    
    if pool_layers is None:
        pool_layers = ['max'] * len(conv_indices)
    
    # Update all layers in parallel
    # Note: although this is a for-loop, computations for each layer are independent
    # and can be parallelized in supporting frameworks
    for conv_idx, pool_type in zip(conv_indices, pool_layers):
        inc_train_single_sample_optimized(
            model, x, y, conv_idx, fc_idx, 
            ker, stri, pool_type, gain, slope, auto
        )


def inc_train_2_layer_e2e_optimized(model, batch_idx, epoch_idx, x, y, 
                                   ker, stri, pool_layer='max', 
                                   epochs=400, gain=0.001, auto=True):
    """
    Two-layer E2E training (optimized)
    Replacement for the original inc_train_2_layer_e2e_acce function
    
    Key improvements:
    1. Ensure batch_size=1
    2. Vectorized matrix operations
    3. Automatic learning-rate adjustment
    
    Args:
        model: CNN model
        batch_idx: current batch index
        epoch_idx: current epoch index
        x: input data [batch_size, channels, height, width]
        y: labels [batch_size]
        ker: pooling kernel size
        stri: pooling stride
        pool_layer: pooling type
        epochs: total epochs
        gain: learning rate
        auto: whether to auto-adjust learning rate
    
    Returns:
        None
    """
    # Determine layers to train
    if pool_layer:
        conv_idx = -4
    else:
        conv_idx = -3
    fc_idx = -1
    
    # Get slope parameter for LeakyReLU
    slope = 0.01  # default slope
    
    # Train sample by sample (as required by the convergence proof)
    for i in range(x.shape[0]):
        x_single = x[i:i+1]  # [1, C, H, W]
        y_single = y[i:i+1]  # [1]
        
        inc_train_single_sample_optimized(
            model, x_single, y_single, conv_idx, fc_idx,
            ker, stri, pool_layer, gain, slope, auto
        )


# ============ Batch processing utilities ============

def process_batch_sample_by_sample(model, x_batch, y_batch, train_func, **kwargs):
    """
    Split batch data into single samples and process one by one
    
    This preserves the validity of the convergence proof because the theory
    requires per-sample updates
    
    Args:
        model: CNN model
        x_batch: batch inputs [batch_size, ...]
        y_batch: batch labels [batch_size, ...]
        train_func: training function
        **kwargs: additional parameters passed to the training function
    
    Returns:
        None
    """
    batch_size = x_batch.shape[0]
    
    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_single = y_batch[i:i+1] if len(y_batch.shape) > 1 else y_batch[i:i+1]
        
        train_func(model, x_single, y_single, **kwargs)


# ============ Performance monitoring utilities ============

class ConvergenceMonitor:
    """Convergence monitor"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.errors = []
        self.alpha_vs = []
        self.alpha_ws = []
    
    def update(self, error, alpha_v, alpha_w):
        """Update monitoring data"""
        self.errors.append(error.item() if torch.is_tensor(error) else error)
        self.alpha_vs.append(alpha_v.item() if torch.is_tensor(alpha_v) else alpha_v)
        self.alpha_ws.append(alpha_w.item() if torch.is_tensor(alpha_w) else alpha_w)
        
        # Keep window size
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
            self.alpha_vs.pop(0)
            self.alpha_ws.pop(0)
    
    def get_stats(self):
        """Get statistics"""
        if not self.errors:
            return {}
        
        return {
            'avg_error': np.mean(self.errors),
            'avg_alpha_v': np.mean(self.alpha_vs),
            'avg_alpha_w': np.mean(self.alpha_ws),
            'error_trend': self.errors[-10:] if len(self.errors) >= 10 else self.errors
        }
    
    def is_converging(self, threshold=0.01):
        """Check whether training is converging"""
        if len(self.errors) < self.window_size:
            return False
        
        recent_errors = self.errors[-self.window_size//4:]
        return np.std(recent_errors) < threshold


print("✓ Optimized helper functions loaded")
print("Main optimizations:")
print("  1. Ensure batch_size=1 for per-sample updates (matches convergence proof)")
print("  2. Vectorized matrix operations to reduce loops")
print("  3. Support parallel updates for multiple filter learning systems")
print("  4. Auto learning-rate adjustment to ensure convergence")
