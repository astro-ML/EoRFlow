import time
import torch
import torch.nn as nn
import numpy as np

# For testing, we create a dummy base class for invertible modules.
class InvertibleModule(nn.Module):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__()
    def output_dims(self, input_dims):
        return input_dims

# Original loop-based version
class SigmoidModule(InvertibleModule):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            # Forward pass: apply inverse sigmoid (logit)
            y = []
            log_jacobian = []
            for xi in x:
                # Clamp to avoid taking log(0) or log(1)
                xi = torch.clamp(xi, min=1e-6, max=1 - 1e-6)
                yi = torch.log(xi / (1 - xi))
                y.append(yi)
                if jac:
                    # Jacobian determinant: derivative of logit is 1/(x(1-x))
                    log_det_jac = - (torch.log(xi) + torch.log(1 - xi)).sum(dim=1)
                    log_jacobian.append(log_det_jac)
            if jac:
                total_log_jacobian = sum(log_jacobian)
                return y, total_log_jacobian
            else:
                return y
        else:
            # Reverse pass: apply sigmoid
            y = []
            log_jacobian = []
            for xi in x:
                yi = torch.sigmoid(xi)
                y.append(yi)
                if jac:
                    # Jacobian: derivative of sigmoid is sigma(x)*(1-sigma(x))
                    log_det_jac = (torch.log(yi) + torch.log(1 - yi)).sum(dim=1)
                    log_jacobian.append(log_det_jac)
            if jac:
                total_log_jacobian = sum(log_jacobian)
                return y, total_log_jacobian
            else:
                return y

    def output_dims(self, input_dims):
        return input_dims

# Vectorized version: processes the entire tensor at once.
class SigmoidModuleVectorized(InvertibleModule):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)

    def forward(self, x, rev=False, jac=True):
        # Assume x is a list with one tensor.
        x_tensor = x[0]
        if not rev:
            x_clamped = torch.clamp(x_tensor, min=1e-6, max=1 - 1e-6)
            y = torch.log(x_clamped / (1 - x_clamped))
            if jac:
                log_det_jac = - (torch.log(x_clamped) + torch.log(1 - x_clamped)).sum(dim=1)
                return [y], log_det_jac
            else:
                return [y]
        else:
            y = torch.sigmoid(x_tensor)
            if jac:
                log_det_jac = (torch.log(y) + torch.log(1 - y)).sum(dim=1)
                return [y], log_det_jac
            else:
                return [y]

    def output_dims(self, input_dims):
        return input_dims

# ---------------------------
# Testing the modules
# ---------------------------
def test_module(module, x, rev_flag, jac_flag):
    start = time.time()
    y_list, log_jac = module.forward([x], rev=rev_flag, jac=jac_flag)
    end = time.time()
    # y_list is a list; take the first element.
    y = y_list[0]
    return y, log_jac, end - start

def main():
    # Create dummy input in (0,1)
    batch_size = 100  # Increase batch for timing differences
    dims = 50  # number of features/dimensions
    x = torch.rand(batch_size, dims)  # uniformly distributed between 0 and 1

    print("Original Input (x):")
    print(x)

    # Instantiate both modules.
    module_loop = SigmoidModule(dims)
    module_vect = SigmoidModuleVectorized(dims)

    # Run forward pass (logit) and measure time
    y_loop, log_jac_loop, time_loop_forward = test_module(module_loop, x, rev_flag=False, jac_flag=True)
    y_vect, log_jac_vect, time_vect_forward = test_module(module_vect, x, rev_flag=False, jac_flag=True)

    print("\nForward Pass (Logit):")
    print("Loop-based output:")
    print(y_loop)
    print("Vectorized output:")
    print(y_vect)
    print("Time (loop-based): {:.6f} seconds".format(time_loop_forward))
    print("Time (vectorized): {:.6f} seconds".format(time_vect_forward))

    # Run reverse pass (sigmoid) and measure time
    x_hat_loop, log_jac_loop_rev, time_loop_reverse = test_module(module_loop, y_loop, rev_flag=True, jac_flag=True)
    x_hat_vect, log_jac_vect_rev, time_vect_reverse = test_module(module_vect, y_vect, rev_flag=True, jac_flag=True)

    print("\nReverse Pass (Sigmoid):")
    print("Loop-based reconstruction:")
    print(x_hat_loop)
    print("Vectorized reconstruction:")
    print(x_hat_vect)
    print("Time (loop-based): {:.6f} seconds".format(time_loop_reverse))
    print("Time (vectorized): {:.6f} seconds".format(time_vect_reverse))

    # Check reconstruction error (should be very low)
    error_loop = torch.norm(x - x_hat_loop) / torch.norm(x)
    error_vect = torch.norm(x - x_hat_vect) / torch.norm(x)
    print("\nRelative reconstruction error (loop-based): {:.2e}".format(error_loop.item()))
    print("Relative reconstruction error (vectorized): {:.2e}".format(error_vect.item()))

if __name__ == '__main__':
    main()