"""
Loss function utilities for hydrological model training.

This module provides several loss functions used in hydrological
model training, including mixed RMSE-based losses and
Nash–Sutcliffe efficiency (NSE) style losses. They are mainly used
for training differentiable hydrological models such as dPLHBV
and XAJ-based models.

Main components:
- Mixed RMSE loss: combines standard RMSE and a log–sqrt RMSE term.
- NSE loss: batch loss based on Nash–Sutcliffe efficiency.
- Sqrt-NSE loss: NSE-style loss using RMSE and standard deviation.
- Support for multi-variable outputs and missing-value handling.
"""

import torch
import numpy as np


class RmseLossComb(torch.nn.Module):
    """
    Mixed RMSE loss.

    This loss combines standard RMSE with a log–sqrt RMSE term for
    hydrological model training. It aims to improve performance for
    both high and low flows, which is particularly useful for streamflow
    prediction.

    Mathematical form:
        Loss = (1 - α) * RMSE + α * Log-Sqrt-RMSE
    where
        - RMSE = sqrt(mean((pred - true)²))
        - Log-Sqrt-RMSE =
          sqrt(mean((log10(sqrt(pred + β) + 0.1) -
                     log10(sqrt(true + β) + 0.1))²))

    Args:
        alpha (float): weight for the log–sqrt RMSE term
                       - range: [0, 1]
                       - 0: pure standard RMSE
                       - 1: pure log–sqrt RMSE
                       - 0.5: equal weights
        beta (float): numerical stability parameter to avoid log(0)
                       - default: 1e-6
                       - used to handle near-zero predictions

    Input shapes:
        output: [batch_size, seq_len, n_variables] – model predictions
        target: [batch_size, seq_len, n_variables] – ground truth

    Features:
        - supports multi-variable outputs
        - masks NaN values in targets
        - includes numerical stability safeguards
        - suitable for streamflow prediction tasks
    """
    
    def __init__(self, alpha, beta=1e-6):
        """
        Initialize the mixed RMSE loss.

        Args:
            alpha (float): weight for the log–sqrt RMSE term
            beta (float): numerical stability parameter (default: 1e-6)
        """
        super(RmseLossComb, self).__init__()
        self.alpha = alpha  # weight for log–sqrt RMSE
        self.beta = beta    # numerical stability parameter

    def forward(self, output, target):
        """
        Compute the mixed RMSE loss.

        Args:
            output (torch.Tensor): predictions, shape
                [batch_size, seq_len, n_variables]
            target (torch.Tensor): targets, shape
                [batch_size, seq_len, n_variables]

        Returns:
            torch.Tensor: scalar loss value

        Steps:
            1. Loop over variables.
            2. Compute standard RMSE.
            3. Compute log–sqrt RMSE.
            4. Combine the two with weight `alpha`.
            5. Sum over variables.
        """
        ny = target.shape[2]  # number of variables
        loss = 0

        # compute loss for each output variable separately
        for k in range(ny):
            # extract predictions and targets for variable k
            p0 = output[:, :, k]
            t0 = target[:, :, k]

            # log–sqrt transform with beta and offset 0.1 for stability
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)

            # build mask of valid values (exclude NaNs in target)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]

            # standard RMSE
            loss1 = torch.sqrt(((p - t)**2).mean())

            # mask for log–sqrt RMSE
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]

            # log–sqrt RMSE
            loss2 = torch.sqrt(((pa - ta)**2).mean())

            # weighted combination
            temp = (1.0 - self.alpha) * loss1 + self.alpha * loss2

            # accumulate over variables
            loss = loss + temp
            
        return loss


class NSELossBatch(torch.nn.Module):
    """
    Batch NSE-style loss.

    NSE-like batch loss based on Nash–Sutcliffe efficiency, following
    a formulation similar to Fredrik (2019). The loss is normalized by
    the standard deviation of basin discharge and is suitable for
    multi-basin batch training.

    Mathematical form:
        Loss = mean((pred - true)² / (std + ε)²)
    where
        - std: standard deviation of discharge for each basin
        - ε: numerical stability parameter

    Args:
        stdarray (numpy.ndarray): standard deviation of discharge for
            all basins, shape [n_basins], used for normalization
        eps (float): numerical stability parameter (default: 0.1),
            added to the standard deviation to avoid division by zero

    Input shapes:
        output: [batch_size, n_basins, 1] – predictions
        target: [batch_size, n_basins, 1] – ground truth
        igrid:  [batch_size] – basin indices

    Features:
        - supports multi-basin batch training
        - automatically normalizes losses across basins
        - handles missing values via masking
        - suitable for streamflow prediction tasks
    """
    
    def __init__(self, stdarray, eps=0.1):
        """
        Initialize the batch NSE loss.

        Args:
            stdarray (numpy.ndarray): per-basin discharge standard deviation
            eps (float): numerical stability parameter (default: 0.1)
        """
        super(NSELossBatch, self).__init__()
        self.std = stdarray  # per-basin discharge standard deviation
        self.eps = eps       # numerical stability parameter

    def forward(self, output, target, igrid):
        """
        Compute the batch NSE-style loss.

        Args:
            output (torch.Tensor): predictions, shape
                [batch_size, n_basins, 1]
            target (torch.Tensor): targets, shape
                [batch_size, n_basins, 1]
            igrid (torch.Tensor): basin indices, shape [batch_size]

        Returns:
            torch.Tensor: scalar loss value

        Steps:
            1. Expand per-basin standard deviation to batch and time.
            2. Compute squared errors.
            3. Normalize by (std + eps)².
            4. Take the mean over valid entries.
        """
        nt = target.shape[0]  # number of time steps

        # replicate std for each time step according to basin indices
        stdse = np.tile(self.std[igrid], (nt, 1))

        # move standard deviation array to the correct device
        device = output.device
        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(device)

        # extract predictions and targets (assume a single output variable)
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]

        # mask valid values (exclude NaNs in target)
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]

        # squared error
        sqRes = (p - t)**2

        # normalize by standard deviation (add eps to avoid division by zero)
        normRes = sqRes / (stdw + self.eps)**2

        # mean loss
        loss = torch.mean(normRes)
        
        return loss


class NSESqrtLossBatch(torch.nn.Module):
    """
    Sqrt-NSE batch loss.

    NSE-style batch loss that uses the square root of squared errors
    (absolute error) normalized by standard deviation. This follows a
    formulation similar to Fredrik (2019) and is suitable for
    multi-basin batch training.

    Mathematical form:
        Loss = mean( sqrt((pred - true)²) / (std + ε) )
    where
        - sqrt((pred - true)²): absolute error
        - std: standard deviation of discharge for each basin
        - ε: numerical stability parameter

    Args:
        stdarray (numpy.ndarray): standard deviation of discharge for
            all basins, shape [n_basins], used for normalization
        eps (float): numerical stability parameter (default: 0.1),
            added to the standard deviation to avoid division by zero

    Input shapes:
        output: [batch_size, n_basins, 1] – predictions
        target: [batch_size, n_basins, 1] – ground truth
        igrid:  [batch_size] – basin indices

    Features:
        - uses absolute error instead of squared error
        - supports multi-basin batch training
        - automatically normalizes across basins
        - handles missing values
        - more robust to outliers than pure squared-error formulations
    """
    
    def __init__(self, stdarray, eps=0.1):
        """
        Initialize the sqrt-NSE batch loss.

        Args:
            stdarray (numpy.ndarray): per-basin discharge standard deviation
            eps (float): numerical stability parameter (default: 0.1)
        """
        super(NSESqrtLossBatch, self).__init__()
        self.std = stdarray  # per-basin discharge standard deviation
        self.eps = eps       # numerical stability parameter

    def forward(self, output, target, igrid):
        """
        Compute the sqrt-NSE batch loss.

        Args:
            output (torch.Tensor): predictions, shape
                [batch_size, n_basins, 1]
            target (torch.Tensor): targets, shape
                [batch_size, n_basins, 1]
            igrid (torch.Tensor): basin indices, shape [batch_size]

        Returns:
            torch.Tensor: scalar loss value

        Steps:
            1. Expand per-basin standard deviation to batch and time.
            2. Compute absolute errors (sqrt of squared errors).
            3. Normalize by (std + eps).
            4. Take the mean over valid entries.
        """
        nt = target.shape[0]  # number of time steps

        # replicate std for each time step according to basin indices
        stdse = np.tile(self.std[igrid], (nt, 1))

        # move standard deviation array to the correct device
        device = output.device
        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(device)

        # extract predictions and targets (assume a single output variable)
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]

        # mask valid values (exclude NaNs in target)
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]

        # absolute error (sqrt of squared error)
        sqRes = torch.sqrt((p - t)**2)

        # normalize by standard deviation (add eps to avoid division by zero)
        normRes = sqRes / (stdw + self.eps)

        # mean loss
        loss = torch.mean(normRes)
        
        return loss
