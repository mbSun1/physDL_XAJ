# -*- coding: utf-8 -*-
"""
Model Training and Testing Module

This module contains core utilities for training and testing deep-learning
hydrological models, focusing on XAJ (static/dynamic parameter) variants.

Main functionality:
1. Model training (trainModel) - supports static and dynamic XAJ parameter models
2. Model testing (testModel) - batch testing and saving results
3. Model loading (loadModel) - resume from checkpoints
4. Data sampling (randomIndex, selectSubset) - random sampling for training
"""

import numpy as np
import torch
import time
import os
from hydroMLlib.model import xaj_static, xaj_dynamic
from hydroMLlib.utils import criterion as loss_module
import pandas as pd


def trainModel(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               bufftime=0):
    """
    Train deep-learning hydrological models.

    Trains XAJ static-parameter and dynamic-parameter variants with optional
    GPU acceleration and checkpoint saving.

    Args:
        model: Model to train. Supports `xaj_static.MultiInv_XAJModel` and
            `xaj_dynamic.MultiInv_XAJTDModel`.
        x: Input data of shape (ngrid, nt, nx) or a tuple (x, z) where:
            - x: meteorological forcings (e.g., precipitation, temperature)
            - z: additional inputs (e.g., static attributes)
        y: Target data of shape (ngrid, nt, ny), typically observed streamflow.
        c: Constant inputs of shape (ngrid, nc), e.g., basin attributes.
        lossFun: Loss function. Supports `loss_module.NSELossBatch`,
            `loss_module.NSESqrtLossBatch`, etc.
        nEpoch: Number of epochs.
        miniBatch: [batchSize, rho] batch size and sequence length.
        saveEpoch: Checkpoint saving interval (epochs).
        saveFolder: Output directory for checkpoints/logs; None disables saving.
        bufftime: Buffer (warmup) time steps used for model initialization.

    Returns:
        torch.nn.Module: Trained model.

    Notes:
        - Auto-detects GPU and moves model/loss accordingly.
        - Includes NaN/Inf checks for numerical stability.
        - Prints training progress and basic performance/memory info.
        - Computes iterations per epoch to ensure sufficient sampling coverage.
    """
    print("=" * 50)
    print("Starting model training")
    print("=" * 50)
    
    # Parse batch parameters
    batchSize, rho = miniBatch
    
    # Handle input data format
    if type(x) is tuple or type(x) is list:
        x, z = x
    
    # Get data dimension information
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    
    # Adjust batch size
    if batchSize >= ngrid:
        batchSize = ngrid

    # Calculate iterations per epoch to reach ~99% sampling probability coverage
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    
    # If the model removes conditional time steps, adjust iteration count
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    # Display training configuration
    print("Training info:")
    print(f"- Data shape: {ngrid} grids x {nt} timesteps x {nx} variables")
    print(f"- Batch size: {batchSize}")
    print(f"- Sequence length (rho): {rho}")
    print(f"- Iterations per epoch: {nIterEp}")
    print(f"- Total epochs: {nEpoch}")
    print(f"- Buffer time: {bufftime}")
    print("-" * 50)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        lossFun = lossFun.to(device)
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Optimizer configuration
    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    
    # Training log file setup
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')
    # Start training loop
    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        t0 = time.time()
        
        # Iteration loop within each epoch
        for iIter in range(0, nIterEp):
            # Training iteration (supports XAJ static/dynamic variants)
            if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
                # Random sampling of training data
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                
                # Physical model: include buffer time
                xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
                
                # Select target subset
                yTrain = selectSubset(y, iGrid, iT, rho)
                
                # Select additional input subset
                if type(model) is xaj_static.MultiInv_XAJModel:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c)
                else:  # xaj_dynamic.MultiInv_XAJTDModel
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
                # Numerical stability check
                if torch.isnan(xTrain).any() or torch.isinf(xTrain).any():
                    print(f"Warning: xTrain contains NaN/Inf at iteration {iIter}")
                if torch.isnan(zTrain).any() or torch.isinf(zTrain).any():
                    print(f"Warning: zTrain contains NaN/Inf at iteration {iIter}")
                
                # Forward pass
                yP = model(xTrain, zTrain)
                
                # Check model outputs
                if torch.isnan(yP).any() or torch.isinf(yP).any():
                    print(f"Warning: yP contains NaN/Inf at iteration {iIter}")
                    print(f"yP stats: min={yP.min()}, max={yP.max()}, mean={yP.mean()}")
            else:
                Exception('unknown model')
            # Consider buffer time for initialization (kept as reference)
            # if bufftime > 0:
            #     yP = yP[bufftime:,:,:]
            
            # Calculate loss (only for total runoff Qr, column 0)
            if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
                yP_loss = yP[:, :, 0:1]
                yT_loss = yTrain[:, :, 0:1]
            else:
                yP_loss = yP
                yT_loss = yTrain

            # Loss computation by loss type
            if type(lossFun) in [loss_module.NSELossBatch, loss_module.NSESqrtLossBatch]:
                loss = lossFun(yP_loss, yT_loss, iGrid)
            else:
                loss = lossFun(yP_loss, yT_loss)
            # Backward pass and update
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
            
            # Loss numerical stability check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at iteration {iIter}")
                print(f"Loss value: {loss.item()}")
                break

            # Enhanced progress bar
            pct = int((iIter + 1) * 100 / max(1, nIterEp))
            filled = pct // 2
            bar = '█' * filled + '░' * (50 - filled)
            print('\rEpoch {}/{} [{}] {}% Iter {}/{} Loss {:.5f}'.format(
                iEpoch, nEpoch, bar, pct, iIter + 1, nIterEp, loss.item()), end='', flush=True)
        # End of epoch processing
        print()
        lossEp = lossEp / nIterEp
        elapsed_time = time.time() - t0
        remaining_epochs = nEpoch - iEpoch
        est_remaining_time = elapsed_time * remaining_epochs
        
        # GPU memory monitoring at end of each epoch
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(current_device)/1024**3
            reserved = torch.cuda.memory_reserved(current_device)/1024**3
            total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            gpu_util = f"GPU memory: {allocated:.2f}GB / {total:.2f}GB (cached: {reserved:.2f}GB)"
        else:
            gpu_util = "Using CPU"
        
        # Generate training log string
        logStr = 'Epoch {} | Loss {:.5f} | Time {:.2f}s | Est. Remaining {:.2f}s | {}'.format(
            iEpoch, lossEp, elapsed_time, est_remaining_time, gpu_util)
        print(logStr)
        
        # Save model and loss records
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                modelFile = os.path.join(saveFolder, 'model_Ep' + str(iEpoch) + '.pt')
                print(f"Saving checkpoint: {modelFile}")
                torch.save(model, modelFile)
    
    # Close training log file
    if saveFolder is not None:
        rf.close()
    
    # Training completion message
    print("=" * 50)
    print("Model training completed")
    print("=" * 50)
    return model


def loadModel(outFolder, epoch, modelName='model'):
    """
    Load a trained model checkpoint.

    Args:
        outFolder: Directory containing saved checkpoints.
        epoch: Epoch number to load.
        modelName: Model filename prefix.

    Returns:
        torch.nn.Module: Loaded model.

    Notes:
        - Use weights_only=False to load the full model object/state.
        - Supports resuming from any epoch checkpoint.
    """
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile, weights_only=False)
    return model


def testModel(model, x, c, *, batchSize=None, filePathLst=None):
    """
    Test a trained deep-learning hydrological model.

    Batch-tests trained XAJ models with optional GPU acceleration and output
    saving.

    Args:
        model: Model to test (supports XAJ static/dynamic variants).
        x: Input data of shape (ngrid, nt, nx) or a tuple (x, z).
        c: Constant inputs of shape (ngrid, nc) or None.
        batchSize: Batch size for testing; None uses all basins.
        filePathLst: List of output file paths; None auto-generates names.

    Returns:
        torch.Tensor or None: Test outputs (returned only when batchSize==ngrid).

    Notes:
        - Auto-detects GPU and manages memory.
        - Can capture and save XAJ physical parameters for analysis/plotting.
        - Saves outputs as CSV.
    """
    print("=" * 50)
    print("Starting model testing")
    print("=" * 50)
    
    # Handle input data format
    if type(x) is tuple or type(x) is list:
        x, z = x
    else:
        z = None
    
    # Get data dimension information
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    
    # Determine number of outputs
    if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
        ny = 5  # XAJ outputs: Qs, QSave, QIave, QGave, ETave
    else:
        ny = model.ny
    
    # Set batch size
    if batchSize is None:
        batchSize = ngrid
        
    # Display testing configuration
    print("Testing info:")
    print(f"- Data shape: {ngrid} grids x {nt} timesteps x {nx} variables")
    print(f"- Batch size: {batchSize}")
    print(f"- Output variables: {ny}")
    print(f"- Model type: {type(model).__name__}")
    print("-" * 50)
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set model to evaluation mode
    model.train(mode=False)
    
    # Handle conditional time step removal
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct

    # Batch index calculation
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    total_batches = len(iS)

    # Handle output file names
    if filePathLst is None:
        filePathLst = ['out' + str(x) for x in range(ny)]
    print(f"Output files: {filePathLst}")
    
    # Create output file list
    fLst = list()
    for filePath in filePathLst:
        # Ensure the directory exists
        dirPath = os.path.dirname(filePath)
        if dirPath and not os.path.exists(dirPath):
            os.makedirs(dirPath, exist_ok=True)
        
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        fLst.append(f)

    # Start batch forward pass
    t0_total = time.time()
    for i in range(0, len(iS)):
        t0_batch = time.time()
        
        # Progress
        pct = int((i + 1) * 100 / total_batches)
        filled = pct // 2
        bar = '█' * filled + '░' * (50 - filled)
        print(f"\rBatch {i+1}/{total_batches} [{bar}] {pct}%", end='', flush=True)

        # Prepare batch inputs
        xTemp = x[iS[i]:iE[i], :, :]
        
        if c is not None:
            # Expand constant inputs to all time steps
            cTemp = np.repeat(
                np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
            xTest = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp, cTemp], 2), 1, 0)).float()
        else:
            xTest = torch.from_numpy(
                np.swapaxes(xTemp, 1, 0)).float()
        
        # Move to device
        if torch.cuda.is_available():
            xTest = xTest.to(device)

        # Handle additional input data
        if z is not None:
            zTemp = z[iS[i]:iE[i], :, :]
            zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.to(device)

        # XAJ model forward pass and parameter capture
        if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
            with torch.no_grad():
                yP = model(xTest, zTest)
                
                # Capture XAJ parameters for analysis/plotting
                if type(model) == xaj_static.MultiInv_XAJModel:
                    Gen = model.lstminv(zTest)
                    Params0 = Gen[-1, :, :]
                    para0 = Params0[:, 0:model.nxajpm]
                    phy_params = torch.sigmoid(para0).view(Params0.shape[0], model.nfea, model.nmul)
                    model.last_phy_params = phy_params.detach().cpu().numpy()
                    model.last_phy_name = 'XAJ'
                elif type(model) == xaj_dynamic.MultiInv_XAJTDModel:
                    Params0 = model.lstminv(zTest)
                    para0 = Params0[:, :, 0:model.nxajpm]
                    phy_params = torch.sigmoid(para0).view(Params0.shape[0], Params0.shape[1], model.nfea, model.nmul)
                    model.last_phy_params = phy_params.detach().cpu().numpy()
                    model.last_phy_name = 'XAJTD'

        # Convert output to numpy [batch, time, var]
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)

        # Save outputs
        for k in range(ny):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        
        # Save XAJ parameters if captured
        if hasattr(model, 'last_phy_params') and model.last_phy_params is not None:
            phy_params = model.last_phy_params
            output_dir = os.path.dirname(filePathLst[0]) if filePathLst else '.'
            params_file = os.path.join(output_dir, f'phy_params_{getattr(model, "last_phy_name", "PHY")}_batch_{i}.npy')
            np.save(params_file, phy_params)

        # Batch timing
        batch_time = time.time() - t0_batch
        print(f" - Time: {batch_time:.2f}s", flush=True)
        
        # Memory cleanup
        model.zero_grad()
        torch.cuda.empty_cache()

    # Total timing summary
    total_time = time.time() - t0_total
    print("\n" + "-" * 50)
    print("Testing completed")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per batch: {total_time/total_batches:.2f}s")
    print("=" * 50)
    
    # Close all output files
    for f in fLst:
        f.close()

    # Return test results under specific conditions
    if batchSize == ngrid:
        yOut = torch.from_numpy(yOut)
        return yOut

def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    """
    Generate random indices for training data sampling.

    Randomly selects grid indices and time indices for each training batch
    to ensure stochastic coverage of the dataset.

    Parameters
    ----------
    ngrid : int
        Total number of grids.
    nt : int
        Total number of time steps.
    dimSubset : list
        Mini-batch configuration as [batchSize, rho].
    bufftime : int, default=0
        Number of warm-up/buffer time steps.

    Returns
    -------
    iGrid : numpy.ndarray
        Randomly selected grid indices of shape [batchSize].
    iT : numpy.ndarray
        Randomly selected time indices of shape [batchSize].

    Notes
    -----
    - Time index range accounts for buffer time and sequence length.
    - Ensures indices stay within data bounds.
    """
    batchSize, rho = dimSubset
    # Randomly select grid indices
    iGrid = np.random.randint(0, ngrid, [batchSize])
    # Randomly select time indices, considering buffer time
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    """
    Select a subset from the input data.

    According to the given grid and time indices, select the corresponding
    subset from the original data, supporting buffer time, constant inputs,
    and multiple output formats.

    Parameters
    ----------
    x : numpy.ndarray
        Input data with shape (ngrid, nt, nx).
    iGrid : numpy.ndarray
        Array of grid indices.
    iT : numpy.ndarray
        Array of time indices.
    rho : int
        Sequence length.
    c : numpy.ndarray or None, default=None
        Constant input data with shape (ngrid, nc).
    tupleOut : bool, default=False
        If True, return a tuple separating main and constant inputs.
    LCopt : bool, default=False
        If True, use local calibration options.
    bufftime : int, default=0
        Number of buffer time steps.

    Returns
    -------
    out : torch.Tensor or tuple
        Selected subset data with shape (rho+bufftime, batchSize, nx)
        or a tuple (xTensor, cTensor).

    Notes
    -----
    - Automatically handles GPU device transfer.
    - Supports multiple data formats and output options.
    - Includes boundary checks and special-case handling.
    """
    nx = x.shape[-1]  # number of input variables
    nt = x.shape[1]   # number of time steps
    
    # Special case handling
    if x.shape[0] == len(iGrid):   # hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if nt <= rho:
        iT.fill(0)  # if there are not enough time steps, set time index to 0

    batchSize = iGrid.shape[0]  # batch size
    
    if iT is not None:
        # Standard case: select data subset based on indices
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            # For each batch, select the corresponding time-window data
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        # Special case: handling when iT is None
        if LCopt is True:
            # Used for local calibration kernels: FDC, SMAP, etc.
            if len(x.shape) == 2:
                # FDC local calibration kernel (x = Ngrid * Ntime)
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # LC-SMAP with x = Ngrid * Ntime * Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Case where rho equals the full time-series length
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]  # update rho to the actual tensor length
    # Handle constant input data
    if c is not None:
        nc = c.shape[-1]  # number of constant input variables
        # Expand constant input to all time steps
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            # Return as tuple - separate main and constant inputs
            if torch.cuda.is_available():
                device = torch.device("cuda")
                xTensor = xTensor.to(device)
                cTensor = cTensor.to(device)
            out = (xTensor, cTensor)
        else:
            # Concatenate main and constant inputs
            out = torch.cat((xTensor, cTensor), 2)
    else:
        # No constant input, return main input directly
        out = xTensor

    # Device conversion
    if torch.cuda.is_available() and type(out) is not tuple:
        device = torch.device("cuda")
        out = out.to(device)  # move output to GPU
    
    return out
