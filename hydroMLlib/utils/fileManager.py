"""
File management utilities.

This module provides file management utilities used during model
training and prediction, including reading/writing configuration
files, loading trained models, and generating standardized file
names for prediction outputs.

Main functionalities:
- Configuration management: read/write the `master.json` configuration
  file, storing data/model/loss/training options.
- Model loading: load trained models given the output directory
  and epoch index.
- Prediction file naming: generate standardized file names and
  paths for prediction results.
- Support naming rules for different loss functions and
  uncertainty quantification outputs.
"""

import os
import hydroMLlib
from collections import OrderedDict
import json


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    """
    Wrap configuration dictionaries into a master config.

    Collect data/model/loss/training options into an ordered
    dictionary that can be written to `master.json`. This is
    the central entry point for creating configuration files,
    ensuring that all options are recorded consistently.

    Args:
        out (str): output directory for models and configuration files
        optData (dict): data configuration (dataset paths, variable lists, etc.)
        optModel (dict): model configuration (architecture, hyperparameters, etc.)
        optLoss (dict): loss configuration (loss type, weights, etc.)
        optTrain (dict): training configuration (learning rate, batch size, epochs, etc.)

    Returns:
        OrderedDict: ordered master configuration with keys:
            - out: output directory
            - data: data config
            - model: model config
            - loss: loss config
            - train: training config

    Example:
        >>> config = wrapMaster('/path/to/output', dataOpt, modelOpt, lossOpt, trainOpt)
        >>> writeMasterFile(config)
    """
    # Use OrderedDict to keep a stable and readable order in JSON.
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return mDict


def readMasterFile(out):
    """
    Read the master configuration file (`master.json`).

    Load the configuration dictionary from the given output
    directory. This is used when loading models, running
    predictions, or analyzing results.

    Args:
        out (str): output directory containing `master.json`

    Returns:
        OrderedDict: configuration dictionary with keys:
            - out: output directory
            - data: data configuration
            - model: model configuration
            - loss: loss configuration
            - train: training configuration

    Raises:
        FileNotFoundError: if `master.json` does not exist
        json.JSONDecodeError: if the JSON file is malformed

    Example:
        >>> config = readMasterFile('/path/to/output')
        >>> print(config['train']['nEpoch'])
    """
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'r') as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    
    print('read master file ' + mFile)
    return mDict


def writeMasterFile(mDict):
    """
    Write the master configuration file (`master.json`).

    Save the master configuration dictionary to the specified
    output directory. If the directory does not exist, it will
    be created. The JSON file is indented for readability.

    Args:
        mDict (OrderedDict): master configuration dictionary. It must
            contain the key 'out' specifying the output directory; the
            remaining keys typically include 'data', 'model', 'loss',
            and 'train'.

    Returns:
        str: output directory path

    Example:
        >>> config = wrapMaster('/path/to/output', dataOpt, modelOpt, lossOpt, trainOpt)
        >>> outputDir = writeMasterFile(config)
    """
    out = mDict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'w') as fp:
        json.dump(mDict, fp, indent=4)
    
    print('write master file ' + mFile)
    return out


def loadModel(out, epoch=None):
    """
    Load a trained model.

    Load a trained model from the given output directory and epoch.
    If no epoch is provided, the last training epoch specified in
    `master.json` will be used.

    Args:
        out (str): output directory containing model files and config
        epoch (int, optional): training epoch to load
            - None: use the last epoch from the config (default)
            - int: specific training epoch

    Returns:
        torch.nn.Module: PyTorch model with loaded weights, ready for
        prediction or continued training.

    Raises:
        FileNotFoundError: if the model file does not exist
        KeyError: if required fields are missing in the config

    Example:
        >>> model = loadModel('/path/to/output')
        >>> model = loadModel('/path/to/output', epoch=100)
    """
    if epoch is None:
        mDict = readMasterFile(out)
        epoch = mDict['train']['nEpoch']

    model = hydroMLlib.model.train.loadModel(out, epoch)
    return model


def namePred(out, tRange, subset, epoch=None, targLst=None):
    """
    Generate standardized prediction file names.

    Build standardized file names and full paths for prediction
    outputs based on configuration, time range, subset, and epoch.
    Supports naming rules for different loss functions and
    uncertainty quantification outputs, ensuring both uniqueness
    and readability.

    Args:
        out (str): output directory to store prediction files
        tRange (list): time range [start_time, end_time] as
            [YYYYMMDD, YYYYMMDD]
        subset (str or list): subset identifier
            - str: used directly as subset label
            - list: use its length as subset label
        epoch (int, optional): training epoch
            - None: use the last epoch from the config (default)
            - int: specific training epoch
        targLst (list, optional): target variable list
            - None: read from config (default)
            - list: user-defined targets

    Returns:
        list: list of full file paths for prediction outputs
              (CSV files, including directory and file name)

    Naming rules (base examples):
        Base: {subset}_{startTime}_{endTime}_ep{epoch}_{target}.csv
        With uncertainty (SigmaLoss):
              {subset}_{startTime}_{endTime}_ep{epoch}_{target}_SigmaX.csv

    Example:
        >>> files = namePred('/output', [20200101, 20201231], 'test', epoch=100)
        >>> # ['/output/test_20200101_20201231_ep100_Streamflow.csv']

    """
    mDict = readMasterFile(out)

    # determine target variable list
    if targLst is not None:
        target = targLst
    else:
        if 'name' in mDict['data'].keys() and mDict['data']['name'] == 'hydroMLlib.utils.camels.DataframeCamels':
            target = ['Streamflow']
        else:
            target = mDict['data']['target']

    # ensure target is a list
    if type(target) is not list:
        target = [target]

    # number of target variables
    nt = len(target)

    # loss name (used for uncertainty naming rules)
    lossName = mDict['loss']['name']

    # determine training epoch
    if epoch is None:
        epoch = mDict['train']['nEpoch']

    # deal with subset identifier
    if type(subset) is list:
        subset = str(len(subset))

    # generate base file name list
    fileNameLst = list()
    for k in range(nt):
        # base test name: subset_startTime_endTime_ep{epoch}
        testName = '_'.join(
            [subset, str(tRange[0]),
             str(tRange[1]), 'ep' + str(epoch)])

        # append target variable name
        fileName = '_'.join([testName, target[k]])
        fileNameLst.append(fileName)

        # add uncertainty output if using SigmaLoss
        if lossName == 'hydroMLlib.utils.loss.SigmaLoss':
            fileName = '_'.join([testName, target[k], 'SigmaX'])
            fileNameLst.append(fileName)

    # build full file paths
    filePathLst = list()
    for fileName in fileNameLst:
        filePath = os.path.join(out, fileName + '.csv')
        filePathLst.append(filePath)
    
    return filePathLst
