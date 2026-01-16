"""
Default configuration module.

This module provides default configuration dictionaries for the dPLHBV
main workflow, using a pure functional design. All configurations are
returned by functions to avoid exposing module-level mutable state.

Main functionalities:
- provide default configuration for CAMELS-style datasets
- provide default configuration for model training
- provide default configuration for loss functions
- support updating and force-updating configuration dictionaries
"""

from collections import OrderedDict
from hydroMLlib.utils import camels


def get_data_config():
    """
    Get default configuration for the CAMELS dataset.

    This returns the default CAMELS configuration used by the dPLHBV
    workflow, including data source, variable selection, time range,
    normalization options, etc.

    Returns:
        OrderedDict: CAMELS data configuration with the following keys:
            - name: fully-qualified dataset class name
            - subset: basin subset name ('All' means all basins)
            - varT: list of time-series variables (meteorological forcing)
            - varC: list of constant variables (catchment attributes)
            - target: list of target variables (e.g., streamflow)
            - tRange: time range [start_date, end_date] in yyyymmdd
            - doNorm: [normalize time series, normalize constants]
            - rmNan: [handle NaNs in time series, handle NaNs in constants]
            - basinNorm: whether to apply basin-wise normalization
            - forType: meteorological forcing data type

    Note:
        - This function is called by the dPLHBV workflow to obtain the
          default data configuration.
        - The default time range is 1990–1995 (training period).
        - The default meteorological forcing is NLDAS.
        - Basin normalization can be used to convert discharge into
          runoff coefficients.
    """
    data_config = OrderedDict()
    data_config['name'] = 'hydroMLlib.utils.camels.DataframeCamels'
    data_config['subset'] = 'All'
    data_config['varT'] = camels.get_camels_config()['forcingLst']   # meteorological forcing variables
    data_config['varC'] = camels.get_camels_config()['attrLstSel']   # catchment attribute variables
    data_config['target'] = ['Streamflow']                           # target variable (discharge)
    data_config['tRange'] = [19900101, 19950101]                     # default time range
    data_config['doNorm'] = [True, True]                             # normalization options
    data_config['rmNan'] = [True, False]                             # NaN handling options
    data_config['basinNorm'] = True                                  # basin-wise normalization
    data_config['forType'] = 'nldas'                                 # forcing data type
    return data_config


def get_train_config():
    """
    Get default configuration for model training.

    This returns the default training configuration used by the dPLHBV
    workflow, including batch size, number of epochs, save frequency,
    random seed, etc.

    Returns:
        OrderedDict: training configuration with the following keys:
            - miniBatch: list [batch_size, sequence_length (rho)]
            - nEpoch: number of training epochs
            - saveEpoch: save frequency (save every N epochs)
            - seed: random seed (None means use default RNG state)
            - trainBuff: training buffer size

    Note:
        - This function is called by the dPLHBV workflow to obtain the
          default training configuration.
        - Default training is 100 epochs, saving every 50 epochs.
        - Different batch sizes for training and validation can be supported.
        - The default seed is None (system default randomness).
    """
    train_config = OrderedDict()
    train_config['miniBatch'] = [100, 200]   # [batch_size, sequence_length (rho)]
    train_config['nEpoch'] = 100             # number of epochs
    train_config['saveEpoch'] = 50           # save frequency
    train_config['seed'] = None              # random seed
    train_config['trainBuff'] = 0            # training buffer size
    return train_config


def get_loss_config():
    """
    Get default configuration for the loss function.

    This returns the default loss configuration used by the dPLHBV
    workflow, including loss type, prior distribution, and weight.

    Returns:
        OrderedDict: loss configuration with the following keys:
            - name: fully-qualified loss class name
            - prior: prior distribution type
            - weight: prior weight parameter

    Note:
        - This function is called by the dPLHBV workflow to obtain the
          default loss configuration.
        - Uses `RmseLossComb`, a combination of RMSE and log-sqrt RMSE.
        - The prior distribution is Gaussian.
        - The default prior weight is 0.0 (no prior term).
    """
    loss_config = OrderedDict()
    loss_config['name'] = 'hydroMLlib.utils.criterion.RmseLossComb'  # loss class name
    loss_config['prior'] = 'gauss'                                   # prior distribution type
    loss_config['weight'] = 0.0                                      # prior weight
    return loss_config


def update_config(opt, **kw):
    """
    Update a configuration dictionary.

    Update the configuration dictionary based on keyword arguments,
    with basic type checking and error handling.

    Args:
        opt (OrderedDict): configuration dictionary to be updated
        **kw: configuration items to update, as key–value pairs

    Returns:
        OrderedDict: updated configuration dictionary

    Note:
        - This function is called by the dPLHBV workflow to update
          default configurations.
        - Performs simple type checking to keep types consistent with
          the original configuration.
        - Special keys (such as 'subset' and 'seed') are assigned
          directly without type casting.
        - If a key does not exist or type casting fails, the item is
          skipped and a warning is printed.
        - Keeps the original ordering of the OrderedDict.
    """
    for key in kw:
        if key in opt:
            try:
                if key in ['subset', 'seed']:
                    opt[key] = kw[key]
                else:
                    opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print(f'skip key {key}: type casting error')
        else:
            print(f'skip key {key}: not in configuration dictionary')
    return opt


def force_update_config(opt, **kw):
    """
    Force-update a configuration dictionary.

    Directly update the configuration dictionary without any type
    checking. This is intended for cases where a full override is
    desired.

    Args:
        opt (OrderedDict): configuration dictionary to be updated
        **kw: configuration items to update, as key–value pairs

    Returns:
        OrderedDict: updated configuration dictionary

    Note:
        - This function is called by the dPLHBV workflow to forcibly
          override default configurations.
        - No type checking is performed; values are overwritten as-is.
        - New keys will be added if they do not already exist.
        - Keeps the original ordering of the OrderedDict.
    """
    for key in kw:
        opt[key] = kw[key]
    return opt

