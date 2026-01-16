"""
dPLXAJ Dynamic Parameter Training Script
======================================
- Train dynamic-parameter XAJ model
- Use LSTM/GRU/etc. as parameter-prediction backbone
- Some XAJ parameters are dynamic (time-varying)
- Support multiple RNN types and configuration options
"""

# =============================================================================
# Imports and setup
# =============================================================================
from hydroMLlib.utils import camels, defaultSetting, fileManager, time, criterion
from hydroMLlib.model import xaj_static, xaj_dynamic, train

import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
import datetime as dt
import argparse


def parse_args():
    """
    Parse command-line arguments for dynamic-parameter training.

    All arguments must match the testing script to ensure consistent
    model configuration. In dynamic mode, td_opt defaults to True and
    XAJ has 12 parameters.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='dPLXAJ Dynamic Parameter Training')

    # =============================================================================
    # Experiment / basin split (must match testing)
    # =============================================================================

    parser.add_argument('--pu_opt', type=int, default=0, choices=[0, 1, 2],
                       help='Basin split: 0=ALL, 1=PUB, 2=PUR')
    parser.add_argument('--buff_opt', type=int, default=0, choices=[0, 1, 2],
                       help='Warmup: 0=first year for next, 1=repeat first year, 2=load extra year')
    parser.add_argument('--td_opt', action='store_true', default=True,
                       help='Parameter mode: False=static, True=dynamic (default True for dynamic mode)')
    parser.add_argument('--for_type', type=str, default='daymet',
                       choices=['daymet', 'nldas', 'maurer'],
                       help='Meteorological forcing: daymet, nldas, or maurer')

    # =============================================================================
    # Model architecture (XAJ has 12 parameters)
    # =============================================================================

    parser.add_argument('--rnn_type', type=str, default='cnnlstm',
                       choices=['lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'],
                       help='RNN type: lstm, gru, bilstm, bigru, rnn, cnnlstm, cnnbilstm')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Training batch size (basins per batch)')
    parser.add_argument('--rho', type=int, default=365,
                       help='Sequence length in days')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='RNN hidden size')
    parser.add_argument('--nfea', type=int, default=12,
                       help='XAJ parameter count (12)')

    # Time configuration
    parser.add_argument('--train_start', type=str, default='19801001',
                       help='Training start date (YYYYMMDD)')
    parser.add_argument('--train_end', type=str, default='20041001',
                       help='Training end date (YYYYMMDD)')
    parser.add_argument('--inv_start', type=str, default='19801001',
                       help='Inversion start date (YYYYMMDD)')
    parser.add_argument('--inv_end', type=str, default='20041001',
                       help='Inversion end date (YYYYMMDD)')
    parser.add_argument('--buff_time', type=int, default=365,
                       help='Warmup length in days')

    # XAJ configuration
    parser.add_argument('--routing', action='store_true', default=True,
                       help='Enable routing')
    parser.add_argument('--nmul', type=int, default=10,
                       help='Number of XAJ components')
    parser.add_argument('--comprout', action='store_true', default=False,
                       help='Component routing')
    parser.add_argument('--compwts', action='store_true', default=False,
                       help='Component weights')

    # Training configuration
    parser.add_argument('--epoch', type=int, default=30,
                       help='Total training epochs')
    parser.add_argument('--save_epoch', type=int, default=1,
                       help='Model save interval (epochs)')
    parser.add_argument('--random_seed', type=int, default=111111,
                       help='Random seed')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--test_fold', type=int, default=1,
                       help='Fold index for PUB/PUR')

    # =============================================================================
    # Dynamic-parameter configuration (core for dynamic mode)
    # Dynamic params: ke, wm, c, sm, ki (indices 1, 5, 6, 7, 9)
    # =============================================================================
    parser.add_argument('--td_rep', type=int, nargs='+', default=[1, 5, 6, 7, 9],
                       help='Dynamic parameter indices (1–12)')
    parser.add_argument('--dy_drop', type=float, default=0.0,
                       help='Dynamic dropout: 0=always dynamic, 1=always static, in-between=mixed')
    parser.add_argument('--sta_ind', type=int, default=-1,
                       help='Static parameter time step: -1=last, else index')

    return parser.parse_args()

def setup_environment(args):
    """
    Set up environment and random seed for reproducible training.

    Args:
        args: Parsed command-line arguments (contains random_seed and gpu_id).
    """
    # Random seeds
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    
    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        print("CUDA not available, using CPU")

# =============================================================================
# Path configuration
# =============================================================================
def init_camels_path():
    """
    Initialize CAMELS/data and output paths with priority rules.

    Priority (high to low):
    1) Local project directories:
       - Data:  <project_root>/Camels or <project_root>/CAMELS
       - Out:   <project_root>/outputs (created if missing)
    2) If no local Camels/CAMELS, fall back to generic defaults:
       - Data:  /scratch/Camels
       - Out:   /data/rnnStreamflow

    Returns:
        OrderedDict: {'DB': data_root, 'Out': output_root}
    """
    # Project root: parent of experiments directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    
    # Prefer local Camels/CAMELS and outputs
    candidate_camels = [
        os.path.join(project_root, 'Camels'),
        os.path.join(project_root, 'CAMELS')
    ]
    local_camels = next((p for p in candidate_camels if os.path.isdir(p)), None)
    local_outputs = os.path.join(project_root, 'outputs')
    
    # Ensure outputs directory exists
    if not os.path.isdir(local_outputs):
        try:
            os.makedirs(local_outputs, exist_ok=True)
        except Exception:
            pass

    if local_camels is not None:
        pathCamels = OrderedDict(
            DB=local_camels,
            Out=local_outputs
        )
    else:
        pathCamels = OrderedDict(
            DB=os.path.join(os.path.sep, 'scratch', 'Camels'),
            Out=os.path.join(os.path.sep, 'data', 'rnnStreamflow')
        )

    return pathCamels


def setup_basin_split(args, pathCamels):
    """
    Set up basin split for training and testing.

    Args:
        args: Parsed command-line arguments (pu_opt, test_fold).
        pathCamels: Path dictionary with 'DB' and 'Out'.

    Returns:
        tuple: (rootDatabase, rootOut, gageinfo, puN,
                TrainLS, TrainInd, TestLS, TestInd, gageDic)
    """
    # CAMELS database and output roots
    rootDatabase = pathCamels['DB']
    rootOut = pathCamels['Out']
    
    # Read basin information
    gageinfo = camels.readGageInfo(rootDatabase)
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()

    if args.pu_opt == 0:  # ALL mode
        puN = 'ALL'
        TrainLS = gageidLst
        TrainInd = [gageidLst.index(j) for j in TrainLS]
        TestLS = gageidLst
        TestInd = [gageidLst.index(j) for j in TestLS]
        gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    elif args.pu_opt == 1:  # PUB mode
        puN = 'PUB'
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst = json.load(fp)
        TestLS = testIDLst[args.test_fold - 1]
        TestInd = [gageidLst.index(j) for j in TestLS]
        TrainLS = list(set(gageid.tolist()) - set(TestLS))
        TrainInd = [gageidLst.index(j) for j in TrainLS]
        gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    elif args.pu_opt == 2:  # PUR mode
        puN = 'PUR'
        regionID = list()
        regionDivide = [[1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17]]

        for ii in range(len(regionDivide)):
            tempcomb = regionDivide[ii]
            tempregid = list()
            for ih in tempcomb:
                tempid = gageid[hucinfo == ih].tolist()
                tempregid = tempregid + tempid
            regionID.append(tempregid)

        iexp = args.test_fold - 1
        TestLS = regionID[iexp]
        TestInd = [gageidLst.index(j) for j in TestLS]
        TrainLS = list(set(gageid.tolist()) - set(TestLS))
        TrainInd = [gageidLst.index(j) for j in TrainLS]
        gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    return rootDatabase, rootOut, gageinfo, puN, TrainLS, TrainInd, TestLS, TestInd, gageDic

def load_and_preprocess_data(args, rootDatabase, gageinfo, TrainLS, TrainInd):
    """
    Load and preprocess CAMELS data for training.

    This includes:
    1. Meteorological forcings.
    2. Observed streamflow.
    3. Basin attributes.
    4. Potential evapotranspiration (PET).
    5. Unit conversion and normalization.
    6. Applying warmup options.

    Returns:
        tuple: (optData, statDict, forcTuple, yTrainIn, attrs)
    """
    # Time ranges
    Ttrain = [int(args.train_start), int(args.train_end)]
    Tinv = [int(args.inv_start), int(args.inv_end)]
    print(f"Training period: {Ttrain[0]} - {Ttrain[1]}")
    print(f"Inversion period: {Tinv[0]} - {Tinv[1]}")
    
    # Warmup options
    if args.buff_opt == 2:
        print(f"Warmup option 2: load extra {args.buff_time} days for warmup")
        sd = time.t2dt(Ttrain[0]) - dt.timedelta(days=args.buff_time)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]
        TinvLoad = [sdint, Tinv[1]]
        print(f"Extended training period: {TtrainLoad[0]} - {TtrainLoad[1]}")
    else:
        TtrainLoad = Ttrain
        TinvLoad = Tinv
        print(f"Warmup option {args.buff_opt}: use original period")

    # Forcing variables
    if args.for_type == 'daymet':
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
        print("Forcing: Daymet (precip, tmean)")
    else:
        varF = ['prcp', 'tmax']
        varFInv = ['prcp', 'tmax']
        print(f"Forcing: {args.for_type.upper()} (precip, tmax)")
    
    attrnewLst = [
        'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
        'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
        'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
        'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
        'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class',
        'glim_1st_class_frac', 'geol_2nd_class', 'glim_2nd_class_frac',
        'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
    ]
    print(f"Number of basin attributes: {len(attrnewLst)}")
    
    # Data config and loading
    optData = defaultSetting.get_data_config()
    optData = defaultSetting.update_config(optData,
                            tRange=TtrainLoad,
                            varT=varFInv,
                            varC=attrnewLst,
                            subset=TrainLS,
                            forType=args.for_type)
    
    # Forcings and observed flow
    dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=args.for_type,
                                     dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    
    # Inversion data (for parameter learning)
    dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=args.for_type,
                                   dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)
    
    # Unit conversion: ft3/s -> mm/day
    areas = gageinfo['area'][TrainInd]
    temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1], 1))
    obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3

    # PET (Potential Evapotranspiration)
    varLstNL = ['PEVAP']
    usgsIdLst = gageinfo['id']
    
    # PET time range by forcing type
    if args.for_type == 'maurer':
        tPETRange = [19800101, 20090101]
    else:
        tPETRange = [19800101, 20150101]
    
    tPETLst = time.tRange2Array(tPETRange)
    
    # PET directory, e.g. Camels/pet_harg/daymet/
    PETDir = rootDatabase + '/pet_harg/' + args.for_type + '/'

    ntime = len(tPETLst)
    PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
    
    for k in range(len(usgsIdLst)):
        dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
        PETfull[k, :, :] = dataTemp
    
    # PET for train and inversion periods
    TtrainLst = time.tRange2Array(TtrainLoad)
    TinvLst = time.tRange2Array(TinvLoad)
    _, _, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    PETUN = PETfull[:, ind2, :]
    PETUN = PETUN[TrainInd, :, :]
    _, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInvUN = PETfull[:, ind2inv, :]
    PETInvUN = PETInvUN[TrainInd, :, :]
    
    # Normalization and NaN handling
    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    seriesvarLst = varFInv + ['pet']

    statDict = camels.getStatDic(attrLst=attrnewLst, attrdata=attrsUN,
                                seriesLst=seriesvarLst, seriesdata=series_inv)
    
    # Normalize
    attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    # Prepare inputs
    zTrain = series_norm
    xTrain = np.concatenate([forcUN, PETUN], axis=2)
    xTrain[np.isnan(xTrain)] = 0.0

    # Apply warmup option
    if args.buff_opt == 1:
        zTrainIn = np.concatenate([zTrain[:, 0:args.buff_time, :], zTrain], axis=1)
        xTrainIn = np.concatenate([xTrain[:, 0:args.buff_time, :], xTrain], axis=1)
        yTrainIn = np.concatenate([obsUN[:, 0:args.buff_time, :], obsUN], axis=1)
    else:
        zTrainIn = zTrain
        xTrainIn = xTrain
        yTrainIn = obsUN

    forcTuple = (xTrainIn, zTrainIn)
    attrs = attr_norm

    return optData, statDict, forcTuple, yTrainIn, attrs

def train_model(args, rootOut, puN, optData, statDict, forcTuple, yTrainIn, attrs):
    """
    Build and train the dPLXAJ dynamic-parameter model.

    Args:
        args: Parsed command-line arguments.
        rootOut: Output root directory.
        puN: Basin mode name ('ALL'/'PUB'/'PUR').
        optData: Data loading options.
        statDict: Normalization statistics.
        forcTuple: Tuple (xTrainIn, zTrainIn).
        yTrainIn: Target discharge.
        attrs: Basin attributes.

    Returns:
        tuple: (trainedModel, out)
    """
    # Dynamic vs static parameter mode
    if args.td_opt:
        tdRep = args.td_rep
        tdRepS = [str(ix) for ix in tdRep]
        Nfea = args.nfea
        dydrop = args.dy_drop
        staind = args.sta_ind
        TDN = '/TDTestforc/' + 'TD' + "_".join(tdRepS) + '/'
        print(f"Dynamic mode: dynamic params {tdRep}, Nfea={Nfea}")
    else:
        TDN = '/Testforc/'
        Nfea = args.nfea
        print(f"Static mode: all params static, Nfea={Nfea}")
    
    # Loss configuration
    alpha = 0.25
    optLoss = defaultSetting.update_config(defaultSetting.get_loss_config(), name='hydroMLlib.utils.criterion.RmseLossComb', weight=alpha)
    lossFun = criterion.RmseLossComb(alpha=alpha)
    print(f"Loss: RMSE alpha={alpha}")
    
    # Training options
    optTrain = defaultSetting.update_config(defaultSetting.get_train_config(),
                             miniBatch=[args.batch_size, args.rho],
                             nEpoch=args.epoch,
                             saveEpoch=args.save_epoch)
    print(f"Train config: batch={args.batch_size}, rho={args.rho}, epochs={args.epoch}")
    
    # Output path components
    exp_name = 'CAMELSDemo'
    rnn_type_str = args.rnn_type.upper()
    
    # Experiment config path
    exp_disp = (f'dPLXAJ/{puN}{TDN}{args.for_type}/BuffOpt{args.buff_opt}/RMSE_para{alpha}/'
               f'{args.random_seed}/Fold{args.test_fold}/{rnn_type_str}')
    
    # Model-info path
    exp_info = (f'T_{args.train_start}_{args.train_end}_BS_{args.batch_size}_HS_{args.hidden_size}_RHO_{args.rho}_'
               f'NF_{Nfea}_Buff_{args.buff_time}_Mul_{args.nmul}_{rnn_type_str}')
    
    # Full save path
    save_path = os.path.join(exp_name, exp_disp)
    out = os.path.join(rootOut, save_path, exp_info)

    # Model definition
    Ninv = forcTuple[1].shape[-1] + attrs.shape[-1]
    print(f"Model input dim: {Ninv} (series: {forcTuple[1].shape[-1]}, attrs: {attrs.shape[-1]})")
    
    if not args.td_opt:
        print("Building static-parameter XAJ model...")
        model = xaj_static.MultiInv_XAJModel(
            ninv=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddeninv=args.hidden_size, inittime=args.buff_time,
            routOpt=args.routing, comprout=args.comprout,
            compwts=args.compwts, pcorr=None,
            rnn_type=args.rnn_type)

        optModel = OrderedDict(
            name='dPLXAJ', nx=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddenSize=args.hidden_size, doReLU=True,
            Tinv=[int(args.inv_start), int(args.inv_end)],
            Trainbuff=args.buff_time, routOpt=args.routing,
            comprout=args.comprout, compwts=args.compwts,
            pcorr=None, buffOpt=args.buff_opt, TDOpt=args.td_opt,
            rnn_type=args.rnn_type)
    else:
        print("Building dynamic-parameter XAJ model...")
        print(f"Dynamic config: params={tdRep}, dropout={dydrop}")
        model = xaj_dynamic.MultiInv_XAJTDModel(
            ninv=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddeninv=args.hidden_size, inittime=args.buff_time,
            routOpt=args.routing, comprout=args.comprout,
            compwts=args.compwts, staind=staind, tdlst=tdRep,
            dydrop=dydrop, rnn_type=args.rnn_type)

        optModel = OrderedDict(
            name='dPLXAJDP', nx=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddenSize=args.hidden_size, doReLU=True,
            Tinv=[int(args.inv_start), int(args.inv_end)],
            Trainbuff=args.buff_time, routOpt=args.routing,
            comprout=args.comprout, compwts=args.compwts,
            pcorr=None, staind=staind, tdlst=tdRep, dydrop=dydrop,
            buffOpt=args.buff_opt, TDOpt=args.td_opt,
            rnn_type=args.rnn_type)

    # Show model info
    print("\n" + "=" * 50)
    print(f"Using {args.rnn_type.upper()} as parameter prediction backbone")
    print("=" * 50 + "\n")
    
    # Save configs and stats
    print("Saving model configuration...")
    masterDict = fileManager.wrapMaster(out, optData, optModel, optLoss, optTrain)
    fileManager.writeMasterFile(masterDict)
    print("Model configuration saved")
    
    print("Saving normalization statistics...")
    statFile = os.path.join(out, 'statDict.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)
    print("Normalization statistics saved")
    
    # Training
    print("=" * 50)
    print("Starting model training")
    print("=" * 50)
    if args.td_opt:
        print(f"Dynamic training: params {tdRep} will learn temporal variation")
    else:
        print("Static training: all parameters fixed")

    trainedModel = train.trainModel(
        model,
        forcTuple,
        yTrainIn,
        attrs,
        lossFun,
        nEpoch=args.epoch,
        miniBatch=[args.batch_size, args.rho],
        saveEpoch=args.save_epoch,
        saveFolder=out,
        bufftime=args.buff_time
    )

    print("=" * 50)
    print("Model training completed")
    print("=" * 50)
    if args.td_opt:
        print(f"Dynamic-parameter training finished: params {tdRep}")
    else:
        print("Static-parameter training finished")

    return trainedModel, out

def main():
    """
    Main entry point for dynamic-parameter training.

    Coordinates the full training workflow:
    1. Parse command-line arguments.
    2. Set environment (seeds, GPU).
    3. Initialize paths.
    4. Set basin split.
    5. Load and preprocess data.
    6. Train dynamic-parameter model.

    Returns:
        tuple: (trainedModel, out)
    """
    args = parse_args()
    print("=" * 60)
    print("dPLXAJ dynamic-parameter model training start")
    print("=" * 60)
    print(f"Experiment: mode={args.pu_opt}, rnn={args.rnn_type.upper()}, forcing={args.for_type}")
    print(f"Train period: {args.train_start} - {args.train_end}")
    print(f"Model: batch_size={args.batch_size}, hidden_size={args.hidden_size}, rho={args.rho}")
    print(f"Dynamic parameters: {args.td_rep}")

    # Environment
    setup_environment(args)

    # Paths
    pathCamels = init_camels_path()

    rootDatabase, rootOut, gageinfo, puN, TrainLS, TrainInd, TestLS, TestInd, gageDic = setup_basin_split(args, pathCamels)
    print(f"Basin split: train={len(TrainLS)}, test={len(TestLS)}")

    # Data
    print("Loading and preprocessing data...")
    optData, statDict, forcTuple, yTrainIn, attrs = load_and_preprocess_data(
        args, rootDatabase, gageinfo, TrainLS, TrainInd)
    print("Data loading and preprocessing done")

    # Training
    print("Training dynamic-parameter model...")
    print(f"Dynamic config: params {', '.join(map(str, args.td_rep))} will vary over time")
    trainedModel, out = train_model(
        args, rootOut, puN, optData, statDict, forcTuple, yTrainIn, attrs)

    # Summary
    print("=" * 60)
    print("Dynamic-parameter model training completed")
    print("=" * 60)
    print(f"Model saved at: {out}")
    print(f"Dynamic parameters {args.td_rep} learned temporal variation")
    return trainedModel, out

if __name__ == "__main__":
    main()