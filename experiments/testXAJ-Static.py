"""
dPLXAJ Static Parameter Testing Script
====================================
- Load trained static-parameter XAJ model
- Run prediction and evaluation on test dataset
- Generate result plots and statistics
"""

# =============================================================================
# Imports and setup
# =============================================================================
from hydroMLlib.utils import camels, fileManager, time, metrics
from hydroMLlib.model import train

import os
import numpy as np
import torch
import pandas as pd
import random
import json
import datetime as dt
import argparse
from collections import OrderedDict
# Result saving and plotting
import save_results_plot



def parse_args():
    """
    Parse command-line arguments for static-parameter testing.

    All arguments must match the training script to ensure correct model loading.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='dPLXAJ Static Parameter Testing')

    # =============================================================================
    # Experiment / basin split (must match training)
    # =============================================================================

    parser.add_argument('--pu_opt', type=int, default=0, choices=[0, 1, 2],
                       help='Basin split: 0=ALL, 1=PUB, 2=PUR')
    parser.add_argument('--buff_opt', type=int, default=0, choices=[0, 1, 2],
                       help='Warmup: 0=first year for next, 1=repeat first year, 2=load extra year')
    parser.add_argument('--for_type', type=str, default='daymet',
                       choices=['daymet', 'nldas', 'maurer'],
                       help='Meteorological forcing: daymet, nldas, or maurer')

    # =============================================================================
    # Model architecture (must match training)
    # =============================================================================

    parser.add_argument('--rnn_type', type=str, default='bilstm',
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

    # =============================================================================
    # Time ranges (must match training)
    # =============================================================================

    parser.add_argument('--train_start', type=str, default='19801001',
                       help='Training start date (YYYYMMDD)')
    parser.add_argument('--train_end', type=str, default='20041001',
                       help='Training end date (YYYYMMDD)')
    parser.add_argument('--test_start', type=str, default='20041001',
                       help='Test start date (YYYYMMDD)')
    parser.add_argument('--test_end', type=str, default='20101001',
                       help='Test end date (YYYYMMDD)')
    parser.add_argument('--inv_start', type=str, default='19801001',
                       help='Inversion start date (YYYYMMDD)')
    parser.add_argument('--inv_end', type=str, default='20041001',
                       help='Inversion end date (YYYYMMDD)')
    parser.add_argument('--buff_time', type=int, default=365,
                       help='Warmup length in days')

    # =============================================================================
    # XAJ configuration (must match training)
    # =============================================================================
    
    parser.add_argument('--routing', action='store_true', default=True,
                       help='Enable routing')
    parser.add_argument('--nmul', type=int, default=10,
                       help='Number of XAJ components')
    parser.add_argument('--comprout', action='store_true', default=False,
                       help='Component routing')
    parser.add_argument('--compwts', action='store_true', default=False,
                       help='Component weights')

    # =============================================================================
    # Testing configuration
    # =============================================================================

    parser.add_argument('--test_batch', type=int, default=50,
                       help='Test batch size')
    parser.add_argument('--test_epoch', type=int, default=20,
                       help='Epoch index of model to load for testing')
    parser.add_argument('--random_seed', type=int, default=111111,
                       help='Random seed (must match training)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID (-1 for CPU)')
    parser.add_argument( '--test_fold',type=int,default=1,
        help=('In PUB/PUR: fold index (1 = first list in PUBsplitLst.txt); <=0 to loop all folds.'))


    # =============================================================================
    # Dynamic-parameter arguments kept for compatibility with training scripts
    # =============================================================================
    
    parser.add_argument('--td_opt', action='store_true', default=False,
                       help='Parameter mode: False=static, True=dynamic')
    parser.add_argument('--td_rep', type=int, nargs='+', default=[1, 12],
                       help='Dynamic parameter indices (1-12)')
    parser.add_argument('--dy_drop', type=float, default=0.0,
                       help='Dynamic dropout: 0=always dynamic, 1=always static')
    parser.add_argument('--sta_ind', type=int, default=-1,
                       help='Static parameter time step: -1=last, else index')

    return parser.parse_args()

def setup_environment(args):
    """
    Set up environment and random seed for reproducible testing.
    """
    # =============================================================================
    # Random seeds
    # =============================================================================

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # =============================================================================
    # PyTorch deterministic settings
    # =============================================================================

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # =============================================================================
    # GPU device
    # =============================================================================

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        print("CUDA not available, using CPU")

def init_camels_path():
    """
    Initialize CAMELS data and output paths.

    Prefer project_root/Camels (or CAMELS) and project_root/outputs;
    fallback to /scratch/Camels and /data/rnnStreamflow.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    
    candidate_camels = [
        os.path.join(project_root, 'Camels'),
        os.path.join(project_root, 'CAMELS')
    ]
    local_camels = next((p for p in candidate_camels if os.path.isdir(p)), None)
    local_outputs = os.path.join(project_root, 'outputs')
    
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

    # Return unified path dict: {'DB': data root, 'Out': output root}
    return pathCamels
    
def setup_basin_split(args, pathCamels):
    """
    Set up basin split for training/testing under ALL/PUB/PUR modes.

    Args:
        args: Parsed command-line arguments (contains pu_opt, test_fold, etc.).
        pathCamels: Path config dict with 'DB' and 'Out'.
        
    Returns:
        tuple:
            - rootDatabase: CAMELS database root path.
            - rootOut: Model output root path.
            - gageinfo: Basin information dictionary.
            - puN: Basin split mode name ('ALL'/'PUB'/'PUR').
            - tarIDLst: List of test-basin ID lists (one per fold).
            - gageidLst: List of all basin IDs.
    """
    # CAMELS dataset paths
    rootDatabase = pathCamels['DB']
    rootOut = pathCamels['Out']

    # Read basin info
    gageinfo = camels.readGageInfo(rootDatabase)
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()

    if args.pu_opt == 0:  # ALL mode
        puN = 'ALL'
        tarIDLst = [gageidLst]

    elif args.pu_opt == 1:  # PUB mode
        puN = 'PUB'
        # load the subset ID
        # splitPath saves the basin ID of random groups
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst = json.load(fp)
        tarIDLst = testIDLst

    elif args.pu_opt == 2:  # PUR mode
        puN = 'PUR'
        # Divide CAMELS dataset into 7 PUR regions
        # get the id list of each region
        regionID = list()
        regionNum = list()
        regionDivide = [[1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17]] # seven regions
        for ii in range(len(regionDivide)):
            tempcomb = regionDivide[ii]
            tempregid = list()
            for ih in tempcomb:
                tempid = gageid[hucinfo==ih].tolist()
                tempregid = tempregid + tempid
            regionID.append(tempregid)
            regionNum.append(len(tempregid))
        tarIDLst = regionID

    return rootDatabase, rootOut, gageinfo, puN, tarIDLst, gageidLst

def load_test_data(args, rootDatabase, gageinfo):
    """
    Load test data and perform preprocessing.

    Includes:
    1. Setting time ranges (training, inversion, testing)
    2. Configuring meteorological and attribute variables
    3. Loading PET data
    4. Applying warmup options
    
    Args:
        args: Parsed command-line arguments.
        rootDatabase: CAMELS database root path.
        gageinfo: Basin information dictionary.
        
    Returns:
        tuple:
            - Ttrain: Training time range.
            - Tinv: Inversion time range.
            - Ttest: Test time range.
            - TtestLst: Test time stamps.
            - TtestLoad: Time range for loading test data.
            - TtrainLoad: Time range for loading training data.
            - TinvLoad: Time range for loading inversion data.
            - varF: Forcing variable names.
            - varFInv: Inversion forcing variable names.
            - attrnewLst: Attribute variable names.
            - PETfull: Full PET array.
            - tPETLst: PET time stamps.
    """
    # Time ranges: training, inversion, testing
    Ttrain = [int(args.train_start), int(args.train_end)]
    Tinv = [int(args.inv_start), int(args.inv_end)]
    Ttest = [int(args.test_start), int(args.test_end)]
    TtestLst = time.tRange2Array(Ttest)
    TtestLoad = [int(args.test_start), int(args.test_end)]
    print(f"Training range: {Ttrain[0]} - {Ttrain[1]}")
    print(f"Inversion range: {Tinv[0]} - {Tinv[1]}")
    print(f"Test range: {Ttest[0]} - {Ttest[1]}")
    
    # Apply warmup option
    if args.buff_opt == 2:
        print(f"Buff opt 2: loading extra {args.buff_time} days for warmup")
        sd = time.t2dt(Ttrain[0]) - dt.timedelta(days=args.buff_time)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]
        TinvLoad = [sdint, Tinv[1]]
        print(f"Extended training range: {TtrainLoad[0]} - {TtrainLoad[1]}")
    else:
        TtrainLoad = Ttrain
        TinvLoad = Tinv
        print(f"Buff opt {args.buff_opt}: using original time range")

    # =============================================================================
    # Meteorological variable configuration
    
    if args.for_type == 'daymet':
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
        print("Forcing source: Daymet (prcp, tmean)")
    else:
        varF = ['prcp', 'tmax']
        varFInv = ['prcp', 'tmax']
        print(f"Forcing source: {args.for_type.upper()} (prcp, tmax)")

    # Basin attribute variable list
    
    attrnewLst = [
        'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity',
        'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
        'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
        'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover',
        'root_depth_50', 'soil_depth_pelletier', 'soil_depth_statsgo',
        'soil_porosity', 'soil_conductivity', 'max_water_content',
        'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class',
        'glim_1st_class_frac', 'geol_2nd_class', 'glim_2nd_class_frac',
        'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
    ]
    print(f"Number of basin attributes: {len(attrnewLst)}")

    # Show data-loading banner
    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    # Load potential evapotranspiration (PET)

    varLstNL = ['PEVAP']
    usgsIdLst = gageinfo['id']
    
    # PET time range depends on forcing source
    if args.for_type == 'maurer':
        tPETRange = [19800101, 20090101]
    else:
        tPETRange = [19800101, 20150101]
    
    tPETLst = time.tRange2Array(tPETRange)
    
    # PET directory: Camels/pet_harg/{for_type}/
    PETDir = rootDatabase + '/pet_harg/' + args.for_type + '/'

    ntime = len(tPETLst)
    PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
    
    # Load PET for each basin
    for k in range(len(usgsIdLst)):
        dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
        PETfull[k, :, :] = dataTemp

    return Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad, varF, varFInv, attrnewLst, PETfull, tPETLst

def test_model_fold(args, rootDatabase, rootOut, puN, gageinfo, TestLS, TestInd, TrainLS, TrainInd, 
                   Ttest, TtestLoad, TtrainLoad, TinvLoad, 
                   varF, varFInv, attrnewLst, PETfull, tPETLst, testfold):
    """
    Test a single fold of the static-parameter XAJ model.
    
    Steps:
    1. Load trained model.
    2. Load train and test data.
    3. Preprocess and normalize data.
    4. Warm up model and run prediction.
    5. Save prediction results.
    
    Args:
        args: Parsed command-line arguments.
        rootOut: Model output root directory.
        puN: Basin mode name.
        gageinfo: Basin information dictionary.
        TestLS: Test basin ID list.
        TestInd: Test basin index list.
        TrainLS: Train basin ID list.
        TrainInd: Train basin index list.
        Ttest: Test time range.
        TtestLoad: Time range for loading test data.
        TtrainLoad: Time range for loading training data.
        TinvLoad: Time range for loading inversion data.
        varF: Forcing variable names.
        varFInv: Inversion forcing variable names.
        attrnewLst: Basin attribute variable names.
        PETfull: Full PET array.
        tPETLst: PET time stamps.
        testfold: Fold index.
        
    Returns:
        tuple: (dataPred, obs, TestLS, TestBuff, testout, filePathLst,
                forcTestUN, PETTestUN)
    """
    # Model path components

    rnn_type_str = args.rnn_type.upper()
    testsave_path = (f'CAMELSDemo/dPLXAJ/{puN}/Testforc/{args.for_type}/BuffOpt{args.buff_opt}/'
                    f'RMSE_para0.25/{args.random_seed}')
    
    foldstr = 'Fold' + str(testfold)
    exp_info = (f'T_{args.train_start}_{args.train_end}_BS_{args.batch_size}_HS_{args.hidden_size}_'
               f'RHO_{args.rho}_NF_{args.nfea}_Buff_{args.buff_time}_Mul_{args.nmul}_{rnn_type_str}')
    
    foldpath = os.path.join(rootOut, testsave_path, foldstr)
    testout = os.path.join(foldpath, rnn_type_str, exp_info)
    
    print("\n" + "=" * 50)
    print(f"Trying to load model from: {testout}")
    print("=" * 50 + "\n")
    
    if not os.path.isdir(testout):
        raise FileNotFoundError(f'Trained model directory not found: {testout}')
    
    model_file = os.path.join(testout, f'model_Ep{args.test_epoch}.pt')
    if not os.path.isfile(model_file):
        print(f"Warning: Cannot find model file for epoch {args.test_epoch}")
    
    testmodel = fileManager.loadModel(testout, epoch=args.test_epoch)

    dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=args.for_type,
                                     dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    print("- Training forcing data loaded")
    dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=args.for_type,
                                   dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)
    print("- Inversion training data loaded")
    dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=args.for_type,
                                    dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)
    print("- Test data loaded")

    # Discharge unit conversion: ft3/s -> mm/day
    areas = gageinfo['area'][TestInd]  # km^2
    temparea = np.tile(areas[:, None, None], (1, obsTestUN.shape[1], 1))
    obsTestUN = (obsTestUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3

    # Extract PET for train / inversion / test periods
    TtrainLst = time.tRange2Array(TtrainLoad)
    TinvLst = time.tRange2Array(TinvLoad)
    TtestLoadLst = time.tRange2Array(TtestLoad)

    _, _, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    PETUN = PETfull[:, ind2, :]
    PETUN = PETUN[TrainInd, :, :]
    
    _, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInvUN = PETfull[:, ind2inv, :]
    PETInvUN = PETInvUN[TrainInd, :, :]
    
    _, _, ind2test = np.intersect1d(TtestLoadLst, tPETLst, return_indices=True)
    PETTestUN = PETfull[:, ind2test, :]
    PETTestUN = PETTestUN[TestInd, :, :]

    print("- Evapotranspiration data loaded")
    
    statFile = os.path.join(testout, 'statDict.json')
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)
    print("- Statistics loaded")
    
    # Prepare inversion series (for parameter learning)
    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    seriesvarLst = varFInv + ['pet']
    
    # Normalize training data
    attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    # Normalize test data
    attrtest_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict, toNorm=True)
    attrtest_norm[np.isnan(attrtest_norm)] = 0.0
    seriestest_inv = np.concatenate([forcTestUN, PETTestUN], axis=2)
    seriestest_norm = camels.transNormbyDic(seriestest_inv, seriesvarLst, statDict, toNorm=True)
    seriestest_norm[np.isnan(seriestest_norm)] = 0.0
    
    print("- Data normalization completed")
    print("\n" + "=" * 50)
    print("Data loading completed!")
    print("=" * 50 + "\n")

    # Prepare model inputs
    zTrain = series_norm
    xTrain = np.concatenate([forcUN, PETUN], axis=2)
    xTrain[np.isnan(xTrain)] = 0.0

    # Test buffer length for warmup (use full training period)
    TestBuff = xTrain.shape[1]
    
    # Prediction output paths (Qr, Q0, Q1, Q2, ET)
    filePathLst = fileManager.namePred(
          testout, Ttest, 'All_Buff'+str(TestBuff), epoch=args.test_epoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])

    # Prepare test inputs depending on mode
    # ALL: temporal generalization, warmup with training data
    # PUB/PUR: spatial generalization, warmup from test data only
    if args.pu_opt == 0:  # ALL
        zTest = series_norm
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)
        xTestBuff = xTrain[:, -TestBuff:, :]
        xTest = np.concatenate([xTestBuff, xTest], axis=1)
        obs = obsTestUN[:, 0:, :]
    else:  # PUB/PUR
        zTest = seriestest_norm[:, 0:TestBuff, :]
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)
        obs = obsTestUN[:, TestBuff:, :]
    
    # Set model initialization time
    testmodel.inittime = TestBuff

    # Build final test inputs: append attributes to time series
    # Replace NaNs
    xTest[np.isnan(xTest)] = 0.0
    attrtest = attrtest_norm
    
    cTemp = np.repeat(
        np.reshape(attrtest, [attrtest.shape[0], 1, attrtest.shape[-1]]), zTest.shape[1], axis=1)
    zTest = np.concatenate([zTest, cTemp], 2)
    
    # Final test tuple: xTest (XAJ forcing), zTest (LSTM parameter input)
    testTuple = (xTest, zTest)
    
    # Show model info
    print("\n" + "=" * 50)
    print(f"Using {args.rnn_type.upper()} as parameter prediction backbone")
    print("=" * 50 + "\n")

    print("=" * 50)
    print("Starting model testing")
    print("=" * 50)

    train.testModel(
        testmodel, testTuple, c=None, batchSize=args.test_batch, filePathLst=filePathLst)

    # Read prediction results
    dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(
            filePath, dtype=float, header=None).values

    return dataPred, obs, TestLS, TestBuff, testout, filePathLst, forcTestUN, PETTestUN

def save_test_results(args, rootOut, puN, Ttrain, Ttest, TestBuff, evaDict, obstestALL, predtestALL):
    """
    Save test results (static-parameter mode).
    
    Creates the output directory and saves metrics, observations, and predictions.
    
    Args:
        args: Parsed command-line arguments.
        rootOut: Model output root directory.
        puN: Basin mode name.
        Ttrain: Training time range.
        Ttest: Test time range.
        TestBuff: Test buffer length.
        evaDict: Evaluation metrics dictionary.
        obstestALL: Observations for all test basins.
        predtestALL: Predictions for all test basins.
        
    Returns:
        str: Output path.
    """
    rnn_type_str = args.rnn_type.upper()
    seStr = (f'Train{Ttrain[0]}_{Ttrain[1]}Test{Ttest[0]}_{Ttest[1]}'
             f'Buff{TestBuff}Nmul{args.nmul}_{rnn_type_str}')
    testsave_path = (f'CAMELSDemo/dPLXAJ/{puN}/Testforc/{args.for_type}/BuffOpt{args.buff_opt}/'
                    f'RMSE_para0.25/{args.random_seed}')
    outpath = os.path.join(rootOut, testsave_path, seStr)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    # Save metrics, observations, and predictions
    EvaFile = os.path.join(outpath, f'Eva{args.test_epoch}.npy')
    np.save(EvaFile, evaDict)

    obsFile = os.path.join(outpath, 'obs.npy')
    np.save(obsFile, obstestALL)

    predFile = os.path.join(outpath, f'pred{args.test_epoch}.npy')
    np.save(predFile, predtestALL)

    return outpath

def process_xaj_parameters(outpath, logtestIDLst, filePathLst):
    """
    Process static XAJ parameters.
    
    Steps:
    1. Find XAJ parameter files.
    2. Load and merge all batches.
    3. Save merged parameters.
    4. Print parameter statistics.
    5. Save parameters to CSV files.
    
    Args:
        outpath: Output directory.
        logtestIDLst: Test basin ID list.
        filePathLst: Prediction file paths.
    """
    print("\n" + "=" * 50)
    print("XAJ Parameters Info")
    print("=" * 50)

    # Find XAJ parameter files
    xaj_params_files = []
    for file in os.listdir(os.path.dirname(filePathLst[0])):
        if file.startswith('phy_params_XAJ') and file.endswith('.npy'):
            xaj_params_files.append(file)
    
    if xaj_params_files:
        print(f"Found {len(xaj_params_files)} XAJ parameter files")
        
        # Load all batches of XAJ parameters
        all_xaj_params = []
        for file in sorted(xaj_params_files):
            file_path = os.path.join(os.path.dirname(filePathLst[0]), file)
            xaj_params = np.load(file_path)
            all_xaj_params.append(xaj_params)
            print(f"  {file}: shape {xaj_params.shape}")
        
        # Merge all batches
        if len(all_xaj_params) > 1:
            combined_xaj_params = np.concatenate(all_xaj_params, axis=0)
        else:
            combined_xaj_params = all_xaj_params[0]
        
        print(f"\nCombined XAJ parameter shape: {combined_xaj_params.shape}")
        
        # Save merged parameters
        combined_params_file = os.path.join(outpath, 'xaj_parameters_static.npy')
        np.save(combined_params_file, combined_xaj_params)
        print(f"XAJ parameters saved to: {combined_params_file}")
        
        # Print parameter statistics
        print(f"\nXAJ Parameters Statistics:")
        print(f"- Number of basins: {combined_xaj_params.shape[0]}")
        print(f"- Number of parameters: {combined_xaj_params.shape[1]}")
        print(f"- Number of components: {combined_xaj_params.shape[2]}")
        print(f"- Parameter range: [{combined_xaj_params.min():.4f}, {combined_xaj_params.max():.4f}]")
        print(f"- Parameter mean: {combined_xaj_params.mean():.4f}")
        print(f"- Parameter std: {combined_xaj_params.std():.4f}")
        
        # Save XAJ parameters to CSV files
        from save_results_plot import save_xaj_parameters
        xaj_csv_dir = save_xaj_parameters(outpath, logtestIDLst, combined_xaj_params, model_type='static')
        print(f"XAJ parameter CSV files saved to: {xaj_csv_dir}")
        
    else:
        print("No XAJ parameter files found")

def run_all_folds_test(args, rootDatabase, rootOut, puN, gageinfo, gageidLst, tarIDLst, 
                      Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad,
                      varF, varFInv, attrnewLst, PETfull, tPETLst):
    """
    Run tests for all folds (static-parameter mode).
    
    Args:
        args: Parsed command-line arguments.
        rootOut: Model output root directory.
        puN: Basin mode name.
        gageinfo: Basin information dictionary.
        gageidLst: All basin IDs.
        tarIDLst: List of test-basin ID lists (one per fold).
        Ttest: Test time range.
        TtestLst: Test time stamps.
        TtestLoad: Time range for loading test data.
        TtrainLoad: Time range for loading training data.
        TinvLoad: Time range for loading inversion data.
        varF: Forcing variable names.
        varFInv: Inversion forcing variable names.
        attrnewLst: Basin attribute variable names.
        PETfull: Full PET array.
        tPETLst: PET time stamps.
        
    Returns:
        tuple: (predtestALL, obstestALL, logtestIDLst, TestBuff,
                filePathLst, forcTestUN_all, PETTestUN_all)
    """
    # Initialize result matrices
    predtestALL = np.full([len(gageidLst), len(TtestLst), 5], np.nan)
    obstestALL = np.full([len(gageidLst), len(TtestLst), 1], np.nan)
    
    # Store test-period forcing data (for later plotting)
    forcTestUN_all = None
    PETTestUN_all = None

    # Loop over all folds
    nstart = 0
    logtestIDLst = []
    for ifold in range(1, len(tarIDLst)+1):
        testfold = ifold
        TestLS = tarIDLst[testfold - 1]
        TestInd = [gageidLst.index(j) for j in TestLS]
        if args.pu_opt == 0:  # Train and test on ALL basins
            TrainLS = gageidLst
            TrainInd = [gageidLst.index(j) for j in TrainLS]
        else:
            TrainLS = list(set(gageinfo['id'].tolist()) - set(TestLS))
            TrainInd = [gageidLst.index(j) for j in TrainLS]

        nbasin = len(TestLS) # number of basins for testing

    # Test model for this fold
        dataPred, obs, TestLS, TestBuff, testout, filePathLst, forcTestUN, PETTestUN = test_model_fold(
            args, rootDatabase, rootOut, puN, gageinfo, TestLS, TestInd, TrainLS, TrainInd,
            Ttest, TtestLoad, TtrainLoad, TinvLoad,
            varF, varFInv, attrnewLst, PETfull, tPETLst, testfold)

        # Save predictions into global matrices
        
        predtestALL[nstart:nstart + nbasin, :, :] = dataPred
        obstestALL[nstart:nstart + nbasin, :, :] = obs
        
        # Save test-period forcing data once (same for all folds)
        if forcTestUN_all is None:
            forcTestUN_all = forcTestUN
            PETTestUN_all = PETTestUN
        
        nstart = nstart + nbasin
        logtestIDLst = logtestIDLst + TestLS

    return predtestALL, obstestALL, logtestIDLst, TestBuff, filePathLst, forcTestUN_all, PETTestUN_all

def compute_evaluation_metrics(predtestALL, obstestALL, logtestIDLst):
    """
    Compute evaluation metrics for model performance.
    
    Computes NSE, RMSE, MAE, and KGE for all test basins.
    
    Args:
        predtestALL: Predictions for all test basins.
        obstestALL: Observations for all test basins.
        logtestIDLst: Test basin ID list.
        
    Returns:
        list: List containing one evaluation-metric dictionary.
    """
    # Compute basic metrics for Q0 (runoff)
    evaDict = [metrics.statError(predtestALL[:,:,0], obstestALL.squeeze())]
    
    # Compute KGE if not already present
    if 'KGE' not in evaDict[0]:
        print("Computing KGE metrics for all basins...")
        kge_values = []
        for i in range(len(logtestIDLst)):
            obs = obstestALL[i, :, 0]
            sim = predtestALL[i, :, 0]
            kge = save_results_plot.calculate_kge(obs, sim)
            kge_values.append(kge)
        evaDict[0]['KGE'] = np.array(kge_values)

    return evaDict

def generate_results_and_plots(args, outpath, logtestIDLst, evaDict, obstestALL, predtestALL, 
                              TestBuff, Ttest, filePathLst, forcTestUN_all):
    """
    Generate results and plots (static-parameter mode).
    
    Args:
        args: Parsed command-line arguments.
        outpath: Output directory.
        logtestIDLst: Test basin ID list.
        evaDict: Evaluation metrics (list of dicts).
        obstestALL: Observations for all test basins.
        predtestALL: Predictions for all test basins.
        TestBuff: Test buffer length.
        Ttest: Test time range.
        filePathLst: Prediction file paths.
        forcTestUN_all: Test-period forcing for all basins.
    """
    # Summary
    print('=' * 50)
    print('Testing completed!')
    print('=' * 50)
    print('Evaluation results saved in:\n', outpath)
    print('For all basins, NSE median:', np.nanmedian(evaDict[0]['NSE']))

    # Boxplot for all basins
    print(f"Generating boxplot for all {len(logtestIDLst)} basins...")
    save_results_plot.plot_metrics_boxplot(outpath, evaDict[0], args.rnn_type)

    # Process XAJ parameters
    process_xaj_parameters(outpath, logtestIDLst, filePathLst)

    # Save basin data and generate streamflow comparison plots
    print("\nSaving basin data and generating streamflow comparison plots...")
    
    save_results_plot.save_results_and_plot(
        outpath,
        args.test_epoch,
        logtestIDLst,
        forcTestUN_all,
        obstestALL,
        predtestALL,
        max_plots=None,
        test_start_date=Ttest[0]
    )

    print("Result saving and plot generation completed!")
    print(f"- CSV data saved in: {os.path.join(outpath, 'csv_results')}")
    print(f"- Streamflow plots saved in: {os.path.join(outpath, 'plots')}")   

def main():
    """
    Main entry point for static-parameter testing.

    Coordinates the full testing workflow:
    1. Parse command-line arguments.
    2. Set environment (seeds, GPU).
    3. Set basin split.
    4. Load test data.
    5. Run model testing for all folds.
    6. Compute evaluation metrics.
    7. Save results.
    8. Generate plots and summaries.
    """
    args = parse_args()
    print("=" * 60)
    print("dPLXAJ static-parameter model testing start")
    print("=" * 60)
    print(f"Experiment: mode={args.pu_opt}, rnn={args.rnn_type.upper()}, forcing={args.for_type}")
    print(f"Test period: {args.test_start} - {args.test_end}")
    print(f"Model: batch_size={args.batch_size}, hidden_size={args.hidden_size}, rho={args.rho}")
    print(f"Test config: test_batch={args.test_batch}, epoch={args.test_epoch}")
    
    # Environment
    setup_environment(args)

    # Paths
    pathCamels = init_camels_path()

    # Basin split
    rootDatabase, rootOut, gageinfo, puN, tarIDLst, gageidLst = setup_basin_split(args, pathCamels)
    print(f"Basin split done: folds={len(tarIDLst)}, total_basins={len(gageidLst)}")
    
    # Load test data
    print("Loading test data...")
    Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad, varF, varFInv, attrnewLst, PETfull, tPETLst = load_test_data(
        args, rootDatabase, gageinfo)
    print("Test data loaded")
    
    # Run tests for all folds
    print("Running model testing for all folds...")
    predtestALL, obstestALL, logtestIDLst, TestBuff, filePathLst, forcTestUN_all, PETTestUN_all = run_all_folds_test(
        args, rootDatabase, rootOut, puN, gageinfo, gageidLst, tarIDLst,
        Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad,
        varF, varFInv, attrnewLst, PETfull, tPETLst)
    print("All folds testing completed")
    
    # Metrics
    print("Computing evaluation metrics...")
    evaDict = compute_evaluation_metrics(predtestALL, obstestALL, logtestIDLst)
    print("Evaluation metrics computed")
    
    # Save results
    print("Saving test results...")
    outpath = save_test_results(args, rootOut, puN, Ttrain, Ttest, TestBuff, evaDict, obstestALL, predtestALL)
    print("Test results saved")
    
    # Plots and summaries
    print("Generating result plots...")
    generate_results_and_plots(args, outpath, logtestIDLst, evaDict, obstestALL, predtestALL, TestBuff, Ttest, filePathLst, forcTestUN_all)
    
    print("=" * 60)
    print("Static-parameter model testing completed!")
    print("=" * 60)
    print(f"Results saved in: {outpath}")
    print(f"Number of test basins: {len(logtestIDLst)}")
    print(f"NSE median: {np.nanmedian(evaDict[0]['NSE']):.4f}")
    print(f"KGE median: {np.nanmedian(evaDict[0]['KGE']):.4f}")
    print("All plots generated")

if __name__ == "__main__":
    main()