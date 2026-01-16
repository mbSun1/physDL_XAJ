"""
dPLXAJ Dynamic Parameter Testing Script
=====================================
- Load trained dynamic-parameter XAJ model
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
    Parse command-line arguments for dynamic-parameter testing.

    Args must match the training script exactly so the model loads correctly.
    In dynamic mode td_opt defaults to True; XAJ has 12 parameters.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='dPLXAJ Dynamic Parameter Testing')

    # Experiment / basin split (must match training)
    parser.add_argument('--pu_opt', type=int, default=2, choices=[0, 1, 2],
                       help='Basin split: 0=ALL, 1=PUB, 2=PUR')
    parser.add_argument('--buff_opt', type=int, default=0, choices=[0, 1, 2],
                       help='Warmup: 0=first year for next, 1=repeat first year, 2=load extra year')
    parser.add_argument('--for_type', type=str, default='daymet',
                       choices=['daymet', 'nldas', 'maurer'],
                       help='Meteorological forcing: daymet, nldas, or maurer')

    # Model architecture (must match training)
    parser.add_argument('--rnn_type', type=str, default='bigru',
                       choices=['lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'],
                       help='RNN type')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Training batch size (basins per batch)')
    parser.add_argument('--rho', type=int, default=365,
                       help='Sequence length in days')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='RNN hidden size')
    parser.add_argument('--nfea', type=int, default=12,
                       help='XAJ parameter count (12)')

    # Time ranges (must match training)
    parser.add_argument('--train_start', type=str, default='19801001',
                       help='Training start (YYYYMMDD)')
    parser.add_argument('--train_end', type=str, default='20041001',
                       help='Training end (YYYYMMDD)')
    parser.add_argument('--test_start', type=str, default='20041001',
                       help='Test start (YYYYMMDD)')
    parser.add_argument('--test_end', type=str, default='20101001',
                       help='Test end (YYYYMMDD)')
    parser.add_argument('--inv_start', type=str, default='19801001',
                       help='Inversion start (YYYYMMDD)')
    parser.add_argument('--inv_end', type=str, default='20041001',
                       help='Inversion end (YYYYMMDD)')
    parser.add_argument('--buff_time', type=int, default=365,
                       help='Warmup length in days')

    # XAJ config (must match training)
    parser.add_argument('--routing', action='store_true', default=True,
                       help='Enable routing')
    parser.add_argument('--nmul', type=int, default=10,
                       help='Number of XAJ components')
    parser.add_argument('--comprout', action='store_true', default=False,
                       help='Component routing')
    parser.add_argument('--compwts', action='store_true', default=False,
                       help='Component weights')

    # Test config
    parser.add_argument('--test_batch', type=int, default=30,
                       help='Test batch size')
    parser.add_argument('--test_epoch', type=int, default=5,
                       help='Epoch index of model to load for testing')
    parser.add_argument('--random_seed', type=int, default=111111,
                       help='Random seed (must match training)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID (-1 for CPU)')
    parser.add_argument('--test_fold', type=int, default=1,
                       help='In PUB/PUR: fold index (1 = first list in PUBsplitLst.txt); <=0 to loop all folds.')

    # Dynamic-parameter config (td_opt default True)
    parser.add_argument('--td_opt', action='store_true', default=True,
                       help='Dynamic params: True=dynamic, False=static')
    parser.add_argument('--td_rep', type=int, nargs='+', default=[1, 5, 6, 7, 9],
                       help='Dynamic parameter indices (1-12), e.g. ke,wm,c,sm,ki')
    parser.add_argument('--dy_drop', type=float, default=0.0,
                       help='Dynamic dropout: 0=always dynamic, 1=always static')
    parser.add_argument('--sta_ind', type=int, default=-1,
                       help='Static param time step: -1=last, else index')

    return parser.parse_args()

def setup_environment(args):
    """
    Set up environment and random seed for reproducible testing.
    """
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

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
        pathCamels = OrderedDict(DB=local_camels, Out=local_outputs)
    else:
        pathCamels = OrderedDict(
            DB=os.path.join(os.path.sep, 'scratch', 'Camels'),
            Out=os.path.join(os.path.sep, 'data', 'rnnStreamflow')
        )
    return pathCamels

def setup_basin_split(args, pathCamels):
    """
    Set up basin train/test split by mode (ALL/PUB/PUR).
    Returns rootDatabase, rootOut, gageinfo, puN, tarIDLst, gageidLst.
    """
    rootDatabase = pathCamels['DB']
    rootOut = pathCamels['Out']

    gageinfo = camels.readGageInfo(rootDatabase)
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()

    if args.pu_opt == 0:  # ALL
        puN = 'ALL'
        tarIDLst = [gageidLst]

    elif args.pu_opt == 1:  # PUB
        puN = 'PUB'
        # load the subset ID
        # splitPath saves the basin ID of random groups
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst = json.load(fp)
        tarIDLst = testIDLst

    elif args.pu_opt == 2:  # PUR
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
    Load test data: set time ranges, met/attr vars, PET; apply warmup option.
    Returns Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad,
    varF, varFInv, attrnewLst, PETfull, tPETLst.
    """
    Ttrain = [int(args.train_start), int(args.train_end)]
    Tinv = [int(args.inv_start), int(args.inv_end)]
    Ttest = [int(args.test_start), int(args.test_end)]
    TtestLst = time.tRange2Array(Ttest)
    TtestLoad = [int(args.test_start), int(args.test_end)]
    print(f"Train range: {Ttrain[0]} - {Ttrain[1]}")
    print(f"Inversion range: {Tinv[0]} - {Tinv[1]}")
    print(f"Test range: {Ttest[0]} - {Ttest[1]}")

    if args.buff_opt == 2:
        print(f"Buff opt 2: loading extra {args.buff_time} days for warmup")
        sd = time.t2dt(Ttrain[0]) - dt.timedelta(days=args.buff_time)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]
        TinvLoad = [sdint, Tinv[1]]
        print(f"Extended train range: {TtrainLoad[0]} - {TtrainLoad[1]}")
    else:
        TtrainLoad = Ttrain
        TinvLoad = Tinv
        print(f"Buff opt {args.buff_opt}: using original time range")

    if args.for_type == 'daymet':
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
        print("Forcing: Daymet (prcp, tmean)")
    else:
        varF = ['prcp', 'tmax']
        varFInv = ['prcp', 'tmax']
        print(f"Forcing: {args.for_type.upper()} (prcp, tmax)")

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
    print(f"Attribute variables: {len(attrnewLst)}")

    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    varLstNL = ['PEVAP']
    usgsIdLst = gageinfo['id']

    if args.for_type == 'maurer':
        tPETRange = [19800101, 20090101]
    else:
        tPETRange = [19800101, 20150101]

    tPETLst = time.tRange2Array(tPETRange)
    PETDir = rootDatabase + '/pet_harg/' + args.for_type + '/'

    ntime = len(tPETLst)
    PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])

    for k in range(len(usgsIdLst)):
        dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
        PETfull[k, :, :] = dataTemp

    return Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad, varF, varFInv, attrnewLst, PETfull, tPETLst

def test_model_fold(args, rootDatabase, rootOut, puN, gageinfo, TestLS, TestInd, TrainLS, TrainInd, 
                   Ttest, TtestLoad, TtrainLoad, TinvLoad, 
                   varF, varFInv, attrnewLst, PETfull, tPETLst, testfold):
    """
    Test one fold of the dynamic-parameter XAJ model.
    
    This function tests one fold of the dynamic-parameter XAJ model, including:
    1. Loading the trained dynamic-parameter model
    2. Loading train and test data
    3. Preprocessing and normalization
    4. Warmup and prediction
    5. Saving test results
    
    Args:
        args: Parsed command-line arguments.
        rootOut: Root output directory.
        puN: Basin split mode name.
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
                testmodel, forcTestUN, PETTestUN)
    """
    # Build model path: rootOut/exp_name/exp_disp/exp_info/

    rnn_type_str = args.rnn_type.upper()
    tdRepS = [str(ix) for ix in args.td_rep]
    TDN = 'TD' + "_".join(tdRepS)
    
    testsave_path = (f'CAMELSDemo/dPLXAJ/{puN}/TDTestforc/{TDN}/{args.for_type}/BuffOpt{args.buff_opt}/'
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
    
    # Extract PET for test period
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
    
    # Normalize train series
    attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    # Normalize test series
    attrtest_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict, toNorm=True)
    attrtest_norm[np.isnan(attrtest_norm)] = 0.0
    seriestest_inv = np.concatenate([forcTestUN, PETTestUN], axis=2)
    seriestest_norm = camels.transNormbyDic(seriestest_inv, seriesvarLst, statDict, toNorm=True)
    seriestest_norm[np.isnan(seriestest_norm)] = 0.0
    
    print("- Data normalization completed")
    print("\n" + "=" * 50)
    print("Data loading completed!")
    print("=" * 50 + "\n")

    # Prepare model input data
    zTrain = series_norm
    xTrain = np.concatenate([forcUN, PETUN], axis=2)
    xTrain[np.isnan(xTrain)] = 0.0

    # Test buffer length for warmup (use full training period)
    TestBuff = xTrain.shape[1]
    runBUFF = 0
    testmodel.inittime = runBUFF
    testmodel.dydrop = args.dy_drop
    
    # Output file paths for predictions (Qr, Q0, Q1, Q2, ET)
    filePathLst = fileManager.namePred(
          testout, Ttest, 'All_Buff'+str(TestBuff), epoch=args.test_epoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])

    # Prepare test inputs depending on mode (dynamic parameters)
    # ALL: temporal generalization, warmup with train data
    # PUB/PUR: spatial generalization, warmup from test data only
    if args.pu_opt == 0:  # ALL
        testmodel.staind = TestBuff - 1
        zTest = np.concatenate([series_norm[:, -TestBuff:, :], seriestest_norm], axis=1)
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)
        xTestBuff = xTrain[:, -TestBuff:, :]
        xTest = np.concatenate([xTestBuff, xTest], axis=1)
        obs = obsTestUN[:, 0:, :]
    else:  # PUB/PUR
        testmodel.staind = TestBuff - 1
        zTest = seriestest_norm
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)
        obs = obsTestUN
    
    # Build final test inputs (dynamic mode): append basin attributes
    xTest[np.isnan(xTest)] = 0.0
    attrtest = attrtest_norm
    
    # Expand basin attributes along time dimension
    cTemp = np.repeat(
        np.reshape(attrtest, [attrtest.shape[0], 1, attrtest.shape[-1]]), zTest.shape[1], axis=1)
    zTest = np.concatenate([zTest, cTemp], 2)
    
    testTuple = (xTest, zTest)  # xTest: XAJ forcing; zTest: parameter-learning LSTM input
    
    print("\n" + "=" * 50)
    print(f"Using {args.rnn_type.upper()} as parameter prediction backbone")
    print(f"Dynamic parameter indices: {args.td_rep}")
    print("=" * 50 + "\n")
    
    print("=" * 50)
    print("Starting model testing")
    print("=" * 50)

    train.testModel(
        testmodel, testTuple, c=None, batchSize=args.test_batch, filePathLst=filePathLst)

    # Read prediction results
    pred_time_len = obs.shape[1] + TestBuff - runBUFF
    dataPred = np.full(
        (obs.shape[0], pred_time_len, len(filePathLst)),
        np.nan,
        dtype=float
    )
    for k, filePath in enumerate(filePathLst):
        pred_df = pd.read_csv(
            filePath,
            header=None,
            comment="(",
            engine="python",
            dtype=str
        )
        pred_arr = pd.to_numeric(pred_df.stack(), errors="coerce").unstack().to_numpy()
        if pred_arr.shape[1] > pred_time_len:
            raise ValueError(
                f"Prediction file {os.path.basename(filePath)} has time length {pred_arr.shape[1]} "
                f"exceeding expected length {pred_time_len}"
            )
        dataPred[:, -pred_arr.shape[1]:, k] = pred_arr

    return dataPred, obs, TestLS, TestBuff, runBUFF, testout, filePathLst, testmodel, forcTestUN, PETTestUN

def save_test_results(args, rootOut, puN, Ttrain, Ttest, TestBuff, testmodel, evaDict, obstestALL, predtestALL):
    """
    Save dynamic-parameter test results.

    Creates the output directory, saves evaluation metrics, observations,
    and predictions.

    Args:
        args: Parsed command-line arguments.
        rootOut: Root output directory.
        puN: Basin split mode name.
        Ttrain: Training time range.
        Ttest: Test time range.
        TestBuff: Test buffer length.
        testmodel: Tested model object.
        evaDict: Evaluation metrics dictionary.
        obstestALL: Observations for all test basins.
        predtestALL: Predictions for all test basins.
        
    Returns:
        str: Output path.
    """
    rnn_type_str = args.rnn_type.upper()
    seStr = (f'Train{Ttrain[0]}_{Ttrain[1]}Test{Ttest[0]}_{Ttest[1]}'
             f'Buff{TestBuff}Staind{testmodel.staind}_{rnn_type_str}')
    
    tdRepS = [str(ix) for ix in args.td_rep]
    TDN = 'TD' + "_".join(tdRepS)
    testsave_path = (f'CAMELSDemo/dPLXAJ/{puN}/TDTestforc/{TDN}/{args.for_type}/BuffOpt{args.buff_opt}/'
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

def process_xaj_parameters(outpath, logtestIDLst, filePathLst, test_start_date=19951001, test_end_date=20101001):
    """
    Process XAJ parameters (dynamic mode).
    
    Steps:
    1. Find XAJ parameter files
    2. Load and merge all batches (along basin dimension)
    3. Save merged parameters
    4. Print parameter statistics
    5. Save parameters to CSV files
    6. Generate dynamic parameter time series plots
    
    Args:
        outpath: Output directory.
        logtestIDLst: Test basin ID list.
        filePathLst: Prediction result file paths.
        test_start_date: Test start date.
        test_end_date: Test end date.
    """
    print("\n" + "=" * 50)
    print("XAJ Parameters Info")
    print("=" * 50)

    # Find XAJ parameter files (dynamic: XAJTD)
    xaj_params_files = []
    for file in os.listdir(os.path.dirname(filePathLst[0])):
        if file.startswith('phy_params_XAJTD') and file.endswith('.npy'):
            xaj_params_files.append(file)

    if xaj_params_files:
        print(f"Found {len(xaj_params_files)} XAJ parameter files")
        
        # Load all batches of XAJ parameters
        all_xaj_params = []
        for file in sorted(xaj_params_files):
            file_path = os.path.join(os.path.dirname(filePathLst[0]), file)
            xaj_params = np.load(file_path)
            all_xaj_params.append(xaj_params)
            print(f"- {file}: shape {xaj_params.shape}")
        
        # Merge all batches along basin dimension (axis=1)
        if len(all_xaj_params) > 1:
            combined_xaj_params = np.concatenate(all_xaj_params, axis=1)
        else:
            combined_xaj_params = all_xaj_params[0]
        
        print(f"\nCombined XAJ parameter shape: {combined_xaj_params.shape}")
        
        # Save merged parameters
        combined_params_file = os.path.join(outpath, 'xaj_parameters_dynamic.npy')
        np.save(combined_params_file, combined_xaj_params)
        print(f"XAJ parameters saved to: {combined_params_file}")
        
        # Print parameter statistics
        print(f"\nXAJ Parameters Statistics:")
        print(f"- Number of time steps: {combined_xaj_params.shape[0]}")
        print(f"- Number of basins: {combined_xaj_params.shape[1]}")
        print(f"- Number of parameters: {combined_xaj_params.shape[2]}")
        print(f"- Number of components: {combined_xaj_params.shape[3]}")
        print(f"- Parameter range: [{combined_xaj_params.min():.4f}, {combined_xaj_params.max():.4f}]")
        print(f"- Parameter mean: {combined_xaj_params.mean():.4f}")
        print(f"- Parameter std: {combined_xaj_params.std():.4f}")
        
        # Save XAJ parameters to CSV files
        xaj_csv_dir = save_results_plot.save_xaj_parameters(outpath, logtestIDLst, combined_xaj_params, model_type='dynamic')
        print(f"XAJ parameter CSV files saved to: {xaj_csv_dir}")
        
        # Generate dynamic parameter time series plots
        param_plots_dir = save_results_plot.plot_dynamic_parameters(outpath, logtestIDLst, combined_xaj_params, model_type='dynamic', max_basins=20, test_start_date=test_start_date, test_end_date=test_end_date)
        print(f"Dynamic parameter plots saved to: {param_plots_dir}")
        
    else:
        print("No XAJ parameter files found")

def run_all_folds_test(args, rootDatabase, rootOut, puN, gageinfo, gageidLst, tarIDLst, 
                      Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad,
                      varF, varFInv, attrnewLst, PETfull, tPETLst):
    """
    Run tests for all folds (dynamic-parameter mode).
    
    Args:
        args: Parsed command-line arguments.
        rootOut: Root output directory.
        puN: Basin split mode name.
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
        tuple: (predtestALL, obstestALL, logtestIDLst, TestBuff, testmodel,
                filePathLst, forcTestUN_all, PETTestUN_all)
    """
    # Initialize result matrices
    predtestALL = np.full([len(gageidLst), len(TtestLst), 5], np.nan)
    obstestALL = np.full([len(gageidLst), len(TtestLst), 1], np.nan)
    
    # Store test-period forcing data (for later plotting)
    forcTestUN_all = None
    PETTestUN_all = None

    # Decide which folds to test: >0 for specific fold, else loop all
    if args.test_fold is not None and args.test_fold > 0:
        fold_indices = [args.test_fold]
    else:
        fold_indices = list(range(1, len(tarIDLst) + 1))

    nstart = 0
    logtestIDLst = []
    for testfold in fold_indices:
        if testfold < 1 or testfold > len(tarIDLst):
            raise ValueError(f"test_fold={testfold} out of valid range 1~{len(tarIDLst)}")
        TestLS = tarIDLst[testfold - 1]
        TestInd = [gageidLst.index(j) for j in TestLS]
        if args.pu_opt == 0:  # Train and test on ALL basins
            TrainLS = gageidLst
            TrainInd = [gageidLst.index(j) for j in TrainLS]
        else:
            TrainLS = list(set(gageinfo['id'].tolist()) - set(TestLS))
            TrainInd = [gageidLst.index(j) for j in TrainLS]

        nbasin = len(TestLS) # number of basins for testing

        # Test model
        dataPred, obs, TestLS, TestBuff, runBUFF, testout, filePathLst, testmodel, forcTestUN, PETTestUN = test_model_fold(
            args, rootDatabase, rootOut, puN, gageinfo, TestLS, TestInd, TrainLS, TrainInd,
            Ttest, TtestLoad, TtrainLoad, TinvLoad,
            varF, varFInv, attrnewLst, PETfull, tPETLst, testfold)

        # Save predictions into global matrices
        predtestALL[nstart:nstart + nbasin, :, :] = dataPred[:, TestBuff-runBUFF:, :]
        obstestALL[nstart:nstart + nbasin, :, :] = obs
        
        # Save test-period forcing data (only once; same for all folds)
        if forcTestUN_all is None:
            forcTestUN_all = forcTestUN
            PETTestUN_all = PETTestUN
        
        nstart = nstart + nbasin
        logtestIDLst = logtestIDLst + TestLS

    return predtestALL, obstestALL, logtestIDLst, TestBuff, runBUFF, testmodel, filePathLst, forcTestUN_all, PETTestUN_all

def compute_evaluation_metrics(predtestALL, obstestALL, logtestIDLst):
    """
    Compute evaluation metrics for model performance.
    
    Computes NSE, RMSE, MAE and KGE for all test basins and returns a list
    of metric dictionaries.
    
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
    Generate results and plots (dynamic-parameter mode).
    
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
        forcTestUN_all: Test-period forcing data for all basins.
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
    process_xaj_parameters(outpath, logtestIDLst, filePathLst, Ttest[0], Ttest[1])

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
    Main entry point for dynamic-parameter testing.

    Runs the full testing pipeline:
    parse args, set environment, split basins, load data, test all folds,
    compute metrics, save outputs, and generate plots.
    """
    args = parse_args()
    print("=" * 60)
    print("dPLXAJ dynamic-parameter model testing start")
    print("=" * 60)
    print(f"Experiment: mode={args.pu_opt}, rnn={args.rnn_type.upper()}, forcing={args.for_type}")
    print(f"Test period: {args.test_start} - {args.test_end}")
    print(f"Model: batch_size={args.batch_size}, hidden_size={args.hidden_size}, rho={args.rho}")
    print(f"Test config: test_batch={args.test_batch}, epoch={args.test_epoch}")
    print(f"Dynamic params: {args.td_rep} (indices {', '.join(map(str, args.td_rep))})")

    # Environment
    setup_environment(args)

    # Paths
    pathCamels = init_camels_path()

    # Basin split
    rootDatabase, rootOut, gageinfo, puN, tarIDLst, gageidLst = setup_basin_split(args, pathCamels)
    print(f"Basin split done: folds={len(tarIDLst)}, total_basins={len(gageidLst)}")

    # Load data
    print("Loading test data...")
    Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad, varF, varFInv, attrnewLst, PETfull, tPETLst = load_test_data(
        args, rootDatabase, gageinfo)
    print("Test data loaded")

    # Test all folds
    print("Running model testing for all folds...")
    print(f"Dynamic params (time-varying): {', '.join(map(str, args.td_rep))}")
    predtestALL, obstestALL, logtestIDLst, TestBuff, runBUFF, testmodel, filePathLst, forcTestUN_all, PETTestUN_all = run_all_folds_test(
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
    outpath = save_test_results(args, rootOut, puN, Ttrain, Ttest, TestBuff, testmodel, evaDict, obstestALL, predtestALL)
    print("Test results saved")

    # Plots
    print("Generating plots...")
    generate_results_and_plots(args, outpath, logtestIDLst, evaDict, obstestALL, predtestALL, TestBuff, Ttest, filePathLst, forcTestUN_all)
    
    print("=" * 60)
    print("Dynamic-parameter model testing completed!")
    print("=" * 60)
    print(f"Results saved in: {outpath}")
    print(f"Number of test basins: {len(logtestIDLst)}")
    print(f"NSE median: {np.nanmedian(evaDict[0]['NSE']):.4f}")
    print(f"KGE median: {np.nanmedian(evaDict[0]['KGE']):.4f}")
    print("All plots generated")

if __name__ == "__main__":
    main()
