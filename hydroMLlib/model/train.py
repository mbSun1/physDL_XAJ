# -*- coding: utf-8 -*-
"""
模型训练和测试模块 (Model Training and Testing Module)

本模块包含用于训练和测试深度学习水文模型的核心函数，面向 XAJ（静态/动态）参数模型。
主要功能包括：
1. 模型训练 (trainModel) - 支持静态与动态 XAJ 参数模型
2. 模型测试 (testModel) - 批量测试和结果保存
3. 模型加载 (loadModel) - 从检查点恢复模型
4. 数据采样 (randomIndex, selectSubset) - 训练数据随机采样
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
    训练深度学习水文模型 (Train Deep Learning Hydrological Model)
    
    本函数用于训练 XAJ 模型的静态与动态参数变体，支持GPU加速和自动检查点保存。
    
    参数 (Parameters):
    ----------
    model : torch.nn.Module
        要训练的模型，支持 xaj_static.MultiInv_XAJModel 与 xaj_dynamic.MultiInv_XAJTDModel
    x : numpy.ndarray or tuple
        输入数据，形状为 (ngrid, nt, nx) 或 (x, z) 元组
        - x: 气象强迫数据 (降雨、温度等)
        - z: 额外输入数据 (静态属性等)
    y : numpy.ndarray
        目标数据，形状为 (ngrid, nt, ny)，通常是径流观测值
    c : numpy.ndarray or None
        常数输入数据，形状为 (ngrid, nc)，如流域静态属性
    lossFun : torch.nn.Module
        损失函数，支持 loss_module.NSELossBatch, loss_module.NSESqrtLossBatch 等
    nEpoch : int, default=500
        训练轮数
    miniBatch : list, default=[100, 30]
        批次大小和时间步长 [batchSize, rho]
    saveEpoch : int, default=100
        模型保存间隔（每多少轮保存一次）
    saveFolder : str or None, default=None
        模型保存目录，None表示不保存
    bufftime : int, default=0
        缓冲区时间步数，用于模型初始化
    
    返回 (Returns):
    -------
    model : torch.nn.Module
        训练完成的模型
    
    注意 (Notes):
    -----
    - 支持GPU自动检测和切换
    - 包含数值稳定性检查和NaN/Inf检测
    - 提供详细的训练进度显示和性能监控
    - 自动计算每轮迭代次数以确保充分训练
    """
    print("=" * 50)
    print("开始模型训练 (Starting Model Training)")
    print("=" * 50)
    
    # 解析批次参数 (Parse batch parameters)
    batchSize, rho = miniBatch
    
    # 处理输入数据格式 (Handle input data format)
    # x: 主要输入数据; z: 额外输入数据; y: 目标数据; c: 常数输入数据
    if type(x) is tuple or type(x) is list:
        x, z = x  # 解包元组或列表格式的输入
    
    # 获取数据维度信息 (Get data dimension information)
    ngrid, nt, nx = x.shape  # 网格数, 时间步数, 输入变量数
    if c is not None:
        nx = nx + c.shape[-1]  # 如果有常数输入，增加变量维度
    
    # 调整批次大小 (Adjust batch size)
    if batchSize >= ngrid:
        # 如果批次大小大于等于总网格数，使用全量数据
        batchSize = ngrid

    # 计算每轮迭代次数 (Calculate iterations per epoch)
    # 基于概率理论，确保每个样本被选中的概率达到99%
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    
    # 如果模型有条件时间步移除功能，调整迭代次数计算
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            # 考虑条件时间步的影响，调整有效时间步长
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    # 显示训练配置信息 (Display training configuration)
    print(f"训练信息 (Training Info):")
    print(f"- 数据形状 (Data Shape): {ngrid} 网格 x {nt} 时间步 x {nx} 变量")
    print(f"- 批次大小 (Batch Size): {batchSize}")
    print(f"- 时间步长 (Time Steps): {rho}")
    print(f"- 每轮迭代数 (Iterations per Epoch): {nIterEp}")
    print(f"- 总训练轮数 (Total Epochs): {nEpoch}")
    print(f"- 缓冲区时间 (Buffer Time): {bufftime}")
    print("-" * 50)

    # 设备配置 (Device configuration)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # 将模型和损失函数移动到GPU
        lossFun = lossFun.to(device)
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("使用CPU训练 (Using CPU)")

    # 优化器配置 (Optimizer configuration)
    optim = torch.optim.Adadelta(model.parameters())  # 使用Adadelta优化器
    model.zero_grad()  # 清零梯度
    
    # 训练日志文件设置 (Training log file setup)
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')  # 训练过程记录文件
        rf = open(runFile, 'w+')  # 打开日志文件
    # 开始训练循环 (Start training loop)
    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0  # 当前轮次累计损失
        t0 = time.time()  # 记录轮次开始时间
        
        # 每轮次内的迭代循环 (Iteration loop within each epoch)
        for iIter in range(0, nIterEp):
            # 训练迭代 (Training iterations)
            # 支持 XAJ 的静态与动态参数变体
            if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
                # 随机采样训练数据 (Random sampling of training data)
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                
                # 物理模型：使用缓冲区时间
                xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
                
                # 选择目标数据子集 (Select target data subset)
                yTrain = selectSubset(y, iGrid, iT, rho)
                
                # 根据模型类型选择额外输入数据 (Select additional input data based on model type)
                if type(model) is xaj_static.MultiInv_XAJModel:
                    # 静态 XAJ：使用常数输入
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c)
                else:  # xaj_dynamic.MultiInv_XAJTDModel
                    # 动态 XAJ：使用常数输入与缓冲区
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
                # 数值稳定性检查 (Numerical stability check)
                # 检查输入数据是否包含NaN或Inf值
                if torch.isnan(xTrain).any() or torch.isinf(xTrain).any():
                    print(f"Warning: xTrain contains NaN/Inf at iteration {iIter}")
                if torch.isnan(zTrain).any() or torch.isinf(zTrain).any():
                    print(f"Warning: zTrain contains NaN/Inf at iteration {iIter}")
                
                # 前向传播 (Forward pass)
                yP = model(xTrain, zTrain)
                
                # 检查模型输出数据 (Check model output data)
                if torch.isnan(yP).any() or torch.isinf(yP).any():
                    print(f"Warning: yP contains NaN/Inf at iteration {iIter}")
                    print(f"yP stats: min={yP.min()}, max={yP.max()}, mean={yP.mean()}")
            else:
                # 如果模型类型不在支持列表中，抛出异常
                Exception('unknown model')
            # 考虑缓冲区时间用于初始化 (Consider buffer time for initialization)
            # 注释掉的代码：如果bufftime > 0，从输出中移除缓冲区部分
            # if bufftime > 0:
            #     yP = yP[bufftime:,:,:]
            
            # 计算损失函数 (Calculate loss function)
            # 仅对总径流 Qr（第0列）计损失
            if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
                yP_loss = yP[:, :, 0:1]
                yT_loss = yTrain[:, :, 0:1]
            else:
                yP_loss = yP
                yT_loss = yTrain

            # 根据损失函数类型选择不同的计算方式
            if type(lossFun) in [loss_module.NSELossBatch, loss_module.NSESqrtLossBatch]:
                # NSE类损失需要额外的网格索引信息
                loss = lossFun(yP_loss, yT_loss, iGrid)
            else:
                # 标准损失函数
                loss = lossFun(yP_loss, yT_loss)
            # 反向传播和参数更新 (Backward pass and parameter update)
            loss.backward()  # 计算梯度
            optim.step()     # 更新模型参数
            model.zero_grad()  # 清零梯度，为下次迭代做准备
            lossEp = lossEp + loss.item()  # 累计损失值
            
            # 损失值数值稳定性检查 (Loss value numerical stability check)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到NaN/Inf损失值在迭代 {iIter}")
                print(f"损失值: {loss.item()}")
                break  # 如果损失值异常，提前结束训练

            # 增强型进度条显示 (Enhanced progress bar display)
            pct = int((iIter + 1) * 100 / max(1, nIterEp))  # 计算完成百分比
            filled = pct // 2  # 计算填充的进度条长度
            bar = '█' * filled + '░' * (50 - filled)  # 构建进度条
            print('\r轮次 (Epoch) {}/{} [{}] {}% 迭代 (Iter) {}/{} 损失 (Loss) {:.5f}'.format(
                iEpoch, nEpoch, bar, pct, iIter + 1, nIterEp, loss.item()), end='', flush=True)
        # 轮次结束处理 (End of epoch processing)
        print()  # 进度条后换行
        lossEp = lossEp / nIterEp  # 计算平均损失
        elapsed_time = time.time() - t0  # 计算本轮次耗时
        remaining_epochs = nEpoch - iEpoch  # 计算剩余轮次数
        est_remaining_time = elapsed_time * remaining_epochs  # 估算剩余时间
        
        # GPU内存监控（每个epoch结束时显示）(GPU memory monitoring at end of each epoch)
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()  # 获取当前使用的GPU设备
            allocated = torch.cuda.memory_allocated(current_device)/1024**3
            reserved = torch.cuda.memory_reserved(current_device)/1024**3
            total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            gpu_util = f"GPU内存: {allocated:.2f}GB / {total:.2f}GB (缓存: {reserved:.2f}GB)"
        else:
            gpu_util = "使用CPU训练"
        
        # 生成训练日志字符串 (Generate training log string)
        logStr = 'Epoch {} | Loss {:.5f} | 耗时 (Time) {:.2f}s | 预计剩余 (Est. Remaining) {:.2f}s | {}'.format(
            iEpoch, lossEp, elapsed_time, est_remaining_time, gpu_util)
        print(logStr)
        
        # 保存模型和损失记录 (Save model and loss records)
        if saveFolder is not None:
            rf.write(logStr + '\n')  # 写入训练日志
            if iEpoch % saveEpoch == 0:
                # 按指定间隔保存模型检查点
                modelFile = os.path.join(saveFolder, 'model_Ep' + str(iEpoch) + '.pt')
                print(f"保存模型检查点 (Saving checkpoint): {modelFile}")
                torch.save(model, modelFile)  # 保存模型状态
    
    # 关闭训练日志文件 (Close training log file)
    if saveFolder is not None:
        rf.close()
    
    # 训练完成提示 (Training completion message)
    print("=" * 50)
    print("模型训练完成! (Model Training Completed!)")
    print("=" * 50)
    return model


def loadModel(outFolder, epoch, modelName='model'):
    """
    加载训练好的模型 (Load trained model)
    
    从指定目录加载指定轮次的模型检查点。
    
    参数 (Parameters):
    ----------
    outFolder : str
        模型保存目录路径
    epoch : int
        要加载的模型轮次
    modelName : str, default='model'
        模型文件名前缀
    
    返回 (Returns):
    -------
    model : torch.nn.Module
        加载的模型对象
    
    注意 (Notes):
    -----
    - 使用 weights_only=False 确保加载完整的模型状态
    - 支持从任何轮次的检查点恢复训练
    """
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile, weights_only=False)
    return model


def testModel(model, x, c, *, batchSize=None, filePathLst=None):
    """
    测试深度学习水文模型 (Test Deep Learning Hydrological Model)
    
    本函数用于批量测试训练好的 XAJ 模型，支持GPU加速和结果保存。
    专门针对静态与动态 XAJ 参数模型进行优化。
    
    参数 (Parameters):
    ----------
    model : torch.nn.Module
        要测试的模型，支持 xaj_static.MultiInv_XAJModel 与 xaj_dynamic.MultiInv_XAJTDModel
    x : numpy.ndarray or tuple
        测试输入数据，形状为 (ngrid, nt, nx) 或 (x, z) 元组
    c : numpy.ndarray or None
        常数输入数据，形状为 (ngrid, nc)
    batchSize : int or None, default=None
        测试批次大小，None表示使用全量数据
    filePathLst : list or None, default=None
        输出文件路径列表，None表示自动生成
    
    
    返回 (Returns):
    -------
    yOut : torch.Tensor or None
        测试输出结果（仅在batchSize=ngrid时返回）
    
    注意 (Notes):
    -----
    - 自动检测GPU并优化内存使用
    - 支持 XAJ 参数的自动提取和保存
    - 提供详细的测试进度显示
    - 结果自动保存为CSV格式
    """
    print("=" * 50)
    print("开始模型测试 (Starting Model Testing)")
    print("=" * 50)
    
    # 处理输入数据格式 (Handle input data format)
    if type(x) is tuple or type(x) is list:
        x, z = x  # 解包元组或列表格式的输入
    else:
        z = None  # 如果没有额外输入，设为None
    
    # 获取数据维度信息 (Get data dimension information)
    ngrid, nt, nx = x.shape  # 网格数, 时间步数, 输入变量数
    if c is not None:
        nc = c.shape[-1]  # 常数输入变量数
    
    # 确定输出变量数 (Determine number of output variables)
    if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
        ny = 5  # XAJ 输出5个变量（Qs, QSave, QIave, QGave, ETave）
    else:
        ny = model.ny  # 其他模型使用模型定义的输出数
    
    # 设置批次大小 (Set batch size)
    if batchSize is None:
        batchSize = ngrid  # 默认使用全量数据
        
    # 显示测试配置信息 (Display testing configuration)
    print(f"测试信息 (Testing Info):")
    print(f"- 数据形状 (Data Shape): {ngrid} 网格 x {nt} 时间步 x {nx} 变量")
    print(f"- 批次大小 (Batch Size): {batchSize}")
    print(f"- 输出变量数 (Output Variables): {ny}")
    print(f"- 模型类型 (Model Type): {type(model).__name__}")
    print("-" * 50)
    
    # 设备配置 (Device configuration)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU测试: {torch.cuda.get_device_name(0)}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)  # 将模型移动到GPU
    else:
        device = torch.device("cpu")
        print("使用CPU测试 (Using CPU)")

    # 设置模型为评估模式 (Set model to evaluation mode)
    model.train(mode=False)
    
    # 处理条件时间步移除 (Handle conditional time step removal)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct  # 调整有效时间步数
    # 批次索引计算 (Batch index calculation)
    # yP = torch.zeros([nt, ngrid, ny])  # 注释掉的预分配输出张量
    iS = np.arange(0, ngrid, batchSize)  # 批次起始索引
    iE = np.append(iS[1:], ngrid)        # 批次结束索引
    total_batches = len(iS)              # 总批次数

    # 处理输出文件名 (Handle output file names)
    if filePathLst is None:
        # 如果没有指定输出文件，自动生成文件名
        filePathLst = ['out' + str(x) for x in range(ny)]
    print(f"输出文件 (Output Files): {filePathLst}")
    
    # 创建输出文件列表 (Create output file list)
    fLst = list()
    for filePath in filePathLst:
        # 确保文件所在目录存在 (Ensure the directory exists)
        dirPath = os.path.dirname(filePath)
        if dirPath and not os.path.exists(dirPath):
            os.makedirs(dirPath, exist_ok=True)
        
        if os.path.exists(filePath):
            os.remove(filePath)  # 如果文件已存在，先删除
        f = open(filePath, 'a')  # 以追加模式打开文件
        fLst.append(f)

    # 开始批次前向传播 (Start batch forward pass)
    t0_total = time.time()  # 记录总开始时间
    for i in range(0, len(iS)):
        t0_batch = time.time()  # 记录当前批次开始时间
        
        # 进度信息显示 (Progress information display)
        pct = int((i + 1) * 100 / total_batches)  # 计算完成百分比
        filled = pct // 2  # 计算填充的进度条长度
        bar = '█' * filled + '░' * (50 - filled)  # 构建进度条
        print(f"\r批次 (Batch) {i+1}/{total_batches} [{bar}] {pct}%", end='', flush=True)
        # 准备当前批次的输入数据 (Prepare input data for current batch)
        xTemp = x[iS[i]:iE[i], :, :]  # 提取当前批次的主要输入数据
        
        if c is not None:
            # 如果有常数输入，将其扩展到所有时间步
            cTemp = np.repeat(
                np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
            # 拼接主要输入和常数输入，并转换为PyTorch张量
            xTest = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp, cTemp], 2), 1, 0)).float()
        else:
            # 如果没有常数输入，直接转换主要输入
            xTest = torch.from_numpy(
                np.swapaxes(xTemp, 1, 0)).float()
        
        # 将输入数据移动到指定设备 (Move input data to specified device)
        if torch.cuda.is_available():
            xTest = xTest.to(device)
        # 处理额外输入数据 (Handle additional input data)
        if z is not None:
            zTemp = z[iS[i]:iE[i], :, :]  # 提取当前批次的额外输入数据
            zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()  # 转换为PyTorch张量
            if torch.cuda.is_available():
                zTest = zTest.to(device)  # 移动到GPU
        # XAJ 模型前向传播与参数提取 (XAJ model forward pass and parameter extraction)
        if type(model) in [xaj_static.MultiInv_XAJModel, xaj_dynamic.MultiInv_XAJTDModel]:
            with torch.no_grad():
                # 执行模型前向传播
                yP = model(xTest, zTest)
                
                # 捕获 XAJ 参数用于后续分析和绘图 (Capture XAJ parameters)
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

        # 转换输出数据格式 (Convert output data format)
        # CP-- 标记问题合并的开始 (marks the beginning of problematic merge)
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)  # 分离梯度并转换维度

        # 保存输出结果 (Save output results)
        for k in range(ny):
            f = fLst[k]  # 获取对应的输出文件句柄
            # 将第k个输出变量保存为CSV格式
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        
        # 保存 XAJ 参数（如果存在）(Save XAJ parameters if available)
        if hasattr(model, 'last_phy_params') and model.last_phy_params is not None:
            phy_params = model.last_phy_params
            # 获取输出目录 (Get output directory)
            output_dir = os.path.dirname(filePathLst[0]) if filePathLst else '.'
            params_file = os.path.join(output_dir, f'phy_params_{getattr(model, "last_phy_name", "PHY")}_batch_{i}.npy')
            np.save(params_file, phy_params)  # 保存参数为numpy格式

        # 显示批次处理时间 (Display batch processing time)
        batch_time = time.time() - t0_batch
        print(f" - 耗时 (Time): {batch_time:.2f}s", flush=True)
        
        # 清理内存 (Memory cleanup)
        model.zero_grad()      # 清零梯度
        torch.cuda.empty_cache()  # 清空GPU缓存

    # 计算并显示总测试时间 (Calculate and display total testing time)
    total_time = time.time() - t0_total
    print("\n" + "-" * 50)
    print(f"测试完成! (Testing Completed!)")
    print(f"总耗时 (Total Time): {total_time:.2f}s")
    print(f"平均每批次耗时 (Avg Time per Batch): {total_time/total_batches:.2f}s")
    print("=" * 50)
    
    # 关闭所有输出文件 (Close all output files)
    for f in fLst:
        f.close()

    # 返回测试结果（仅在特定条件下）(Return test results under specific conditions)
    if batchSize == ngrid:
        # 用于Wenping的工作：计算测试数据的损失
        # 仅在没有使用小批次的情况下有效
        yOut = torch.from_numpy(yOut)
        return yOut

def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    """
    生成随机索引用于训练数据采样 (Generate random indices for training data sampling)
    
    为每个训练批次随机选择网格索引和时间索引，确保训练数据的随机性。
    
    参数 (Parameters):
    ----------
    ngrid : int
        总网格数
    nt : int
        总时间步数
    dimSubset : list
        批次参数 [batchSize, rho]
    bufftime : int, default=0
        缓冲区时间步数
    
    返回 (Returns):
    -------
    iGrid : numpy.ndarray
        随机选择的网格索引，形状为 [batchSize]
    iT : numpy.ndarray
        随机选择的时间索引，形状为 [batchSize]
    
    注意 (Notes):
    -----
    - 时间索引范围考虑了缓冲区时间和时间步长
    - 确保不会超出数据边界
    """
    batchSize, rho = dimSubset
    # 随机选择网格索引 (Randomly select grid indices)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    # 随机选择时间索引，考虑缓冲区时间 (Randomly select time indices, considering buffer time)
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    """
    从输入数据中选择子集 (Select subset from input data)
    
    根据给定的网格索引和时间索引，从原始数据中选择对应的子集，
    支持缓冲区时间、常数输入和多种输出格式。
    
    参数 (Parameters):
    ----------
    x : numpy.ndarray
        输入数据，形状为 (ngrid, nt, nx)
    iGrid : numpy.ndarray
        网格索引数组
    iT : numpy.ndarray
        时间索引数组
    rho : int
        时间步长
    c : numpy.ndarray or None, default=None
        常数输入数据，形状为 (ngrid, nc)
    tupleOut : bool, default=False
        是否以元组形式返回（分离主要输入和常数输入）
    LCopt : bool, default=False
        是否使用局部校准选项
    bufftime : int, default=0
        缓冲区时间步数
    
    返回 (Returns):
    -------
    out : torch.Tensor or tuple
        选择的子集数据，形状为 (rho+bufftime, batchSize, nx) 或 (xTensor, cTensor)
    
    注意 (Notes):
    -----
    - 自动处理GPU设备转换
    - 支持多种数据格式和输出选项
    - 包含边界检查和异常处理
    """
    nx = x.shape[-1]  # 输入变量数
    nt = x.shape[1]   # 时间步数
    
    # 特殊情况处理 (Special case handling)
    if x.shape[0] == len(iGrid):   # hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if nt <= rho:
        iT.fill(0)  # 如果时间步数不足，将时间索引设为0

    batchSize = iGrid.shape[0]  # 批次大小
    
    if iT is not None:
        # 标准情况：根据索引选择数据子集 (Standard case: select data subset based on indices)
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            # 为每个批次选择对应的时间窗口数据
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        # 特殊情况：iT为None时的处理 (Special case: handling when iT is None)
        if LCopt is True:
            # 用于局部校准核：FDC, SMAP等 (Used for local calibration kernel: FDC, SMAP...)
            if len(x.shape) == 2:
                # 用于FDC局部校准核 (Used for local calibration kernel as FDC)
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # 用于LC-SMAP x=Ngrid*Ntime*Nvar (used for LC-SMAP x=Ngrid*Ntime*Nvar)
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # 用于rho等于整个时间序列长度的情况 (Used for rho equal to the whole length of time series)
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]  # 更新rho为实际张量长度
    # 处理常数输入数据 (Handle constant input data)
    if c is not None:
        nc = c.shape[-1]  # 常数输入变量数
        # 将常数输入扩展到所有时间步 (Expand constant input to all time steps)
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            # 以元组形式返回（分离主要输入和常数输入）(Return as tuple - separate main and constant inputs)
            if torch.cuda.is_available():
                device = torch.device("cuda")
                xTensor = xTensor.to(device)
                cTensor = cTensor.to(device)
            out = (xTensor, cTensor)
        else:
            # 拼接主要输入和常数输入 (Concatenate main and constant inputs)
            out = torch.cat((xTensor, cTensor), 2)
    else:
        # 没有常数输入，直接返回主要输入 (No constant input, return main input directly)
        out = xTensor

    # 设备转换 (Device conversion)
    if torch.cuda.is_available() and type(out) is not tuple:
        device = torch.device("cuda")
        out = out.to(device)  # 将输出移动到GPU
    
    return out
