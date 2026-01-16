import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def save_hydro(outpath, testepoch, logtestIDLst, forcTestUN, obsTestUN, predtestALL):
    """
    保存测试期的降雨、观测流量、预测流量和其他水文过程变量到hydro_variables_csv
    
    参数:
        outpath: 输出路径
        testepoch: 测试使用的epoch
        logtestIDLst: 测试站点ID列表
        forcTestUN: 测试期的输入数据 [basin, time, var]，包括降雨等
        obsTestUN: 测试期的观测流量 [basin, time, 1]
        predtestALL: 测试期的预测结果 [basin, time, var]，包括流量和其他水文过程变量
    """
    print("=" * 50)
    print("Saving hydro variables")
    print("=" * 50)
    
    # 创建保存数据的文件夹
    hydro_variables_csv = os.path.join(outpath, 'hydro_variables_csv')
    if not os.path.exists(hydro_variables_csv):
        os.makedirs(hydro_variables_csv)
    
    # 获取变量名称
    var_names = ['Qr', 'Q0', 'Q1', 'Q2', 'ET']  # 预测变量名称
    
    # 遍历每个站点，保存数据
    n_basins = len(logtestIDLst)
    
    print(f"Processing {n_basins} basins")
    
    for i in range(n_basins):
        basin_id = logtestIDLst[i]
        
        # 进度显示
        progress = (i + 1) / n_basins * 100
        print(f"\rSaving data: {i+1}/{n_basins} [{basin_id}] - {progress:.1f}%", end="", flush=True)
        
        # 提取该站点的数据
        rainfall = forcTestUN[i, :, 0]  # 第一个变量是降雨量(prcp)
        obs_flow = obsTestUN[i, :, 0]   # 观测流量
        pred_flow = predtestALL[i, :, 0]  # 预测流量 (Q0)
        
        # 创建DataFrame保存所有数据
        data = {
            'Rainfall': rainfall,
            'Observed_Flow': obs_flow,
            'Predicted_Flow': pred_flow
        }
        
        # 添加其他水文过程变量
        for j, var_name in enumerate(var_names):
            if j > 0:  # 跳过Q0，因为已经作为Predicted_Flow保存
                data[var_name] = predtestALL[i, :, j]
        
        # 保存为CSV
        df = pd.DataFrame(data)
        csv_file = os.path.join(hydro_variables_csv, f"{basin_id}.csv")
        
        # 直接使用UTF-8-sig编码，如果遇到无效字符则跳过
        try:
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        except UnicodeEncodeError:
            # 如果遇到无效字符，跳过该文件
            print(f"\n警告: 跳过 {basin_id} (包含无效字符)", end="", flush=True)
            continue
    
    print("\n" + "=" * 50)
    print(f"Hydro variables saved to: {hydro_variables_csv}")
    print("=" * 50)
    
    # 生成README文件，说明数据格式
    with open(os.path.join(outpath, 'README.txt'), 'w') as f:
        f.write("Data Description\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test epoch: {testepoch}\n")
        f.write(f"Total basins: {n_basins}\n\n")
        f.write("CSV File Format:\n")
        f.write("- Rainfall: Precipitation (mm/day)\n")
        f.write("- Observed_Flow: Observed streamflow (mm/day)\n")
        f.write("- Predicted_Flow: Predicted streamflow (mm/day)\n")
        f.write("- Qr, Q1, Q2, ET: Other hydrological process variables\n\n")
    
    return hydro_variables_csv

def generate_streamflow_plots(outpath, logtestIDLst, obsTestUN, predtestALL, max_plots=None, test_start_date='19951001'):
    """
    为每个站点生成径流对比图（不包含降雨）
    
    参数:
        outpath: 输出路径
        logtestIDLst: 测试站点ID列表
        obsTestUN: 测试期的观测流量 [basin, time, 1]
        predtestALL: 测试期的预测结果 [basin, time, var]，包括流量和其他水文过程变量
        max_plots: 最大生成图片数量，None表示生成全部图片
        test_start_date: 测试开始日期，格式为 'YYYYMMDD'
    """
    print("=" * 50)
    print("Starting to generate streamflow comparison plots")
    print("=" * 50)
    
    # 设置matplotlib使用中文字体
    
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 创建保存图表的文件夹
    streamflow_plots = os.path.join(outpath, 'streamflow_plots')
    if not os.path.exists(streamflow_plots):
        os.makedirs(streamflow_plots)
    
    # 仅使用 basin_id 标题，不依赖外部全局路径
    
    # 生成日期序列（根据测试开始日期动态生成）
    # 处理test_start_date，可能是字符串或整数
    if isinstance(test_start_date, int):
        test_start_date = str(test_start_date)
    
    start_year = int(test_start_date[:4])
    start_month = int(test_start_date[4:6])
    start_day = int(test_start_date[6:8])
    start_date = datetime(start_year, start_month, start_day)
    
    n_days = obsTestUN.shape[1]
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # 设置x轴刻度（5个刻度）
    x_ticks = np.linspace(0, len(dates)-1, 5, dtype=int)
    x_labels = [dates[i].strftime('%Y-%m') for i in x_ticks]
    
    # 遍历每个站点，生成图表
    n_basins = len(logtestIDLst)
    
    # 确定要处理的站点数量
    if max_plots is not None and max_plots > 0 and max_plots < n_basins:
        plot_count = max_plots
        print(f"Generating plots for first {max_plots} basins only")
    else:
        plot_count = n_basins
        print(f"Generating plots for all {n_basins} basins")
    
    for i in range(plot_count):
        basin_id = logtestIDLst[i]
        
        # 可选：根据 basin_id 设置可读名称（此处直接使用 basin_id）
        
        # 进度显示
        progress = (i + 1) / n_basins * 100
        print(f"\rGenerating plot: {i+1}/{n_basins} [{basin_id}] - {progress:.1f}%", end="", flush=True)
        
        # 提取该站点的数据
        obs_flow = obsTestUN[i, :, 0]   # 观测流量
        pred_flow = predtestALL[i, :, 0]  # 预测流量 (Qr)
        
        # 计算NSE
        nse = calculate_nse(obs_flow, pred_flow)
        
        # 生成径流对比图（单图）
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 绘制观测和预测径流
        ax.plot(dates, obs_flow, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
        ax.plot(dates, pred_flow, 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
        
        # 设置标签和标题
        ax.set_ylabel('Streamflow (mm/day)', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title(f'Basin {basin_id} - NSE: {nse:.4f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴刻度和标签
        ax.set_xticks([dates[j] for j in x_ticks])
        ax.set_xticklabels(x_labels, rotation=0)
        
        # 保存图表
        plt.tight_layout()
        # 清理文件名，移除可能导致问题的字符
        safe_basin_id = str(basin_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(os.path.join(streamflow_plots, f"{safe_basin_id}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 50)
    print(f"Streamflow plots saved to: {streamflow_plots}")
    print("=" * 50)
    
    # 更新README文件，添加图表说明
    with open(os.path.join(outpath, 'README.txt'), 'a') as f:
        f.write("\nPlot Description:\n")
        f.write("- Single panel: Observed vs Predicted Streamflow\n")
        f.write("- X-axis: Time (Year-Month) with 5 ticks\n")
        f.write("- Title contains Basin ID and NSE value\n")
    
    return streamflow_plots

def save_results_and_plot(outpath, testepoch, logtestIDLst, forcTestUN, obsTestUN, predtestALL, max_plots=None, test_start_date='19951001'):
    """
    保存测试期的降雨、观测流量、预测流量和其他水文过程变量，并生成径流对比图
    (整合函数，同时调用save_hydro和generate_streamflow_plots)
    
    参数:
        outpath: 输出路径
        testepoch: 测试使用的epoch
        logtestIDLst: 测试站点ID列表
        forcTestUN: 测试期的输入数据 [basin, time, var]，包括降雨等
        obsTestUN: 测试期的观测流量 [basin, time, 1]
        predtestALL: 测试期的预测结果 [basin, time, var]，包括流量和其他水文过程变量
        max_plots: 最大生成图片数量，None表示生成全部图片，0表示不生成图片
        test_start_date: 测试开始日期，格式为 'YYYYMMDD'
    """
    # 保存数据
    hydro_variables_csv = save_hydro(outpath, testepoch, logtestIDLst, forcTestUN, obsTestUN, predtestALL)
    
    # 生成图表（如果max_plots为0则跳过）
    streamflow_plots = None
    if max_plots is None or max_plots > 0:
        streamflow_plots = generate_streamflow_plots(outpath, logtestIDLst, obsTestUN, predtestALL, max_plots, test_start_date)
    else:
        print("Skipping plot generation")
    
    return hydro_variables_csv, streamflow_plots

def calculate_nse(obs, sim):
    """计算Nash-Sutcliffe效率系数"""
    # 去除NaN值
    idx = np.where(~np.isnan(obs) & ~np.isnan(sim))[0]
    obs = obs[idx]
    sim = sim[idx]
    
    if len(obs) == 0:
        return np.nan
    
    mean_obs = np.mean(obs)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - mean_obs) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return 1 - (numerator / denominator)

def calculate_rmse(obs, sim):
    """计算均方根误差"""
    # 去除NaN值
    idx = np.where(~np.isnan(obs) & ~np.isnan(sim))[0]
    obs = obs[idx]
    sim = sim[idx]
    
    if len(obs) == 0:
        return np.nan
    
    return np.sqrt(np.mean((obs - sim) ** 2))

def calculate_kge(obs, sim):
    """计算Kling-Gupta效率系数"""
    # 去除NaN值
    idx = np.where(~np.isnan(obs) & ~np.isnan(sim))[0]
    obs = obs[idx]
    sim = sim[idx]
    
    if len(obs) == 0:
        return np.nan
    
    # 计算相关系数
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs)
    std_sim = np.std(sim)
    
    if std_obs == 0 or std_sim == 0:
        return np.nan
    
    # 计算相关系数r
    cov = np.mean((obs - mean_obs) * (sim - mean_sim))
    r = cov / (std_obs * std_sim)
    
    # 计算偏差比率beta
    beta = mean_sim / mean_obs
    
    # 计算变异系数比率alpha
    alpha = std_sim / std_obs
    
    # 计算KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge



def save_xaj_parameters(outpath, logtestIDLst, xaj_params, model_type='static'):
    """
    保存XAJ参数到CSV文件，便于后续分析和绘图
    
    参数:
        outpath: 输出路径
        logtestIDLst: 测试站点ID列表
        xaj_params: XAJ参数数组
        model_type: 模型类型 ('static' 或 'dynamic')
    """
    print("=" * 50)
    print("Saving XAJ parameters")
    print("=" * 50)

    # 创建保存XAJ参数的文件夹
    xaj_params_dir = os.path.join(outpath, 'xaj_parameters_csv')
    if not os.path.exists(xaj_params_dir):
        os.makedirs(xaj_params_dir)

    # XAJ 参数名称（12个）
    param_names = ['ke', 'b', 'wum', 'wlm', 'wm', 'c', 'sm', 'ex', 'ki', 'kg', 'ci', 'cg']

    n_basins = len(logtestIDLst)
    if model_type == 'static':
        # 静态模型：xaj_params.shape = [ngage, nfea, nmul]
        for i in range(n_basins):
            basin_id = logtestIDLst[i]
            progress = (i + 1) / n_basins * 100
            print(f"\rSaving XAJ parameters: {i+1}/{n_basins} [{basin_id}] - {progress:.1f}%", end="", flush=True)

            basin_params = xaj_params[i, :, :]  # [nfea, nmul]
            data = {}
            for j, pname in enumerate(param_names):
                if j < basin_params.shape[0]:
                    for k in range(basin_params.shape[1]):
                        data[f'{pname}_comp{k+1}'] = [basin_params[j, k]]

            df = pd.DataFrame(data)
            csv_file = os.path.join(xaj_params_dir, f"{basin_id}_xaj_params.csv")
            try:
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            except UnicodeEncodeError:
                # 如果遇到无效字符，跳过该文件
                print(f"\n警告: 跳过 {basin_id} (包含无效字符)", end="", flush=True)
                continue
    else:
        # 动态模型：xaj_params.shape = [ntstep, ngage, nfea, nmul]
        for i in range(n_basins):
            basin_id = logtestIDLst[i]
            progress = (i + 1) / n_basins * 100
            print(f"\rSaving XAJ parameters: {i+1}/{n_basins} [{basin_id}] - {progress:.1f}%", end="", flush=True)

            basin_params = xaj_params[:, i, :, :]  # [ntstep, nfea, nmul]
            data = {}
            for j, pname in enumerate(param_names):
                if j < basin_params.shape[1]:
                    for k in range(basin_params.shape[2]):
                        data[f'{pname}_comp{k+1}'] = basin_params[:, j, k]

            df = pd.DataFrame(data)
            csv_file = os.path.join(xaj_params_dir, f"{basin_id}_xaj_params.csv")
            try:
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            except UnicodeEncodeError:
                # 如果遇到无效字符，跳过该文件
                print(f"\n警告: 跳过 {basin_id} (包含无效字符)", end="", flush=True)
                continue

    print("\n" + "=" * 50)
    print(f"XAJ parameters saved to: {xaj_params_dir}")
    print("=" * 50)

    # 生成README文件，说明XAJ参数格式
    with open(os.path.join(xaj_params_dir, 'README.txt'), 'w') as f:
        f.write("XAJ Parameters Description\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Total basins: {n_basins}\n")
        f.write(f"Parameter shape: {xaj_params.shape}\n\n")
        f.write("CSV File Format:\n")
        f.write("- Each file contains XAJ parameters for one basin\n")
        f.write("- Parameter names: " + ", ".join(param_names) + "\n")
        f.write("- Each parameter has multiple components (comp1, comp2, ...)\n")

    return xaj_params_dir

def plot_dynamic_parameters(outpath, logtestIDLst, params, model_type='dynamic', max_basins=None, test_start_date='19951001', test_end_date='20101001', train_start_date='19801001'):
    """
    绘制动态参数时间序列（通用）
    每个参数的Nmul个组件用灰色显示，中位数组件用不同颜色显示
    
    参数:
        outpath: 输出路径
        logtestIDLst: 测试站点ID列表
        params: 参数数组 [ntime, nbasin, nparam, ncomp]
        model_type: 模型类型 ('static' 或 'dynamic')
        max_basins: 最大绘制站点数量，None表示绘制全部站点
        test_start_date: 测试开始日期，格式为 'YYYYMMDD'
        test_end_date: 测试结束日期，格式为 'YYYYMMDD'
        train_start_date: 训练开始日期（参数序列起点），格式为 'YYYYMMDD'
    """
    if model_type != 'dynamic':
        print("此函数仅适用于动态模型参数")
        return
    
    print("=" * 50)
    print("Generating dynamic parameter plots")
    print("=" * 50)
    
    # 创建保存图表的文件夹
    param_plots_dir = os.path.join(outpath, 'parameter_plots')
    if not os.path.exists(param_plots_dir):
        os.makedirs(param_plots_dir)
    
    # 参数名称（优先支持12个参数：XAJ）
    if params.shape[2] == 12:
        param_names = ['ke', 'b', 'wum', 'wlm', 'wm', 'c', 'sm', 'ex', 'ki', 'kg', 'ci', 'cg']
    else:
        # 回退：通用名称
        param_names = [f'P{i+1}' for i in range(params.shape[2])]
    
    # 为每个参数分配不同颜色
    param_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 
                   'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'navy']
    
    # 生成时间轴
    # 处理test_start_date、test_end_date、train_start_date，可能是字符串或整数
    if isinstance(test_start_date, int):
        test_start_date = str(test_start_date)
    if isinstance(test_end_date, int):
        test_end_date = str(test_end_date)
    if isinstance(train_start_date, int):
        train_start_date = str(train_start_date)

    # 计算训练期的开始日期
    train_year = int(train_start_date[:4])
    train_month = int(train_start_date[4:6])
    train_day = int(train_start_date[6:8])
    train_start_datetime = datetime(train_year, train_month, train_day)
    
    # 计算测试期的开始和结束日期与训练起点
    start_year = int(test_start_date[:4])
    start_month = int(test_start_date[4:6])
    start_day = int(test_start_date[6:8])
    test_start_datetime = datetime(start_year, start_month, start_day)
    
    end_year = int(test_end_date[:4])
    end_month = int(test_end_date[4:6])
    end_day = int(test_end_date[6:8])
    test_end_datetime = datetime(end_year, end_month, end_day)

 
    
    # 计算测试期的天数
    test_days = (test_end_datetime - test_start_datetime).days
    
    # 截取测试期的参数数据
    # 假设参数的时间维度包含了整个序列，我们需要找到测试期对应的索引
    # 这里假设参数的时间维度从训练开始，我们需要找到测试期开始的位置
    total_time_steps = params.shape[0]
    
    # 如果参数的时间步数等于测试期天数，说明已经是测试期数据
    if total_time_steps == test_days:
        print(f"参数已经是测试期数据: {total_time_steps} 天")
        test_params = params
        time_axis = [test_start_datetime + timedelta(days=i) for i in range(test_days)]
    else:
        # 如果参数包含更多时间步，需要截取测试期部分
        # 这里假设参数从某个基准日期开始，我们需要计算测试期的偏移
        print(f"参数包含 {total_time_steps} 天，测试期为 {test_days} 天")
        print("截取测试期参数数据...")
        
        # 依据传入的训练开始日期计算到测试期开始的天数
        days_to_test_start = (test_start_datetime - train_start_datetime).days
        
        # 截取测试期的参数
        start_idx = days_to_test_start
        end_idx = start_idx + test_days
        
        if end_idx <= total_time_steps:
            test_params = params[start_idx:end_idx, :, :, :]
            time_axis = [test_start_datetime + timedelta(days=i) for i in range(test_days)]
            print(f"成功截取测试期参数: 索引 {start_idx} 到 {end_idx-1}")
        else:
            print(f"警告: 无法截取完整的测试期数据 (需要到索引 {end_idx-1}，但只有 {total_time_steps} 天)")
            # 截取到可用的最大范围
            available_days = total_time_steps - start_idx
            test_params = params[start_idx:, :, :, :]
            time_axis = [test_start_datetime + timedelta(days=i) for i in range(available_days)]
            print(f"截取可用数据: {available_days} 天")
    
    ntime = test_params.shape[0]
    
    # 获取组件数量
    n_components = test_params.shape[3]  # 第4个维度是组件数量
    print(f"Number of components (Nmul): {n_components}")
    
    # 确定要绘制的站点数量
    if max_basins is None:
        n_basins_to_plot = len(logtestIDLst)
        print(f"Generating parameter plots for all {n_basins_to_plot} basins")
    else:
        n_basins_to_plot = min(len(logtestIDLst), max_basins)
        print(f"Generating parameter plots for {n_basins_to_plot} basins (limited by max_basins={max_basins})")
    
    for basin_idx in range(n_basins_to_plot):
        basin_id = logtestIDLst[basin_idx]
        
        # 调试信息：打印前几个站点ID的详细信息
        if basin_idx < 3:
            print(f"\nDebug: basin_idx={basin_idx}, basin_id={repr(basin_id)}, type={type(basin_id)}")
        
        # 进度显示
        progress = (basin_idx + 1) / n_basins_to_plot * 100
        print(f"\rGenerating plots: {basin_idx+1}/{n_basins_to_plot} [{basin_id}] - {progress:.1f}%", end="", flush=True)
        
        # 提取该站点的参数 [ntime, nparam, ncomp]
        basin_params = test_params[:, basin_idx, :, :]
        
        # 为每个参数创建子图，所有参数放在同一页
        n_params = basin_params.shape[1]
        
        # 计算子图布局：尽量接近正方形
        n_cols = int(np.ceil(np.sqrt(n_params)))
        n_rows = int(np.ceil(n_params / n_cols))
        
        # 创建子图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        
        # 确保axes是二维数组
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # 为每个参数绘制子图
        for param_idx in range(n_params):
            row = param_idx // n_cols
            col = param_idx % n_cols
            ax = axes[row, col]
            
            param_name = param_names[param_idx]
            param_color = param_colors[param_idx % len(param_colors)]
            
            # 提取该参数的所有组件时间序列 [ntime, ncomp]
            param_components = basin_params[:, param_idx, :]
            
            # 绘制所有组件的参数（灰色线条）
            for comp in range(n_components):
                ax.plot(time_axis, param_components[:, comp], 
                       color='lightgray', alpha=0.7, linewidth=0.8)
            
            # 计算每个组件的时间平均参数值
            time_averages = np.mean(param_components, axis=0)
            
            # 找到中位数组件
            median_avg = np.median(time_averages)
            median_comp_idx = np.argmin(np.abs(time_averages - median_avg))
            
            # 绘制中位数组件（指定颜色实线）
            ax.plot(time_axis, param_components[:, median_comp_idx], 
                   color=param_color, linewidth=2, 
                   label=f'Median (Comp {median_comp_idx+1})')
            
            # 设置子图属性
            ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{param_name}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            
            # 设置x轴格式 - 5个刻度，年月格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # 计算5个等间距的刻度位置
            n_ticks = 5
            tick_positions = np.linspace(0, len(time_axis)-1, n_ticks, dtype=int)
            ax.set_xticks([time_axis[i] for i in tick_positions])
            ax.set_xticklabels([time_axis[i].strftime('%Y-%m') for i in tick_positions], 
                              rotation=0, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
        
        # 隐藏多余的子图
        for param_idx in range(n_params, n_rows * n_cols):
            row = param_idx // n_cols
            col = param_idx % n_cols
            axes[row, col].set_visible(False)
        
        # 添加总标题
        fig.suptitle(f'Dynamic Parameters - Basin {basin_id}', fontsize=14, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 为总标题留出空间
        
        # 保存图表 - 清理文件名，移除可能导致问题的字符
        safe_basin_id = str(basin_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_file = os.path.join(param_plots_dir, f"{safe_basin_id}_dynamic_parameters.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 50)
    print(f"Dynamic parameter plots saved to: {param_plots_dir}")
    print("=" * 50)
    
    # 生成README文件，说明参数图表格式
    with open(os.path.join(param_plots_dir, 'README.txt'), 'w') as f:
        f.write("Dynamic Parameter Plots Description\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Total basins plotted: {n_basins_to_plot}\n")
        f.write(f"Original parameter shape: {params.shape}\n")
        f.write(f"Test period parameter shape: {test_params.shape}\n")
        f.write(f"Test period: {test_start_date} to {test_end_date}\n\n")
        f.write("Plot Format:\n")
        f.write(f"- Gray lines: Parameter values from all {n_components} components\n")
        f.write("- Colored lines: Component with median temporal average parameter value\n")
        f.write("- Each parameter has a different color for the median component\n")
        f.write(f"- Time axis: From {test_start_datetime.strftime('%Y-%m-%d')} to {(test_start_datetime + timedelta(days=ntime-1)).strftime('%Y-%m-%d')}\n\n")
        f.write("Parameter Colors:\n")
        for i, (name, color) in enumerate(zip(param_names, param_colors)):
            f.write(f"- {name}: {color}\n")
        f.write("\nInterpretation:\n")
        f.write("- Seasonal patterns indicate dynamic parameter behavior\n")
        f.write("- Median component represents typical basin response\n")
        f.write("- Component spread shows parameter uncertainty/variability\n")
    
    return param_plots_dir

def plot_metrics_boxplot(outpath, evaDict, rnn_type):
    """
    绘制NSE和KGE的箱线图，并标明深度学习模型名称
    
    参数:
        outpath: 输出路径
        evaDict: 评估指标字典，包含NSE和KGE
        rnn_type: 深度学习模型类型
    """
    print("=" * 50)
    print("Generating metrics boxplot")
    print("=" * 50)
    
    # 设置matplotlib样式
    plt.style.use('default')  # 使用默认样式
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置Times New Roman字体
    
    # 准备数据
    nse_values = evaDict['NSE']
    kge_values = evaDict['KGE']
    
    # 计算中位数和四分位数
    nse_median = np.nanmedian(nse_values)
    kge_median = np.nanmedian(kge_values)
    
    # 创建数据框
    data = {
        'Metric': ['NSE'] * len(nse_values) + ['KGE'] * len(kge_values),
        'Value': np.concatenate([nse_values, kge_values]),
    }
    df = pd.DataFrame(data)
    
    # 打印中位数值，用于调试
    print(f"NSE中位数: {nse_median:.3f}")
    print(f"KGE中位数: {kge_median:.3f}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用matplotlib的boxplot绘制箱线图
    metrics = ['NSE', 'KGE']
    # 确保NSE和KGE数据对应正确的标签位置
    boxplot_data = [
        nse_values,  # 第一个箱体是NSE
        kge_values   # 第二个箱体是KGE
    ]
    
    # 绘制箱线图
    bp = ax.boxplot(boxplot_data, labels=metrics, patch_artist=True, widths=0.6)
    
    # 设置颜色
    colors = ['#66c2a5', '#fc8d62']  # 好看的颜色
    
    # 为每个箱体设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置边框颜色
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('#2c3e50')
            item.set_linewidth(1.5)
    
    # 设置异常值点的样式
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('#e74c3c')
        flier.set_markeredgecolor('none')
        flier.set_markersize(4)
        flier.set_alpha(0.5)
    
    # 添加模型类型和中位数信息
    model_name_map = {
        'lstm': 'LSTM',
        'gru': 'GRU',
        'bilstm': 'BiLSTM',
        'bigru': 'BiGRU',
        'rnn': 'RNN',
        'cnnlstm': 'CNN-LSTM',
        'cnnbilstm': 'CNN-BiLSTM',
    }
    
    model_name = model_name_map.get(rnn_type.lower(), rnn_type)
    
    # 添加标题和标签
    ax.set_title(f'Performance Metrics for {model_name} Model\n(n={len(nse_values)} basins)', 
                fontsize=16, family='Times New Roman')
    ax.set_ylabel('Value', fontsize=14, family='Times New Roman')
    ax.set_xlabel('', fontsize=14, family='Times New Roman')
    
    # 设置y轴范围，确保能看到所有数据
    y_min = min(np.nanpercentile(nse_values, 1), np.nanpercentile(kge_values, 1))
    y_max = max(np.nanpercentile(nse_values, 99), np.nanpercentile(kge_values, 99))
    
    # 确保y轴范围能容纳所有数据点
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    
    # 在图框右上角添加NSE和KGE的中位数值
    # 获取坐标轴的范围
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # 计算文本位置 - 右上角，留出一定边距
    # 由于x轴只有两个标签(0和1)，使用固定位置而不是相对位置
    text_x = 1.45  # 位于KGE箱体右侧
    text_y_nse = y_max * 0.95  # 靠近上边界
    text_y_kge = y_max * 0.88  # 稍低一点
    
    # 添加中位数标签到右上角，添加背景框使文本更清晰
    nse_text = ax.text(text_x, text_y_nse, f'NSE Median: {nse_median:.3f}', 
            ha='right', va='top', fontsize=12, fontweight='bold', family='Times New Roman')
    kge_text = ax.text(text_x, text_y_kge, f'KGE Median: {kge_median:.3f}', 
            ha='right', va='top', fontsize=12, fontweight='bold', family='Times New Roman')
            
    # 为文本添加背景框
    nse_text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    kge_text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # 不需要额外的空间，因为标签已经放在图框内部
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plot_path = os.path.join(outpath, f'Metrics_Boxplot_{model_name}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics boxplot saved to: {plot_path}")
    print("=" * 50)
    
    return plot_path

