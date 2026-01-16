import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def save_hydro(outpath, testepoch, logtestIDLst, forcTestUN, obsTestUN, predtestALL):
    """
    Save test-period rainfall, observed flow, predicted flow and other hydrological variables to hydro_variables_csv.

    Args:
        outpath: Output directory path.
        testepoch: Epoch used for testing.
        logtestIDLst: List of test basin IDs.
        forcTestUN: Test input data [basin, time, var], e.g. rainfall.
        obsTestUN: Test observed flow [basin, time, 1].
        predtestALL: Test predictions [basin, time, var], flow and other hydrological variables.
    """
    print("=" * 50)
    print("Saving hydro variables")
    print("=" * 50)
    
    # Create output directory for hydro variables
    hydro_variables_csv = os.path.join(outpath, 'hydro_variables_csv')
    if not os.path.exists(hydro_variables_csv):
        os.makedirs(hydro_variables_csv)
    
    # Variable names for predicted outputs
    var_names = ['Qr', 'Q0', 'Q1', 'Q2', 'ET']

    # Iterate over basins and save data
    n_basins = len(logtestIDLst)
    
    print(f"Processing {n_basins} basins")
    
    for i in range(n_basins):
        basin_id = logtestIDLst[i]
        
        # Progress
        progress = (i + 1) / n_basins * 100
        print(f"\rSaving data: {i+1}/{n_basins} [{basin_id}] - {progress:.1f}%", end="", flush=True)

        # Extract data for this basin
        rainfall = forcTestUN[i, :, 0]   # First variable is precipitation (prcp)
        obs_flow = obsTestUN[i, :, 0]    # Observed flow
        pred_flow = predtestALL[i, :, 0] # Predicted flow (Q0)

        # Build DataFrame with all variables
        data = {
            'Rainfall': rainfall,
            'Observed_Flow': obs_flow,
            'Predicted_Flow': pred_flow
        }
        
        # Add other hydrological process variables
        for j, var_name in enumerate(var_names):
            if j > 0:  # Skip Q0, already saved as Predicted_Flow
                data[var_name] = predtestALL[i, :, j]

        # Save to CSV
        df = pd.DataFrame(data)
        csv_file = os.path.join(hydro_variables_csv, f"{basin_id}.csv")
        
        # Use UTF-8-sig; skip file if invalid characters
        try:
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        except UnicodeEncodeError:
            print(f"\nWarning: Skipping {basin_id} (invalid characters)", end="", flush=True)
            continue
    
    print("\n" + "=" * 50)
    print(f"Hydro variables saved to: {hydro_variables_csv}")
    print("=" * 50)
    
    # Write README describing data format
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
    Generate streamflow comparison plots (no rainfall) for each basin.

    Args:
        outpath: Output directory path.
        logtestIDLst: List of test basin IDs.
        obsTestUN: Test observed flow [basin, time, 1].
        predtestALL: Test predictions [basin, time, var], flow and other variables.
        max_plots: Max number of plots to generate; None = all basins.
        test_start_date: Test start date, format 'YYYYMMDD'.
    """
    print("=" * 50)
    print("Starting to generate streamflow comparison plots")
    print("=" * 50)
    
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Create directory for streamflow plots
    streamflow_plots = os.path.join(outpath, 'streamflow_plots')
    if not os.path.exists(streamflow_plots):
        os.makedirs(streamflow_plots)
    
    # Build date sequence from test start date (handles str or int)
    if isinstance(test_start_date, int):
        test_start_date = str(test_start_date)
    
    start_year = int(test_start_date[:4])
    start_month = int(test_start_date[4:6])
    start_day = int(test_start_date[6:8])
    start_date = datetime(start_year, start_month, start_day)
    
    n_days = obsTestUN.shape[1]
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # X-axis ticks (5 ticks)
    x_ticks = np.linspace(0, len(dates)-1, 5, dtype=int)
    x_labels = [dates[i].strftime('%Y-%m') for i in x_ticks]
    
    n_basins = len(logtestIDLst)

    # Number of basins to plot
    if max_plots is not None and max_plots > 0 and max_plots < n_basins:
        plot_count = max_plots
        print(f"Generating plots for first {max_plots} basins only")
    else:
        plot_count = n_basins
        print(f"Generating plots for all {n_basins} basins")
    
    for i in range(plot_count):
        basin_id = logtestIDLst[i]
        
        # Progress
        progress = (i + 1) / n_basins * 100
        print(f"\rGenerating plot: {i+1}/{n_basins} [{basin_id}] - {progress:.1f}%", end="", flush=True)
        
        obs_flow = obsTestUN[i, :, 0]   # Observed flow
        pred_flow = predtestALL[i, :, 0]  # Predicted flow (Qr)

        nse = calculate_nse(obs_flow, pred_flow)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot observed and predicted streamflow
        ax.plot(dates, obs_flow, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
        ax.plot(dates, pred_flow, 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
        
        # Labels and title
        ax.set_ylabel('Streamflow (mm/day)', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title(f'Basin {basin_id} - NSE: {nse:.4f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # X-axis ticks and labels
        ax.set_xticks([dates[j] for j in x_ticks])
        ax.set_xticklabels(x_labels, rotation=0)
        
        plt.tight_layout()
        # Sanitize filename
        safe_basin_id = str(basin_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(os.path.join(streamflow_plots, f"{safe_basin_id}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 50)
    print(f"Streamflow plots saved to: {streamflow_plots}")
    print("=" * 50)
    
    # Append plot description to README
    with open(os.path.join(outpath, 'README.txt'), 'a') as f:
        f.write("\nPlot Description:\n")
        f.write("- Single panel: Observed vs Predicted Streamflow\n")
        f.write("- X-axis: Time (Year-Month) with 5 ticks\n")
        f.write("- Title contains Basin ID and NSE value\n")
    
    return streamflow_plots

def save_results_and_plot(outpath, testepoch, logtestIDLst, forcTestUN, obsTestUN, predtestALL, max_plots=None, test_start_date='19951001'):
    """
    Save test-period data (rainfall, observed/predicted flow, etc.) and generate streamflow comparison plots.
    Wrapper that calls save_hydro and generate_streamflow_plots.

    Args:
        outpath: Output directory path.
        testepoch: Epoch used for testing.
        logtestIDLst: List of test basin IDs.
        forcTestUN: Test input data [basin, time, var].
        obsTestUN: Test observed flow [basin, time, 1].
        predtestALL: Test predictions [basin, time, var].
        max_plots: Max number of plots; None=all, 0=skip plots.
        test_start_date: Test start date, format 'YYYYMMDD'.
    """
    hydro_variables_csv = save_hydro(outpath, testepoch, logtestIDLst, forcTestUN, obsTestUN, predtestALL)

    # Generate plots (skip if max_plots is 0)
    streamflow_plots = None
    if max_plots is None or max_plots > 0:
        streamflow_plots = generate_streamflow_plots(outpath, logtestIDLst, obsTestUN, predtestALL, max_plots, test_start_date)
    else:
        print("Skipping plot generation")
    
    return hydro_variables_csv, streamflow_plots

def calculate_nse(obs, sim):
    """Compute Nash-Sutcliffe efficiency coefficient."""
    # Remove NaN values
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
    """Compute root mean square error."""
    # Remove NaN values
    idx = np.where(~np.isnan(obs) & ~np.isnan(sim))[0]
    obs = obs[idx]
    sim = sim[idx]
    
    if len(obs) == 0:
        return np.nan
    
    return np.sqrt(np.mean((obs - sim) ** 2))

def calculate_kge(obs, sim):
    """Compute Kling-Gupta efficiency coefficient."""
    # Remove NaN values
    idx = np.where(~np.isnan(obs) & ~np.isnan(sim))[0]
    obs = obs[idx]
    sim = sim[idx]
    
    if len(obs) == 0:
        return np.nan
    
    # Correlation and moments
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs)
    std_sim = np.std(sim)
    
    if std_obs == 0 or std_sim == 0:
        return np.nan
    
    cov = np.mean((obs - mean_obs) * (sim - mean_sim))
    r = cov / (std_obs * std_sim)
    
    beta = mean_sim / mean_obs  # Bias ratio
    alpha = std_sim / std_obs  # Variability ratio
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge



def save_xaj_parameters(outpath, logtestIDLst, xaj_params, model_type='static'):
    """
    Save XAJ parameters to CSV files for analysis and plotting.

    Args:
        outpath: Output directory path.
        logtestIDLst: List of test basin IDs.
        xaj_params: XAJ parameter array.
        model_type: 'static' or 'dynamic'.
    """
    print("=" * 50)
    print("Saving XAJ parameters")
    print("=" * 50)

    xaj_params_dir = os.path.join(outpath, 'xaj_parameters_csv')
    if not os.path.exists(xaj_params_dir):
        os.makedirs(xaj_params_dir)

    param_names = ['ke', 'b', 'wum', 'wlm', 'wm', 'c', 'sm', 'ex', 'ki', 'kg', 'ci', 'cg']

    n_basins = len(logtestIDLst)
    if model_type == 'static':
        # Static model: xaj_params.shape = [ngage, nfea, nmul]
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
                print(f"\nWarning: Skipping {basin_id} (invalid characters)", end="", flush=True)
                continue
    else:
        # Dynamic model: xaj_params.shape = [ntstep, ngage, nfea, nmul]
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
                print(f"\nWarning: Skipping {basin_id} (invalid characters)", end="", flush=True)
                continue

    print("\n" + "=" * 50)
    print(f"XAJ parameters saved to: {xaj_params_dir}")
    print("=" * 50)

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
    Plot dynamic parameter time series.

    All Nmul components for each parameter are shown in gray; the median
    component (based on time-averaged value) is highlighted with color.

    Args:
        outpath: Output directory path.
        logtestIDLst: List of test basin IDs.
        params: Parameter array [ntime, nbasin, nparam, ncomp].
        model_type: 'static' or 'dynamic'.
        max_basins: Max basins to plot; None=all.
        test_start_date: Test start date, format 'YYYYMMDD'.
        test_end_date: Test end date, format 'YYYYMMDD'.
        train_start_date: Training start (parameter sequence start), format 'YYYYMMDD'.
    """
    if model_type != 'dynamic':
        print("This function is for dynamic model parameters only")
        return
    
    print("=" * 50)
    print("Generating dynamic parameter plots")
    print("=" * 50)
    
    param_plots_dir = os.path.join(outpath, 'parameter_plots')
    if not os.path.exists(param_plots_dir):
        os.makedirs(param_plots_dir)
    
    if params.shape[2] == 12:
        param_names = ['ke', 'b', 'wum', 'wlm', 'wm', 'c', 'sm', 'ex', 'ki', 'kg', 'ci', 'cg']
    else:
        param_names = [f'P{i+1}' for i in range(params.shape[2])]

    # Color for each parameter
    param_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 
                   'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'navy']
    
    # Handle dates (can be str or int)
    if isinstance(test_start_date, int):
        test_start_date = str(test_start_date)
    if isinstance(test_end_date, int):
        test_end_date = str(test_end_date)
    if isinstance(train_start_date, int):
        train_start_date = str(train_start_date)

    train_year = int(train_start_date[:4])
    train_month = int(train_start_date[4:6])
    train_day = int(train_start_date[6:8])
    train_start_datetime = datetime(train_year, train_month, train_day)
    
    start_year = int(test_start_date[:4])
    start_month = int(test_start_date[4:6])
    start_day = int(test_start_date[6:8])
    test_start_datetime = datetime(start_year, start_month, start_day)
    
    end_year = int(test_end_date[:4])
    end_month = int(test_end_date[4:6])
    end_day = int(test_end_date[6:8])
    test_end_datetime = datetime(end_year, end_month, end_day)

    
    # Determine test-period length (in days)
    test_days = (test_end_datetime - test_start_datetime).days
    total_time_steps = params.shape[0]

    if total_time_steps == test_days:
        print(f"Parameters are already test-period data: {total_time_steps} days")
        test_params = params
        time_axis = [test_start_datetime + timedelta(days=i) for i in range(test_days)]
    else:
        print(f"Parameters span {total_time_steps} days, test period: {test_days} days")
        print("Extracting test-period parameter data...")

        # Offset from training start to test start
        days_to_test_start = (test_start_datetime - train_start_datetime).days
        start_idx = days_to_test_start
        end_idx = start_idx + test_days
        
        if end_idx <= total_time_steps:
            test_params = params[start_idx:end_idx, :, :, :]
            time_axis = [test_start_datetime + timedelta(days=i) for i in range(test_days)]
            print(f"Extracted test-period parameters: indices {start_idx} to {end_idx-1}")
        else:
            print(f"Warning: Cannot extract full test period (need index {end_idx-1}, only {total_time_steps} days)")
            available_days = total_time_steps - start_idx
            test_params = params[start_idx:, :, :, :]
            time_axis = [test_start_datetime + timedelta(days=i) for i in range(available_days)]
            print(f"Extracted available data: {available_days} days")
    
    ntime = test_params.shape[0]
    
    n_components = test_params.shape[3]
    print(f"Number of components (Nmul): {n_components}")
    
    # Number of basins to plot
    if max_basins is None:
        n_basins_to_plot = len(logtestIDLst)
        print(f"Generating parameter plots for all {n_basins_to_plot} basins")
    else:
        n_basins_to_plot = min(len(logtestIDLst), max_basins)
        print(f"Generating parameter plots for {n_basins_to_plot} basins (limited by max_basins={max_basins})")
    
    for basin_idx in range(n_basins_to_plot):
        basin_id = logtestIDLst[basin_idx]
        
        # Debug: first few basins
        if basin_idx < 3:
            print(f"\nDebug: basin_idx={basin_idx}, basin_id={repr(basin_id)}, type={type(basin_id)}")
        
        # Progress
        progress = (basin_idx + 1) / n_basins_to_plot * 100
        print(f"\rGenerating plots: {basin_idx+1}/{n_basins_to_plot} [{basin_id}] - {progress:.1f}%", end="", flush=True)
        
        basin_params = test_params[:, basin_idx, :, :]

        n_params = basin_params.shape[1]

        # Layout: approximately square grid
        n_cols = int(np.ceil(np.sqrt(n_params)))
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

        # Ensure axes is 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Plot each parameter
        for param_idx in range(n_params):
            row = param_idx // n_cols
            col = param_idx % n_cols
            ax = axes[row, col]
            
            param_name = param_names[param_idx]
            param_color = param_colors[param_idx % len(param_colors)]
            
            param_components = basin_params[:, param_idx, :]
            
            # All components (gray)
            for comp in range(n_components):
                ax.plot(time_axis, param_components[:, comp], 
                       color='lightgray', alpha=0.7, linewidth=0.8)

            # Time-mean for each component
            time_averages = np.mean(param_components, axis=0)
            
            # Find median component (by time-mean)
            median_avg = np.median(time_averages)
            median_comp_idx = np.argmin(np.abs(time_averages - median_avg))

            # Plot median component (colored line)
            ax.plot(time_axis, param_components[:, median_comp_idx], 
                   color=param_color, linewidth=2, 
                   label=f'Median (Comp {median_comp_idx+1})')
            
            # Subplot styling
            ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{param_name}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Time axis formatting (5 evenly spaced ticks, YYYY-MM)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            n_ticks = 5
            tick_positions = np.linspace(0, len(time_axis)-1, n_ticks, dtype=int)
            ax.set_xticks([time_axis[i] for i in tick_positions])
            ax.set_xticklabels([time_axis[i].strftime('%Y-%m') for i in tick_positions], 
                              rotation=0, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
        
        # Hide unused subplots
        for param_idx in range(n_params, n_rows * n_cols):
            row = param_idx // n_cols
            col = param_idx % n_cols
            axes[row, col].set_visible(False)
        
        # Global title
        fig.suptitle(f'Dynamic Parameters - Basin {basin_id}', fontsize=14, fontweight='bold')
        
        # Layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Sanitize filename
        safe_basin_id = str(basin_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_file = os.path.join(param_plots_dir, f"{safe_basin_id}_dynamic_parameters.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 50)
    print(f"Dynamic parameter plots saved to: {param_plots_dir}")
    print("=" * 50)
    
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
    Plot boxplots of NSE and KGE metrics for a given deep learning model.

    Args:
        outpath: Output directory path.
        evaDict: Dictionary with evaluation metrics; must contain 'NSE' and 'KGE'.
        rnn_type: Deep learning model type string (e.g., 'lstm', 'gru').
    """
    print("=" * 50)
    print("Generating metrics boxplot")
    print("=" * 50)
    
    # Matplotlib style
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Prepare data
    nse_values = evaDict['NSE']
    kge_values = evaDict['KGE']
    
    # Medians
    nse_median = np.nanmedian(nse_values)
    kge_median = np.nanmedian(kge_values)
    
    # DataFrame for potential future use (not strictly required for boxplot)
    data = {
        'Metric': ['NSE'] * len(nse_values) + ['KGE'] * len(kge_values),
        'Value': np.concatenate([nse_values, kge_values]),
    }
    df = pd.DataFrame(data)
    
    # Print medians for logging
    print(f"NSE median: {nse_median:.3f}")
    print(f"KGE median: {kge_median:.3f}")
    
    # Figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare boxplot data (NSE first, then KGE)
    metrics = ['NSE', 'KGE']
    boxplot_data = [
        nse_values,
        kge_values
    ]
    
    # Draw boxplot
    bp = ax.boxplot(boxplot_data, labels=metrics, patch_artist=True, widths=0.6)
    
    # Colors
    colors = ['#66c2a5', '#fc8d62']
    
    # Box colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Line colors
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('#2c3e50')
            item.set_linewidth(1.5)
    
    # Outlier markers
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('#e74c3c')
        flier.set_markeredgecolor('none')
        flier.set_markersize(4)
        flier.set_alpha(0.5)
    
    # Map internal type to pretty model name
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
    
    # Titles and labels
    ax.set_title(f'Performance Metrics for {model_name} Model\n(n={len(nse_values)} basins)', 
                fontsize=16, family='Times New Roman')
    ax.set_ylabel('Value', fontsize=14, family='Times New Roman')
    ax.set_xlabel('', fontsize=14, family='Times New Roman')
    
    # Y-axis range to include most data
    y_min = min(np.nanpercentile(nse_values, 1), np.nanpercentile(kge_values, 1))
    y_max = max(np.nanpercentile(nse_values, 99), np.nanpercentile(kge_values, 99))
    
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    
    # Put NSE/KGE medians in upper-right corner
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # For 2 metrics, use a fixed x-position near the right edge
    text_x = 1.45
    text_y_nse = y_max * 0.95
    text_y_kge = y_max * 0.88
    
    # Add median labels with background box
    nse_text = ax.text(text_x, text_y_nse, f'NSE Median: {nse_median:.3f}', 
            ha='right', va='top', fontsize=12, fontweight='bold', family='Times New Roman')
    kge_text = ax.text(text_x, text_y_kge, f'KGE Median: {kge_median:.3f}', 
            ha='right', va='top', fontsize=12, fontweight='bold', family='Times New Roman')
            
    nse_text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    kge_text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # save
    plot_path = os.path.join(outpath, f'Metrics_Boxplot_{model_name}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics boxplot saved to: {plot_path}")
    print("=" * 50)
    
    return plot_path

