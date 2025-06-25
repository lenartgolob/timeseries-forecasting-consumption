"""
Visualizations for energy forecasting analysis.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from .config import PROCESSED_FILE

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def generate_plots(run_dir, model_name):
    """Generate all visualizations for the experiment."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics and forecast data
    metrics_df = pd.read_csv(run_dir / "metrics.csv")
    
    # Clean up any infinite or extreme values in the metrics dataframe
    print(f"\n📊 Generating visualizations...")
    print(f"   Original data shape: {metrics_df.shape}")
    
    # Replace infinite values with NaN
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    
    # Check for problematic values
    for col in ['MAE', 'RMSE', 'MAPE', 'nRMSE']:
        if col in metrics_df.columns:
            invalid_count = (~np.isfinite(metrics_df[col])).sum()
            if invalid_count > 0:
                print(f"   Warning: {invalid_count} invalid values in {col}")
    
    try:
        # 1. Multi-Metric Performance Timeline
        plot_performance_timeline(metrics_df, fig_dir)
        
        # 2. Hourly Performance Heatmap
        plot_hourly_heatmap(run_dir, model_name, fig_dir)
        
        # 3. Error Distribution
        plot_error_distribution(metrics_df, fig_dir)
        
        # 4. Best/Worst Case Studies
        plot_case_studies(run_dir, model_name, metrics_df, fig_dir)
        
        # 5. Consumption vs Error Analysis
        plot_consumption_analysis(metrics_df, fig_dir)
        
        print(f"📈 All visualizations saved to: {fig_dir}")
        
    except Exception as e:
        print(f"   Error in visualization generation: {e}")
        # Try to generate individual plots to identify which one is failing
        plots = [
            ("performance_timeline", lambda: plot_performance_timeline(metrics_df, fig_dir)),
            ("hourly_heatmap", lambda: plot_hourly_heatmap(run_dir, model_name, fig_dir)),
            ("error_distribution", lambda: plot_error_distribution(metrics_df, fig_dir)),
            ("case_studies", lambda: plot_case_studies(run_dir, model_name, metrics_df, fig_dir)),
            ("consumption_analysis", lambda: plot_consumption_analysis(metrics_df, fig_dir))
        ]
        
        for plot_name, plot_func in plots:
            try:
                plot_func()
                print(f"   ✓ {plot_name}")
            except Exception as plot_error:
                print(f"   ✗ {plot_name}: {plot_error}")
        
        raise e


def plot_performance_timeline(metrics_df, fig_dir):
    """Multi-metric performance across all walks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    walks = metrics_df['walk']
    ax.plot(walks, metrics_df['MAE'], 'o-', label='MAE', linewidth=2, markersize=4)
    ax.plot(walks, metrics_df['RMSE'], 's-', label='RMSE', linewidth=2, markersize=4)
    
    # Filter MAPE for plotting (remove inf/extreme values and NaN)
    mape_filtered = metrics_df['MAPE'].replace([np.inf, -np.inf], np.nan).dropna()
    mape_filtered = mape_filtered.clip(upper=100)  # Cap at 100% for better visualization
    mape_walks = metrics_df.loc[mape_filtered.index, 'walk']
    ax.plot(mape_walks, mape_filtered, '^-', label='MAPE (%)', linewidth=2, markersize=4)
    
    ax.plot(walks, metrics_df['nRMSE'], 'd-', label='nRMSE', linewidth=2, markersize=4)
    
    ax.set_xlabel('Walk Number (Days into 2025)', fontsize=12)
    ax.set_ylabel('Error Metric Value', fontsize=12)
    ax.set_title('Model Performance Evolution Across Testing Period', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "performance_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Performance timeline")


def plot_hourly_heatmap(run_dir, model_name, fig_dir):
    """Heatmap of prediction errors by hour of day."""
    # Load all forecast files and combine with actual data
    df_truth = pd.read_parquet(PROCESSED_FILE)
    df_truth['ds'] = pd.to_datetime(df_truth['ds'])
    
    all_errors = []
    forecast_files = list((run_dir / "forecasts").glob(f"{model_name}_*.csv"))
    
    for fpath in forecast_files:
        cut_idx = int(fpath.stem.split('_')[-1])
        forecast_df = pd.read_csv(fpath)
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        
        # Get corresponding actual values
        actual_slice = df_truth.iloc[cut_idx:cut_idx+24].copy()
        
        # Combine and calculate errors
        combined = forecast_df.merge(actual_slice[['ds', 'y']], on='ds', how='inner')
        combined['error'] = np.abs(combined['y'] - combined['y_hat'])
        combined['hour'] = combined['ds'].dt.hour
        
        all_errors.append(combined[['hour', 'error']])
    
    # Combine all errors
    error_data = pd.concat(all_errors, ignore_index=True)
    
    # Create heatmap data
    hourly_stats = error_data.groupby('hour')['error'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Mean errors by hour
    hours = hourly_stats['hour']
    mean_errors = hourly_stats['mean']
    
    bars1 = ax1.bar(hours, mean_errors, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (kWh)', fontsize=12)
    ax1.set_title('Prediction Accuracy by Hour of Day', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, mean_errors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Standard deviation by hour
    bars2 = ax2.bar(hours, hourly_stats['std'], color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Error Standard Deviation (kWh)', fontsize=12)
    ax2.set_title('Prediction Variability by Hour of Day', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "hourly_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Hourly performance heatmap")


def plot_error_distribution(metrics_df, fig_dir):
    """Distribution of prediction errors across all walks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Helper function to safely plot histogram
    def safe_hist(ax, data, column_name, color, xlabel, title):
        clean_data = data.dropna()
        clean_data = clean_data[np.isfinite(clean_data)]
        
        if len(clean_data) > 0:
            ax.hist(clean_data, bins=20, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(clean_data.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {clean_data.mean():.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No valid {column_name} values', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # MAE distribution
    safe_hist(axes[0], metrics_df['MAE'], 'MAE', 'steelblue', 'MAE (kWh)', 'MAE Distribution')
    
    # RMSE distribution  
    safe_hist(axes[1], metrics_df['RMSE'], 'RMSE', 'green', 'RMSE (kWh)', 'RMSE Distribution')
    
    # MAPE distribution (with additional filtering)
    mape_data = metrics_df['MAPE'].dropna()
    mape_data = mape_data[np.isfinite(mape_data)]
    mape_data = mape_data[(mape_data >= 0) & (mape_data <= 200)]  # Remove negative and extreme values
    
    if len(mape_data) > 0:
        axes[2].hist(mape_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[2].axvline(mape_data.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mape_data.mean():.2f}%')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No valid MAPE values', ha='center', va='center',
                    transform=axes[2].transAxes, fontsize=12)
    
    axes[2].set_xlabel('MAPE (%)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('MAPE Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # nRMSE distribution
    safe_hist(axes[3], metrics_df['nRMSE'], 'nRMSE', 'purple', 'nRMSE', 'nRMSE Distribution')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Error distribution")


def plot_case_studies(run_dir, model_name, metrics_df, fig_dir):
    """Best vs worst performing days comparison."""
    # Find best and worst days (use cut_idx for file lookup)
    best_walks = metrics_df.nsmallest(3, 'MAE')['cut_idx'].values
    worst_walks = metrics_df.nlargest(3, 'MAE')['cut_idx'].values
    
    df_truth = pd.read_parquet(PROCESSED_FILE)
    df_truth['ds'] = pd.to_datetime(df_truth['ds'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Best vs Worst Prediction Days Comparison', fontsize=16, fontweight='bold')
    
    # Plot best days
    for i, cut_idx in enumerate(best_walks):
        ax = axes[0, i]
        mae_val = metrics_df[metrics_df['cut_idx'] == cut_idx]['MAE'].iloc[0]
        
        # Load forecast and actual data
        forecast_df = pd.read_csv(run_dir / "forecasts" / f"{model_name}_{cut_idx}.csv")
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        actual_slice = df_truth.iloc[cut_idx:cut_idx+24].copy()
        
        # Plot
        ax.plot(actual_slice['ds'], actual_slice['y'], 'o-', label='Actual', linewidth=2, markersize=6)
        ax.plot(forecast_df['ds'], forecast_df['y_hat'], 's--', label='Predicted', linewidth=2, markersize=6)
        
        ax.set_title(f'Best Day #{i+1}\nMAE: {mae_val:.3f} kWh', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy (kWh)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot worst days
    for i, cut_idx in enumerate(worst_walks):
        ax = axes[1, i]
        mae_val = metrics_df[metrics_df['cut_idx'] == cut_idx]['MAE'].iloc[0]
        
        # Load forecast and actual data
        forecast_df = pd.read_csv(run_dir / "forecasts" / f"{model_name}_{cut_idx}.csv")
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        actual_slice = df_truth.iloc[cut_idx:cut_idx+24].copy()
        
        # Plot
        ax.plot(actual_slice['ds'], actual_slice['y'], 'o-', label='Actual', linewidth=2, markersize=6)
        ax.plot(forecast_df['ds'], forecast_df['y_hat'], 's--', label='Predicted', linewidth=2, markersize=6)
        
        ax.set_title(f'Worst Day #{i+1}\nMAE: {mae_val:.3f} kWh', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Energy (kWh)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "case_studies.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Best/worst case studies")


def plot_consumption_analysis(metrics_df, fig_dir):
    """Consumption vs error analysis with dual panels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Scatter plot - Consumption vs Error (filter NaN values)
    valid_mask = ~(np.isnan(metrics_df['avg_consumption']) | np.isnan(metrics_df['MAE']))
    valid_data = metrics_df[valid_mask]
    
    if len(valid_data) > 0:
        scatter = ax1.scatter(valid_data['avg_consumption'], valid_data['MAE'], 
                             c=range(len(valid_data)), cmap='viridis', alpha=0.7, s=50)
    else:
        ax1.text(0.5, 0.5, 'No valid data for scatter plot', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
    ax1.set_xlabel('Average Consumption (kWh/hour)', fontsize=12)
    ax1.set_ylabel('MAE (kWh/hour)', fontsize=12)
    ax1.set_title('Prediction Error vs Consumption Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Filter data for correlation and trend analysis (remove NaN values)
    valid_mask = ~(np.isnan(metrics_df['avg_consumption']) | np.isnan(metrics_df['MAE']))
    if valid_mask.sum() > 1:  # Need at least 2 points for correlation
        valid_consumption = metrics_df.loc[valid_mask, 'avg_consumption']
        valid_mae = metrics_df.loc[valid_mask, 'MAE']
        
        # Add correlation coefficient
        correlation = np.corrcoef(valid_consumption, valid_mae)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax1.transAxes, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add trend line
        z = np.polyfit(valid_consumption, valid_mae, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_consumption.min(), valid_consumption.max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend line')
        ax1.legend()
    else:
        ax1.text(0.05, 0.95, 'Insufficient data for correlation', 
                transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Walk Number (Days into 2025)', fontsize=10)
    
    # Panel 2: Timeline - Consumption and MAE over time
    ax2_twin = ax2.twinx()
    
    # Plot consumption on left axis
    line1 = ax2.plot(metrics_df['walk'], metrics_df['avg_consumption'], 
                    'b-o', linewidth=2, markersize=4, label='Avg Consumption', alpha=0.7)
    ax2.set_xlabel('Walk Number (Days into 2025)', fontsize=12)
    ax2.set_ylabel('Average Consumption (kWh/hour)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Plot MAE on right axis (filter out NaN values)
    mae_mask = ~np.isnan(metrics_df['MAE'])
    mae_walks = metrics_df.loc[mae_mask, 'walk']
    mae_values = metrics_df.loc[mae_mask, 'MAE']
    line2 = ax2_twin.plot(mae_walks, mae_values, 
                         'r-s', linewidth=2, markersize=4, label='MAE', alpha=0.7)
    ax2_twin.set_ylabel('MAE (kWh/hour)', fontsize=12, color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    ax2.set_title('Consumption and Error Timeline', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "consumption_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Consumption vs error analysis")
