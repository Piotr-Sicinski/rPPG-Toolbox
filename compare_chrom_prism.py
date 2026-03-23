#!/usr/bin/env python3
"""
Script to create histograms comparing CHROM and PRISM methods
from nature_mae_all.csv data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data():
    """Load the nature_mae_all.csv data"""
    data_file = Path("nature_mae_all.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found!")
    
    df = pd.read_csv(data_file)
    return df

def create_histograms(df):
    """Create histograms for CHROM and PRISM methods"""
    
    # Extract CHROM and PRISM data
    chrom_data = df['CHROM'].dropna()
    prism_data = df['prism'].dropna()
    
    print(f"CHROM data points: {len(chrom_data)}")
    print(f"PRISM data points: {len(prism_data)}")
    print(f"CHROM range: {chrom_data.min():.3f} - {chrom_data.max():.3f}")
    print(f"PRISM range: {prism_data.min():.3f} - {prism_data.max():.3f}")
    
    # Create figure with two subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # First histogram: CHROM
    ax1.hist(chrom_data, bins=30, alpha=0.7, color='darkmagenta', edgecolor='black')
    ax1.set_title('CHROM - MAE Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics to CHROM plot
    chrom_mean = chrom_data.mean()
    chrom_std = chrom_data.std()
    chrom_median = chrom_data.median()
    ax1.axvline(chrom_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {chrom_mean:.3f}')
    ax1.axvline(chrom_median, color='blue', linestyle=':', linewidth=2, 
                label=f'Median: {chrom_median:.3f}')
    ax1.legend()
    
    # Second histogram: PRISM
    ax2.hist(prism_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('PRISM - MAE Distribution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to PRISM plot
    prism_mean = prism_data.mean()
    prism_std = prism_data.std()
    prism_median = prism_data.median()
    ax2.axvline(prism_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {prism_mean:.3f}')
    ax2.axvline(prism_median, color='blue', linestyle=':', linewidth=2, 
                label=f'Median: {prism_median:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('chrom_vs_prism_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nCHROM Statistics:")
    print(f"  Mean: {chrom_mean:.3f}")
    print(f"  Median: {chrom_median:.3f}")
    print(f"  Std Dev: {chrom_std:.3f}")
    print(f"  Min: {chrom_data.min():.3f}")
    print(f"  Max: {chrom_data.max():.3f}")
    print(f"  25th percentile: {chrom_data.quantile(0.25):.3f}")
    print(f"  75th percentile: {chrom_data.quantile(0.75):.3f}")
    
    print(f"\nPRISM Statistics:")
    print(f"  Mean: {prism_mean:.3f}")
    print(f"  Median: {prism_median:.3f}")
    print(f"  Std Dev: {prism_std:.3f}")
    print(f"  Min: {prism_data.min():.3f}")
    print(f"  Max: {prism_data.max():.3f}")
    print(f"  25th percentile: {prism_data.quantile(0.25):.3f}")
    print(f"  75th percentile: {prism_data.quantile(0.75):.3f}")
    
    # Compare the methods
    print(f"\nComparison:")
    print(f"  CHROM mean vs PRISM mean: {chrom_mean:.3f} vs {prism_mean:.3f}")
    print(f"  Difference (CHROM - PRISM): {chrom_mean - prism_mean:.3f}")
    print(f"  PRISM is {'better' if prism_mean < chrom_mean else 'worse'} on average")
    
    return chrom_data, prism_data

def create_comparison_histogram(chrom_data, prism_data):
    """Create an overlaid comparison histogram"""
    
    plt.figure(figsize=(12, 8))
    
    # Determine common bins for fair comparison
    min_val = min(chrom_data.min(), prism_data.min())
    max_val = max(chrom_data.max(), prism_data.max())
    bins = np.linspace(min_val, max_val, 30)
    
    # Create overlaid histograms
    plt.hist(chrom_data, bins=bins, alpha=0.6, color='darkmagenta', 
             label=f'CHROM (n={len(chrom_data)})', edgecolor='black')
    plt.hist(prism_data, bins=bins, alpha=0.6, color='blue', 
             label=f'PRISM (n={len(prism_data)})', edgecolor='black')
    
    plt.title('CHROM vs PRISM - MAE Distribution Comparison', 
              fontweight='bold', fontsize=16)
    plt.xlabel('MAE (Mean Absolute Error)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add mean lines
    plt.axvline(chrom_data.mean(), color='darkmagenta', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'CHROM Mean: {chrom_data.mean():.3f}')
    plt.axvline(prism_data.mean(), color='blue', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'PRISM Mean: {prism_data.mean():.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('chrom_vs_prism_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_difference_plot(chrom_data, prism_data):
    """Create a difference plot showing PRISM - CHROM counts per bin"""
    
    plt.figure(figsize=(14, 8))
    
    # Use smaller bins for more detailed comparison
    min_val = min(chrom_data.min(), prism_data.min())
    max_val = max(chrom_data.max(), prism_data.max())
    bins = np.linspace(min_val, max_val, 50)  # More bins for finer detail
    
    # Calculate histograms
    chrom_counts, bin_edges = np.histogram(chrom_data, bins=bins)
    prism_counts, _ = np.histogram(prism_data, bins=bins)
    
    # Calculate difference (PRISM - CHROM)
    diff_counts = prism_counts - chrom_counts
    
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Create the difference bar plot
    colors = ['red' if diff < 0 else 'green' for diff in diff_counts]
    bars = plt.bar(bin_centers, diff_counts, width=bin_width * 0.8, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.title('Difference Plot: PRISM - CHROM Sample Counts per Bin', 
              fontweight='bold', fontsize=16)
    plt.xlabel('MAE (Mean Absolute Error)', fontsize=12)
    plt.ylabel('Difference in Sample Count (PRISM - CHROM)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='PRISM has more samples'),
                       Patch(facecolor='red', alpha=0.7, label='CHROM has more samples')]
    plt.legend(handles=legend_elements)
    
    # Add summary statistics as text
    positive_diff = np.sum(diff_counts > 0)
    negative_diff = np.sum(diff_counts < 0)
    zero_diff = np.sum(diff_counts == 0)
    
    stats_text = f'Bins where PRISM > CHROM: {positive_diff}\n'
    stats_text += f'Bins where CHROM > PRISM: {negative_diff}\n'
    stats_text += f'Bins with equal counts: {zero_diff}\n'
    stats_text += f'Max difference: {np.max(diff_counts)}\n'
    stats_text += f'Min difference: {np.min(diff_counts)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prism_chrom_difference_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some analysis
    print(f"\nDifference Plot Analysis:")
    print(f"  Total bins: {len(diff_counts)}")
    print(f"  Bins where PRISM has more samples: {positive_diff}")
    print(f"  Bins where CHROM has more samples: {negative_diff}")
    print(f"  Bins with equal sample counts: {zero_diff}")
    print(f"  Maximum difference (PRISM advantage): {np.max(diff_counts)}")
    print(f"  Minimum difference (CHROM advantage): {np.min(diff_counts)}")
    print(f"  Sum of all differences: {np.sum(diff_counts)} (should equal difference in total samples)")
    
    return diff_counts, bin_centers

def main():
    """Main function"""
    try:
        # Load data
        print("Loading data from nature_mae_all.csv...")
        df = load_data()
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Create histograms
        print("\nCreating histograms...")
        chrom_data, prism_data = create_histograms(df)
        
        # Create comparison histogram
        print("\nCreating comparison histogram...")
        create_comparison_histogram(chrom_data, prism_data)
        
        # Create difference plot
        print("\nCreating difference plot...")
        diff_counts, bin_centers = create_difference_plot(chrom_data, prism_data)
        
        print("\nDone! Generated plots:")
        print("  - chrom_vs_prism_histograms.png (separate histograms)")
        print("  - chrom_vs_prism_comparison.png (overlaid comparison)")
        print("  - prism_chrom_difference_plot.png (difference plot)")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
