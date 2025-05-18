
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
from pathlib import Path

# Define RSA operations in OpenSSL 
RSA_OPERATIONS = {
    "sign": {
        "description": "Digital signature operation",
        "complexity": "O(n²)",
        "mathematical_operation": "m^d mod n, where d is the private key, n is the modulus"
    },
    "verify": {
        "description": "Signature verification operation",
        "complexity": "O(n)",
        "mathematical_operation": "s^e mod n, where e is the public key, n is the modulus"
    },
    "encrypt": {
        "description": "RSA encryption operation",
        "complexity": "O(n)",
        "mathematical_operation": "m^e mod n, where e is the public key, n is the modulus"
    },
    "decrypt": {
        "description": "RSA decryption operation",
        "complexity": "O(n²)",
        "mathematical_operation": "c^d mod n, where d is the private key, n is the modulus"
    }
}

# For RSA-2048, operations per test based on OpenSSL benchmarks
RSA_2048_OPERATIONS_PER_TEST = {
    "sign": 213,    # Approximate number of RSA-2048 sign operations in a typical 20s test
    "verify": 4096, # Approximate number of RSA-2048 verify operations in a typical 20s test
    "encrypt": 2048, # Approximate number of RSA-2048 encrypt operations in a typical 20s test
    "decrypt": 213   # Approximate number of RSA-2048 decrypt operations in a typical 20s test
}

# Total mathematical operations for RSA-2048 (based on modular exponentiation)
# For each RSA operation using naive implementation (not counting optimizations like CRT):
MATHEMATICAL_OPS_PER_RSA = {
    "sign": 3_160_000,    # ~3.16M arithmetic operations for RSA-2048 sign 
    "verify": 170_000,    # ~170K arithmetic operations for RSA-2048 verify (public exponent is small)
    "encrypt": 170_000,   # ~170K arithmetic operations for RSA-2048 encrypt (public exponent is small)
    "decrypt": 3_160_000  # ~3.16M arithmetic operations for RSA-2048 decrypt
}

def calculate_ops_per_second(row):
    """Calculate estimated operations per second based on thread count"""
    # This is a rough estimate of total RSA operations per second
    total_ops = 0
    for op_type, count in RSA_2048_OPERATIONS_PER_TEST.items():
        # Multiply by thread count to estimate parallel operations
        total_ops += count * row['threads']
    
    # Divide by runtime to get ops per second
    return total_ops / row['runtime_seconds']

def calculate_total_calculations(row):
    """Calculate total mathematical operations performed during the test"""
    total_math_ops = 0
    for op_type, count in RSA_2048_OPERATIONS_PER_TEST.items():
        # For each RSA operation type, calculate the arithmetic operations
        math_ops_for_type = count * MATHEMATICAL_OPS_PER_RSA[op_type]
        # Multiply by thread count for parallel operations
        total_math_ops += math_ops_for_type * row['threads']
    
    # Return operations in billions for readability
    return total_math_ops / 1_000_000_000

def parse_openssl_csv(csv_path):
    """Parse the OpenSSL speed test results CSV file"""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Clean the data - handle NaN or invalid power values
        df['power_watts'] = pd.to_numeric(df['power_watts'], errors='coerce')
        
        # Filter out invalid power values (negative or extremely high)
        df = df[(df['power_watts'] > 0) & (df['power_watts'] < 1000)]
        
        # Calculate additional metrics
        df['ops_per_second'] = df.apply(calculate_ops_per_second, axis=1)
        df['billions_calculations'] = df.apply(calculate_total_calculations, axis=1)
        df['ops_per_watt'] = df['ops_per_second'] / df['power_watts']
        
        if 'energy_joules' not in df.columns:
            # Calculate energy if not in the CSV
            df['energy_joules'] = df['power_watts'] * df['runtime_seconds']
        
        return df
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)

def plot_results(df, output_dir):
    """Create various plots from the analyzed data"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    
    # Figure 1: Thread scaling overview (4 subplots)
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Operations per second vs. Threads
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['threads'], df['ops_per_second'], marker='o', color=colors[0], linewidth=2)
    ax1.set_title('RSA Operations per Second vs. Thread Count', fontsize=14)
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Operations per Second')
    ax1.grid(True)
    
    # Plot 2: Power consumption vs. Threads
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['threads'], df['power_watts'], marker='o', color=colors[1], linewidth=2)
    ax2.set_title('Power Consumption vs. Thread Count', fontsize=14)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Power (Watts)')
    ax2.grid(True)
    
    # Plot 3: Operations per Watt vs. Threads (Energy Efficiency)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['threads'], df['ops_per_watt'], marker='o', color=colors[2], linewidth=2)
    ax3.set_title('Energy Efficiency vs. Thread Count', fontsize=14)
    ax3.set_xlabel('Thread Count')
    ax3.set_ylabel('Operations per Watt')
    ax3.grid(True)
    
    # Plot 4: Total Mathematical Calculations vs Threads
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['threads'], df['billions_calculations'], marker='o', color=colors[3], linewidth=2)
    ax4.set_title('Total Mathematical Operations vs. Thread Count', fontsize=14)
    ax4.set_xlabel('Thread Count')
    ax4.set_ylabel('Billions of Operations')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'openssl_overview.png'), dpi=300)
    
    # Figure 2: Efficiency focus - normalized performance per thread
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate normalized efficiency (ops per second per thread)
    df['ops_per_thread'] = df['ops_per_second'] / df['threads']
    
    ax.plot(df['threads'], df['ops_per_thread'], marker='o', color=colors[4], linewidth=2)
    ax.set_title('RSA Performance Scaling Efficiency', fontsize=16)
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Operations per Second per Thread', fontsize=12)
    ax.grid(True)
    
    # Add efficiency trend line
    z = np.polyfit(df['threads'], df['ops_per_thread'], 1)
    p = np.poly1d(z)
    ax.plot(df['threads'], p(df['threads']), "r--", alpha=0.7)
    
    # Add annotation about efficiency trend
    slope = z[0]
    if slope < 0:
        trend_text = f"Decreasing efficiency: {slope:.2f} ops/thread lost per additional thread"
    else:
        trend_text = f"Increasing efficiency: {slope:.2f} ops/thread gained per additional thread"
    
    ax.annotate(trend_text, xy=(0.5, 0.05), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'openssl_efficiency.png'), dpi=300)
    
    # Figure 3: Power efficiency analysis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set up twin axes for power and performance
    ax2 = ax.twinx()
    
    # Plot power consumption
    lns1 = ax.plot(df['threads'], df['power_watts'], marker='o', color='red', label='Power (W)', linewidth=2)
    ax.set_ylabel('Power (Watts)', color='red', fontsize=12)
    ax.tick_params(axis='y', labelcolor='red')
    
    # Plot operations per second
    lns2 = ax2.plot(df['threads'], df['ops_per_second'], marker='s', color='blue', label='Ops/s', linewidth=2)
    ax2.set_ylabel('Operations per Second', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Add legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    
    ax.set_title('Power Consumption vs. Performance', fontsize=16)
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.grid(True)
    
    # Find and mark the optimal thread count for energy efficiency
    max_efficiency_idx = df['ops_per_watt'].idxmax()
    optimal_threads = df.loc[max_efficiency_idx, 'threads']
    ax.axvline(x=optimal_threads, color='green', linestyle='--', alpha=0.7)
    
    # Annotate optimal point
    ax.annotate(f'Best Efficiency\n{optimal_threads} Threads',
                xy=(optimal_threads, df.loc[max_efficiency_idx, 'power_watts']),
                xytext=(optimal_threads + 2, df.loc[max_efficiency_idx, 'power_watts'] * 1.2),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'openssl_power_analysis.png'), dpi=300)
    
    # Generate a table with complete results
    fig, ax = plt.subplots(figsize=(12, len(df)*0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data with better formatting
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"{row['threads']:.0f}",
            f"{row['runtime_seconds']:.2f}",
            f"{row['power_watts']:.2f}",
            f"{row['ops_per_second']:.0f}",
            f"{row['ops_per_watt']:.0f}",
            f"{row['billions_calculations']:.1f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Threads', 'Runtime (s)', 'Power (W)', 
                               'Ops/Second', 'Ops/Watt', 'Billion Calcs'],
                     loc='center', cellLoc='center',
                     colWidths=[0.1, 0.15, 0.15, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('OpenSSL RSA-2048 Performance Results', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'openssl_results_table.png'), dpi=300)
    
    print(f"Plots saved to {output_dir}")
    
    # Generate explanatory text file about mathematical operations
    generate_math_explanation(output_dir)

def generate_math_explanation(output_dir):
    """Generate a text file explaining the mathematical operations in RSA"""
    explanation = """# Mathematical Operations in OpenSSL RSA Benchmarks

## RSA-2048 Operations

The OpenSSL 'speed' benchmark for RSA-2048 performs four main cryptographic operations:

1. **Sign**: Uses private key to create digital signatures
   - Mathematical representation: m^d mod n
   - Private key operation (slower)
   - Estimated arithmetic operations: ~3.16 million per RSA-2048 sign

2. **Verify**: Uses public key to verify signatures
   - Mathematical representation: s^e mod n
   - Public key operation (faster due to small public exponent)
   - Estimated arithmetic operations: ~170,000 per RSA-2048 verify

3. **Encrypt**: Uses public key to encrypt data
   - Mathematical representation: m^e mod n
   - Public key operation (faster)
   - Estimated arithmetic operations: ~170,000 per RSA-2048 encrypt

4. **Decrypt**: Uses private key to decrypt data
   - Mathematical representation: c^d mod n
   - Private key operation (slower)
   - Estimated arithmetic operations: ~3.16 million per RSA-2048 decrypt

## Core Mathematical Operations

The fundamental mathematical operation in RSA is **modular exponentiation** (calculating a^b mod n).
This involves:

- Modular multiplications
- Modular squaring operations
- Modular reductions

For a 2048-bit RSA key:
- The modulus n is 2048 bits long
- A full exponentiation requires approximately 3,072 modular multiplications for private key operations
- Each modular multiplication requires approximately 1,000 elementary operations
- Public key operations are much faster because the public exponent is typically small (usually 65537)

## OpenSSL Optimizations

OpenSSL implements several optimizations for these operations:

1. **Chinese Remainder Theorem (CRT)**: Speeds up private key operations by about 4x
2. **Montgomery multiplication**: Reduces the cost of modular reductions
3. **Windowing techniques**: Reduce the number of multiplications needed in exponentiation
4. **Assembly optimizations**: Hardware-specific code paths for maximum performance

## Total Computational Work

A typical 20-second OpenSSL RSA-2048 test on a single thread performs:
- ~213 sign operations
- ~213 decrypt operations
- ~4,096 verify operations
- ~2,048 encrypt operations

This translates to approximately 1.4 billion elementary arithmetic operations per second on a modern CPU core.
When using multiple threads, this computational work scales nearly linearly until hardware limits are reached.
"""

    with open(os.path.join(output_dir, 'rsa_math_explanation.md'), 'w') as f:
        f.write(explanation)
    
    print(f"Mathematical operations explanation saved to {output_dir}/rsa_math_explanation.md")

if __name__ == "__main__":
    # Default paths
    default_input = "./openssl_speed_results/openssl_speed_results.csv"
    default_output = "./openssl_speed_results/plots"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_input
        
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output
    
    print(f"Analyzing OpenSSL performance data from: {csv_path}")
    df = parse_openssl_csv(csv_path)
    plot_results(df, output_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Thread range: {df['threads'].min()} to {df['threads'].max()}")
    print(f"Average power consumption: {df['power_watts'].mean():.2f} W")
    print(f"Maximum operations per second: {df['ops_per_second'].max():.0f}")
    
    # Find optimal thread configuration
    max_ops_idx = df['ops_per_second'].idxmax()
    max_eff_idx = df['ops_per_watt'].idxmax()
    
    print(f"\nOptimal thread count for maximum performance: {df.loc[max_ops_idx, 'threads']:.0f}")
    print(f"Optimal thread count for energy efficiency: {df.loc[max_eff_idx, 'threads']:.0f}")
