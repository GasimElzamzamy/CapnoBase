# plot_signals.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import setup_experiment_folder

def save_signal_comparison():
    print("📈 Locating a sample from the CapnoBase dataset...")
    
    # Dynamically find the first CapnoBase signal file
    signal_files = glob.glob(os.path.join('data', 'raw', '*_signal.csv'))
    if not signal_files:
        print("❌ Could not find any CapnoBase *_signal.csv files in data/raw/")
        return
        
    file_path = signal_files[6]
    print(f"📄 Reading {os.path.basename(file_path)}...")
    
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    
    if 'co2_y' not in df.columns:
        print(f"❌ Column 'co2_y' not found in {file_path}")
        return
        
    co2_signal = pd.to_numeric(df['co2_y'], errors='coerce').dropna().values
    
    # Extract one clean 10-second window (3000 samples at 300Hz)
    window_size = 3000
    clean_sample = None
    
    for i in range(0, len(co2_signal) - window_size, window_size):
        window = co2_signal[i:i+window_size]
        
        # Clinical check: Ensure it's not a flatline (amplitude > 2.0)
        if np.max(window) - np.min(window) > 2.0:
            clean_sample = window
            break

    if clean_sample is None:
        print("❌ Could not isolate a valid breathing window.")
        return

    # Generate the two artificial anomalies
    # Note: Mathematical injections are scaled down to match the smaller CapnoBase amplitude
    
    # 1. Sekresyon (Secretion)
    sekresyon_sample = np.copy(clean_sample)
    noise = np.random.normal(0, 0.4, window_size) 
    sekresyon_sample += noise
    
    # 2. Yoğunlaşma (Condensation)
    yogunlasma_sample = np.copy(clean_sample)
    drift = np.linspace(0, 2.5, window_size) 
    yogunlasma_sample += drift

    # Plotting configuration (Silent saving)
    plt.ioff()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_axis = np.linspace(0, 10, window_size)

    # Plot 1: Clean Baseline
    axes[0].plot(time_axis, clean_sample, color='teal', linewidth=1.5)
    axes[0].set_title('Original Clean Clinical Signal (Class 0)', weight='bold')
    axes[0].set_ylabel('CO2 (Vol%)')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Sekresyon
    axes[1].plot(time_axis, sekresyon_sample, color='crimson', linewidth=1.2)
    axes[1].set_title('Injected Artifact: Sekresyon (Class 1 - High Frequency Noise)', weight='bold')
    axes[1].set_ylabel('CO2 (Vol%)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Yoğunlaşma
    axes[2].plot(time_axis, yogunlasma_sample, color='darkorange', linewidth=1.5)
    axes[2].set_title('Injected Artifact: Yoğunlaşma (Class 1 - Upward Baseline Drift)', weight='bold')
    axes[2].set_xlabel('Time (Seconds)')
    axes[2].set_ylabel('CO2 (Vol%)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Determine experiment folder and save
    save_dir = setup_experiment_folder()
    save_path = os.path.join(save_dir, 'capnobase_modifications_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparison plot successfully saved to: {save_path}")

if __name__ == "__main__":
    save_signal_comparison()