# src/data_prep.py
import os
import glob
import numpy as np
import pandas as pd

def generate_dataset():
    print("🚀 Building the master CapnoBase dataset...")
    
    raw_data_dir = os.path.join('data', 'raw')
    # Find every signal CSV in the directory automatically
    signal_files = glob.glob(os.path.join(raw_data_dir, '*_signal.csv'))
    
    if not signal_files:
        print(f"❌ No signal files found in {raw_data_dir}.")
        return

    print(f"📂 Found {len(signal_files)} CapnoBase files. Processing...")
    
    window_size = 3000  # CapnoBase is 300Hz. 10 seconds = 3000 data points.
    clean_segments = []
    
    for file in signal_files:
        try:
            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
            
            if 'co2_y' not in df.columns:
                continue
                
            co2_signal = pd.to_numeric(df['co2_y'], errors='coerce').dropna().values
            
            # Slice the continuous wave into 10-second windows
            for i in range(0, len(co2_signal) - window_size, window_size):
                window = co2_signal[i:i+window_size]
                
               # Clinical check: Ensure the sensor is not flatlined.
                # We just check that the peak is at least 2.0 units higher than the trough.
                if np.max(window) - np.min(window) > 2.0:
                    clean_segments.append(window)
        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(file)}: {e}")
            
    clean_segments = np.array(clean_segments)
    num_samples = len(clean_segments)
    
    if num_samples == 0:
        print("❌ No valid capnogram segments extracted. Check the files.")
        return
        
    print(f"✅ Extracted {num_samples} pristine 10-second clinical breath windows.")
    
    # Split the massive dataset and inject artifacts
    half = num_samples // 2
    X = np.copy(clean_segments)
    y = np.zeros(num_samples)
    
    print("🧪 Injecting clinical artifacts (Sekresyon & Yoğunlaşma)...")
    for i in range(half, num_samples):
        y[i] = 1 # Label as Artifact
        
        if i % 2 == 0:
            # Yoğunlaşma: Baseline drift
            drift = np.linspace(0, np.random.uniform(5, 12), window_size)
            X[i] = X[i] + drift
        else:
            # Sekresyon: High-frequency noise
            noise = np.random.normal(0, np.random.uniform(1.5, 3.5), window_size)
            X[i] = X[i] + noise

    # Shuffle the dataset
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X.npy', X.astype(np.float32))
    np.save('data/processed/y.npy', y.astype(np.float32))
    print("💾 High-quality dataset processed and saved to data/processed/")

if __name__ == "__main__":
    generate_dataset()