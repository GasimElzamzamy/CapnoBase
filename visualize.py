# visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from src.model import CapnoCNN

def setup_experiment_folder(base_dir="results"):
    """Creates a new incrementally numbered folder like results/1_experiment/"""
    os.makedirs(base_dir, exist_ok=True)
    
    # Find all folders ending in '_experiment'
    existing_dirs = glob.glob(os.path.join(base_dir, '*_experiment'))
    
    if not existing_dirs:
        next_num = 1
    else:
        # Extract the numbers from the folder names and find the max
        nums = []
        for d in existing_dirs:
            folder_name = os.path.basename(d)
            try:
                nums.append(int(folder_name.split('_')[0]))
            except ValueError:
                continue
        next_num = max(nums) + 1 if nums else 1
        
    exp_dir = os.path.join(base_dir, f"{next_num}_experiment")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def run_inference_visualization():
    print("🔍 Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CapnoCNN().to(device)
    
    try:
        model.load_state_dict(torch.load('capno_cnn.pth', map_location=device))
    except FileNotFoundError:
        print("⚠️ Model weights not found. Run main.py first to save 'capno_cnn.pth'.")
        return
        
    model.eval()

    print("📈 Fetching continuous test data...")
    file_path = os.path.join('data', 'raw', 'uq_vsd_case01_fulldata_11.csv')
    df = pd.read_csv(file_path, low_memory=False)
    co2_signal = pd.to_numeric(df['CO2'], errors='coerce').dropna().values
    
    start_idx = 10000 
    end_idx = start_idx + 6000
    test_segment = co2_signal[start_idx:end_idx]
    
    # Inject synthetic artifact for testing
    noise_start, noise_end = 2500, 3500
    test_segment[noise_start:noise_end] += np.random.normal(0, 3.0, 1000)

    window_size = 1000 
    stride = 500       
    predictions = []
    
    print("🧠 Running sliding window inference...")
    with torch.no_grad():
        for i in range(0, len(test_segment) - window_size + 1, stride):
            window = test_segment[i:i+window_size]
            tensor_window = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(tensor_window)
            prob = torch.sigmoid(output).item()
            is_artifact = prob > 0.5
            
            predictions.append({
                'start': i,
                'end': i + window_size,
                'is_artifact': is_artifact,
                'confidence': prob
            })

    print("📊 Generating Plot...")
    
    # Set Matplotlib to non-interactive mode so it doesn't pop up
    plt.ioff() 
    plt.figure(figsize=(15, 5))
    
    time_axis = np.linspace(0, 60, len(test_segment))
    plt.plot(time_axis, test_segment, color='black', linewidth=1.5, label='Raw Capnogram')
    
    for pred in predictions:
        if pred['is_artifact']:
            start_sec = pred['start'] / 100.0
            end_sec = pred['end'] / 100.0
            plt.axvspan(start_sec, end_sec, color='red', alpha=0.3, label='Artifact Detected' if 'Artifact Detected' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(start_sec + 1, max(test_segment) + 2, f"Anomaly\n({pred['confidence']*100:.1f}%)", color='red', weight='bold')

    plt.title('Real-Time AI Artifact Detection via Sliding Window')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('CO2 Amplitude (mmHg)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the file silently instead of showing it
    save_dir = setup_experiment_folder()
    save_path = os.path.join(save_dir, 'inference_sliding_window.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Free up memory
    
    print(f"✅ Success! Plot silently saved to: {save_path}")

if __name__ == "__main__":
    run_inference_visualization()