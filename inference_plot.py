# inference_plot.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from src.model import CapnoCNN
from main import setup_experiment_folder

def generate_inference_visual():
    print("🧠 Model ve veriler yükleniyor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CapnoCNN().to(device)
    
    # Eğitilen ağırlıkları yükle
    try:
        model.load_state_dict(torch.load('capno_cnn.pth', map_location=device))
        model.eval()
    except FileNotFoundError:
        print("❌ Hata: 'capno_cnn.pth' bulunamadı. Önce main.py ile modeli eğitmelisiniz.")
        return

    # Rastgele bir CapnoBase dosyası seç
    signal_files = glob.glob(os.path.join('data', 'raw', '*_signal.csv'))
    df = pd.read_csv(signal_files[0], low_memory=False, on_bad_lines='skip')
    co2_raw = pd.to_numeric(df['co2_y'], errors='coerce').dropna().values
    
    # 60 Saniyelik (18.000 veri noktası) bir kesit al
    full_signal = co2_raw[30000:48000].copy()
    
    # --- ARTEFAKT ENJEKSİYONU (Görselleştirmek için) ---
    # Sinyalin ortasına (20. ve 40. saniyeler arasına) Sekresyon ekleyelim
    noise_start, noise_end = 6000, 12000 # 20sn - 40sn arası
    full_signal[noise_start:noise_end] += np.random.normal(0, 0.8, 6000)

    window_size = 3000 # 10 saniye
    stride = 1500      # 5 saniye adım (Overlapping)
    
    detections = []

    print("🚀 Kayan pencere analizi (Sliding Window Inference) yapılıyor...")
    with torch.no_grad():
        for i in range(0, len(full_signal) - window_size + 1, stride):
            window = full_signal[i:i+window_size]
            # Tensor hazırlığı [Batch, Channel, Length]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(x)
            prob = torch.sigmoid(output).item()
            
            # Eğer olasılık %50'den büyükse artefakt olarak işaretle
            detections.append({
                'start': i,
                'end': i + window_size,
                'prob': prob,
                'is_artifact': prob > 0.5
            })

    # --- PROFESYONEL GRAFİK ÇİZİMİ ---
    plt.ioff()
    plt.figure(figsize=(16, 6))
    time_axis = np.linspace(0, 60, len(full_signal))
    
    plt.plot(time_axis, full_signal, color='#2c3e50', linewidth=1, label='Sürekli CO2 Akışı (300Hz)')
    
    # Tespit edilen bölgeleri boya
    first_flag = True
    for d in detections:
        if d['is_artifact']:
            start_sec = d['start'] / 300
            end_sec = d['end'] / 300
            label = "Yapay Zeka Tespiti: ARTEFAKT" if first_flag else ""
            plt.axvspan(start_sec, end_sec, color='#e74c3c', alpha=0.2, label=label)
            
            # Olasılık değerini kutunun üzerine yaz
            plt.text((start_sec + end_sec)/2, np.max(full_signal)+0.5, 
                     f"%{d['prob']*100:.1f}", color='#c0392b', 
                     fontsize=10, fontweight='bold', ha='center')
            first_flag = False

    plt.title('Slayt 7: Gerçek Zamanlı Anomali Tespiti (Inference Output)', fontsize=16, fontweight='bold')
    plt.xlabel('Zaman (Saniye)', fontsize=12)
    plt.ylabel('CO2 (Vol%)', fontsize=12)
    plt.ylim(np.min(full_signal)-1, np.max(full_signal)+2)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    # Kaydet
    save_dir = setup_experiment_folder()
    save_path = os.path.join(save_dir, 'slayt7_inference_output.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Harika! Slayt 7 görselin hazır: {save_path}")

if __name__ == "__main__":
    generate_inference_visual()