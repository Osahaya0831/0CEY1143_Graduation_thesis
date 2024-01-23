# eeg_analysis.py
import glob
import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm  # 進行度バー表示用


# EEGデータを読み込む関数
def load_eeg_data(filename):
    df = pd.read_excel(filename, usecols=["C3"])
    return df["C3"]


# ウェーブレット変換と画像の保存
def save_wavelet_transform_image(data, start_time, end_time, sampling_rate=250, wavelet="morl", max_scale=100, scales_step=1, filename="wavelet.png"):
    dt = 1 / sampling_rate  # サンプリング間隔
    scales = np.arange(1, max_scale + 1, scales_step)
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, dt)
    plt.imshow(coefficients, extent=[start_time, end_time, 1, max_scale], cmap="PRGn", aspect="auto", vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# 画像の読み込み
def load_images(image_paths, image_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=image_size, color_mode="grayscale")
        img_array = img_to_array(img)
        img_array /= 255.0
        images.append(img_array)
    return np.array(images)

# 異常検出
def detect_abnormalities(model, test_images):
    predictions = model.predict(test_images)
    abnormal_indices = np.where(predictions.ravel() > 0.9)[0]
    return abnormal_indices

def main():
    # モデルのロード
    model = load_model('my_model_2.h5')

    # EEGデータの読み込み
    eeg_data = load_eeg_data('left_test_EEG.xlsx')
    test_image_dir = 'test_wavelet_images'
    os.makedirs(test_image_dir, exist_ok=True)
    
    # 進行度バーの設定
    total_steps = (len(eeg_data) - 250) // 10
    pbar = tqdm(total=total_steps, desc="Generating Images")
    sampling_rate = 250
    # ウェーブレット変換と画像の保存
    for start in range(0, len(eeg_data) - 250, 10):
        end = start + 250
        start_time = start / sampling_rate  # 開始時間（秒）
        end_time = end / sampling_rate  # 終了時間（秒）
        image_path = os.path.join(test_image_dir, f'wavelet_{start}_{end}.png')
        save_wavelet_transform_image(eeg_data[start:end], start_time, end_time, sampling_rate=sampling_rate, filename=image_path)
        pbar.update(1)  # 進行度バーを更新

    pbar.close()  # 進行度バーを閉じる

    

    # 画像の読み込み
    test_images = load_images(glob.glob(os.path.join(test_image_dir, '*.png')))

    # 異常検出
    abnormal_indices = detect_abnormalities(model, test_images)

    # 異常検出結果のプロット
    plt.figure(figsize=(15, 5))
    plt.plot(eeg_data, color='b', label='EEG Data')
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG Value')
    plt.title('EEG Data with Abnormalities Highlighted')

    # 異常部分のハイライトと時間(ms)の出力
    for index in abnormal_indices:
        start_time = index * 10
        end_time = start_time + 1000
        plt.axvspan(start_time, end_time, color='red', alpha=0.3)
        print(f"Abnormality detected from {start_time}ms to {end_time}ms.")

    plt.legend()
    plt.show()

    # 異常検出結果のCSV出力
    pd.DataFrame({'Abnormal Time (ms)': abnormal_indices * 10}).to_csv('abnormal_times.csv', index=False)

if __name__ == "__main__":
    main()
