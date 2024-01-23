import matplotlib
matplotlib.use("Agg")  # GUIバックエンドを使用しないように設定

import glob
import os
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm  # tqdmのインポート


# EEGデータを読み込む関数
def load_eeg_data(filename):
    df = pd.read_excel(filename, usecols=["C3"])
    return df["C3"]


# ウェーブレット変換とグラフ描画を行う関数（メインスレッドで実行される）
def plot_and_save_wavelet(data, filename, wavelet="morl", max_scale=100, scales_step=1):
    scales = np.arange(1, max_scale + 1, scales_step)
    coefficients, frequencies = pywt.cwt(data, scales, wavelet)
    plt.imshow(
        coefficients,
        extent=[0, len(data), 1, max_scale],
        cmap="PRGn",
        aspect="auto",
        vmax=abs(coefficients).max(),
        vmin=-abs(coefficients).max(),
    )
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# 並行処理を行う関数（プロセスで実行される）
def process_range(data_range, label, eeg_data, image_dir, sampling_rate, time_window):
    for start in tqdm(
        range(data_range["start"], data_range["end"], data_range["step"]),
        desc=f"Processing {label} data",
        total=data_range["total"],
    ):
        end = start + sampling_rate * time_window
        if end > len(eeg_data):
            break
        image_filename = os.path.join(image_dir, f"wavelet_{label}_{start}_{end}.png")
        # ウェーブレット変換の結果をグラフとして保存
        plot_and_save_wavelet(eeg_data[start:end], image_filename)


# サンプリングレートと時間窓の設定
sampling_rate = 250
time_window = 1

# 画像を保存するディレクトリを作成
image_dir = "wavelet_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# EEGデータの読み込み
filename = "/Users/hayato/Python_test/wavelet_try/C3_Only_all.xlsx"
eeg_data = load_eeg_data(filename)

# 正常データと異常データの範囲を定義
normal_ranges = [
    {"start": 25000, "end": 44000, "step": 1, "total": 19000},
    {"start": 3000, "end": 5000, "step": 1, "total": 2000},
    {"start": 53000, "end": 73000, "step": 1, "total": 20000},
]
abnormal_ranges = [
    {"start": 118628, "end": 120162, "step": 1, "total": 1534},
    {"start": 48789, "end": 52053, "step": 1, "total": 3264},
    {"start": 84314, "end": 86314, "step": 1, "total": 2000},
    {"start": 117209, "end": 120747, "step": 1, "total": 3538},
]

# 画像生成処理の関数
def process_ranges(ranges, label):
    for data_range in ranges:
        for start in tqdm(
            range(data_range["start"], data_range["end"], data_range["step"]),
            desc=f"Processing {label} data",
            total=data_range["total"],
        ):
            end = start + sampling_rate * time_window
            if end > len(eeg_data):
                break
            image_filename = os.path.join(image_dir, f"wavelet_{label}_{start}_{end}.png")
            plot_and_save_wavelet(eeg_data[start:end], image_filename)

# 正常データと異常データの画像を生成
process_ranges(normal_ranges, "normal")
process_ranges(abnormal_ranges, "abnormal")


# 画像ファイルのパスを取得して分割
image_files = glob.glob(os.path.join(image_dir, "*.png"))
train_image_files, test_image_files = train_test_split(
    image_files, test_size=0.2, random_state=42
)


# ラベル抽出関数
def extract_label_from_filename(filenames):
    labels = []
    for filename in filenames:
        if "abnormal" in filename:
            labels.append(1)  # 異常の場合はラベルを1とする
        else:
            labels.append(0)  # 正常の場合はラベルを0とする
    return np.array(labels)


# ラベルの抽出
y_train = extract_label_from_filename(train_image_files)
y_test = extract_label_from_filename(test_image_files)

'''
# 画像の読み込みと前処理
def load_images(image_paths, image_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=image_size, color_mode="grayscale")
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # 正規化
        images.append(img_array)
    return np.array(images)


X_train = load_images(train_image_files)
X_test = load_images(test_image_files)


# CNNモデルの構築
# モデルの構造を改善
def build_cnn(input_shape):
    model = Sequential()
    # 畳み込み層とプーリング層を追加
    model.add(
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape)
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))  # 追加
    model.add(MaxPooling2D((2, 2)))  # 追加
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))  # ユニット数を増加
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))  # ユニット数を増加
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = build_cnn(input_shape=X_train.shape[1:])

# データ拡張の改良
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,  # シアー変換を追加
    zoom_range=0.15,  # 拡大縮小を追加
    horizontal_flip=True,
)

# ラベルの不均衡対策
# 例: sklearn.utils.class_weight.compute_class_weight を使用してクラスの重みを計算
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# モデルの訓練（データ拡張とクラスの重みを使用）
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=32),
    epochs=1,
    validation_data=(X_test, y_test),
)

# テストデータに対する予測
y_pred = model.predict(X_test).ravel()
y_pred_label = [1 if pred > 0.5 else 0 for pred in y_pred]

# 混同行列の表示
cm = confusion_matrix(y_test, y_pred_label)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# 分類レポートの表示（精度、リコール、F1スコアを含む）
print("Classification Report:")
print(classification_report(y_test, y_pred_label))

# ROC曲線とAUCの計算
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
# 混同行列のグラフ
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
# グラフを一時的なファイルとして保存
plt.savefig("/tmp/confusion_matrix.png")
plt.close()
# MLflowに画像をログ
mlflow.log_artifact("/tmp/confusion_matrix.png", "confusion_matrix_plots")

# ROC曲線のグラフ
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
# グラフを一時的なファイルとして保存
plt.savefig("/tmp/roc_curve.png")
plt.close()
# MLflowに画像をログ
mlflow.log_artifact("/tmp/roc_curve.png", "roc_curve_plots")

# 実験の終了
mlflow.end_run()
'''