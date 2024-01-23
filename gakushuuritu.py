import glob
import multiprocessing
import os
import tempfile

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


def save_and_log_plot(file_name, plot_function):
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".png", delete=False
    ) as tmp_file:
        plot_function()  # グラフを描画する関数
        plt.savefig(tmp_file.name)
        plt.close()
        mlflow.log_artifact(tmp_file.name, file_name)
    os.remove(tmp_file.name)  # 一時ファイルを削除


def load_eeg_data(filename):
    df = pd.read_excel(filename, usecols=["C3"])
    return df["C3"]


def generate_images(eeg_data, image_dir, normal_ranges, abnormal_ranges):
    # EEGデータを読み込む関数
    def load_eeg_data(filename):
        df = pd.read_excel(filename, usecols=["C3"])
        return df["C3"]

    # EEGデータを読み込む
    filename = "/Users/hayato/Python_test/C3_Only_all.xlsx"
    eeg_data = load_eeg_data(filename)

    # ウェーブレット変換を行い、結果を画像ファイルとして保存する関数
    def save_wavelet_transform_image(
        data, wavelet="morl", max_scale=100, scales_step=1, filename="wavelet.png"
    ):
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

    # 並列処理でウェーブレット変換と画像保存を行う関数
    def process_data(args):
        data, start, end, image_filename = args
        save_wavelet_transform_image(data[start:end], filename=image_filename)

    # 正常データの処理範囲
    normal_ranges = [
        {"start": 25000, "end": 44000, "step": 1, "total": 19000},
        {"start": 3000, "end": 5000, "step": 1, "total": 2000},
    ]

    # 異常データの処理範囲
    abnormal_ranges = [
        {"start": 118628, "end": 120162, "step": 1, "total": 1534},
        {"start": 48789, "end": 52053, "step": 1, "total": 3264},
    ]

    # 並列処理の準備
    def prepare_args(ranges, label):
        args_list = []
        for range_info in ranges:
            for start in range(
                range_info["start"], range_info["end"], range_info["step"]
            ):
                end = start + sampling_rate * time_window
                if end > len(eeg_data):
                    break
                image_filename = os.path.join(
                    image_dir, f"wavelet_{label}_{start}_{end}.png"
                )
                args_list.append((eeg_data, start, end, image_filename))
        return args_list

    # 並列処理の実行
    def run_parallel_processing(args_list):
        pool = multiprocessing.Pool()
        for _ in tqdm(
            pool.imap_unordered(process_data, args_list),
            total=len(args_list),
            ascii=True,
        ):
            pass
        pool.close()
        pool.join()

    # 正常データと異常データの処理を並列で実行
    normal_args = prepare_args(normal_ranges, "normal")
    abnormal_args = prepare_args(abnormal_ranges, "abnormal")
    run_parallel_processing(normal_args + abnormal_args)


def train_model(image_dir, learning_rate, epochs):
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

    # 画像読み込み関数のメモリ管理改善
    def load_images(image_paths, image_size=(128, 128)):
        for img_path in image_paths:
            img = load_img(img_path, target_size=image_size, color_mode="grayscale")
            img_array = img_to_array(img)
            img_array /= 255.0  # 正規化
            yield img_array

    # ジェネレータを使ったデータのロード
    X_train = np.array(list(load_images(train_image_files)))
    X_test = np.array(list(load_images(test_image_files)))
    # 以下は既存のCNNモデル構築、訓練、評価のコード

    # CNNモデルの構築
    def build_cnn(input_shape, learning_rate):
        model = Sequential()

        # モデルの構造定義
        model.add(
            Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            )
        )
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

        # 追加の畳み込みブロック
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

        # 全結合層の拡張
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        # オプティマイザに学習率を設定
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    model = build_cnn(input_shape=X_train.shape[1:], learning_rate=learning_rate)

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

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # モデルの訓練
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_test, y_test),
    )

    # 訓練メトリクスのログ記録
    for epoch in range(epochs):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric(
            "train_accuracy", history.history["accuracy"][epoch], step=epoch
        )
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric(
            "val_accuracy", history.history["val_accuracy"][epoch], step=epoch
        )

    # テストデータに対する予測
    y_pred = model.predict(X_test).ravel()
    y_pred_label = [1 if pred > 0.5 else 0 for pred in y_pred]

    # ROC曲線とAUCの計算
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # ROC曲線を描画する関数の定義
    def plot_roc_curve(fpr, tpr, roc_auc):
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")

    # ROC曲線のログ記録関数の定義（変更点）
    def log_roc_curve(fpr, tpr, roc_auc):
        save_and_log_plot(
            "roc_curve_plots/roc_curve.png", lambda: plot_roc_curve(fpr, tpr, roc_auc)
        )

    # ROC曲線をログ記録（変更点）
    log_roc_curve(fpr, tpr, roc_auc)

    # 分類レポートの表示（精度、リコール、F1スコアを含む）
    report = classification_report(y_test, y_pred_label)
    print("Classification Report:")
    print(report)

    # 分類レポートを一時ファイルに保存し、MLflowにログする
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_file.write(report)
        tmp_file_path = tmp_file.name

    # MLflowに分類レポートをログする
    mlflow.log_artifact(tmp_file_path, "classification_report")
    os.remove(tmp_file_path)  # 一時ファイルを削除

    # 混同行列の表示
    cm = confusion_matrix(y_test, y_pred_label)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # 混同行列のログ記録
    save_and_log_plot(
        "confusion_matrix_plots/confusion_matrix.png",
        lambda: sns.heatmap(cm, annot=True, fmt="d"),
    )




N = 0


def main():
    # MLflow実験の開始
    mlflow.start_run()

    # 実験設定
    sampling_rate = 250  # サンプリングレート
    time_window = 1  # 時間窓
    epochs = 10
    batch_size = 32
    learning_rate = 0.000002  # 学習率

    # MLflowへのパラメータのログ記録
    mlflow.log_param("sampling_rate", sampling_rate)
    mlflow.log_param("time_window", time_window)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # EEGデータの読み込み
    filename = "/Users/hayato/Python_test/C3_Only_all.xlsx"
    eeg_data = load_eeg_data(filename)

    image_dir = "wavelet_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    N = 0
    if N == 1:
        generate_images(eeg_data, image_dir, normal_ranges, abnormal_ranges)

    # モデルの訓練
    train_model(image_dir, learning_rate, epochs)

    mlflow.end_run()


if __name__ == "__main__":
    main()
