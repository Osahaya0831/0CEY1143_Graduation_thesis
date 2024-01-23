import glob
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.regularizers import l1, l2


def save_and_log_plot(file_name, plot_function):
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".png", delete=False
    ) as tmp_file:
        plot_function()
        plt.savefig(tmp_file.name)
        plt.close()
        mlflow.log_artifact(tmp_file.name, file_name)
    os.remove(tmp_file.name)


def load_eeg_data(filename):
    df = pd.read_excel(filename, usecols=["C3"])
    return df["C3"]


# 省略: generate_images 関数
image_dir = "wavelet_images"


def build_cnn(
    input_shape,
    n_filters,
    n_dense_units,
    dropout_rate,
    learning_rate,
    regularization,
    reg_lambda,
):
    model = Sequential()
    model.add(
        Conv2D(
            n_filters[0],
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D((2, 2)))

    for filters in n_filters[1:]:
        model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    for units in n_dense_units:
        if regularization == "l1":
            model.add(
                Dense(units, activation="relu", kernel_regularizer=l1(reg_lambda))
            )
        else:
            model.add(
                Dense(units, activation="relu", kernel_regularizer=l2(reg_lambda))
            )
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def objective(trial):
    # ここで image_dir をグローバル変数として使用
    global image_dir

    # Optunaのトライアルからハイパーパラメータを取得
    params = {
        "n_filters": [trial.suggest_int(f"n_filters_{i}", 32, 128) for i in range(2)],
        "n_dense_units": [
            trial.suggest_int(f"n_dense_units_{i}", 64, 256) for i in range(2)
        ],
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "regularization": trial.suggest_categorical("regularization", ["l1", "l2"]),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e-3),
        "epochs": trial.suggest_int("epochs", 5, 20),
    }

    # train_model 関数を呼び出し、パラメータを渡す
    history = train_model(image_dir, params)

    # バリデーション精度を返す
    return history.history["val_accuracy"][-1]


def train_model(image_dir, params):
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

    # 画像読み込み関数
    def load_images(image_paths, image_size=(128, 128)):
        images = []
        for img_path in image_paths:
            img = load_img(img_path, target_size=image_size, color_mode="grayscale")
            img_array = img_to_array(img)
            img_array /= 255.0  # 正規化
            images.append(img_array)
        return np.array(images)

    # 画像データのロード
    X_train = load_images(train_image_files)
    X_test = load_images(test_image_files)

    # データ拡張の設定
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    # クラスの重みを計算（ラベルの不均衡対策）
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # モデルの構築
    model = build_cnn(
        X_train.shape[1:],
        params["n_filters"],
        params["n_dense_units"],
        params["dropout_rate"],
        params["learning_rate"],
        params["regularization"],
        params["reg_lambda"],
    )

    # モデルの訓練
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=32),
        epochs=params["epochs"],
        validation_data=(X_test, y_test),
        class_weight=class_weights,
    )

    # 訓練メトリクスのログ記録
    for epoch in range(params["epochs"]):
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

    # ROC曲線のログ記録
    save_and_log_plot(
        "roc_curve_plots/roc_curve.png", lambda: plot_roc_curve(fpr, tpr, roc_auc)
    )

    # 分類レポートの表示とログ記録
    report = classification_report(y_test, y_pred_label)
    print("Classification Report:\n", report)
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_file.write(report)
        tmp_file_path = tmp_file.name
    mlflow.log_artifact(tmp_file_path, "classification_report")
    os.remove(tmp_file_path)

    # 混同行列の表示とログ記録
    cm = confusion_matrix(y_test, y_pred_label)
    save_and_log_plot(
        "confusion_matrix_plots/confusion_matrix.png",
        lambda: sns.heatmap(cm, annot=True, fmt="d"),
    )

    return history


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


def main():
    mlflow.start_run()

    # Optunaの研究を作成して最適化を実行
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective, n_trials=15
    )  # ここで objective は1つの引数のみを受け取る

    # 最適なハイパーパラメータを表示
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # MLflowへの最適なパラメータのログ記録
    mlflow.log_params(trial.params)

    # 最適なパラメータでモデルを再トレーニング
    train_model(image_dir, trial.params)

    mlflow.end_run()


if __name__ == "__main__":
    main()
