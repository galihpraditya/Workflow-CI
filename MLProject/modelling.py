import sys
from pathlib import Path
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("file:./mlruns")
print(f"[MLflow] Tracking URI: ./mlruns")

mlflow.autolog()
print(f"[MLflow] Autolog ENABLED - SEMUA logging otomatis!")


def load_preprocessed_data(data_dir: Path):
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def train_model_simple(model, model_name: str, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        model.fit(X_train, y_train)

        print(f"Model {model_name} trained successfully!")
        print(f"SEMUA di-log OTOMATIS oleh mlflow.autolog()")
        print(f"{'='*60}\n")

    return model
def main():
    experiment_name = "Telco_Churn_Prediction_Galih-Praditya-Kurniawan"
    mlflow.set_experiment(experiment_name)

    print(f"\n{'='*60}")
    print(f"MLflow Experiment: {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"{'='*60}\n")

    data_dir = Path(__file__).parent.parent / "data_preprocessing"

    if not data_dir.exists():
        print(f"Error: Data directory tidak ditemukan: {data_dir}")
        print("   Pastikan Anda sudah menjalankan preprocessing terlebih dahulu.")
        sys.exit(1)

    print(f"Loading data from: {data_dir}")
    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape:  {y_test.shape}\n")

    # Model 1: Logistic Regression
    print("\n[1/3] Logistic Regression")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    train_model_simple(lr_model, "Logistic_Regression", X_train, X_test, y_train, y_test)

    # Model 2: Random Forest
    print("\n[2/3] Random Forest")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    train_model_simple(rf_model, "Random_Forest", X_train, X_test, y_train, y_test)

    # Model 3: Gradient Boosting
    print("\n[3/3] Gradient Boosting")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    train_model_simple(gb_model, "Gradient_Boosting", X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("Semua model berhasil dilatih dan di-log ke MLflow!")
    print(f"Lihat hasil di: {mlflow.get_tracking_uri()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()