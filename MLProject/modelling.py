import os
import sys
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Konfigurasi Global
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Eksperimen_Churn_Prediction_Galih"

def get_dataset(folder_path):
    """
    Membaca dataset dari folder preprocessing.
    Menggunakan os.path.join sebagai alternatif pathlib.
    """
    try:
        files = ['X_train', 'X_test', 'y_train', 'y_test']
        data = {}
        
        for f in files:
            file_path = os.path.join(folder_path, f"{f}.csv")
            if 'y_' in f:
                data[f] = pd.read_csv(file_path).values.ravel()
            else:
                data[f] = pd.read_csv(file_path)
                
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
    except FileNotFoundError as e:
        print(f"[ERROR] File dataset tidak ditemukan: {e}")
        sys.exit(1)

def run_training_cycle(model, run_title, X_train, y_train):
    """
    Eksekusi training model.
    Mengandalkan mlflow.autolog() untuk menangkap metrics & params secara otomatis.
    """
    mlflow.autolog(log_models=True)

    with mlflow.start_run(run_name=run_title):
        print(f"--> Memulai Training: {run_title}")
        
        model.fit(X_train, y_train)
        
        mlflow.set_tag("training_type", "automated")
        print(f"    [DONE] Training {run_title} selesai.")

def main():
    # 1. Setup MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment ID set to: {EXPERIMENT_NAME}")

    # 2. Lokasi Data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'preprocessing', 'churn_data_preprocessing')
    
    print(f"Mencari data di: {data_path}")
    X_train, X_test, y_train, y_test = get_dataset(data_path)

    # 3. Definisi Model
    models_collection = {
        "LogReg_Baseline": LogisticRegression(max_iter=500, solver='liblinear', random_state=123),
        
        "RandomForest_Core": RandomForestClassifier(
            n_estimators=120,   
            max_depth=8,       
            random_state=123    
        ),
        
        "GradientBoosting_V1": GradientBoostingClassifier(
            learning_rate=0.05, 
            n_estimators=150,
            random_state=123
        )
    }

    # 4. Eksekusi Loop
    print("\n=== Memulai Batch Training ===")
    for model_name, model_obj in models_collection.items():
        run_training_cycle(model_obj, model_name, X_train, y_train)
        
    print("\n=== Seluruh Eksperimen Selesai ===")
    print(f"Cek hasil dengan mengetik: mlflow ui")

if __name__ == "__main__":
    main()