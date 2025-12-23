import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train():
    # 1. Setup Tracking (Pastikan URI DagsHub benar jika ingin sinkron online)
    # Jika hanya untuk Docker build di CI, MLflow akan simpan di folder 'mlruns' lokal runner
    
    mlflow.sklearn.autolog()
    
    # Memastikan path dataset benar
    data_file = "exam_score_preprocessed.csv"
    if not os.path.exists(data_file):
        data_file = os.path.join("MLProject", data_file)

    df = pd.read_csv(data_file)
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Proses Training
    # JANGAN gunakan nested=True di dalam mlflow run
    with mlflow.start_run(run_name="Training_CI_Docker"):
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Evaluasi singkat untuk memicu autolog
        model.score(X_test, y_test)
        
        print("Training completed. Model saved to mlruns.")

if __name__ == "__main__":
    train()
