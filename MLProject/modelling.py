import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train():
    # Gunakan autolog agar MLflow otomatis membuat folder 'model'
    mlflow.sklearn.autolog()
    
    # Path data yang aman untuk GitHub Actions
    data_file = "exam_score_preprocessed.csv"
    if not os.path.exists(data_file):
        data_file = os.path.join("MLProject", data_file)

    df = pd.read_csv(data_file)
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mulai Run (tanpa URI DagsHub jika hanya ingin ke Docker)
    with mlflow.start_run(run_name="Docker_Build_Run"):
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        # Memicu autolog mencatat evaluasi
        model.score(X_test, y_test)
        print("Training selesai, model tersimpan di folder mlruns lokal runner.")

if __name__ == "__main__":
    train()
