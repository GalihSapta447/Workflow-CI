import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train():
    mlflow.sklearn.autolog()
    
    # Path relatif terhadap root folder MLProject
    df = pd.read_csv("exam_score_preprocessed.csv")
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        print("Training inside MLProject completed.")

if __name__ == "__main__":
    train()