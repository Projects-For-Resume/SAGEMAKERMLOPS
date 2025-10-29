import json
import pathlib
import pickle as pkl
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt
from sklearn.metrics import roc_curve, auc

def load_model(model_tar_path):
    # Untar the model.tar.gz and load model
    with tarfile.open(model_tar_path) as tar:
        member_list = tar.getnames()
        # Find model file (e.g. "model.pkl", "model.joblib", "xgboost-model")
        model_files = [m for m in member_list if m.endswith('.pkl') or m.endswith('.joblib')]
        if model_files:
            model_file = model_files[0]
            tar.extract(model_file, path="/tmp")  # extract to temp
            model_path = f"/tmp/{model_file}"
            if model_file.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pkl.load(f)
            else:
                model = joblib.load(model_path)
            return model
        else:
            # For xgboost saver
            for m in member_list:
                if "xgboost" in m or m.endswith(".model"):
                    tar.extract(m, path="/tmp")
                    model_path = f"/tmp/{m}"
                    return xgb.Booster({'nthread':4}, model_file=model_path)
            raise ValueError("No model file found in tar")
    raise ValueError("No valid tar file available.")

def run_predictions(model, test_df):
    if hasattr(model, 'predict'):
        # sklearn/xgboost sklearn API
        preds = model.predict(test_df.drop("target", axis=1, errors="ignore"))
    else:
        # xgboost Booster API
        dmat = xgb.DMatrix(test_df.drop("target", axis=1, errors="ignore"))
        preds = model.predict(dmat)
    return preds

def evaluate_predictions(test_df, preds):
    # Use ROC-AUC as example, requires a "target" column
    y_true = test_df.get("target")
    if y_true is not None:
        fpr, tpr, thresholds = roc_curve(y_true, preds)
        score = auc(fpr, tpr)
        metrics = {
            "roc_auc": score,
            "timestamp": dt.datetime.now().isoformat()
        }
    else:
        metrics = {
            "detail": "Target column not found for evaluation."
        }
    return metrics

if __name__ == "__main__":
    # Paths would be passed as arguments in real pipeline code
    model_tar_path = "/opt/ml/processing/model/model.tar.gz"
    test_data_path = "/opt/ml/processing/test/test.csv"
    report_path = "/opt/ml/processing/evaluation/eval.json"

    print("Loading model...")
    model = load_model(model_tar_path)
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    print("Running predictions...")
    preds = run_predictions(model, test_df)
    print("Evaluating predictions...")
    metrics = evaluate_predictions(test_df, preds)
    print("Saving evaluation report...")
    pathlib.Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f)
    print("Done.")
