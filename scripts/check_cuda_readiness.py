
import sys
import xgboost as xgb
import numpy as np

def check_xgboost_gpu():
    print(f"XGBoost version: {xgb.__version__}")
    try:
        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        # Try to train with GPU
        print("Attempting to train with tree_method='gpu_hist'...")
        clf = xgb.XGBClassifier(tree_method="gpu_hist", n_estimators=10)
        clf.fit(X, y)
        print("SUCCESS: XGBoost trained with GPU support!")
        return True
    except xgb.core.XGBoostError as e:
        print(f"FAILURE: XGBoost GPU training failed. Error: {e}")
        return False
    except Exception as e:
        print(f"FAILURE: Unexpected error during XGBoost GPU check. Error: {e}")
        return False

if __name__ == "__main__":
    if check_xgboost_gpu():
        sys.exit(0)
    else:
        sys.exit(1)
