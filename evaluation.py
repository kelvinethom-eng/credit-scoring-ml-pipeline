import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

class ModelEvaluator:
    TARGET_NAMES = ['Good', 'Poor', 'Standard']

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

    def compute_metrics(self, y_true, y_pred, prefix: str = 'test') -> dict:
        return {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}_f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}_f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

    def full_evaluate(self, model, X_train, X_test, y_train, y_test, model_name: str = 'Model') -> tuple:
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Compute metrics
        train_metrics = self.compute_metrics(y_train, y_pred_train, 'train')
        test_metrics = self.compute_metrics(y_test, y_pred_test, 'test')
        
        # Combine metrics
        all_metrics = {**train_metrics, **test_metrics}
        
        # Compute overfit gap
        overfit_gap = train_metrics['train_accuracy'] - test_metrics['test_accuracy']
        all_metrics['overfit_gap'] = round(overfit_gap, 4)

        print(f"  {model_name}")
        print(f"  Train Accuracy              : {train_metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy               : {test_metrics['test_accuracy']:.4f}  "
              f"({'OVERFIT ⚠️' if overfit_gap > 0.05 else 'OK ✅'})")
        print(f"  Precision (Weighted)        : {test_metrics['test_precision']:.4f}")
        print(f"  Recall (Weighted)           : {test_metrics['test_recall']:.4f}")
        print(f"  F1 Score (Weighted)         : {test_metrics['test_f1_weighted']:.4f}")
        print(f"  F1 Score (Macro)            : {test_metrics['test_f1_macro']:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=self.TARGET_NAMES,
                                    zero_division=0, digits=4))

        return all_metrics, y_pred_test

   
    @staticmethod
    def evaluate_all_runs(config: dict) -> pd.DataFrame:
        import mlflow
        
        mlflow.set_tracking_uri(config['mlflow_uri'])
        experiment = mlflow.get_experiment_by_name(config['experiment_name'])
        
        if experiment is None:
            print("❌ No experiment found!")
            return None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        results = []
        for _, row in runs.iterrows():
            # Mengambil nama run dari tags MLflow, berikan default string jika tidak ada
            run_name = row.get('tags.mlflow.runName', 'Unknown Run')
            
            if run_name == '__summary__':
                continue
            
            result = {
                'run_id': row.get('run_id'),
                'model_name': run_name, # Gunakan variabel run_name yang sudah di-extract
                'test_accuracy': row.get('metrics.test_accuracy', np.nan),
                'test_f1_weighted': row.get('metrics.test_f1_weighted', np.nan),
                'test_f1_macro': row.get('metrics.test_f1_macro', np.nan),
                'overfit_gap': row.get('metrics.overfit_gap', np.nan),
            }
            results.append(result)
        
        results_df = pd.DataFrame(results).sort_values('test_f1_macro', ascending=False)
        
        # Print results table
        print("  EVALUATION: ALL EXPERIMENTS")
        print("\n📊 Results Table:")
        print(results_df.to_string(index=False))
        
        return results_df

    @staticmethod
    def get_best_model(results_df: pd.DataFrame) -> dict:
        if results_df is None or len(results_df) == 0:
            print("❌ No results to evaluate!")
            return None
        
        best = results_df.iloc[0]
        
        print(f"\n🏆 Best Model (by F1-Macro):")
        print(f"   Name: {best['model_name']}")
        print(f"   Run ID: {best['run_id']}")
        print(f"   F1 Macro: {best['test_f1_macro']:.4f}")
        print(f"   F1 Weighted: {best['test_f1_weighted']:.4f}")
        print(f"   Test Accuracy: {best['test_accuracy']:.4f}")
        print(f"   Overfit Gap: {best['overfit_gap']:+.4f}")
        
        return best.to_dict()