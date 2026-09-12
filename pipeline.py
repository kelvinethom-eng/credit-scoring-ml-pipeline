from pathlib import Path
import pandas as pd
from data_ingestion import DataIngestion
from train import CreditScoreTrainer
from evaluation import ModelEvaluator
import pickle
import mlflow


class CreditScoringPipeline:
    
    def __init__(self, raw_data_path: str | Path, config: dict, accuracy_threshold: float = 0.7):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.config = config
        self.accuracy_threshold = accuracy_threshold
        
        self.ingestor = DataIngestion(self.raw_data_path, config['artifacts_dir'])
        self.trainer = CreditScoreTrainer(config, config['artifacts_dir'])
        self.evaluator = ModelEvaluator(config['artifacts_dir'])

    def execute(self):
        print("\n🚀 Executing Credit Scoring Pipeline...\n")
        
        # Step 1: Data Ingestion
        ingested_file = self.ingestor.run()
        df = pd.read_csv(ingested_file)
        self.ingestor.exploratory_analysis(df)
        
        # Step 2: Preprocessing & Training
        X_train, X_test, y_train, y_test = self.trainer.load_and_preprocess(ingested_file)
        results_df = self.trainer.run_all(X_train, X_test, y_train, y_test)
        
        # Step 3: Evaluation & Comparison
        all_results = ModelEvaluator.evaluate_all_runs(self.config)
        best_model = ModelEvaluator.get_best_model(all_results)
        
        # Step 4: Deployment Approval
        print("--- Step 5: Deployment Approval Decision ---")
        if best_model['test_f1_macro'] >= self.accuracy_threshold:
            print(f"APPROVED: F1-Macro ({best_model['test_f1_macro']:.4f}) ≥ threshold ({self.accuracy_threshold})")
        else:
            print(f"REJECTED: F1-Macro ({best_model['test_f1_macro']:.4f}) < threshold ({self.accuracy_threshold})")
        
        # Save best model
        print("--- Step 6: Save Best Model for Deployment ---")
        # Buat objek Path dari konfigurasi yang sudah ada
        artifacts_path = Path(self.config['artifacts_dir'])
        
        best_model_pkl = artifacts_path / "best_model.pkl"
        best_model_prep = artifacts_path / "preprocessor.pkl"

        # Already saved preprocessor di train.load_and_preprocess()
        print(f"✅ Preprocessor: {best_model_prep}")

        # Save best model dari MLflow
        mlflow.set_tracking_uri(self.config['mlflow_uri'])

        # Load model dari MLflow
        model_uri = f"runs:/{best_model['run_id']}/model"
        try:
            loaded_model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            try:
                loaded_model = mlflow.xgboost.load_model(model_uri)
            except Exception:
                loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Save ke pickle
        with open(best_model_pkl, 'wb') as f:
            pickle.dump(loaded_model, f)

        print(f"✅ Best Model: {best_model_pkl}")
        print(f"   Model: {best_model_pkl}")
        print(f"   Preprocessor: {best_model_prep}")
        # Info
        print("📊 MLflow UI Access:")
        print(f"   $ mlflow ui --backend-store-uri {self.config['mlflow_uri']}")
        print(f"   → http://127.0.0.1:5000")


if __name__ == "__main__":
    from config import CONFIG
    
    DATA_INPUT = Path(__file__).parent / "data" / "data_A_md.csv"
    
    pipeline = CreditScoringPipeline(
        raw_data_path=DATA_INPUT,
        config=CONFIG,
        accuracy_threshold=0.7
    )
    pipeline.execute()