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
        # Hanya run dari eksekusi ini yang dibandingkan. Sebelumnya semua run
        # di experiment ikut dibaca (evaluate_all_runs), sehingga model terbaik
        # bisa berasal dari run lama dengan preprocessor yang sudah berbeda.
        print("  EVALUATION: EXPERIMENTS IN THIS RUN")
        print(results_df[['model_name', 'test_accuracy', 'test_f1_macro',
                          'test_f1_weighted', 'overfit_gap']].to_string(index=False))
        best_model = ModelEvaluator.get_best_model(results_df)

        # Step 4: Deployment Approval
        print("--- Step 5: Deployment Approval Decision ---")
        if best_model['test_f1_macro'] >= self.accuracy_threshold:
            print(f"APPROVED: F1-Macro ({best_model['test_f1_macro']:.4f}) ≥ threshold ({self.accuracy_threshold})")
        else:
            # Gate yang benar-benar menahan deploy: sebelumnya model tetap
            # disimpan walaupun statusnya REJECTED.
            print(f"REJECTED: F1-Macro ({best_model['test_f1_macro']:.4f}) < threshold ({self.accuracy_threshold})")
            print("Model tidak disimpan untuk deployment.")
            return None

        # Save best model
        print("--- Step 6: Save Best Model for Deployment ---")
        artifacts_path = Path(self.config['artifacts_dir'])

        best_model_pkl = artifacts_path / "best_model.pkl"
        best_model_prep = artifacts_path / "preprocessor.pkl"

        # Already saved preprocessor di train.load_and_preprocess()
        print(f"✅ Preprocessor: {best_model_prep}")

        mlflow.set_tracking_uri(self.config['mlflow_uri'])

        # Load model dari MLflow memakai flavor yang dicatat saat training
        # (tag 'mlflow_flavor'), bukan tebak-tebakan lewat try/except.
        model_uri = f"runs:/{best_model['run_id']}/model"
        flavor = mlflow.get_run(best_model['run_id']).data.tags.get('mlflow_flavor', 'sklearn')
        loaded_model = getattr(mlflow, flavor).load_model(model_uri)

        # Save ke pickle
        with open(best_model_pkl, 'wb') as f:
            pickle.dump(loaded_model, f)

        print(f"✅ Best Model: {best_model_pkl} ({best_model['model_name']})")
        print(f"   Preprocessor: {best_model_prep}")
        # Info
        print("📊 MLflow UI Access:")
        print(f"   $ mlflow ui --backend-store-uri {self.config['mlflow_uri']}")
        print(f"   → http://127.0.0.1:5000")

        return best_model


if __name__ == "__main__":
    from config import CONFIG

    DATA_INPUT = Path(__file__).parent / "data" / "data_A_md.csv"

    pipeline = CreditScoringPipeline(
        raw_data_path=DATA_INPUT,
        config=CONFIG,
        accuracy_threshold=0.7
    )
    pipeline.execute()
