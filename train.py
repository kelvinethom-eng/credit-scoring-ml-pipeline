from pathlib import Path
import json
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from preprocessing import CreditDataPreprocessor, ModelTrainer
from evaluation import ModelEvaluator


class CreditScoreTrainer:
    """Orchestrate training semua model dengan MLflow tracking."""

    def __init__(self, config: dict, artifact_dir: str = "artifacts"):
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = CreditDataPreprocessor()
        self.evaluator = ModelEvaluator(artifact_dir)
        self.results = []

        mlflow.set_tracking_uri(config['mlflow_uri'])
        exp = mlflow.set_experiment(config['experiment_name'])
        self.experiment_id = exp.experiment_id

        print(f"\n✅ MLflow experiment: {config['experiment_name']}")

    def load_and_preprocess(self, data_path: str | Path):
        print("\n--- Step 2: Preprocessing ---")

        df = pd.read_csv(data_path)

        X_raw = df.drop('Credit_Score', axis=1)
        y_raw = df['Credit_Score']

        # Pakai TARGET_MAP dari CreditDataPreprocessor -> satu-satunya sumber
        # kebenaran untuk mapping label. Dulu di sini pakai LabelEncoder sendiri
        # yang terpisah dari TARGET_MAP di preprocessing.py (kebetulan hasilnya
        # sama karena alfabetis, tapi berisiko tidak sinkron kalau kelas berubah).
        y_encoded = y_raw.map(CreditDataPreprocessor.TARGET_MAP)
        classes = list(CreditDataPreprocessor.TARGET_MAP.keys())

        print(f"\n✅ Label Mapping - Classes: {classes}")
        print("   Mapping yang akan digunakan saat inference:")
        for cls, idx in CreditDataPreprocessor.TARGET_MAP.items():
            print(f"      {idx} → '{cls}'")

        label_classes_path = self.artifact_dir / "label_classes.json"
        with open(label_classes_path, 'w') as f:
            json.dump(classes, f)
        print(f"✅ Saved label_classes.json → {label_classes_path}")

        y_raw = pd.Series(y_encoded.values, name=y_raw.name)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw,
            test_size    = self.config['test_size'],
            random_state = self.config['random_state'],
            stratify     = y_raw
        )

        X_train = self.preprocessor.fit_transform(X_train_raw)
        print("Shape dari Pipeline:", X_train.shape)
        print("Daftar Kolom X_train:")
        print(self.preprocessor.feature_names)

        X_test = self.preprocessor.transform(X_test_raw)

        prep_path = self.artifact_dir / "preprocessor.pkl"
        self.preprocessor.save(str(prep_path))

        return X_train, X_test, y_train, y_test

    def run_experiment(self, trainer: ModelTrainer, X_train, X_test, y_train, y_test,
                        tune: bool = False) -> dict:
        """Run single experiment: train, evaluate, log ke MLflow."""
        print(f"\n🚀 Training: {trainer.get_name()}")

        if tune and trainer.key == 'dtree':
            trainer.tune_depth(X_train, y_train)

        with mlflow.start_run(run_name=trainer.get_name()):

            trainer.build_model()
            trainer.train(X_train, y_train)
            model = trainer.get_model()

            metrics, y_pred = self.evaluator.full_evaluate(
                model, X_train, X_test, y_train, y_test, trainer.get_name()
            )

            mlflow.log_params(trainer.get_params())
            mlflow.log_metrics(metrics)

            # Flavor diambil dari trainer -> satu sumber kebenaran, juga dicatat
            # sebagai tag supaya pipeline.py bisa load tanpa tebak-tebakan try/except.
            flavor = trainer.mlflow_flavor()
            getattr(mlflow, flavor).log_model(model, artifact_path="model")
            mlflow.set_tag("mlflow_flavor", flavor)

            run_id = mlflow.active_run().info.run_id

        result = {'run_id': run_id, 'model_name': trainer.get_name(), **metrics}
        self.results.append(result)
        print(f"   ✅ Logged to MLflow | run_id: {run_id[:12]}...")

        return result

    def run_all(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        print(" TRAINING ALL EXPERIMENTS")

        experiments = [
            ('logreg', {'C': 1.0, 'class_weight': 'balanced',
                        'max_iter': 1000, 'random_state': 42}, None, False),

            ('dtree', {'criterion': 'gini', 'random_state': 42}, None, True),

            ('rf', {'n_estimators': 300, 'max_depth': 16, 'min_samples_split': 5,
                    'min_samples_leaf': 2, 'max_features': 'sqrt',
                    'n_jobs': -1, 'random_state': 42}, 'Random Forest (Tuned)', False),

            ('rf', {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2,
                    'class_weight': 'balanced_subsample', 'n_jobs': -1, 'random_state': 42},
             'Random Forest (Balanced, Tuned)', False),

            ('gb', {'n_estimators': 250, 'max_depth': 3, 'learning_rate': 0.05,
                    'subsample': 0.8, 'random_state': 42}, None, False),

            ('xgb', {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                     'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5,
                     'reg_lambda': 2.0, 'eval_metric': 'mlogloss',
                     'random_state': 42, 'n_jobs': -1}, None, False),
        ]

        for key, params, name, do_tune in experiments:
            trainer = ModelTrainer(key, params=params, model_name=name)
            self.run_experiment(trainer, X_train, X_test, y_train, y_test, tune=do_tune)

        results_df = pd.DataFrame(self.results).sort_values('test_f1_macro', ascending=False)
        return results_df