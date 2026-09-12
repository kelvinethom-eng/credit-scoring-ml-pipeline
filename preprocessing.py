import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


class CreditDataPreprocessor:
    DROP_COLS = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Month']

    NUMERIC_COLS = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Months',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]

    LABEL_ENCODE_COLS = ['Occupation', 'Payment_Behaviour']

    CREDIT_MIX_MAP = {'Bad': 0, 'Standard': 1, 'Good': 2}
    PAY_MIN_MAP    = {'No': 0, 'Yes': 1}
    TARGET_MAP     = {'Good': 0, 'Poor': 1, 'Standard': 2}
    TARGET_MAP_INV = {v: k for k, v in TARGET_MAP.items()}
    VALUE_BOUNDS = {
        'Age':               (18, 100),
        'Num_of_Loan':       (0, None),
        'Num_Bank_Accounts': (0, 20),
        'Num_Credit_Card':   (0, 20),
        'Interest_Rate':     (1, 100),
    }

    PLACEHOLDER_MAP = {
        'Occupation': '_______',
        'Credit_Mix': '_',
        'Payment_Behaviour': '!@9#%8',
        'Payment_of_Min_Amount': 'NM',
    }

    def __init__(self):
        self.label_encoders    = {}
        self.scaler            = StandardScaler()
        self.numeric_medians   = {}
        self.categorical_modes = {}
        self.feature_names     = None
        self.is_fitted         = False

    @staticmethod
    def _parse_history_age(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', str(val))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan

    def _apply_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Capping outlier / logical bounds, sekarang satu loop untuk semua kolom."""
        for col, (lo, hi) in self.VALUE_BOUNDS.items():
            if col not in df.columns:
                continue
            mask = pd.Series(False, index=df.index)
            if lo is not None:
                mask |= df[col] < lo
            if hi is not None:
                mask |= df[col] > hi
            df.loc[mask, col] = np.nan
        return df

    def _raw_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df.drop(columns=[c for c in self.DROP_COLS if c in df.columns], inplace=True)

        for col in ['Age', 'Annual_Income', 'Num_of_Loan', 'Changed_Credit_Limit']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('_', '', regex=False).str.strip(),
                    errors='coerce'
                )

        if 'Monthly_Balance' in df.columns:
            df['Monthly_Balance'] = pd.to_numeric(
                df['Monthly_Balance'].astype(str)
                                     .str.replace(r'^[^0-9\-\.].*', '', regex=True),
                errors='coerce'
            )

        df = self._apply_bounds(df)

        for col in self.NUMERIC_COLS:
            if col in df.columns and col != 'Credit_History_Months':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col, placeholder in self.PLACEHOLDER_MAP.items():
            if col in df.columns:
                df[col] = df[col].replace(placeholder, np.nan)

        if 'Credit_History_Age' in df.columns:
            df['Credit_History_Months'] = df['Credit_History_Age'].apply(self._parse_history_age)
            df.drop(columns=['Credit_History_Age'], inplace=True)

        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, val in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, val in self.categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()

        if 'Credit_Mix' in df.columns:
            df['Credit_Mix'] = df['Credit_Mix'].map(self.CREDIT_MIX_MAP).fillna(1).astype(int)

        if 'Payment_of_Min_Amount' in df.columns:
            df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map(self.PAY_MIN_MAP).fillna(0).astype(int)

        for col in self.LABEL_ENCODE_COLS:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            if fit:
                le = LabelEncoder()
                le.fit(df[col])
                self.label_encoders[col] = le
            le = self.label_encoders.get(col)
            if le is not None:
                known   = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])

        return df

    def fit(self, df: pd.DataFrame) -> 'CreditDataPreprocessor':
        df_proc = self._raw_clean(df)

        for col in self.NUMERIC_COLS:
            if col in df_proc.columns:
                self.numeric_medians[col] = df_proc[col].median()

        all_cat = self.LABEL_ENCODE_COLS + ['Credit_Mix', 'Payment_of_Min_Amount']
        for col in all_cat:
            if col in df_proc.columns:
                mode = df_proc[col].mode()
                if len(mode) > 0:
                    self.categorical_modes[col] = mode[0]

        df_proc = self._impute(df_proc)
        df_proc = self._encode(df_proc, fit=True)

        expected_features = self.NUMERIC_COLS + all_cat
        feat_cols = [c for c in expected_features if c in df_proc.columns]

        X = df_proc[feat_cols].values.astype(float)
        self.scaler.fit(X)
        self.feature_names = feat_cols
        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame, include_target: bool = True):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df_proc = self._raw_clean(df)
        df_proc = self._impute(df_proc)
        df_proc = self._encode(df_proc, fit=False)

        feat_cols = [c for c in self.feature_names if c in df_proc.columns]

        if len(feat_cols) < len(self.feature_names):
            missing = set(self.feature_names) - set(feat_cols)
            for m in missing:
                df_proc[m] = 0
            feat_cols = self.feature_names

        X = df_proc[feat_cols].values.astype(float)
        X_sc = self.scaler.transform(X)

        if include_target and 'Credit_Score' in df.columns:
            y = df['Credit_Score'].map(self.TARGET_MAP).values
            return X_sc, y
        return X_sc

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df, include_target=True)

    def get_feature_names(self):
        return self.feature_names

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Preprocessor saved → {path}")

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

class BaseModelTrainer(ABC):
    def __init__(self, model_name: str, params: dict = None):
        self.model_name = model_name
        self.params     = params or {}
        self.model      = None
        self.is_trained = False

    @abstractmethod
    def build_model(self) -> 'BaseModelTrainer':
        pass

    def train(self, X_train, y_train) -> 'BaseModelTrainer':
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} belum dilatih.")
        return self.model.predict(X)

    def get_params(self): return self.params
    def get_model(self):  return self.model
    def get_name(self):   return self.model_name


class ModelTrainer(BaseModelTrainer):
    REGISTRY = {
        'logreg': (
            LogisticRegression, "Logistic Regression",
            {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
        ),
        'dtree': (
            DecisionTreeClassifier, "Decision Tree",
            {'criterion': 'gini', 'max_depth': None, 'random_state': 42}
        ),
        'rf': (
            RandomForestClassifier, "Random Forest",
            {'n_estimators': 100, 'criterion': 'gini', 'class_weight': None,
             'n_jobs': -1, 'random_state': 42}
        ),
        'gb': (
            GradientBoostingClassifier, "Gradient Boosting",
            {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1,
             'subsample': 0.8, 'random_state': 42}
        ),
        'xgb': (
            XGBClassifier, "XGBoost",
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
             'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
             'reg_lambda': 1.0, 'eval_metric': 'mlogloss', 'random_state': 42, 'n_jobs': -1}
        ),
    }

    # Model yang butuh flavor MLflow selain 'sklearn' (dipakai train.py & pipeline.py)
    MLFLOW_FLAVOR = {'xgb': 'xgboost'}

    def __init__(self, key: str, params: dict = None, model_name: str = None):
        if key not in self.REGISTRY:
            raise ValueError(f"Unknown model key: {key}. Pilihan: {list(self.REGISTRY)}")
        model_class, default_name, default_params = self.REGISTRY[key]
        super().__init__(model_name or default_name, {**default_params, **(params or {})})
        self.key = key
        self.model_class = model_class

    def build_model(self):
        self.model = self.model_class(**self.params)
        return self

    def mlflow_flavor(self) -> str:
        """Nama modul MLflow (mlflow.sklearn / mlflow.xgboost) untuk log/load model ini."""
        return self.MLFLOW_FLAVOR.get(self.key, 'sklearn')

    def tune_depth(self, X_train, y_train, depths: list = None, cv: int = 5) -> int:
        """Khusus Decision Tree: CV search untuk max_depth terbaik."""
        if self.key != 'dtree':
            raise RuntimeError("tune_depth() hanya berlaku untuk trainer 'dtree'.")

        depths = depths or [3, 5, 7, 10, 15, 20, None]
        cv_skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        print(f"  🔍 CV Depth Tuning (k={cv}):")
        for d in depths:
            dt = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=42)
            sc = cross_val_score(dt, X_train, y_train, cv=cv_skf, scoring='f1_weighted').mean()
            scores.append(sc)
            print(f"      max_depth={str(d):5s} → F1-W: {sc:.4f}")

        best_depth = depths[int(np.argmax(scores))]
        self.params['max_depth'] = best_depth
        self.model_name = f"Decision Tree (max_depth={best_depth})"
        self.build_model()
        print(f"  ✅ Best max_depth: {best_depth}")
        return best_depth