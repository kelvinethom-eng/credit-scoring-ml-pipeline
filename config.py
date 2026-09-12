from pathlib import Path

# Paths
PROJECT_ROOT     = Path(__file__).parent
DATA_DIR         = PROJECT_ROOT / "data"
ARTIFACTS_DIR    = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)
# Configuration
CONFIG = {
    "data_path"        : str(DATA_DIR / "data_A_md.csv"),
    "experiment_name"  : "credit_score_classification",
    
    # FIX: Ubah URI agar menggunakan SQLite Database!
    "mlflow_uri"       : "sqlite:///mlflow.db", 
    
    "test_size"        : 0.2,
    "random_state"     : 42,
    "artifacts_dir"    : str(ARTIFACTS_DIR),
}

print("Config loaded")