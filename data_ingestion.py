from pathlib import Path
import pandas as pd

class DataIngestion:
    """Load, validate, dan explore credit scoring dataset."""
    
    def __init__(self, input_path: str | Path, output_dir: str | Path):
        self.input_file = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "credit_score_ingested.csv"
    
    def run(self) -> Path:
        print("\n--- Step 1: Data Ingestion ---")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read raw data
        df = pd.read_csv(self.input_file)
        
        # Save ingested data
        df.to_csv(self.output_file, index=False)
        print(f"✅ Data ingested | Shape: {df.shape}")
        print(f"   From: {self.input_file}")
        print(f"   To:   {self.output_file}")
        
        return self.output_file
    
    def exploratory_analysis(self, df: pd.DataFrame):
        print("\n📊 Exploratory Data Analysis:")
        print(f"   Shape: {df.shape}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"\n   Target distribution:")
        print(df['Credit_Score'].value_counts())