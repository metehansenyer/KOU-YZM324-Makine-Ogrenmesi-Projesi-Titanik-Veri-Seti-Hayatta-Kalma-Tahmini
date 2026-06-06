import pandas as pd
from src.config import DATA_RAW


def load_data(path=None, verbose=True) -> pd.DataFrame:
    if path is None:
        path = DATA_RAW / "titanic_dataset.csv"
    df = pd.read_csv(path)
    if verbose:
        print(f"Şekil: {df.shape}")
        print(f"Sütunlar: {list(df.columns)}")
        print("\nEksik değerler:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    return df
