import pandas as pd
from src.config import DATA_PROCESSED


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Port"] = df["Port"].fillna(df["Port"].mode()[0])
    return df


def rename_and_drop(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={
        "Pclass": "PassengerClass",
        "SibSp": "SiblingsSpouses",
        "Parch": "ParentsChildren",
        "Fare": "TicketFare",
    })
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    if save:
        out_path = DATA_PROCESSED / "titanic_clean.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Temiz veri kaydedildi: {out_path}")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Port", "Title", "Deck"], drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df
