import re
import pandas as pd


_TITLE_MAP = {
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Miss",
}
_KEEP_TITLES = {"Mr", "Mrs", "Miss", "Master", "Dr"}


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]+)\.", name)
    if not match:
        return "Other"
    title = match.group(1).strip()
    title = _TITLE_MAP.get(title, title)
    return title if title in _KEEP_TITLES else "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["Deck"] = df["Cabin"].apply(lambda c: c[0] if pd.notna(c) else "Unknown")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    embarked_map = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    df["Port"] = df["Embarked"].map(embarked_map)
    return df
