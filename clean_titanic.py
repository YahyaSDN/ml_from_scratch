import re
import pandas as pd

TRAIN_IN = "dataset/train.csv"
TEST_IN = "dataset/test.csv"
TRAIN_OUT = "train_clean.csv"
TEST_OUT = "test_clean.csv"

def extract_title(name):
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"

def map_title(t):
    rare = {"Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"}
    if t in ("Mlle","Ms"): return "Miss"
    if t == "Mme": return "Mrs"
    if t in rare: return "Rare"
    return t

def preprocess(df, title_age_map=None, embarked_mode=None, keep_titles=None):
    df = df.copy()
    # Titles and Age
    df["Title"] = df["Name"].map(extract_title).map(map_title)
    if keep_titles is not None:
        df["Title"] = df["Title"].where(df["Title"].isin(keep_titles), "Rare")
    if title_age_map is not None:
        df["Age"] = df["Age"].fillna(df["Title"].map(title_age_map))
    # Embarked
    if embarked_mode is not None:
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    else:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iat[0])
    # Cabin deck (use fixed deck ordering for reproducibility)
    df["Deck"] = df["Cabin"].fillna("U").str[0]
    deck_order = ["A","B","C","D","E","F","G","T","U"]
    deck_map = {d:i for i,d in enumerate(deck_order)}
    df["Deck"] = df["Deck"].map(deck_map).fillna(deck_map["U"]).astype(int)
    # Family features (these use SibSp internally but SibSp will not be saved)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    # Encode Embarked
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
    # Drop columns we don't need for the cleaned output (still keep PassengerId)
    drop_cols = ["Name", "Ticket", "Cabin", "Fare", "Survived"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    return df

def main():
    train = pd.read_csv(TRAIN_IN)
    test = pd.read_csv(TEST_IN)

    # compute title mapping and relevant titles from train
    train["Title"] = train["Name"].map(extract_title).map(map_title)
    title_age_map = train.groupby("Title")["Age"].median().to_dict()
    overall_age_median = train["Age"].median()
    for k, v in title_age_map.items():
        if pd.isna(v):
            title_age_map[k] = overall_age_median

    # Decide which titles to keep as "relevant" (keep common ones, map rest -> Rare)
    title_counts = train["Title"].value_counts()
    common_titles = set(title_counts[title_counts >= 10].index)  # threshold can be adjusted
    core = {"Mr","Miss","Mrs","Master"}
    keep_titles = set(common_titles) | core

    embarked_mode = train["Embarked"].mode().iat[0]

    train_clean = preprocess(train, title_age_map=title_age_map,
                             embarked_mode=embarked_mode, keep_titles=keep_titles)
    test_clean = preprocess(test, title_age_map=title_age_map,
                            embarked_mode=embarked_mode, keep_titles=keep_titles)

    # Select columns to save (PassengerId kept if present)
    cols = []
    if "PassengerId" in train_clean.columns or "PassengerId" in test_clean.columns:
        cols.append("PassengerId")
    # NOTE: SibSp intentionally removed as requested
    cols += ["Pclass", "Sex", "Age", "Parch", "Embarked", "Title", "Deck", "FamilySize", "IsAlone"]

    # Keep only existing columns (defensive)
    train_cols = [c for c in cols if c in train_clean.columns]
    test_cols  = [c for c in cols if c in test_clean.columns]

    train_clean = train_clean[train_cols]
    test_clean  = test_clean[test_cols]

    train_clean.to_csv(TRAIN_OUT, index=False)
    test_clean.to_csv(TEST_OUT, index=False)
    print(f"Saved cleaned files (SibSp excluded): {TRAIN_OUT}, {TEST_OUT}")
    
if __name__ == "__main__":
    main()