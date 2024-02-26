import fire
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import set_seed

label_maps = {"n": 0, "b": 0.5, "y": 1}


def main(input_file: str, output_dir: str):
    set_seed(42)
    df = pd.read_csv(input_file)
    print(df["comment"].isna().value_counts())
    df = df.dropna(subset=["source"])
    df = df.loc[~df["comment"].isna()]
    df["label"] = df["comment"].map(label_maps)
    print(df["label"].value_counts())
    df["rid"] = list(range(len(df)))
    df = df.set_index("rid")
    train, test = train_test_split(
        df, test_size=0.1, stratify=df["label"], shuffle=True
    )
    train, val = train_test_split(
        train, test_size=0.1, stratify=train["label"], shuffle=True
    )
    train.to_csv(f"{output_dir}/relevance_train.csv")
    val.to_csv(f"{output_dir}/relevance_val.csv")
    test.to_csv(f"{output_dir}/relevance_test.csv")


if __name__ == "__main__":
    fire.Fire(main)
