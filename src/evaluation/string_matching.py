import os
import fire
import pandas as pd
from tqdm import tqdm
from typing import List


S_COLUMNS = ["German_MS", "German_FS"]
P_COLUMNS = ["German_MP", "German_FP"]
FAIR_COLUMNS = ["Gender-Neutral", "Gender-Inclusive"]


def pick_form_to_match(dictionary_df, column, seed_noun) -> List[str]:
    """
    Args:
        dictionary_df: pd.DataFrame, dataframe with english seed nouns as index
        column: str, e.g., "German_MS"
        seed_noun: str, e.g., "dispatchers"
    """

    for idx, row in dictionary_df.iterrows():
        eng_form = idx.lower()  # the dispatchers

        if seed_noun in eng_form:
            form_to_match = row[
                column
            ]  # this can be a list of valid values separated by ";"
            form_to_match = [t.strip() for t in form_to_match.split(";")]
            return form_to_match  # it can also be empty string

    # print(f"Seed noun {seed_noun} not found in dictionary")
    return ""


def main(terms_file: str, translations_file: str, output_dir: str):
    dictionary_df = pd.read_csv(
        terms_file, index_col="English"
    )  # , encoding="latin-1")
    dictionary_df = dictionary_df.fillna("")
    translations_df = pd.read_csv(translations_file, index_col="seed_noun")
    translations_df = translations_df.fillna("")

    if "plural" in terms_file:
        COLS = P_COLUMNS + FAIR_COLUMNS
    else:
        COLS = S_COLUMNS + FAIR_COLUMNS

    print("Dictionary")
    print(dictionary_df.columns)
    print(dictionary_df.head(3))

    # Apply hard_matching function
    form_counts = list()
    for idx, row in translations_df.iterrows():
        translation = row["translation"]  # target sentence to evaluate
        seed_noun = idx  # e.g., "dispatchers"

        col2count = dict()
        for column in COLS:
            # Column is e.g., "German_MS"
            form_to_match = pick_form_to_match(dictionary_df, column, seed_noun)
            col2count[column] = 0
            for f in form_to_match:
                if f:
                    if f.lower() in translation.lower():
                        col2count[column] = 1
                        break  # any form that matches is enough

        form_counts.append(col2count)

    form_counts = pd.DataFrame(form_counts)

    print("Form counts")
    print(form_counts.sum(axis=0))

    for col in COLS:
        translations_df[col + "_count"] = form_counts[col].values

    print(translations_df.columns)
    print(translations_df.head(3))

    basename = os.path.splitext(os.path.basename(translations_file))[0]
    translations_df.to_csv(f"{output_dir}/form_counts_{basename}.csv", index=True)


if __name__ == "__main__":
    fire.Fire(main)
