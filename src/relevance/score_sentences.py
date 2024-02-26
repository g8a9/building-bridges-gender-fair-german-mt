from transformers import pipeline
import pandas as pd
import fire
import json
import os
from tqdm import tqdm


class Scorer:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.regression_model = pipeline(
            "text-classification", model=model_path, device=device
        )

    def __call__(self, sentences: list, show_progress: bool = True):
        results = list()
        for sentence in tqdm(
            sentences, desc="Sentence", total=len(sentences), disable=not show_progress
        ):
            scores = self.regression_model(sentence)[0]["score"]
            results.append({"sentence": sentence, "score": scores})
        return results


def main(model_path: str, sentences_file: str, to_exclude: str):

    to_exclude = pd.read_csv(to_exclude, index_col="rid")
    basename = os.path.splitext(os.path.basename(sentences_file))[0]

    with open(sentences_file, "r") as f:
        sentences = json.load(f)

    regression_model = pipeline(
        "text-classification", model=model_path, device="cuda:0"
    )
    results = list()
    for k, v in tqdm(sentences.items(), desc="Seed", total=len(sentences)):
        sentences = v["selected"] if v["selected"] else list()
        for sentence in sentences:
            if sentence in to_exclude["source"]:
                continue
            # Assign scores to the sentences
            scores = regression_model(sentence)[0]["score"]
            results.append({"source": k, "sentence": sentence, "score": scores})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="score", ascending=False)
    results_df.to_csv(f"./results/relevance/{basename}.csv", index=None)


if __name__ == "__main__":
    fire.Fire(main)
