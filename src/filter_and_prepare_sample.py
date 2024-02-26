import json

import fire
import ftfy
import pandas as pd


def main(sample_file, output_file, max_input_tokens: int = 256):
    with open(sample_file) as fp:
        data = json.load(fp)

    print(f"Found {len(data)} seed words")
    data = {k: v for k, v in data.items() if v["total_selected"] != 0}
    print(f"Found {len(data)} after filtering out empty seeds.")

    results = list()
    for k, v in data.items():
        for t in v["selected"]:
            token_count = len(t.split(" "))
            if token_count <= max_input_tokens:
                results.append(
                    {"seed": k, "token_count": token_count, "text": ftfy.fix_text(t)}
                )
    df = pd.DataFrame(results)

    print(f"Found a total of {len(df)} samples!")
    df.index.name = "rid"
    df.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    fire.Fire(main)
