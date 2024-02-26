import json
import fire
import pandas as pd


def main(input_file, output_file):
    results = list()
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for k, v in data.items():
        seed = k
        if v["selected"]:
            for s, t in zip(v["selected"], v["translation"]):
                results.append(
                    {
                        "seed": seed,
                        "source": s,
                        "translation": t,
                    }
                )

    rids = list(range(len(results)))
    df = pd.DataFrame(results)
    df["rid"] = rids
    df = df.set_index("rid")
    df.to_csv(output_file, encoding="utf-8")


if __name__ == "__main__":
    fire.Fire(main)
