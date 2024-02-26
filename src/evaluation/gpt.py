import fire
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def build_prompt(seed_noun: str, translation: str):
    return f"""
If the following sentence contains the German translation for the English word '{seed_noun}', tell me which overt gender it displays among Masculine, Feminine, Gender-Neutral, or Gender-Inclusive. If no translation is found, reply with None.
Sentence: '{translation}'
"""


def evaluate_with_gpt(
    model_name: str, client: OpenAI, seed_noun: str, translation: str
):
    prompt = build_prompt(seed_noun, translation)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=2048,
        model=model_name,
    )
    return prompt, response.choices[0].message.content


def main(translations_file: str, output_dir: str, model: str = "gpt-3.5-turbo"):
    """
    Evaluate the GPT-3 model using the given translation file.

    Args:
        translation_file (str): The file containing the translations to evaluate. This is the file Manuel annotated.
    """
    df = pd.read_csv(translations_file, sep="\t", index_col="ID")
    client = OpenAI()

    completions = list()
    prompts = list()
    for idx, row in tqdm(df.iterrows(), desc="Row", total=len(df)):

        if row["to_keep"]:
            prompt, response = evaluate_with_gpt(
                model, client, row["seed_noun"], row["translation"]
            )
        else:
            prompt = response = "N/A"

        prompts.append(prompt)
        completions.append(response)
        if (idx + 1) % 60 == 0:
            print("Completed", idx + 1, "prompts. Sleeping for one minute...")
            time.sleep(65)

    df["prompt"] = prompts
    df["response"] = completions

    dataset = "europarl" if "europarl" in translations_file else "wikipedia"

    df.to_csv(f"{output_dir}/{model}_{dataset}.csv")


if __name__ == "__main__":
    fire.Fire(main)
