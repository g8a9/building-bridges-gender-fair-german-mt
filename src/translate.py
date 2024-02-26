import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import deep_translator as dt
from dotenv import load_dotenv
from simple_generation import SimpleGenerator
from tqdm import tqdm
from transformers import HfArgumentParser
import time
from openai import OpenAI
import numpy as np
from utils import DeeplTranslator

load_dotenv()



def get_translator(model_name_or_path: str):
    """Instantiate a translator for commercial systems."""

    args = {"source": "en", "target": "de"}
    if model_name_or_path == "google-translate":
        return dt.GoogleTranslator(**args)
    elif model_name_or_path == "deepl":
        return DeeplTranslator(
            auth_key=os.environ.get("DEEPL_API_KEY"),
            source_lang=args["source"],
            target_lang=args["target"]
        )
        # return dt.DeeplTranslator(
        #     **args, api_key=os.environ.get("DEEPL_API_KEY"), use_free_api=True
        # )
    else:
        raise NotImplementedError()


def get_prompt_template(template_id: str):
    if template_id == "instruction":
        return 'Translate the following sentence into German. Reply only with the translation. Sentence: "{sentence}"'
    elif template_id == "flan":
        return "{sentence}\n\nTranslate this to German?"
    else:
        raise NotImplementedError()


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    system_prompt: Optional[str] = field(default=None)
    lora_weights: Optional[str] = field(default=None)
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    starting_batch_size: Optional[int] = field(default=16)
    max_new_tokens: Optional[int] = field(default=256)


@dataclass
class DataArguments:
    output_file: str = field()
    prompts_file: Optional[str] = field(default=None)
    samples_file: Optional[str] = field(default=None)
    prompt_template: Optional[str] = field(default=None)
    target_col: Optional[str] = field(default="English")
    suffix: Optional[str] = field(default="")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)

    if data_args.prompts_file:
        data = pd.read_csv(data_args.prompts_file, sep=",")
        data.index.name = "ID"
        print("Total rows", len(data))

        final_data = data.loc[~data[data_args.target_col].isna()]
        print(
            f"Total rows after removing NaN on {data_args.target_col} columns",
            len(final_data),
        )

        input_texts = final_data[data_args.target_col].tolist()

    elif data_args.samples_file:
        """
        Logic to handle sample files and filtering e.g., with Wikipedia or Europarl Samples.
        """

        with open(data_args.samples_file) as fp:
            data = json.load(fp)
            input_texts = list()
            for k, v in data.items():
                if v["selected"]:
                    input_texts.extend(v["selected"])

        # final_data = pd.read_csv(data_args.samples_file, sep="\t", index_col="rid")
        # input_texts = final_data.text.tolist()

    else:
        raise ValueError("specify either --prompts_file or --samples_file")

    ##################
    # PROMPT TEMPLATES
    # ################

    if data_args.prompt_template is not None:
        prompt_template = get_prompt_template(data_args.prompt_template)
        input_texts = [prompt_template.format(sentence=p) for p in input_texts]

    print(f"Loaded input texts to translate: {len(input_texts)}")
    print(f"Average words per passage: {np.mean([len(t.split(' ')) for t in input_texts])}")
    print("Some input texts...")
    print(input_texts[:3])

    #############
    # TRANSLATION
    #############

    # Google Translate
    if "google-translate" in model_args.model_name_or_path or "deepl" in model_args.model_name_or_path:
        translator = get_translator(model_args.model_name_or_path)
        print(translator)
        completions = translator.translate_batch(input_texts)

    # GPT-3.5 and 4
    elif "gpt-" in model_args.model_name_or_path:

        client = OpenAI()

        completions = list()
        for count, prompt in tqdm(
            enumerate(input_texts), desc="GPT API calls", total=len(input_texts)
        ):

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt,}],
                max_tokens=2048,
                model=model_args.model_name_or_path,
            )
            translated_text = response.choices[0].message.content
            completions.append(translated_text)
            if (count + 1) % 60 == 0:
                print("Completed", count + 1, "prompts. Sleeping for one minute...")
                time.sleep(65)

    # HF Models
    else:
        generator = SimpleGenerator(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
            # system_prompt=model_args.system_prompt,
        )

        # Model-based setup:
        if (
            ("opus" in model_args.model_name_or_path)
            or ("flan" in model_args.model_name_or_path)
            or ("nllb" in model_args.model_name_or_path)
        ):
            skip_prompt = False
            decoding_args = dict(
                temperature=1,
                num_return_sequences=1,
                max_new_tokens=model_args.max_new_tokens,
                top_k=50,
                top_p=1,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
            )

            # NLLB: force the BOS token to translate in German
            if "nllb" in model_args.model_name_or_path:
                decoding_args[
                    "forced_bos_token_id"
                ] = generator.tokenizer.lang_code_to_id["deu_Latn"]

        else:
            skip_prompt = True
            decoding_args = dict(
                temperature=0,
                max_new_tokens=model_args.max_new_tokens,
                top_p=1.0,
                top_k=50,
                do_sample=True,
            )

        # apply chat templates to chat models
        if (
            "mixtral" in model_args.model_name_or_path
            or "llama" in model_args.model_name_or_path
        ):
            print("Applying chat template to", model_args.model_name_or_path)

            input_texts = [
                generator.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in input_texts
            ]

        completions = generator(
            texts=input_texts,
            skip_prompt=skip_prompt,
            starting_batch_size=model_args.starting_batch_size,
            **decoding_args,
        )

    ##############################
    # POSTPROCESSING AND SAVING
    ##############################

    # basic post processing
    completions = [c.strip() for c in completions]

    model_id = model_args.model_name_or_path.replace("/", "--")

    if data_args.prompts_file:
        final_data[model_id] = completions
        data = data.join(final_data[model_id])

        data.to_csv(f"./results/translations_{model_id}{data_args.suffix}.csv", sep=",")

    elif data_args.samples_file:
        new_dict = dict()
        count = 0
        for k, v in data.items():
            new_dict[k] = v
            new_dict[k]["translation"] = completions[
                count : count + v["total_selected"]
            ]
            count += v["total_selected"]

        with open(data_args.output_file, "w", encoding="utf8") as fp:
            json.dump(new_dict, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
