import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import pandas as pd
import spacy
from datasets import load_dataset
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from relevance.score_sentences import Scorer

logger = logging.getLogger(__name__)


@dataclass
class Args:
    seed_file: str = field()
    corpus: str = field()
    output_file: str = field()
    target_col: str = field()
    seed: int = field(default=42)
    n_samples: int = field(default=3)
    num_workers: Optional[int] = field(default=4)
    context_length: Optional[int] = field(default=3)
    terms_range: Optional[str] = field(default=None)
    remove_article: Optional[bool] = field(default=True)
    apply_relevance: Optional[bool] = field(default=False)
    scorer_path: Optional[str] = field(default=None)
    run_parallel_jobs: Optional[bool] = field(default=True)


def main():
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()
    args = args[0]
    logger.info(args)

    ########
    # Process only a subset of the terms. Convenient for shorter, independent runs
    ########
    if args.terms_range is not None:
        range_l, range_r = args.terms_range.split(",")
        range_l, range_r = int(range_l), int(range_r)
        print("Extracting samples in: ", range_l, range_r)

    ########
    # Resume existing runs
    ########
    stats = dict()
    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} exists already. Loading it...")
        with open(args.output_file) as fp:
            stats = json.load(fp)

    set_seed(args.seed)

    ########
    # Load our seed nouns and filter them
    ########
    terms = pd.read_csv(args.seed_file, sep=",")
    # terms = terms.loc[terms["Exclude"] != "y"]

    ########
    # Remove the article
    ########
    if args.remove_article:
        english_terms = (
            terms[args.target_col].apply(lambda x: " ".join(x.split(" ")[1:])).tolist()
        )
    else:
        english_terms = terms[args.target_col].tolist()

    if args.terms_range is not None:
        english_terms = english_terms[range_l:range_r]

    if args.corpus == "europarl":
        with open("./data/europarl-v7.de-en.en", "r") as f:
            corpus = [l.strip() for l in f.readlines()]
    elif args.corpus == "wikipedia":
        data = load_dataset("wikipedia", "20220301.en", num_proc=args.num_workers)
        data = data["train"]
        corpus = data["text"]

    else:
        raise ValueError(f"Dataset {args.corpus} unrecognized")

    corpus = pd.Series(corpus)
    logger.info("Initial data size:", len(corpus))

    if args.apply_relevance:
        scorer = Scorer(model_path=args.scorer_path, device="cuda:0")

    # en_contexts = en_contexts.iloc[:1000]
    stats = dict()
    for term in tqdm(english_terms, desc="Term", total=len(english_terms)):
        # if term == "":
        #     print("Term is empty string, skipping...")
        #     continue
        if term in stats:
            print("Term found in output json, skipping...")
            continue

        def match_context(context_length: int, word, context) -> List[str]:
            pattern = r"\b" + re.escape(word.lower()) + r"\b"

            # 1. check whether the sample matches
            if re.search(pattern, context.lower()) is not None:
                nlp = spacy.load("en_core_web_sm")

                # 2. split the context into sentences
                sentences = sent_tokenize(context)

                valid_matches = list()
                for i, sent in enumerate(sentences):
                    lower_sent = sent.lower()
                    if re.search(pattern, lower_sent) is not None:
                        for token in nlp(lower_sent):
                            if token.text == word and (
                                token.pos_ == "PROPN" or token.pos_ == "NOUN"
                            ):
                                # 3. build the sample by taking two preceding sentences
                                # and one following
                                #
                                matching_sample = " ".join(
                                    sentences[max(0, i - context_length) : i + 2]
                                )

                                # 4. heuristic: we filter samples with more than three lines
                                # otherwise we retrive lists of references and bullet points
                                # from wikipedia.
                                if len(matching_sample.split("\n")) <= 3:
                                    valid_matches.append(matching_sample)

                return valid_matches
            return None

        # 1. Find all contexts that match
        if args.run_parallel_jobs:
            matched_contexts = Parallel(n_jobs=args.num_workers)(
                delayed(match_context)(args.context_length, term, example)
                for example in corpus
            )
        else:
            matched_contexts = [
                match_context(args.context_length, term, example) for example in corpus
            ]

        # matched_contexts is List[List[str]], hence let's flatten it
        flat_matched_context = [c for m in matched_contexts if m for c in m]

        flat_matched_context = pd.Series(flat_matched_context).fillna("")
        flat_matched_context = flat_matched_context[flat_matched_context != ""]

        if len(flat_matched_context) == 0:
            total_matches = 0
            total_selected = 0
            selected = None
        else:
            total_matches = len(flat_matched_context)
            total_selected = min(total_matches, args.n_samples)

            print(
                f"Term {term}. Total matches: {total_matches}, total selected: {total_selected}"
            )
            # print("Some examples: ", flat_matched_context[:3])

            # if we apply the relevance models, we score the sentences as rank them by decreasing score
            if args.apply_relevance:
                logger.info("Applying relevance model")
                res = scorer(flat_matched_context.tolist(), show_progress=False)
                scores = [r["score"] for r in res]

            # 2. sample "n_samples" sentences
            selected = flat_matched_context.sample(
                n=total_selected, random_state=args.seed, weights=scores
            ).tolist()

        stats[term] = {
            "total_matches": total_matches,
            "total_selected": total_selected,
            "selected": selected,
        }

        with open(args.output_file, "w", encoding="utf8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
