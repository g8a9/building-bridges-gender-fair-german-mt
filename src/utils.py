import inflect
from dataclasses import dataclass
from tqdm import tqdm
import deepl
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


def inflect_to_plural(phrase):
    p = inflect.engine()
    tokens = phrase.split(" ")
    word_to_inflect = tokens[-1]
    plural_word = p.plural(word_to_inflect)
    return " ".join(tokens[:-1] + [plural_word])

@dataclass
class DeeplTranslator:
    def __init__(self, auth_key: str, source_lang: str, target_lang: str):
        self.translator = deepl.Translator(auth_key)
        self.source_lang = source_lang
        self.target_lang = target_lang

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def translate_text(self, text, split_sentences: str = "off"):
        return self.translator.translate_text(
            text,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            split_sentences=split_sentences
        ).text

    def translate_batch(self, texts):
        return [self.translate_text(text) for text in tqdm(texts, desc="Text")]
