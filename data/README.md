## Annotation Files

The processed annotations for Wikipedia and Europarl are located in two separate folders.
Each folder contains three annotation files per translation system, i.e., DeepL and GPT-3.5 (gpt):

- `annot_\[system\]_all_v1.tsv`: all the rows of our initial sample;
- `annot_\[system\]_valid_v1.tsv`: only the rows where `source` matches our sampling criteria, i.e., there are four sentences, where the first two are the preceding context, the third the sentence that contains the seed noun, and the fourth the trailing context. This file is a subset of the previous one;
- `annot_\[system\]_common_v1.tsv`: only the rows where `translation` has been annotated on gender netruality for both the systems, i.e., DeepL and GPT-3.5. This file is a subset of the previous one.