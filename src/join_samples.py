import json

files = [
    "./sample_en_wikipedia_plural__0-40.json",
    "./sample_en_wikipedia_plural__40-80.json",
    "./sample_en_wikipedia_plural__80-200.json",
]
d = dict()
for f in files:
    with open(f) as fp:
        d.update(json.load(fp))

print(len(d.keys()))
with open("./sample_en_plural_wikipedia.json", "w") as fp:
    json.dump(d, fp, indent=2)
