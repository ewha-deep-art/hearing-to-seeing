"""JSON -> ASS.   uv run python make_ass.py --in x.json --out x.ass"""
import argparse
import json

from hearing_to_seeing.converter.ass import generate_ass
from hearing_to_seeing.schema import Transcript

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--in", dest="src", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

with open(a.src, encoding="utf-8") as f:
    transcript = Transcript.from_dict(json.load(f))
with open(a.out, "w", encoding="utf-8") as f:
    f.write(generate_ass(transcript))
print(f"{a.out} written")
