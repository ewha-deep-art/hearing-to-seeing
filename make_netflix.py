"""Netflix-like plain subtitle:  python make_netflix.py --in x.json --out x_netflix.ass [--font "Malgun Gothic"]"""
import argparse
import json

from hearing_to_seeing.converter.ass import generate_plain_ass
from hearing_to_seeing.schema import Transcript

p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", required=True)
p.add_argument("--out", required=True)
p.add_argument("--font", default="Malgun Gothic", help="font family installed on the viewing PC")
a = p.parse_args()

with open(a.src, encoding="utf-8") as f:
    transcript = Transcript.from_dict(json.load(f))
with open(a.out, "w", encoding="utf-8") as f:
    f.write(generate_plain_ass(transcript, a.font))
print(f"{a.out}: plain white / font={a.font}")
