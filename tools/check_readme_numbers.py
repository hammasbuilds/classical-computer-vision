"""Check that every number quoted in a project README exists in its results.json.

Written because numbers were shipped twice that had been copied from console output of
an older run instead of from the results file, so the README and the record disagreed
and nothing caught it.

Every numeric literal in the README is extracted and matched against the values in
results.json. A README number matches if some value in the JSON equals it when rounded
to the number of decimal places the README used - the same value may legitimately
appear as 0.00123, 0.0012 or 0.1% - or if it equals that value times 100 (a percentage)
or as a plain integer. Anything left over is printed and the script exits non-zero.

    python tools/check_readme_numbers.py projects/60_surface_reconstruction

Structural numbers that are not results (section numbers, "3-D", a project's own
number) are declared in the project's own `readme_numbers_allow.txt`, one literal per
line, so that the exemptions are reviewable rather than invisible.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk(v)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        for m in NUM.finditer(obj):
            try:
                yield float(m.group())
            except ValueError:
                pass


def decimals(tok):
    """Decimal places the literal commits to, counting an exponent.

    1.4e-3 states four decimal places, not one: rounding the stored value to one would
    make 1.4e-3 match 0.0007 as easily as 0.0014.
    """
    mant, _, exp = tok.lower().partition("e")
    d = len(mant.split(".")[1]) if "." in mant else 0
    if exp:
        d -= int(exp)
    return max(d, 0)


def matches(tok, values):
    """Does any value in the results file round to this literal?"""
    try:
        x = float(tok)
    except ValueError:
        return False
    d = decimals(tok)
    for v in values:
        for cand in (v, v * 100.0, v * 1000.0, abs(v)):
            if round(cand, d) == x:
                return True
            # a ratio quoted to one decimal from two stored values is handled by the
            # project storing the ratio; nothing is inferred here on purpose
    return False


def readme_numbers(text):
    """Numbers from the prose and tables, skipping fenced code and image links."""
    out = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)    # images
        line = re.sub(r"\]\([^)]*\)", "] ", line)            # link targets
        line = re.sub(r"`[^`]*`", " ", line)                 # inline code
        for m in NUM.finditer(line):
            out.append((m.group(), line.strip()))
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    proj = os.path.abspath(argv[1])
    res = json.load(io.open(os.path.join(proj, "results", "results.json"),
                            encoding="utf-8"))
    values = sorted(set(walk(res)))
    allow_path = os.path.join(proj, "readme_numbers_allow.txt")
    allow = set()
    if os.path.exists(allow_path):
        for line in io.open(allow_path, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                allow.add(line)
    text = io.open(os.path.join(proj, "README.md"), encoding="utf-8").read()
    bad = []
    n = 0
    for tok, line in readme_numbers(text):
        n += 1
        if tok in allow or matches(tok, values):
            continue
        bad.append((tok, line))
    name = os.path.basename(proj)
    # A one- or two-digit integer will match something in any large results file, so
    # the counts are split: the guarantee is strong for the quoted measurements and
    # weak for small integers, and saying so is better than implying otherwise.
    precise = sum(1 for tok, _ in readme_numbers(text)
                  if tok not in allow and decimals(tok) >= 2)
    if bad:
        print(f"{name}: {len(bad)} of {n} numbers are NOT in results.json")
        for tok, line in bad:
            print(f"  {tok:>12}   {line[:96]}")
        return 1
    print(f"{name}: all {n} README numbers are backed by results.json "
          f"({precise} of them quoted to 2+ decimals, so matched on their full "
          f"precision; {len(values)} distinct values in the file, "
          f"{len(allow)} declared exemptions)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
