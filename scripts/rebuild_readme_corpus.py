"""Recover and clean the README corpus that `preprocessing/miner.py` mangled.

`miner.py:27` does `res.write(str(projectRepo.get_readme().decoded_content))`. `str()` of a
`bytes` object returns its *repr*, so every file in `repo_readme/` literally starts with `b'`,
stores newlines as the two characters `\\n`, and stores UTF-8 as `\\xNN` escapes.

With sklearn's default `token_pattern=r"(?u)\\b\\w\\w+\\b"` the damage is large: over a 120-file
sample, 42.9% of the tokens CNBN trains on exist only in the mangled text (`xe5`, `xe6`, `x80`,
`xbc` ... fragments of UTF-8), and every token following a `\\n` escape is fused to a leading `n`
(`\\nInstall` -> `nInstall`). The real vocabulary is 19% larger than the one the model can see.

None of this needs re-mining: the original bytes survive intact inside the repr and round-trip
through `ast.literal_eval`.

On top of the decode, the text is normalised for a bag-of-words classifier: code, badges and
markup carry no topic signal, while URL path segments carry a lot (`github.com/pytorch/pytorch`,
`pypi.org/project/manimgl`), so URLs are split into tokens rather than dropped.

The output goes to a NEW directory. `repo_readme/` is never modified, so the previous results stay
reproducible and the change is a clean A/B.

Usage:
    python scripts/rebuild_readme_corpus.py [--src repo_readme] [--out repo_readme_clean]
                                            [--extra-src DIR ...] [--max-bytes 32768]
                                            [--descriptions repo_descriptions.csv]
"""

import argparse
import ast
import re
import statistics
from pathlib import Path

# Order matters: fenced blocks before inline spans, images before links.
FENCED = re.compile(r"```.*?```", re.S)
INDENTED = re.compile(r"^(?: {4}|\t).*$", re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)\s]*\)")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)\)")
HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
BARE_URL = re.compile(r"https?://\S+")
ENTITY = re.compile(r"&[a-z]{2,10};")
WHITESPACE = re.compile(r"\s+")
# Structural markdown that survives as punctuation noise.
MD_NOISE = re.compile(r"[|*_>#~=+\-]{2,}")


def decode(text):
    """Undo the bytes-repr if present; otherwise return the text unchanged.

    Idempotent, so the script is safe to re-run over its own output.
    """
    stripped = text.lstrip()
    if stripped[:2] in ("b'", 'b"'):
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return text, False
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace"), True
    return text, False


def url_tokens(url):
    """Keep a URL's informative path segments as plain words.

    `https://pypi.org/project/manimgl` -> `pypi org project manimgl`. Dropping URLs outright
    loses topic-indicative names; keeping them whole gives TF-IDF one useless long token.
    """
    url = re.sub(r"^https?://", "", url)
    parts = re.split(r"[/._\-?=&#:,]+", url)
    return " ".join(p for p in parts if len(p) > 1 and not p.isdigit())


def clean(text, level="light"):
    """Normalise markdown for a bag-of-words classifier.

    Two levels, because discarding markup also discards signal. Measured over 300 repos, the
    share of topic labels that still appear literally in the text is:

        decode only  57.9%   |   light  52.1%*  |   strict  51.4%

    (*light retains more than strict at equal token counts because code identifiers and badge
    URLs -- `pip install torch`, `img.shields.io/pypi/v/manimgl` -- name the very topics being
    predicted. `strict` deletes them.)

    light  : keep prose AND code identifiers AND URL path segments; drop only HTML, entities
             and markdown punctuation. This is the default.
    strict : additionally delete fenced/indented/inline code and image markup.
    """
    text = HTML_COMMENT.sub(" ", text)
    if level == "strict":
        text = FENCED.sub(" ", text)
        text = INDENTED.sub(" ", text)
        text = INLINE_CODE.sub(" ", text)
        text = IMAGE.sub(" ", text)
    else:
        # Keep the code's words, drop only the fence/backtick markers.
        text = text.replace("```", " ").replace("`", " ")
        # An image's alt text and URL both name things; keep them as tokens.
        text = IMAGE.sub(lambda m: url_tokens(m.group(0)), text)
    # Link text is prose worth keeping; the target becomes path tokens.
    text = LINK.sub(lambda m: f"{m.group(1)} {url_tokens(m.group(2))}", text)
    text = HTML_TAG.sub(" ", text)
    text = BARE_URL.sub(lambda m: url_tokens(m.group(0)), text)
    text = ENTITY.sub(" ", text)
    text = MD_NOISE.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def load_descriptions(path):
    """repo -> (description, languages) from scripts/extract_descriptions.py."""
    if not path or not Path(path).exists():
        return {}
    import csv
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                out[row[0]] = (row[1].strip(), row[2].strip() if len(row) > 2 else "")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="repo_readme")
    parser.add_argument("--out", default="repo_readme_clean")
    parser.add_argument("--extra-src", action="append", default=[],
                        help="additional source directory; files not already present are added")
    parser.add_argument("--max-bytes", type=int, default=32768,
                        help="truncate cleaned text; 32 KB loses 1.7%% of first topic mentions")
    parser.add_argument("--descriptions", default=None,
                        help="CSV of repo,description to prepend as the first line")
    parser.add_argument("--level", choices=("light", "strict", "decode-only"), default="light",
                        help="how much markup to remove; see clean()")
    parser.add_argument("--description-repeat", type=int, default=0,
                        help="repeat the description N times as a prefix (field up-weighting)")
    parser.add_argument("--with-languages", action="store_true",
                        help="append the repo's language list to the prefix")
    parser.add_argument("--with-name", action="store_true",
                        help="append the owner/name tokens to the prefix")
    parser.add_argument("--no-readme", action="store_true",
                        help="emit ONLY the metadata prefix, discarding the README body; tests "
                             "how much signal the README carries beyond description+languages")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    descriptions = load_descriptions(args.descriptions)

    sources = [(src, False)] + [(Path(d), True) for d in args.extra_src]
    seen = set()
    decoded_count = 0
    added_from_extra = 0
    before_len, after_len, truncated = [], [], 0

    for directory, is_extra in sources:
        if not directory.exists():
            print(f"warning: {directory} does not exist, skipping")
            continue
        # Extra sources may be fold trees (one topic directory per label, the same README
        # copied into each), so walk them recursively and keep the first copy of each name.
        entries = sorted(directory.rglob("*.txt")) if is_extra else sorted(directory.iterdir())
        for path in entries:
            if path.suffix != ".txt" or not path.is_file():
                continue
            # Fold-membership listings, not READMEs.
            if is_extra and re.fullmatch(r"test_\d+\.txt", path.name):
                continue
            if path.name in seen:
                continue
            seen.add(path.name)
            if is_extra:
                added_from_extra += 1

            raw = path.read_text(encoding="utf-8", errors="ignore")
            text, was_encoded = decode(raw)
            decoded_count += was_encoded
            before_len.append(len(text))

            text = text if args.level == 'decode-only' else clean(text, args.level)
            repo = path.stem.replace(",", "/", 1)
            description, languages = descriptions.get(repo, ("", ""))
            # Field up-weighting by repetition. TF-IDF vectors are L2-normalised, so a 63-char
            # description prepended once to an 8 KB README carries no measurable weight;
            # repeating it is the cheap equivalent of a FeatureUnion with a per-field weight,
            # without restructuring load_data (which flattens each document to a plain string
            # before the model ever sees it).
            prefix = []
            if description and args.description_repeat:
                prefix.extend([description] * args.description_repeat)
            if languages and args.with_languages:
                prefix.append(languages)
            if args.with_name:
                prefix.append(repo.replace("/", " ").replace("-", " ").replace("_", " "))
            if args.no_readme:
                text = " ".join(prefix)
            elif prefix:
                text = " ".join(prefix) + " " + text

            if len(text) > args.max_bytes:
                text = text[:args.max_bytes]
                truncated += 1
            after_len.append(len(text))

            (out / path.name).write_text(text, encoding="utf-8")

    print(f"wrote {len(seen):,} files to {out}/")
    print(f"  recovered from bytes-repr : {decoded_count:,}")
    if added_from_extra:
        print(f"  added from --extra-src    : {added_from_extra:,}")
    if descriptions:
        print(f"  descriptions prepended    : "
              f"{sum(1 for n in seen if n[:-4].replace(',', '/', 1) in descriptions):,}")
    print(f"  truncated at {args.max_bytes:,} bytes : {truncated:,}")
    print(f"  median chars {statistics.median(before_len):,.0f} -> "
          f"{statistics.median(after_len):,.0f} "
          f"({statistics.median(after_len) / statistics.median(before_len) * 100:.0f}% retained)")
    short = sum(1 for n in after_len if n < 500)
    print(f"  documents under 500 chars : {short:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
