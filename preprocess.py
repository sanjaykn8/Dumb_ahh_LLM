#!/usr/bin/env python3
"""
clean_raw_train.py
Single-tool pipeline to turn a noisy raw text file into data/train.txt.

Usage:
  python clean_raw_train.py --raw raw_train.txt
  python clean_raw_train.py --raw raw_train.txt --out data/train.txt --preview 20

Output:
  data/train.txt  (one cleaned paragraph per line)

Notes:
 - This is aggressive but safe for training a byte-level from-scratch model.
 - It attempts to fix hyphenation across lines, remove headers/footers, remove citations/URLs,
   collapse line-break noise, dedupe, and optionally create input->output pairs.
"""
import re, argparse, html, random, sys
from pathlib import Path

random.seed(42)

# ----------------- Helpers -----------------
def read_file(path):
    return Path(path).read_text(encoding="utf8", errors="ignore")

def norm_unicode(s: str) -> str:
    s = html.unescape(s)
    # normalize newlines
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # remove weird control chars
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    # normalize fancy quotes
    s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    # normalize weird non-breaking spaces
    s = s.replace('\u00A0', ' ')
    return s

def fix_hyphenation_and_linebreaks(s: str) -> str:
    # 1) fix hyphenations across line breaks: "measur-\ned" -> "measured"
    s = re.sub(r'(\w)-\n(\w)', r'\1\2', s)
    # 2) remove single newlines inside paragraphs (preserve paragraph breaks)
    # Convert 3+ newlines to '\n\n' paragraph sep, then convert single newlines to space.
    s = re.sub(r'\n{3,}', '\n\n', s)
    # mark paragraph boundaries
    s = s.replace('\n\n', '<<<PARA>>>')
    s = s.replace('\n', ' ')
    s = s.replace('<<<PARA>>>', '\n\n')
    # 3) remove spurious spaces inserted inside words due to OCR like "mea sured" heuristically
    # Merge tokens where a short token (<=3) is followed by another token and the concatenation is plausible.
    # This heuristic is conservative: only merge when a short token is attached to a following token
    # and the short token length <=3 and next token starts with lowercase.
    def collapse_internal_spaces(par):
        tokens = par.split()
        i = 0
        out = []
        while i < len(tokens):
            t = tokens[i]
            if i+1 < len(tokens) and len(t) <= 3 and re.match(r'^[a-z]', tokens[i+1]):
                # merge t + next
                merged = t + tokens[i+1]
                out.append(merged)
                i += 2
            else:
                out.append(t)
                i += 1
        return " ".join(out)
    # apply to each paragraph
    paras = [p.strip() for p in s.split('\n\n') if p.strip()]
    paras = [collapse_internal_spaces(p) for p in paras]
    return "\n\n".join(paras)

HEADER_FOOTER_PATTERNS = [
    r'^\s*chapter\b', r'^\s*\d+\s+introduction\b', r'^\s*introduction\b',
    r'^\s*abstract\b', r'^\s*keywords\b', r'^\s*references\b', r'^\s*figure\b',
    r'^\s*table\b', r'^\s*image\b', r'^\s*contents\b', r'^\s*acknowledg', r'^\s*copyright',
]

def likely_header_footer(line: str) -> bool:
    ln = line.strip().lower()
    if len(ln) == 0:
        return False
    # page numbers like "12" on their own
    if re.fullmatch(r'\d{1,4}', ln):
        return True
    # lines with too many non-letters
    letters = sum(1 for ch in ln if ch.isalpha())
    if letters / max(1, len(ln)) < 0.25:
        return True
    for pat in HEADER_FOOTER_PATTERNS:
        if re.match(pat, ln):
            return True
    return False

def remove_citations_urls(s: str) -> str:
    # remove [1], [12], (2019), doi:, http(s)://..., footnote markers like ^1, see note X
    s = re.sub(r'\[\s*\d+(?:,\s*\d+)*\s*\]', ' ', s)
    s = re.sub(r'\(\s*\d{4}\s*\)', ' ', s)
    s = re.sub(r'\bdoi:\S+\b', ' ', s, flags=re.I)
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'\bsee note\b.*', ' ', s, flags=re.I)
    s = re.sub(r'\bfigure\s*\d+', ' ', s, flags=re.I)
    s = re.sub(r'\bpage\s*\d+\b', ' ', s, flags=re.I)
    return s

def clean_paragraph(p: str) -> str:
    p = p.strip()
    # remove repeated runs of non-informative chars
    p = re.sub(r'[_\*]{2,}', ' ', p)
    p = re.sub(r'\s+', ' ', p)
    # space before punctuation -> remove
    p = re.sub(r'\s+([,.;:?!%])', r'\1', p)
    # fix stray spaces around hyphens
    p = re.sub(r'\s*-\s*', '-', p)
    # remove leading/trailing punctuation
    p = p.strip(' \n\t"\'-–—')
    return p

def paragraph_is_garbage(p: str) -> bool:
    # too short
    if len(p) < 40:
        return True
    # too few letters
    letters = sum(1 for ch in p if ch.isalpha())
    if letters / max(1, len(p)) < 0.25:
        return True
    # excessive repetition
    if re.search(r'(.)\1{10,}', p):
        return True
    return False

# ----------------- Main pipeline -----------------
def make_train(raw_text: str, out_path: str, pairs=False, max_examples=0, preview=0):
    s = norm_unicode(raw_text)

    # remove some common OCR junk lines early
    s = re.sub(r'[\u200b\u200c\u200d]', '', s)  # zero-width
    s = re.sub(r'^\s*Image\s*$', '', s, flags=re.MULTILINE)
    s = re.sub(r'\uFFFD', '', s)  # replacement char

    s = fix_hyphenation_and_linebreaks(s)
    s = remove_citations_urls(s)

    # split into paragraphs (two newlines)
    raw_paras = [p.strip() for p in re.split(r'\n{2,}', s) if p.strip()]
    cleaned = []
    for p in raw_paras:
        # drop if header/footer-like
        # check each line of paragraph for header patterns
        lines = [ln.strip() for ln in re.split(r'[\n\.]', p) if ln.strip()]
        if any(likely_header_footer(ln) for ln in lines[:2]):  # check start
            continue
        if any(likely_header_footer(ln) for ln in lines[-1:]):  # check end
            # drop trailing header/footer markers
            p = " ".join([ln for ln in lines if not likely_header_footer(ln)])
        p = clean_paragraph(p)
        if paragraph_is_garbage(p):
            continue
        cleaned.append(p)

    # dedupe exact
    seen = set()
    uniq = []
    for p in cleaned:
        k = re.sub(r'\s+', ' ', p).strip()
        if k in seen: 
            continue
        seen.add(k)
        uniq.append(k)

    # optionally create INPUT <|SEP|> OUTPUT pairs using first sentence as prompt
    out_lines = []
    if pairs:
        for p in uniq:
            sents = re.split(r'(?<=[.!?])\s+', p)
            sents = [x.strip() for x in sents if x.strip()]
            if len(sents) < 2:
                continue
            prompt = sents[0]
            output = " ".join(sents[1:])
            out_lines.append(f"INPUT: {prompt} <|SEP|> OUTPUT: {output}")
    else:
        out_lines = uniq

    # shuffle and truncate if requested
    random.shuffle(out_lines)
    if max_examples and max_examples > 0:
        out_lines = out_lines[:max_examples]

    # preview
    if preview:
        for i, ln in enumerate(out_lines[:preview]):
            print(f'[{i+1}] {ln[:1000]}')
        print(f'--- previewed {min(preview,len(out_lines))} / {len(out_lines)} examples ---')
        return None

    # write
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf8') as f:
        for ln in out_lines:
            f.write(ln.strip() + '\n')
    return len(out_lines)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='path to raw text (raw_train.txt)')
    ap.add_argument('--out', default='data/train.txt', help='output train file')
    ap.add_argument('--pairs', action='store_true', help='emit INPUT<|SEP|>OUTPUT pairs')
    ap.add_argument('--max-examples', type=int, default=0)
    ap.add_argument('--preview', type=int, default=0, help='print preview and exit')
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        # try common uploaded path fallback
        alt = Path('/mnt/data/train.txt')
        if alt.exists():
            raw_path = alt
        else:
            print("raw file not found:", args.raw); sys.exit(1)

    raw_text = read_file(raw_path)
    result = make_train(raw_text, args.out, pairs=args.pairs, max_examples=args.max_examples, preview=args.preview)
    if result is not None:
        print(f"Wrote {result} cleaned lines to {args.out}")

if __name__ == '__main__':
    main()
