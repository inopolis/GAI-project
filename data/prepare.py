import os
import re
import argparse
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import numpy as np

from src.utils import ensure_dir, save_json


START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)

# fallback markers if above don't match
ALT_START = re.compile(r"START OF THIS PROJECT GUTENBERG EBOOK", re.IGNORECASE)
ALT_END   = re.compile(r"END OF THIS PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def fetch_gutenberg_text(book_id: int) -> str:
    """
    Try a few standard Gutenberg URLs. Not all IDs work the same way.
    """
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    last_err = None
    for url in candidates:
        try:
            with urlopen(url) as r:
                raw = r.read()
            # Gutenberg texts can be utf-8; sometimes latin-1; try utf-8 then fallback
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
        except (HTTPError, URLError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not download book_id={book_id}. Last error: {last_err}")


def strip_gutenberg_boilerplate(text: str) -> str:
    # Try the common "*** START OF ... ***" blocks first
    m1 = START_RE.search(text)
    m2 = END_RE.search(text)
    if m1 and m2 and m2.start() > m1.end():
        text = text[m1.end():m2.start()]
    else:
        # fallback: line-based markers
        lines = text.splitlines()
        start_i = 0
        end_i = len(lines)
        for i, line in enumerate(lines):
            if ALT_START.search(line):
                start_i = i + 1
                break
        for i in range(len(lines) - 1, -1, -1):
            if ALT_END.search(lines[i]):
                end_i = i
                break
        text = "\n".join(lines[start_i:end_i])

    return text


def normalize_whitespace(text: str) -> str:
    # normalize CRLF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # collapse 3+ newlines to 2 newlines (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing spaces on each line
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    # trim
    return text.strip() + "\n"


def build_vocab(train_text: str):
    chars = sorted(list(set(train_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> np.ndarray:
    # unknown chars should not happen if vocab built from train and val/test cleaned similarly;
    # but just in case, we map unknowns to space.
    unk = stoi.get(" ", 0)
    ids = [stoi.get(ch, unk) for ch in text]
    return np.array(ids, dtype=np.uint16 if len(stoi) < 65535 else np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data_out", help="Output directory")
    ap.add_argument("--book_ids", type=int, nargs="+", required=True, help="Gutenberg book IDs")
    ap.add_argument("--split", type=str, default="train,val,test",
                    help="Comma list of split names; default train,val,test")
    ap.add_argument("--train_books", type=int, default=None, help="How many books for train (book-level split)")
    ap.add_argument("--val_books", type=int, default=1, help="How many books for val (book-level split)")
    ap.add_argument("--test_books", type=int, default=1, help="How many books for test (book-level split)")
    ap.add_argument("--max_chars_per_book", type=int, default=0,
                    help="If >0, truncate each book to this many chars for CPU-friendly runs")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    raw_dir = os.path.join(args.out_dir, "books_raw")
    clean_dir = os.path.join(args.out_dir, "books_clean")
    split_dir = os.path.join(args.out_dir, "splits")
    ensure_dir(raw_dir); ensure_dir(clean_dir); ensure_dir(split_dir)

    # Download + clean each book
    books = []
    for bid in args.book_ids:
        print(f"Downloading {bid} ...")
        txt = fetch_gutenberg_text(bid)
        with open(os.path.join(raw_dir, f"{bid}.txt"), "w", encoding="utf-8") as f:
            f.write(txt)

        txt = strip_gutenberg_boilerplate(txt)
        txt = normalize_whitespace(txt)
        if args.max_chars_per_book and args.max_chars_per_book > 0:
            txt = txt[:args.max_chars_per_book]

        clean_path = os.path.join(clean_dir, f"{bid}.txt")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(txt)

        books.append((bid, txt))

    # Book-level split
    # default: train = remaining after allocating val/test
    n = len(books)
    if args.train_books is None:
        train_books = n - args.val_books - args.test_books
    else:
        train_books = args.train_books

    assert train_books >= 1, "Need at least 1 training book."
    assert train_books + args.val_books + args.test_books <= n, "Not enough books for requested split sizes."

    # deterministic split: keep user-provided order
    train = books[:train_books]
    val = books[train_books:train_books + args.val_books]
    test = books[train_books + args.val_books:train_books + args.val_books + args.test_books]

    train_text = "".join([t for _, t in train])
    val_text = "".join([t for _, t in val])
    test_text = "".join([t for _, t in test])

    stoi, itos = build_vocab(train_text)
    save_json({"stoi": stoi, "itos": itos}, os.path.join(args.out_dir, "vocab.json"))

    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)
    test_ids = encode(test_text, stoi)

    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.out_dir, "val.bin"))
    test_ids.tofile(os.path.join(args.out_dir, "test.bin"))

    meta = {
        "book_ids": args.book_ids,
        "train_books": [bid for bid, _ in train],
        "val_books": [bid for bid, _ in val],
        "test_books": [bid for bid, _ in test],
        "vocab_size": len(stoi),
        "train_chars": int(train_ids.shape[0]),
        "val_chars": int(val_ids.shape[0]),
        "test_chars": int(test_ids.shape[0]),
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))
    print("Done.")
    print(meta)


if __name__ == "__main__":
    main()