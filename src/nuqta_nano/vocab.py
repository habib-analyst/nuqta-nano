"""CharVocab: a character-level vocabulary with fixed special tokens.

Vocabularies are built from the training split only and stored as JSON
({"specials": [...], "chars": [...]}) so they are fully inspectable.
Encoding maps unknown characters to UNK; decoding drops special tokens.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class CharVocab:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {c: i + len(SPECIALS) for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        for i, s in enumerate(SPECIALS):
            self.itos[i] = s

    def __len__(self) -> int:
        return len(SPECIALS) + len(self.chars)

    @classmethod
    def from_texts(cls, texts) -> "CharVocab":
        seen = set()
        for t in texts:
            seen.update(nfc(t))
        return cls(sorted(seen))

    @classmethod
    def from_json(cls, path) -> "CharVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])

    def to_json(self, path) -> None:
        Path(path).write_text(
            json.dumps({"specials": SPECIALS, "chars": self.chars}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def encode(self, text: str, add_specials: bool = True) -> list[int]:
        ids = [self.stoi.get(c, UNK_ID) for c in nfc(text)]
        if add_specials:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids, stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in (PAD_ID, BOS_ID):
                continue
            if i == EOS_ID:
                if stop_at_eos:
                    break
                continue
            out.append(self.itos.get(i, ""))
        return "".join(out)


class VocabPair:
    """Source (roman) + target (Urdu) vocabularies, stored in one JSON file."""

    def __init__(self, src: CharVocab, tgt: CharVocab):
        self.src = src
        self.tgt = tgt

    @classmethod
    def build(cls, src_texts, tgt_texts) -> "VocabPair":
        return cls(CharVocab.from_texts(src_texts), CharVocab.from_texts(tgt_texts))

    @classmethod
    def load(cls, path) -> "VocabPair":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CharVocab(data["src"]["chars"]), CharVocab(data["tgt"]["chars"]))

    def save(self, path) -> None:
        data = {
            "src": {"specials": SPECIALS, "chars": self.src.chars},
            "tgt": {"specials": SPECIALS, "chars": self.tgt.chars},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
