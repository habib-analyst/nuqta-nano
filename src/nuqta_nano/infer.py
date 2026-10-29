"""Load a packaged Nuqta-Nano artifact and transliterate with it.

The artifact directory holds config.json, vocab.json and
model.safetensors (or a checkpoint passed explicitly). Decoding is
greedy and deterministic: the same input always yields the same output.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import Seq2SeqConfig, TranslitTransformer, greedy_decode
from .vocab import BOS_ID, EOS_ID, PAD_ID, VocabPair


class Transliterator:
    def __init__(self, model: TranslitTransformer, vocabs: VocabPair, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.vocabs = vocabs
        self.device = device

    @classmethod
    def from_dir(cls, artifact_dir, device: str = "cpu") -> "Transliterator":
        artifact_dir = Path(artifact_dir)
        cfg = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        vocabs = VocabPair.load(artifact_dir / "vocab.json")
        model_cfg = Seq2SeqConfig(
            src_vocab_size=len(vocabs.src),
            tgt_vocab_size=len(vocabs.tgt),
            d_model=cfg["d_model"], n_head=cfg["n_head"],
            n_encoder_layers=cfg["n_encoder_layers"],
            n_decoder_layers=cfg["n_decoder_layers"],
            d_ff=cfg["d_ff"], dropout=0.0,
            max_src_len=cfg["max_src_len"], max_tgt_len=cfg["max_tgt_len"],
        )
        model = TranslitTransformer(model_cfg)
        weights = artifact_dir / "model.safetensors"
        if weights.exists():
            state = load_file(str(weights))
        else:
            ckpt = torch.load(artifact_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
            state = ckpt["model_state"]
        model.load_state_dict(state)
        return cls(model, vocabs, device=device)

    def transliterate(self, roman: str) -> str:
        """Transliterate one romanized word (or whitespace-separated text).

        Input is lowercased and stripped; multi-word input is handled
        word by word and rejoined with single spaces. Characters outside
        the source vocabulary become UNK and generally degrade the word.
        """
        roman = roman.strip().lower()
        if not roman:
            return ""
        return " ".join(self.transliterate_words(roman.split()))

    def transliterate_words(self, words: list[str]) -> list[str]:
        encoded = [self.vocabs.src.encode(w) for w in words]
        decoded = greedy_decode(
            self.model, encoded, BOS_ID, EOS_ID, PAD_ID,
            max_len=self.model.cfg.max_tgt_len,
        )
        return [self.vocabs.tgt.decode(ids) for ids in decoded]
