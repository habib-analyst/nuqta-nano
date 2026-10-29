"""Nuqta-Nano: a tiny from-scratch Roman Urdu -> Urdu script transliteration model."""

from .infer import Transliterator
from .metrics import alignment_ops, edit_distance, group_refs, score_predictions
from .model import Seq2SeqConfig, TranslitTransformer
from .vocab import CharVocab, VocabPair

__version__ = "0.1.0"
__all__ = [
    "Transliterator",
    "TranslitTransformer",
    "Seq2SeqConfig",
    "CharVocab",
    "VocabPair",
    "edit_distance",
    "alignment_ops",
    "group_refs",
    "score_predictions",
]
