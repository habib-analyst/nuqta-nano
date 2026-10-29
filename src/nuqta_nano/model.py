"""A compact character-level Transformer encoder-decoder for transliteration.

Standard architecture only — learned token + positional embeddings,
4 encoder and 4 decoder layers, pre-norm off (PyTorch defaults), no
novelty claimed. The whole point of the artifact is the transparent,
reproducible training pipeline around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Seq2SeqConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 192
    n_head: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_src_len: int = 48
    max_tgt_len: int = 40


class TranslitTransformer(nn.Module):
    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=0)
        self.src_pos = nn.Embedding(cfg.max_src_len, cfg.d_model)
        self.tgt_pos = nn.Embedding(cfg.max_tgt_len, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_encoder_layers, enable_nested_tensor=False
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_decoder_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        h = self.drop(self.src_emb(src) + self.src_pos(pos))
        return self.encoder(h, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor,
               src_pad_mask: torch.Tensor, tgt_pad_mask: torch.Tensor) -> torch.Tensor:
        t = tgt_in.size(1)
        pos = torch.arange(t, device=tgt_in.device).unsqueeze(0)
        h = self.drop(self.tgt_emb(tgt_in) + self.tgt_pos(pos))
        causal = torch.full((t, t), float("-inf"), device=tgt_in.device).triu(1)
        out = self.decoder(
            h, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out(self.final_norm(out))

    def forward(self, src, tgt_in, src_pad_mask, tgt_pad_mask):
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt_in, memory, src_pad_mask, tgt_pad_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def greedy_decode(model: TranslitTransformer, src_ids: list[list[int]],
                  bos_id: int, eos_id: int, pad_id: int,
                  max_len: int = 40) -> list[list[int]]:
    """Batched greedy decoding. Deterministic on CPU in eval mode."""
    model.eval()
    device = next(model.parameters()).device
    n = len(src_ids)
    src_len = max(len(s) for s in src_ids)
    src = torch.full((n, src_len), pad_id, dtype=torch.long)
    for i, s in enumerate(src_ids):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    src = src.to(device)
    src_pad = src == pad_id
    memory = model.encode(src, src_pad)

    tgt = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(n)]
    for _ in range(max_len):
        tgt_pad = tgt == pad_id
        logits = model.decode(tgt, memory, src_pad, tgt_pad)
        nxt = logits[:, -1].argmax(dim=-1)
        for i in range(n):
            if not finished[i]:
                tok = int(nxt[i])
                if tok == eos_id:
                    finished[i] = True
                else:
                    outputs[i].append(tok)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    return outputs
