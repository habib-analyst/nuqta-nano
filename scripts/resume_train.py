"""Resume Nuqta-Nano training after the VM reboot killed run 1.

Run 1 (scripts/train.py) completed epochs 1-4 and was killed by a VM
reboot during epoch 5 (~20:45 UTC, 2026-10-28). Its best checkpoint
(epoch 4, dev CER 0.2231) survived and is the initialization here.

This resumed run trains epochs 5-12 with a FRESH AdamW optimizer and a
restarted cosine schedule (peak 7e-4, warmup 100 steps, cosine to 5%
over the remaining steps), re-seeded data order, and the same
dev-CER checkpoint selection. These differences from the uninterrupted
run are disclosed here and in REPORT.md — the same recovery protocol
used for Sukhan-Nano. A checkpoint_last.pt is written every epoch so a
second interruption loses at most one epoch.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from nuqta_nano.model import Seq2SeqConfig, TranslitTransformer  # noqa: E402
from nuqta_nano.vocab import PAD_ID, VocabPair  # noqa: E402
from train import BATCH_SIZE, SEED, dev_eval, epoch_order, load_pairs, make_batches  # noqa: E402

EPOCHS = 8           # epochs 5..12
START_EPOCH = 4
PEAK_LR = 7e-4


def main() -> None:
    torch.manual_seed(SEED + 1)   # resumed run: its own seed, disclosed
    torch.set_num_threads(2)
    vocabs = VocabPair.load(PKG / "vocab.json")
    train_pairs = load_pairs("train")
    dev_pairs = load_pairs("dev")
    encoded = [(vocabs.src.encode(r), vocabs.tgt.encode(u)) for r, u in train_pairs]

    base = torch.load(PKG / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = Seq2SeqConfig(**{k: base["config"][k] for k in
                           ("src_vocab_size", "tgt_vocab_size", "d_model", "n_head",
                            "n_encoder_layers", "n_decoder_layers", "d_ff", "dropout",
                            "max_src_len", "max_tgt_len")})
    model = TranslitTransformer(cfg)
    model.load_state_dict(base["model_state"])
    best_cer = base["dev_corpus_cer"]
    best_epoch = base["epoch"]
    print(f"resuming from epoch {best_epoch} (dev CER {best_cer:.4f})", flush=True)

    steps_per_epoch = math.ceil(len(encoded) / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    warmup = 100

    def lr_at(step: int) -> float:
        if step < warmup:
            return PEAK_LR * (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return PEAK_LR * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    gen = torch.Generator().manual_seed(SEED + 1)

    log = open(PKG / "training_log_resumed.txt", "a", encoding="utf-8")
    step = 0
    t_start = time.time()
    for i in range(EPOCHS):
        epoch = START_EPOCH + i + 1
        model.train()
        order = epoch_order(encoded, gen)
        total_loss, total_tok = 0.0, 0
        for src, tgt in make_batches(encoded, order, BATCH_SIZE):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in, src == 0, tgt_in == 0)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ntok = int((tgt_out != PAD_ID).sum())
            total_loss += float(loss.detach()) * ntok
            total_tok += ntok
            step += 1
        train_loss = total_loss / total_tok
        do_eval = epoch % 2 == 0 or i == EPOCHS - 1
        scores = dev_eval(model, vocabs, dev_pairs) if do_eval else None
        record = {"run": "resumed", "epoch": epoch, "train_loss": round(train_loss, 4),
                  "dev_corpus_cer": round(scores["corpus_cer"], 4) if scores else None,
                  "dev_exact_match": round(scores["exact_match"], 4) if scores else None,
                  "elapsed_s": round(time.time() - t_start, 1)}
        line = json.dumps(record)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        torch.save({"model_state": model.state_dict(), "config": vars(cfg),
                    "epoch": epoch, "seed": SEED + 1, "run": "resumed"},
                   PKG / "checkpoint_last.pt")
        if scores and scores["corpus_cer"] < best_cer:
            best_cer = scores["corpus_cer"]
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "config": vars(cfg),
                        "epoch": epoch, "seed": SEED + 1, "run": "resumed",
                        "dev_corpus_cer": best_cer}, PKG / "checkpoint.pt")
            print(f"  new best dev CER {best_cer:.4f} (epoch {epoch}) — checkpoint saved", flush=True)
    log.close()

    # Finalize release artifacts from the best checkpoint (either run).
    ckpt = torch.load(PKG / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    save_file(model.state_dict(), str(PKG / "model.safetensors"), metadata={"format": "pt"})
    config = {"d_model": cfg.d_model, "n_head": cfg.n_head,
              "n_encoder_layers": cfg.n_encoder_layers,
              "n_decoder_layers": cfg.n_decoder_layers, "d_ff": cfg.d_ff,
              "dropout": cfg.dropout, "max_src_len": cfg.max_src_len,
              "max_tgt_len": cfg.max_tgt_len, "direction": "roman->urdu",
              "seed": SEED, "parameters": model.count_parameters(),
              "selected_epoch": ckpt["epoch"],
              "selected_run": ckpt.get("run", "initial"),
              "dev_corpus_cer": ckpt["dev_corpus_cer"]}
    (PKG / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    print(f"done: best dev CER {best_cer:.4f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
